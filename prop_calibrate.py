#!/usr/bin/env python3
"""
Calibrate the prop probability, and measure what the feature would ACTUALLY have shown.

Two problems with the raw probability from prop_analysis.py:

  1. It is overconfident, by 2 to 5 points, monotonically worse the more confident it gets.
     It claims 60.8% overall and wins 58.4%. Shipping that number next to a sportsbook's
     implied probability would systematically overstate the edge -- and the edge is the
     entire output of the feature. The game model has the same bias and the site already
     says so in prose; here it can be corrected instead, because the error is smooth and in
     one direction.

  2. Nothing yet measures the subset the feature would actually recommend. Aggregate hit
     rate over 48,000 props is not what a user sees; they see the handful that clear the
     filters. That subset is selected on exactly the quantity being measured, so it needs
     measuring separately or the published number describes a different population than the
     one on screen.

The correction is a single shrink toward 0.5, fitted walk-forward. Deliberately the simplest
thing that can work: one parameter, no shape, nothing that can memorise a season. Isotonic
was tried on the game model and made it worse, which is a reasonable prior for not reaching
for something flexible here.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from prop_oos import build
from test_prop_edge import BREAKEVEN, PRETTY, NOT_OVER_UNDER, half_point, wilson

OU = [k for k in PRETTY if k not in NOT_OVER_UNDER]
REC_MIN_P = 0.65        # the spec's confidence floor



def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_platt(p_raw, won):
    """
    Slope and intercept in log-odds space, chosen by log loss on a coarse grid.

    Two parameters, deliberately. A slope alone cannot move a tail that is wrong in the
    opposite direction to the other one, and anything more flexible would start fitting
    seasons rather than a bias.
    """
    x = logit(p_raw)
    best, ab = 1e18, (1.0, 0.0)
    for a in np.arange(0.30, 1.61, 0.05):
        for b in np.arange(-0.60, 0.61, 0.05):
            q = np.clip(sigmoid(a * x + b), 1e-6, 1 - 1e-6)
            ll = -np.mean(won * np.log(q) + (1 - won) * np.log(1 - q))
            if ll < best:
                best, ab = ll, (float(a), float(b))
    return ab


def fit_scale(p, res):
    X = np.column_stack([np.ones_like(p), p])
    coef, *_ = np.linalg.lstsq(X, np.abs(res), rcond=None)
    return float(coef[0]), float(coef[1])


def scale_of(a, b, p):
    return np.maximum(a + b * np.asarray(p), 1e-6) * np.sqrt(np.pi / 2)


def ece_of(p, won, bins=8):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -1, 2
    tot = 0.0
    for i in range(bins):
        s = (p >= edges[i]) & (p < edges[i + 1])
        if s.sum() < 50:
            continue
        tot += s.sum() * abs(p[s].mean() - won[s].mean())
    return float(tot / len(p))


def oos_probabilities(d):
    """Walk-forward raw probabilities, plus the shrink fitted only on prior seasons."""
    seasons = sorted(d.season.unique())
    rows = []
    for s in seasons:
        hist, cur_ = d[d.season < s], d[d.season == s]
        if len(hist) < 2000 or not len(cur_):
            continue
        # --- raw probability for this season, from prior seasons' residuals ---
        parts = []
        for st in OU:
            h, c = hist[hist.stat == st], cur_[cur_.stat == st]
            if len(h) < 300 or not len(c):
                continue
            res = h.actual.values - h.proj.values
            a, b = fit_scale(h.proj.values, res)
            zs = np.sort(res / scale_of(a, b, h.proj.values))
            thr = (c.line.values - c.proj.values) / scale_of(a, b, c.proj.values)
            p_over = 1.0 - np.searchsorted(zs, thr) / len(zs)
            parts.append(pd.DataFrame({
                "season": s, "stat": st,
                "p_raw": np.where(c.over.values, p_over, 1 - p_over),
                "won": c.won.values, "over": c.over.values,
                "edge_pct": c.edge_pct.values, "proj": c.proj.values,
                "line": c.line.values}))
        if not parts:
            continue

        # --- shrink, fitted on prior seasons only, using THEIR out-of-sample probabilities ---
        k = 1.0
        if len(rows):
            prev = pd.concat(rows, ignore_index=True)
            best = 1e9
            for cand in np.arange(0.30, 1.31, 0.02):
                q = np.clip(0.5 + (prev.p_raw.values - 0.5) * cand, 0.01, 0.99)
                ll = -np.mean(prev.won.values * np.log(q) +
                              (1 - prev.won.values) * np.log(1 - q))
                if ll < best:
                    best, k = ll, float(cand)
        cur_out = pd.concat(parts, ignore_index=True)
        cur_out["k"] = k
        cur_out["p_shrink"] = np.clip(0.5 + (cur_out.p_raw - 0.5) * k, 0.01, 0.99)

        # PER-SIDE calibration, also fitted on prior seasons only. The single shrink above
        # is kept for comparison because it is what shipped first, and because seeing the
        # two side by side is the only reason anyone noticed it was wrong: its aggregate
        # error was 0.80% while it was overstating extreme overs by six points.
        cur_out["p_cal"] = cur_out.p_shrink
        if len(rows):
            prev = pd.concat(rows, ignore_index=True)
            for side in (True, False):
                h = prev[prev.over == side]
                sel = cur_out.over == side
                if len(h) < 500 or not sel.any():
                    continue
                a, b = fit_platt(h.p_raw.values, h.won.values.astype(float))
                cur_out.loc[sel, "p_cal"] = np.clip(
                    sigmoid(a * logit(cur_out.loc[sel, "p_raw"].values) + b), 0.01, 0.99)
        rows.append(cur_out)
    return pd.concat(rows, ignore_index=True)


def table(cal, col, title):
    print(f"\n  {title}")
    print(f"    {'says':<12}{'n':>8}{'claimed':>10}{'actual':>9}{'gap':>8}")
    edges = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01]
    names = ["50-55%", "55-60%", "60-65%", "65-70%", "70-80%", "80%+"]
    for i, nm in enumerate(names):
        q = cal[(cal[col] >= edges[i]) & (cal[col] < edges[i + 1])]
        if len(q) < 100:
            continue
        print(f"    {nm:<12}{len(q):>8}{q[col].mean():>9.1%}{q.won.mean():>9.1%}"
              f"{q.won.mean()-q[col].mean():>+8.1%}")
    print(f"    ECE {ece_of(cal[col].values, cal.won.values):.2%}"
          f"   claims {cal[col].mean():.1%}, wins {cal.won.mean():.1%}")


def main():
    d = build()
    d = d[d.stat.isin(OU)].copy()
    d["line"] = half_point(d["median"])
    d["over"] = d.proj > d.line
    d["won"] = np.where(d.over, d.actual > d.line, d.actual < d.line)
    d["edge_pct"] = (d.proj - d.line).abs() / d.line.clip(lower=0.5)

    cal = oos_probabilities(d)
    print("=" * 80)
    print("CALIBRATION, walk-forward")
    table(cal, "p_raw", "RAW -- straight off the residual distribution")
    table(cal, "p_shrink", "ONE GLOBAL SHRINK -- what shipped first")
    table(cal, "p_cal", "PER SIDE -- what ships now")
    print(f"\n  shrink chosen per season: "
          + "  ".join(f"{int(s)}:{cal[cal.season==s].k.iloc[0]:.2f}"
                      for s in sorted(cal.season.unique())))

    print("\n" + "=" * 80)
    print("WHAT THE FEATURE WOULD ACTUALLY HAVE SHOWN")
    print(f"The filters only surface props at {REC_MIN_P:.0%} confidence or better. That subset")
    print("is selected on the thing being measured, so it gets measured on its own.\n")
    rec = cal[cal.p_cal >= REC_MIN_P]
    lo, hi = wilson(int(rec.won.sum()), len(rec))
    print(f"  {'recommended props':<28}{len(rec):>8}  ({len(rec)/len(cal):.0%} of all props)")
    print(f"  {'they won':<28}{rec.won.mean():>8.1%}  (95% CI {lo:.1%}-{hi:.1%})")
    print(f"  {'they claimed':<28}{rec.p_cal.mean():>8.1%}")
    print(f"  {'break-even at -110':<28}{BREAKEVEN:>8.1%}")
    print(f"  {'everything else':<28}{cal[cal.p_cal < REC_MIN_P].won.mean():>8.1%}")

    share_u = 1 - rec.over.mean()
    print(f"\n  {'share of them on the UNDER':<28}{share_u:>8.1%}")
    if share_u > 0.75:
        print("    ^ the filters almost only ever fire on unders. Worth saying on the page:")
        print("      it is the shape of these distributions as much as the model.")
    for side, s in [("over", rec[rec.over]), ("under", rec[~rec.over])]:
        if len(s) < 50:
            continue
        l2, h2 = wilson(int(s.won.sum()), len(s))
        print(f"  {'  recommended ' + side + 's':<28}{len(s):>8}  won {s.won.mean():.1%}"
              f"  (CI {l2:.1%}-{h2:.1%})")

    print("\n  By stat, among recommended:")
    print(f"    {'':<20}{'n':>7}{'won':>8}{'claimed':>10}")
    for st in OU:
        q = rec[rec.stat == st]
        if len(q) < 50:
            continue
        print(f"    {PRETTY[st]:<20}{len(q):>7}{q.won.mean():>7.1%}{q.p_cal.mean():>10.1%}")

    print("\n" + "=" * 80)
    print("THE CAVEAT THAT HAS TO TRAVEL WITH THESE NUMBERS")
    nu = (d.actual < d.line).mean()
    print(f"  Betting the under blindly on every one of these lines returns {nu:.1%},")
    print(f"  comfortably past the {BREAKEVEN:.1%} break-even. A real sportsbook does not")
    print("  leave that lying there. So this line is soft, every number above is measured")
    print("  against a soft line, and none of it demonstrates an edge over FanDuel.")

    out = {
        "n_all": int(len(cal)), "hit_all": float(cal.won.mean()),
        "hit_rest": float(cal[cal.p_cal < REC_MIN_P].won.mean()),
        "n_rec": int(len(rec)), "hit_rec": float(rec.won.mean()),
        "claimed_rec": float(rec.p_cal.mean()),
        "ci_rec": [float(lo), float(hi)],
        "under_share_rec": float(share_u),
        "ece_raw": ece_of(cal.p_raw.values, cal.won.values),
        "ece_cal": ece_of(cal.p_cal.values, cal.won.values),
        "calibration": "per-side-platt",
        "shrink_legacy": float(cal.k.iloc[-1]),
        "null_under": float(nu), "breakeven": BREAKEVEN,
        "by_side_rec": {s: {"n": int(len(g)), "hit": float(g.won.mean())}
                        for s, g in [("over", rec[rec.over]), ("under", rec[~rec.over])]},
        "seasons": [int(cal.season.min()), int(cal.season.max())],
    }
    json.dump(out, open("data/prop_verdict.json", "w"), indent=1)
    print("\nwrote data/prop_verdict.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
