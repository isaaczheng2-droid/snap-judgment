#!/usr/bin/env python3
"""
Does the standalone model beat the market? Measured, not asserted.

Walk-forward by season (train strictly before the test season, 8-seed ensemble, the shipped
feature set, no market input anywhere in the model). For each test game the model's p_home is
compared with the closing market's no-vig home probability on the SAME games:

  accuracy      winner picks, model vs market favourite vs always-home
  brier, logloss probability quality
  calibration   reliability table by predicted-probability decile, model and market
  margin MAE    model expected margin vs the closing spread, against the real margin
  paired tests  model minus market on every metric, with a cluster bootstrap over
                season-weeks (games in the same week are not independent) and the p-value
                of the sign test on paired Brier
  slices        by season, by favourite strength, by week band, with sample sizes
  coverage      games dropped and why (no line, missing features), so a gain cannot come
                from hiding hard games

Selection vs holdout, honestly: the game model's feature ladder was chosen with 2019-2025
walk-forward audits in view, so 2025 is reported separately but is NOT an untouched holdout
for the game model (it is for the player models). The only untouched test is the forward
ledger (tracker + prediction_versions), locked before kickoff from 2026 on.

Market benchmark: nflverse `home_moneyline`/`away_moneyline`, a consensus closing price.
De-vig: implied probabilities normalised to sum to one (multiplicative). It is CLOSING
information, i.e. later than a pregame model run; a same-timestamp benchmark needs a
pregame odds archive this project does not have (documented limitation).
Writes data/standalone_eval.json; the site carries it as the fresh key `standalone_eval`.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

import run_pipeline as rp
from regen_accuracy import walk, P
from adjusted_ratings import add_adjusted_cols, team_adjusted
from elo import add_elo_cols
from scheme_features import team_scheme, add_scheme_cols

SELECTION = (2019, 2024)
HOLDOUT = 2025


def ll(p, y):
    pc = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))


def cluster_boot(diff, clusters, B=4000, seed=11):
    """Mean of `diff` with a 95% interval from resampling whole clusters (season-weeks)."""
    rng = np.random.default_rng(seed)
    cl = pd.Series(clusters)
    groups = [np.where(cl.values == c)[0] for c in cl.unique()]
    means = []
    for _ in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        means.append(diff[idx].mean())
    means = np.array(means)
    return {"mean": float(diff.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))), "clusters": int(len(groups))}


def calibration(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        sel = (p >= edges[i]) & (p < edges[i + 1] + (1e-9 if i == bins - 1 else 0))
        if sel.sum() < 15:
            continue
        rows.append({"bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "n": int(sel.sum()), "predicted": float(p[sel].mean()), "actual": float(y[sel].mean())})
    ece = sum(r["n"] * abs(r["predicted"] - r["actual"]) for r in rows) / max(1, sum(r["n"] for r in rows))
    return rows, float(ece)


def block(p, mk, y, sp, am, wk_id, label):
    hit_m = ((p > 0.5).astype(int) == y); hit_k = ((mk > 0.5).astype(int) == y); hit_h = (y == 1)
    bm, bk = (p - y) ** 2, (mk - y) ** 2
    lm, lk = ll(p, y), ll(mk, y)
    out = {"label": label, "n": int(len(y)),
           "model": {"acc": float(hit_m.mean()), "brier": float(bm.mean()), "logloss": float(lm.mean())},
           "market": {"acc": float(hit_k.mean()), "brier": float(bk.mean()), "logloss": float(lk.mean())},
           "always_home": {"acc": float(hit_h.mean())}}
    if len(y) >= 30:
        out["paired"] = {"acc": cluster_boot(hit_m.astype(float) - hit_k.astype(float), wk_id),
                         "brier": cluster_boot(bk - bm, wk_id),     # positive = model better
                         "logloss": cluster_boot(lk - lm, wk_id)}
        b, c = int((hit_m & ~hit_k).sum()), int((~hit_m & hit_k).sum())
        out["mcnemar_p"] = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
        out["picks_only_model_right"], out["picks_only_market_right"] = b, c
    return out


def main():
    sched, team, plyr, inj, snap, rost, depth, cur = rp.load_all("data", None)
    reg = sched[(sched.season == cur) & (sched.game_type == "REG")]
    un = reg[reg.home_score.isna()]
    tw = int(un.week.min()) if len(un) else int(reg.week.max())
    ratings, _ = rp.team_ratings(team, sched, cur, tw)
    pw = rp.player_form(plyr, sched, ratings)
    ctx = rp.context_features(sched, pw, inj, snap, rost, depth, cur)
    df = rp.build_games(sched, ratings, ctx)
    scheme, _ = team_scheme(team, sched, cur, tw)
    df = add_scheme_cols(df, scheme)
    df = add_adjusted_cols(df, team_adjusted(team, sched, cur, tw))
    df = add_elo_cols(df)

    played_all = df[df.home_win.notna() & (df.season < cur)]
    d = played_all.dropna(subset=rp.FEATS + ["home_margin"])          # training uses every prior season
    n_all = int((played_all.season >= 2019).sum())
    n_feat = int((d.season >= 2019).sum())
    tests = list(range(2019, int(cur)))
    p, m, y, mk, sp, am, sn, te = walk(d, rp.FEATS, tests, keep=True)
    wk = te.week.values
    has_mk = ~np.isnan(mk)
    n_mk = int(has_mk.sum())
    coverage = {"played_2019_to_last_season": n_all, "with_complete_features": n_feat,
                "scored_walk_forward": int(len(y)), "with_market_line": n_mk,
                "dropped_no_market": int(len(y) - n_mk), "dropped_missing_features": n_all - n_feat,
                "ties": int((am == 0).sum()), "tie_handling": "a tie is a home non-win for accuracy and Brier; excluded from ATS"}
    # everything below on the identical set of games that have a market line
    P_, M_, Y_, K_, S_, A_, SN_, W_ = p[has_mk], m[has_mk], y[has_mk], mk[has_mk], sp[has_mk], am[has_mk], sn[has_mk], wk[has_mk]
    wk_id = np.array([f"{s}-{w}" for s, w in zip(SN_, W_)])

    sel = (SN_ >= SELECTION[0]) & (SN_ <= SELECTION[1])
    hold = SN_ == HOLDOUT
    res = {"generated": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
           "protocol": {"folds": "train on seasons strictly before the test season; 8-seed ensemble", "features": list(rp.FEATS),
                        "market_in_model": False, "selection_seasons": list(SELECTION), "holdout_season": HOLDOUT,
                        "holdout_caveat": "2025 is reported separately but the game-model feature set was chosen with 2019-2025 audits in view; it is not an untouched holdout. The untouched test is the 2026 forward ledger.",
                        "benchmark": "nflverse closing moneyline, de-vigged multiplicatively; later information than a pregame run",
                        "clusters": "season-week", "bootstrap_draws": 4000},
           "coverage": coverage,
           "all": block(P_, K_, Y_, S_, A_, wk_id, "2019-2025, all games with a line"),
           "selection": block(P_[sel], K_[sel], Y_[sel], S_[sel], A_[sel], wk_id[sel], "2019-2024 (feature selection seasons)"),
           "holdout": block(P_[hold], K_[hold], Y_[hold], S_[hold], A_[hold], wk_id[hold], f"{HOLDOUT} (reported separately; see holdout_caveat)"),
           "by_season": [], "by_favorite": [], "by_week_band": []}
    for s in sorted(set(SN_)):
        i = SN_ == s
        res["by_season"].append(block(P_[i], K_[i], Y_[i], S_[i], A_[i], wk_id[i], str(int(s))))
    fav = np.maximum(K_, 1 - K_)
    for lo, hi, lab in [(0.5, 0.6, "market 50-60%"), (0.6, 0.7, "60-70%"), (0.7, 0.8, "70-80%"), (0.8, 1.01, "80%+")]:
        i = (fav >= lo) & (fav < hi)
        if i.sum() >= 30:
            res["by_favorite"].append(block(P_[i], K_[i], Y_[i], S_[i], A_[i], wk_id[i], lab))
    for lo, hi, lab in [(1, 4, "weeks 1-3"), (4, 9, "weeks 4-8"), (9, 14, "weeks 9-13"), (14, 23, "weeks 14+")]:
        i = (W_ >= lo) & (W_ < hi)
        if i.sum() >= 30:
            res["by_week_band"].append(block(P_[i], K_[i], Y_[i], S_[i], A_[i], wk_id[i], lab))
    cm, em = calibration(P_, Y_); ck, ek = calibration(K_, Y_)
    res["calibration"] = {"model": cm, "model_ece": em, "market": ck, "market_ece": ek,
                          "note": "expected calibration error over deciles with at least 15 games; the model's probabilities are the raw ensemble output, no post-hoc calibrator is fitted"}
    # margin and ATS on the same games
    cov = np.where(A_ > S_, 1, np.where(A_ < S_, 0, -1)); ok = cov >= 0
    res["margin"] = {"model_mae": float(np.abs(M_ - A_).mean()), "market_mae": float(np.abs(S_ - A_).mean()),
                     "paired_mae": cluster_boot(np.abs(S_ - A_) - np.abs(M_ - A_), wk_id),
                     "ats": float((((M_ > S_).astype(int) == cov)[ok]).mean()), "ats_n": int(ok.sum()),
                     "ats_breakeven": 0.5238, "pushes": int((~ok).sum())}
    a = res["all"]
    verdict = ("The standalone model does NOT beat the closing market: it picks fewer winners "
               f"({a['model']['acc']:.1%} vs {a['market']['acc']:.1%}) and its probabilities are worse "
               f"(Brier {a['model']['brier']:.4f} vs {a['market']['brier']:.4f}, log loss {a['model']['logloss']:.4f} vs {a['market']['logloss']:.4f}) "
               f"on {a['n']:,} identical games, and the paired Brier gap's 95% interval "
               f"({a['paired']['brier']['ci95'][0]:+.4f} to {a['paired']['brier']['ci95'][1]:+.4f}) excludes zero.")
    if a["paired"]["brier"]["ci95"][0] > 0:
        verdict = ("The standalone model beats the closing market on probability quality on these games; "
                   "treat with caution until the forward ledger agrees.")
    elif a["paired"]["brier"]["ci95"][1] > 0 and a["paired"]["brier"]["mean"] > 0:
        verdict = "Inconclusive: the model's point estimate is ahead of the market but the interval includes zero."
    res["verdict"] = verdict
    json.dump(res, open("data/standalone_eval.json", "w"), indent=1)
    rp.log(f"n={a['n']} model acc {a['model']['acc']:.4f} brier {a['model']['brier']:.4f} ll {a['model']['logloss']:.4f} | "
           f"market acc {a['market']['acc']:.4f} brier {a['market']['brier']:.4f} ll {a['market']['logloss']:.4f}")
    rp.log(f"paired brier (market - model) {a['paired']['brier']['mean']:+.4f} CI {a['paired']['brier']['ci95']} p {a['paired']['brier']['p_two_sided']:.3f}; "
           f"acc diff {a['paired']['acc']['mean']:+.4f} CI {a['paired']['acc']['ci95']}")
    rp.log(f"holdout {HOLDOUT}: model {res['holdout']['model']['acc']:.4f}/{res['holdout']['model']['brier']:.4f} market {res['holdout']['market']['acc']:.4f}/{res['holdout']['market']['brier']:.4f}")
    rp.log(verdict)


if __name__ == "__main__":
    main()
