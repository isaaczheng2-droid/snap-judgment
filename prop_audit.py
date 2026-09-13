#!/usr/bin/env python3
"""
Audit of the player-prop backtest: rebuild the walk-forward dataset WITH the identifying
columns the cached one dropped (week, player, game, prior games, actual volume), then
answer, with numbers, the questions the published figures skate over:

  - how the synthetic line was built and what it actually is early in a season
  - whether pushes exist and how they were scored
  - how many "outcomes" are partial games or non-appearances
  - why the blind under wins, and whether it is the distribution or the sample
  - what happens to the recommended subset once those are handled

Writes data/prop_oos_ext.parquet (cached) and data/prop_audit.json, prints the tables.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

import run_pipeline as rp
from test_prop_edge import half_point, NOT_OVER_UNDER, BREAKEVEN

CACHE = "data/prop_oos_ext.parquet"
OU = [k for k in rp.PTARGETS if k not in NOT_OVER_UNDER]
VOLCOL = {"passing_yards": "attempts", "qb_rushing_yards": "attempts", "rushing_yards": "carries",
          "rb_receiving_yards": "carries", "receiving_yards": "targets", "receptions": "targets"}


def build(force=False):
    if os.path.exists(CACHE) and not force:
        return pd.read_parquet(CACHE)
    sched, team, plyr, inj, snap, rost, depth, cur = rp.load_all("data", None)
    reg = sched[(sched.season == cur) & (sched.game_type == "REG")]
    un = reg[reg.home_score.isna()]
    tw = int(un.week.min()) if len(un) else int(reg.week.max())
    ratings, _ = rp.team_ratings(team, sched, cur, tw)
    pw = rp.player_form(plyr, sched, ratings).sort_values(["player_id", "season", "week"])
    for key, cfg in rp.PTARGETS.items():
        st = cfg["stat"]
        pw[f"med_{st}"] = (pw.groupby(["player_id", "season"])[st]
                           .transform(lambda x: x.shift(1).expanding().median()))
        pw[f"n_{st}"] = pw.groupby(["player_id", "season"])[st].cumcount()
    tests = list(range(2019, int(cur)))
    out = []
    keep = ["season", "week", "game_id", "player_id", "player_display_name", "position", "team",
            "opponent_team", "is_home", "gp_prior"]
    for key, cfg in rp.PTARGETS.items():
        oc, pc = rp.OPPCOL[cfg["opp"]], f"proj_{cfg['stat']}"
        med, nst = f"med_{cfg['stat']}", f"n_{cfg['stat']}"
        feats = [pc, oc, "is_home"] + rp.USAGE
        # the median is only needed for the synthetic-line audit; week-1 rows (no same-season
        # median yet) stay in so the real-market backtest can use their projections
        sub = pw[pw.position.isin(cfg["pos"])].dropna(subset=[cfg["stat"]] + feats)
        sub = sub[sub[cfg["vol"]] >= cfg["mn"]]
        if len(sub) < 500:
            continue
        vol = VOLCOL.get(key, "attempts")
        for s in tests:
            tr, te = sub[sub.season < s], sub[sub.season == s]
            if len(tr) < 300 or not len(te):
                continue
            m = Ridge(alpha=5.0).fit(tr[feats], tr[cfg["stat"]])
            d = te[[c for c in keep if c in te.columns]].copy()
            d["stat"] = key
            d["proj"] = m.predict(te[feats])
            d["mean"] = te[pc].values
            d["median"] = te[med].values
            d["n_prior"] = te[nst].values
            d["actual"] = te[cfg["stat"]].values
            d["vol_actual"] = te[vol].values if vol in te.columns else np.nan
            d["vol_proj"] = te[cfg["vol"]].values
            out.append(d)
        rp.log(f"  built {key}")
    a = pd.concat(out, ignore_index=True)
    a.to_parquet(CACHE)
    return a


def wilson(k, n):
    return stats.binomtest(int(k), int(n), 0.5).proportion_ci(0.95) if n else (np.nan, np.nan)


def main():
    d = build()
    d = d[d.stat.isin(OU)].dropna(subset=["median"]).copy()
    d["line"] = half_point(d["median"])
    d["integer_line"] = (d.line % 1 == 0)
    d["push"] = d.actual == d.line
    d["under_wins"] = d.actual < d.line
    d["over"] = d.proj > d.line
    d["won_as_scored"] = np.where(d.over, d.actual > d.line, d.actual < d.line)
    d["no_show"] = d.vol_actual.fillna(0) <= 0
    d["partial"] = (d.vol_actual.fillna(0) > 0) & (d.vol_actual < 0.35 * d.vol_proj)
    A = {}

    print("=" * 84)
    print("1. THE SYNTHETIC LINE")
    print("  line = half_point(expanding median of the player's SAME-SEASON prior games)")
    print("  half_point(x) = floor(2x)/2 + 0.5, which turns a half-integer median into a WHOLE number")
    nb = d.groupby(pd.cut(d.n_prior, [0, 1, 2, 3, 5, 8, 20], right=True)).agg(
        n=("actual", "size"), under=("under_wins", "mean"), push=("push", "mean"),
        int_line=("integer_line", "mean"))
    print("\n  by number of prior games the line is built on:")
    print(nb.to_string(float_format=lambda v: f"{v:.3f}"))
    A["by_prior_games"] = {str(k): {c: float(v) for c, v in r.items()} for k, r in nb.iterrows()}
    A["share_integer_lines"] = float(d.integer_line.mean())
    A["share_push"] = float(d.push.mean())
    print(f"\n  integer (pushable) lines: {d.integer_line.mean():.1%} of rows; pushes: {d.push.mean():.2%}"
          f" ({int(d.push.sum())} rows), scored as a LOSS for whichever side the model took")

    print("\n" + "=" * 84)
    print("2. MISSING AND PARTIAL APPEARANCES")
    print(f"  rows with zero recorded volume (attempts/carries/targets = 0): {d.no_show.mean():.2%} ({int(d.no_show.sum())})")
    print(f"  rows with < 35% of projected volume (left early / role change): {d.partial.mean():.2%} ({int(d.partial.sum())})")
    print(f"  blind under among no-shows: {d[d.no_show].under_wins.mean():.1%}; among partials: {d[d.partial].under_wins.mean():.1%}")
    print("  (nflverse only has a row for a player who appeared; a true DNP is absent, a 5-snap injury is a row of zeros)")
    A["no_show_share"] = float(d.no_show.mean()); A["partial_share"] = float(d.partial.mean())

    print("\n" + "=" * 84)
    print("3. WHY THE UNDER WINS")
    clean = d[~d.no_show & ~d.partial & ~d.push]
    print(f"  blind under, all rows:                 {d.under_wins.mean():.1%}  (n={len(d)})")
    print(f"  blind under, pushes excluded:          {d[~d.push].under_wins.mean():.1%}")
    print(f"  blind under, no-shows/partials/pushes excluded: {clean.under_wins.mean():.1%}  (n={len(clean)})")
    A["blind_under_all"] = float(d.under_wins.mean()); A["blind_under_clean"] = float(clean.under_wins.mean())
    # mean reversion: the line is the player's own recent median; compare actual to it by
    # how far the median sits above the player's projection (regression to the mean)
    d["gap"] = (d["median"] - d["mean"]) / d["mean"].clip(lower=1)
    gb = d[~d.push].groupby(pd.qcut(d["gap"], 5, duplicates="drop")).agg(n=("actual", "size"), under=("under_wins", "mean"))
    print("\n  blind under by (median - rolling mean)/mean quintile: a median far above the mean is a soft over")
    print(gb.to_string(float_format=lambda v: f"{v:.3f}"))
    A["under_by_gap_quintile"] = {str(k): {c: float(v) for c, v in r.items()} for k, r in gb.iterrows()}
    # symmetric check: if it were only skew, the under should win about the same at every prior-game count
    sk = d[~d.push].groupby("stat").agg(n=("actual", "size"), under=("under_wins", "mean"),
                                        skew=("actual", lambda x: float(stats.skew(x))),
                                        mean_over_median=("actual", lambda x: float(x.mean() / max(np.median(x), 1e-9))))
    print("\n  by stat: under rate vs skew of the outcome")
    print(sk.to_string(float_format=lambda v: f"{v:.3f}"))
    A["under_by_stat"] = {k: {c: float(v) for c, v in r.items()} for k, r in sk.iterrows()}
    # by season
    ss = d[~d.push].groupby("season").agg(n=("actual", "size"), under=("under_wins", "mean"))
    print("\n  by season"); print(ss.to_string(float_format=lambda v: f"{v:.3f}"))
    A["under_by_season"] = {int(k): {c: float(v) for c, v in r.items()} for k, r in ss.iterrows()}

    print("\n" + "=" * 84)
    print("4. THE RECOMMENDED SUBSET, RE-SCORED")
    # replicate prop_calibrate's walk-forward probabilities, then re-score with pushes voided
    from prop_calibrate import oos_probabilities, REC_MIN_P
    dd = d.copy()
    dd["won"] = dd.won_as_scored
    dd["edge_pct"] = (dd.proj - dd.line).abs() / dd.line.clip(lower=0.5)
    # oos_probabilities needs season/stat/line/proj/over/won/actual columns; carry the row ids through
    dd["_row"] = np.arange(len(dd))
    cal = oos_probabilities(dd)
    # re-attach the audit flags by position (oos_probabilities preserves order within season/stat)
    parts = []
    for s in sorted(cal.season.unique()):
        for st in OU:
            src = dd[(dd.season == s) & (dd.stat == st)]
            tgt = cal[(cal.season == s) & (cal.stat == st)]
            if len(src) != len(tgt):
                continue
            t = tgt.copy(); t["_row"] = src["_row"].values; parts.append(t)
    cal = pd.concat(parts, ignore_index=True).merge(
        dd[["_row", "push", "no_show", "partial", "week", "game_id", "player_id", "n_prior"]], on="_row")
    rec = cal[cal.p_cal >= REC_MIN_P]
    def rate(x):
        return (float(x.won.mean()), len(x))
    r_all = rate(rec)
    r_np = rate(rec[~rec.push])
    r_clean = rate(rec[~rec.push & ~rec.no_show & ~rec.partial])
    print(f"  as published (pushes = losses):            {r_all[0]:.1%}  n={r_all[1]}")
    print(f"  pushes voided:                              {r_np[0]:.1%}  n={r_np[1]}")
    print(f"  pushes voided, no-shows/partials removed:   {r_clean[0]:.1%}  n={r_clean[1]}")
    print(f"  under share of recommendations:             {1 - rec.over.mean():.1%}")
    e = rec[~rec.push]
    print(f"  recommended unders won {e[~e.over].won.mean():.1%} (n={int((~e.over).sum())}); overs {e[e.over].won.mean():.1%} (n={int(e.over.sum())})")
    # by prior games: how much of the record comes from lines built on 1-2 games
    pg = e.groupby(pd.cut(e.n_prior, [0, 1, 2, 3, 5, 8, 20])).agg(n=("won", "size"), won=("won", "mean"), under=("over", lambda x: 1 - x.mean()))
    print("\n  recommended props by number of prior games behind the line:")
    print(pg.to_string(float_format=lambda v: f"{v:.3f}"))
    A["rec_as_published"] = r_all; A["rec_pushes_void"] = r_np; A["rec_clean"] = r_clean
    A["rec_by_prior_games"] = {str(k): {c: float(v) for c, v in r.items()} for k, r in pg.iterrows()}
    # cluster-robust CI: props from the same game are not independent; count games, not props
    g = e.groupby("game_id").won.agg(["sum", "size"])
    p_hat = g["sum"].sum() / g["size"].sum()
    # ratio estimator variance across clusters
    m = len(g); resid = g["sum"] - p_hat * g["size"]
    se = np.sqrt((resid ** 2).sum() / (m - 1) * m) / g["size"].sum()
    print(f"\n  cluster-robust (by game) 95% CI on the recommended hit rate: {p_hat:.1%} ± {1.96*se:.1%}"
          f"  ({m} games, {len(e)} props, {len(e)/m:.1f} per game)")
    A["rec_cluster_ci"] = [float(p_hat - 1.96 * se), float(p_hat + 1.96 * se)]
    A["rec_games"] = int(m)

    print("\n" + "=" * 84)
    print("5. WHAT THE SHIPPED MODEL FILE WAS FITTED ON")
    m = json.load(open("data/prop_model.json"))
    print("  data/prop_model.json residual quantiles: n per stat =", {k: v["n"] for k, v in m["stats"].items()})
    print("  those counts equal the FULL 2019-2025 walk-forward set, so the shipped scale and the")
    print("  per-side calibration were fitted with 2025 results included. The 2025 rows of the")
    print("  published backtest are therefore not an untouched test of the shipped parameters.")
    A["shipped_model_fit_seasons"] = "2019-2025 (includes the last evaluated season)"
    print("\n" + "=" * 84)
    print("6. BASELINES ON THE IDENTICAL OPPORTUNITIES (pushes voided)")
    # the same rows the model recommended, scored three other ways
    e = e.merge(dd[["_row", "mean", "median", "actual", "line"]], on="_row", suffixes=("", "_d"))
    e["hist_over"] = e["mean"] > e["line"]                       # player-history rule: rolling mean vs the line
    e["hist_won"] = np.where(e.hist_over, e.actual > e.line, e.actual < e.line)
    e["under_won"] = e.actual < e.line
    e["over_won"] = e.actual > e.line
    rows = [("model (as recommended)", e.won.mean()), ("always under", e.under_won.mean()),
            ("always over", e.over_won.mean()), ("player history: rolling mean vs line", e.hist_won.mean())]
    for nm, v in rows:
        print(f"  {nm:<40}{v:>8.1%}   n={len(e)}")
    A["baselines_on_recommended"] = {nm: float(v) for nm, v in rows}
    agree = (e.hist_over == e.over).mean()
    print(f"  the model and the rolling-mean rule pick the same side {agree:.1%} of the time")
    A["model_vs_hist_agreement"] = float(agree)
    # and on ALL eligible rows
    allr = cal[~cal.push].merge(dd[["_row", "mean", "median", "actual", "line"]], on="_row", suffixes=("", "_d"))
    allr["hist_over"] = allr["mean"] > allr["line"]
    allr["hist_won"] = np.where(allr.hist_over, allr.actual > allr.line, allr.actual < allr.line)
    print(f"\n  all eligible rows: model {allr.won.mean():.1%}, always-under {(allr.actual < allr.line).mean():.1%}, "
          f"rolling-mean rule {allr.hist_won.mean():.1%}  (n={len(allr)})")
    A["baselines_all"] = {"model": float(allr.won.mean()), "always_under": float((allr.actual < allr.line).mean()),
                          "hist_rule": float(allr.hist_won.mean()), "n": int(len(allr))}

    print("\n" + "=" * 84)
    print("7. CHRONOLOGICAL TEST: 2025 SCORED WITH PARAMETERS FITTED ON 2019-2024 ONLY (pushes voided)")
    print("  (the walk-forward already refits scale and calibration per season on prior seasons; the")
    print("   0.65 confidence floor was fixed in advance; nothing below was tuned on 2025)")
    t = e[e.season == 2025]
    lo, hi = wilson(int(t.won.sum()), len(t))
    print(f"  recommended 2025: {t.won.mean():.1%} (95% CI {lo:.1%}-{hi:.1%}), n={len(t)}, unders {1-t.over.mean():.0%},"
          f" always-under on the same rows {t.under_won.mean():.1%}, rolling-mean rule {t.hist_won.mean():.1%}")
    bys = t.groupby(["stat", "over"]).agg(n=("won", "size"), won=("won", "mean"), claimed=("p_cal", "mean"), under_base=("under_won", "mean"))
    print(bys.to_string(float_format=lambda v: f"{v:.3f}"))
    A["test_2025"] = {"n": int(len(t)), "won": float(t.won.mean()), "ci": [float(lo), float(hi)],
                      "always_under": float(t.under_won.mean()), "hist_rule": float(t.hist_won.mean()),
                      "by_stat_side": {f"{k[0]}|{'over' if k[1] else 'under'}": {c: float(v) for c, v in r.items()} for k, r in bys.iterrows()}}
    # calibration by probability band on the 2025 test rows, per side
    print("\n  2025 calibration by claimed band and side:")
    for side, nm in [(True, "over"), (False, "under")]:
        q = t[t.over == side]
        for lo_b, hi_b in [(0.65, 0.7), (0.7, 0.8), (0.8, 1.01)]:
            b = q[(q.p_cal >= lo_b) & (q.p_cal < hi_b)]
            if len(b) >= 30:
                print(f"    {nm:<6}{lo_b:.2f}-{min(hi_b,1):.2f}  n={len(b):>5}  claimed {b.p_cal.mean():.1%}  won {b.won.mean():.1%}")
    json.dump(A, open("data/prop_audit.json", "w"), indent=1, default=str)
    print("\nwrote data/prop_audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
