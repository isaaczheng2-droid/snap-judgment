#!/usr/bin/env python3
"""
Seed history.json with the previous complete season so the tracker has something to show
before the current season has played a game.

These rows come from the walk-forward audit: for season S the model saw only seasons < S.
They are genuine out-of-sample predictions, but they were never published in advance, so
they are written with src="backtest" and the site labels them separately from the live
locks. Run once; the daily pipeline only ever appends.

Usage:  python3 seed_tracker.py [--season 2025] [--out history.json]
"""
import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

import run_pipeline as rp
import tracker
from scheme_features import _per_game, team_scheme, add_scheme_cols

P = dict(max_depth=3, n_estimators=150, learning_rate=0.05, subsample=0.8,
         colsample_bytree=0.8, reg_lambda=2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None, help="season to seed (default: last complete)")
    ap.add_argument("--out", default="history.json")
    ap.add_argument("--datadir", default="data")
    a = ap.parse_args()

    sched, team, plyr, inj, snap, rost, depth, cur = rp.load_all(a.datadir, None)
    season = a.season or (cur - 1)
    rp.log(f"seeding {season}")

    reg = sched[(sched.season == cur) & (sched.game_type == "REG")]
    un = reg[reg.home_score.isna()]
    tw = int(un.week.min()) if len(un) else int(reg.week.max())

    ratings, _ = rp.team_ratings(team, sched, cur, tw)
    pw = rp.player_form(plyr, sched, ratings)
    ctx = rp.context_features(sched, pw, inj, snap, rost, depth, cur)
    df = rp.build_games(sched, ratings, ctx)
    scheme, _ = team_scheme(team, sched, cur, tw)
    df = add_scheme_cols(df, scheme)

    d = df.dropna(subset=["home_win", "home_margin"] + rp.FEATS)
    tr = d[d.season < season]
    te = d[d.season == season].copy()
    if not len(te):
        rp.log(f"no completed {season} games — nothing to seed")
        return
    rp.log(f"train {len(tr)} games (< {season}), predict {len(te)}")

    clf = xgb.XGBClassifier(**P, eval_metric="logloss").fit(tr[rp.FEATS], tr.home_win)
    reg_m = xgb.XGBRegressor(**P).fit(tr[rp.FEATS], tr.home_margin)
    te["p_model"] = clf.predict_proba(te[rp.FEATS])[:, 1]
    te["p_market"] = te.market_home_wp
    te["p_blend"] = np.where(te.p_market.notna(), 0.4 * te.p_model + 0.6 * te.p_market, te.p_model)
    te["margin_pred"] = reg_m.predict(te[rp.FEATS])
    te["predicted_winner"] = np.where(te.p_blend > 0.5, te.home_team, te.away_team)

    h = tracker.load(a.out)

    # ---- games, week by week so the locks carry the right week number ----
    n_games = 0
    for w in sorted(te.week.unique()):
        n_games += tracker.lock_week(h, te[te.week == w], None, None, season, int(w),
                                     source="backtest")

    # ---- player props, walk-forward the same way: fit on everything before this season ----
    n_players = 0
    wk_def_all = ratings.set_index(["season", "week", "team"])
    for out_col, cfg in rp.PTARGETS.items():
        oc = rp.OPPCOL[cfg["opp"]]
        pc = f"proj_{cfg['stat']}"
        sub = pw[pw.position.isin(cfg["pos"])].dropna(subset=[cfg["stat"], pc, oc, "is_home"])
        sub = sub[sub[cfg["vol"]] >= cfg["mn"]]
        fit = sub[sub.season < season]
        tgt = sub[sub.season == season]
        if len(fit) < 200 or not len(tgt):
            continue
        f = [pc, oc, "is_home"]
        model = Ridge(alpha=5.0).fit(fit[f], fit[cfg["stat"]])
        tgt = tgt.copy()
        tgt["proj"] = model.predict(tgt[f])
        done_weeks = tracker.folded_weeks(h, season)
        for _, r in tgt.iterrows():
            if int(r.week) in done_weeks:
                continue                     # already folded into the season aggregate
            k = f"{r.game_id}|{r.player_id}|{out_col}"
            if k in h["players"]:
                continue
            h["players"][k] = {
                "s": int(season), "w": int(r.week), "n": r.get("player_display_name"),
                "t": r.team, "pos": r.position, "st": out_col,
                "proj": round(float(r.proj), 1), "src": "backtest", "at": "seed",
            }
            n_players += 1

    # ---- scheme reads: the rating going into each game, which used only prior games ----
    n_scheme = 0
    sc = scheme[scheme.season == season]
    gid_lookup = sched.set_index("game_id")
    for _, r in sc.iterrows():
        if pd.isna(r.get("game_id")):
            continue
        k = f"{season}_{int(r.week)}_{r.team}"
        if k in h["schemes"]:
            continue
        row = {"team": r.team, "game_id": str(r.game_id), "s": int(season),
               "w": int(r.week), "src": "backtest", "at": "seed"}
        ok = False
        for m in tracker.SCHEME_TRACK:
            v = r.get(f"sch_{m}")
            if v is not None and not pd.isna(v):
                row[m] = round(float(v), 4); ok = True
        if ok:
            h["schemes"][k] = row
            n_scheme += 1

    rp.log(f"locked {n_games} games, {n_players} player props, {n_scheme} scheme reads")
    graded = tracker.grade(h, sched, plyr, _per_game(team, sched))
    dropped = tracker.prune_players(h, cur)
    rp.log(f"graded {graded}, folded {dropped} player rows into season aggregates")
    tracker.save(h, a.out)

    s = tracker.summarize(h, cur)
    g = s["by_season"].get(str(season), {}).get("games", {})
    rp.log(f"{season}: {g.get('n')} games, SU {g.get('su')}, ATS {g.get('ats')}, MAE {g.get('mae')}")


if __name__ == "__main__":
    main()
