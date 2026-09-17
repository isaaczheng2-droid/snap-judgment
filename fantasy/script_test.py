#!/usr/bin/env python3
"""
Does the game-script scenario mix improve the projections on games it has not seen?

Same walk-forward harness as fantasy/backtest.py. For test season S the scenario tables are
built from play-by-play of seasons before S, so nothing leaks. Two feature sets on top of
the shipped model:

  +script      the team's expected pass attempts and rush attempts from the scenario mix,
               and the share of late snaps expected with a lead
  +script*use  the same, multiplied by the player's own share (expected targets / carries)

Reported: fantasy-point MAE by position on the selection seasons and the untouched 2025
holdout, per-stat MAE for the volume stats, and how often the side vs the rolling-median
line flips. Reported whichever way it comes out; kept in the number only if it wins.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_pipeline as rp                                   # noqa: E402
from fantasy import backtest, engine, scoring, script     # noqa: E402


def add_script(pw, sched, tg, rates, tests):
    """Per row: own spread, total and the scenario volume from a model of earlier seasons only."""
    g = sched[["game_id", "home_team", "spread_line", "total_line"]].drop_duplicates("game_id")
    pw = pw.merge(g, on=["game_id", "home_team"], how="left")
    home = pw.team == pw.home_team
    pw["own_spread"] = np.where(home, pw.spread_line, -pw.spread_line)
    pw["game_total"] = pw.total_line
    pw["exp_pass_att"] = np.nan; pw["exp_rush_att"] = np.nan; pw["p_lead_late"] = np.nan
    models = {}
    for s in sorted(pw.season.unique()):
        if s < min(tests):
            continue
        models[s] = script.model_from(tg, rates, (2016, int(s) - 1))
    for s, m in models.items():
        idx = pw.index[pw.season == s]
        vals = [script.scenario(m, sp, tot) for sp, tot in zip(pw.loc[idx, "own_spread"], pw.loc[idx, "game_total"])]
        pw.loc[idx, "exp_pass_att"] = [v["exp_pass_att"] if v else np.nan for v in vals]
        pw.loc[idx, "exp_rush_att"] = [v["exp_rush_att"] if v else np.nan for v in vals]
        pw.loc[idx, "p_lead_late"] = [v["p_lead_late"] if v else np.nan for v in vals]
    for c in ["exp_pass_att", "exp_rush_att", "p_lead_late"]:
        pw[c] = pw[c].fillna(pw[c].median())
    pw["exp_targets"] = pw.exp_pass_att * pw.tgt_share
    pw["exp_carries"] = pw.exp_rush_att * pw.car_share
    return pw


def main():
    sched, team, plyr, inj, snap, rost, depth, cur = rp.load_all("data", None)
    reg = sched[(sched.season == cur) & (sched.game_type == "REG")]
    un = reg[reg.home_score.isna()]
    tw = int(un.week.min()) if len(un) else int(reg.week.max())
    ratings, _ = rp.team_ratings(team, sched, cur, tw)
    pw = rp.player_form(plyr, sched, ratings)
    pw, _ = rp.usage_extras(pw, snap, rost, inj, log=rp.log)
    pw = backtest.annotate(pw, inj, rost, plyr)
    tests = list(range(backtest.SEL[0], int(cur)))
    tg, rates = script.team_games(log=rp.log)
    pw = add_script(pw, sched, tg, rates, tests)
    targets = engine.all_targets(rp.PTARGETS)
    settings = scoring.PRESETS[scoring.DEFAULT]

    base_fn = lambda k: rp.player_feature_set(k, True, targets=targets)[0]
    SETS = {
        "shipped": base_fn,
        "+script": lambda k: base_fn(k) + ["exp_pass_att", "exp_rush_att", "p_lead_late"],
        "+script*use": lambda k: base_fn(k) + ["exp_pass_att", "exp_rush_att", "p_lead_late", "exp_targets", "exp_carries"],
    }
    out = {}
    scored = {}
    for name, fn in SETS.items():
        rows = backtest.walk_forward(pw, targets, tests, alpha=rp.RIDGE_ALPHA, feature_fn=fn, log=rp.log)
        sc = backtest.score_rows(rows, settings, targets)
        scored[name] = sc.set_index(["player_id", "season", "week"])
        res = backtest.evaluate(sc)
        out[name] = {"selection": res["selection"]["all"], "holdout": res["holdout"]["all"],
                     "by_position_selection": {p: b["mae"] for p, b in res["selection"]["by_position"].items()},
                     "by_position_holdout": {p: b["mae"] for p, b in res["holdout"]["by_position"].items()},
                     "rank_holdout": res["rank_quality_holdout"], "startsit_holdout": res["startsit_holdout"]}
    base = scored["shipped"]
    for name, sc in scored.items():
        j = sc.join(base[["proj_pts"]].rename(columns={"proj_pts": "base_pts"}), how="inner")
        out[name]["side_flips_vs_shipped"] = float((np.sign(j.proj_pts - j.naive_pts) != np.sign(j.base_pts - j.naive_pts)).mean())
        for split, seas in [("rel_selection", backtest.SEL), ("rel_holdout", (backtest.HOLDOUT, backtest.HOLDOUT))]:
            jj = j[j.index.get_level_values("season").to_series().between(*seas).values]
            out[name][split] = float((jj.act_pts - jj.proj_pts).abs().mean() / (jj.act_pts - jj.base_pts).abs().mean() - 1)
    json.dump(out, open("data/script_test.json", "w"), indent=1)
    print("\nfantasy-point MAE relative to the shipped model (negative = better); selection 2019-2024 / holdout 2025")
    for name, r in out.items():
        print(f"  {name:<14} {r['rel_selection']:>+8.2%} / {r['rel_holdout']:>+8.2%}   by position (holdout): "
              + "  ".join(f"{p} {v:.3f}" for p, v in r["by_position_holdout"].items())
              + f"   side flips {r['side_flips_vs_shipped']:.1%}")
    print("wrote data/script_test.json")


if __name__ == "__main__":
    main()
