#!/usr/bin/env python3
"""
Same coaching / scheme factors, player side: does knowing an offense's early-down pass rate,
play-action rate and pressure allowed (or a defence's pressure and blitz rate) improve the
fantasy stat projections? Runs through learn/candidates.run, i.e. the same walk-forward,
row-paired comparison and gate the weekly learning cycle uses. Selection seasons only
(2019-2024); 2025 stays untouched.
"""
import glob
import json
import pandas as pd

import run_pipeline as rp
import test_coaching as tc
from fantasy import backtest
from learn import candidates, evaluate, gate, registry


def main():
    config = json.load(open("learn/config.json"))
    reg = registry.init(rp)
    active = reg["versions"][reg["active"]]
    pw, inj, plyr, cur = backtest.build_rows("data", log=lambda *a: None)
    sched = pd.read_csv("data/games.csv", low_memory=False)
    sched["gameday"] = pd.to_datetime(sched["gameday"])

    # point-in-time team factors, attached to every player-week: own offence and the opponent's defence
    seasons = list(range(2018, int(cur) + 1))
    pfr = [s for s in seasons if glob.glob(f"data/pfr/pass_{s}.parquet")]
    pg = tc.per_game_pbp(seasons).merge(tc.per_game_pfr(pfr, rp.load_all("data", None)[1]), on=["game_id", "team"], how="outer")
    pit = tc.point_in_time(pg, tc.METRICS, sched).drop(columns="season")
    own = pit.rename(columns={"c_edp_pass": "own_edp_pass", "c_pa_rate": "own_pa_rate", "c_press_allow": "own_press_allow"})[["game_id", "team", "own_edp_pass", "own_pa_rate", "own_press_allow"]]
    opp = pit.rename(columns={"team": "opponent_team", "c_press_made": "opp_press_made", "c_blitz_made": "opp_blitz_made"})[["game_id", "opponent_team", "opp_press_made", "opp_blitz_made"]]
    pw = pw.merge(own, on=["game_id", "team"], how="left").merge(opp, on=["game_id", "opponent_team"], how="left")
    for c in ["own_edp_pass", "own_pa_rate", "own_press_allow", "opp_press_made", "opp_blitz_made"]:
        rp.log(f"  {c}: {pw[c].notna().mean():.0%} of player-weeks")

    pass_targets = ["passing_yards", "passing_tds", "passing_interceptions", "qb_rushing_yards", "qb_rushing_tds"]
    rec_targets = ["receiving_yards", "receptions", "receiving_tds", "rb_receiving_yards", "rb_receptions", "rb_receiving_tds"]
    rush_targets = ["rushing_yards", "rushing_tds"]
    specs = [
        {"id": "edp_pass", "note": "own early-down pass rate on every stat",
         "extra_feats_add": {k: ["own_edp_pass"] for k in pass_targets + rec_targets + rush_targets}},
        {"id": "pressure_qb", "note": "opponent pressure rate + own pressure allowed on passing stats",
         "extra_feats_add": {k: ["opp_press_made", "own_press_allow"] for k in pass_targets}},
        {"id": "blitz_qb", "note": "opponent blitz rate on passing stats",
         "extra_feats_add": {k: ["opp_blitz_made"] for k in pass_targets}},
        {"id": "pass_identity_all", "note": "early-down pass rate + pressure, everywhere they apply",
         "extra_feats_add": {**{k: ["own_edp_pass", "opp_press_made", "own_press_allow"] for k in pass_targets},
                             **{k: ["own_edp_pass"] for k in rec_targets + rush_targets}}},
    ]
    # play action only exists from 2022, so it gets its own folds (2023-2024) against the same base
    pa_specs = [{"id": "play_action", "note": "own play-action rate on passing + receiving stats (2022+)",
                 "extra_feats_add": {k: ["own_pa_rate"] for k in pass_targets + rec_targets}}]

    out = {}
    for label, sp, seasons_ in [("2019-2024", specs, config["selection_seasons"]), ("2023-2024", pa_specs, [2023, 2024])]:
        frames, used = candidates.run(rp, pw, sched, config, active["params"], sp, seasons_, log=lambda *a: None)
        used.pop("_skipped", None)
        cur_mae = float((frames["current"].act_pts - frames["current"].proj_pts).abs().mean())
        rp.log(f"{label}: current MAE {cur_mae:.4f} on {len(frames['current'])} rows")
        res = []
        for s in sp:
            j = evaluate.paired(frames["current"], frames[s["id"]])
            a = gate.assess(j, config)
            a["candidate"] = s["id"]; a["note"] = s["note"]
            res.append(a)
        res = gate.decide(res, config)
        for a in res:
            rp.log(f"  {a['candidate']:<18} gain {a['rel_gain']:+.2%}  CI {a['gain_ci95'][0]:+.2%}..{a['gain_ci95'][1]:+.2%}  p_adj {a['p_adj']:.3f}  "
                   f"worst {a['worst_segment']['segment'] if a['worst_segment'] else '-'} {a['worst_segment']['gain'] if a['worst_segment'] else 0:+.2%}  -> {'ACCEPTED' if a['accepted'] else 'rejected: ' + a['reasons'][0]}")
            out[a["candidate"]] = {k: v for k, v in a.items() if k != "segments"} | {"folds": label, "current_mae": cur_mae}
    json.dump(out, open("data/coaching_players_test.json", "w"), indent=1)
    rp.log("wrote data/coaching_players_test.json")


if __name__ == "__main__":
    main()
