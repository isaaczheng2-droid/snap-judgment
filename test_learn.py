#!/usr/bin/env python3
"""
Checks for the fantasy engine and the learning cycle on synthetic data: scoring under
presets, the promotion gate's four requirements, the ledger's kickoff rule and correction
trail, the start/sit verdict strength, and registry promotion / rollback.
"""
import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from fantasy import scoring, startsit
from learn import gate, ledger, registry

FAILS = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    # ---- scoring
    rb = {"rushing_yards": 80, "rushing_tds": 1, "rb_receptions": 4, "rb_receiving_yards": 30, "rb_receiving_tds": 0}
    check(scoring.points(rb, scoring.PRESETS["full_ppr"], "RB") == 21.0, "Full PPR: 80 rush yds + TD + 4 rec + 30 rec yds = 21.0")
    check(scoring.points(rb, scoring.PRESETS["half_ppr"], "RB") == 19.0, "Half PPR takes 0.5 per catch off (19.0)")
    check(scoring.points(rb, scoring.PRESETS["standard"], "RB") == 17.0, "Standard scores no receptions (17.0)")
    qb = {"passing_yards": 300, "passing_tds": 2, "passing_interceptions": 1, "qb_rushing_yards": 20, "qb_rushing_tds": 0}
    check(scoring.points(qb, scoring.PRESETS["full_ppr"], "QB") == 20.0 and scoring.points(qb, scoring.PRESETS["six_pt_pass_td"], "QB") == 24.0,
          "QB: 12 + 8 - 2 + 2 = 20.0 at 4 per passing TD, 24.0 at 6")
    row = {"passing_yards": 0, "rushing_yards": 100, "rushing_tds": 1, "receptions": 2, "receiving_yards": 10, "rushing_fumbles_lost": 1}
    check(scoring.actual_points(row, scoring.PRESETS["full_ppr"], "RB") == 17.0, "actual points subtract a lost fumble (17.0)")
    check(len(scoring.lineup_slots(scoring.PRESETS["full_ppr"])) == 7, "default lineup fills seven slots")

    # ---- gate
    cfg = {"min_samples": 100, "min_rel_gain": 0.01, "alpha": 0.05, "max_segment_loss": 0.02, "segments": ["position", "season"], "bootstrap": 200}
    rng = np.random.default_rng(1)
    n = 3000
    base = pd.DataFrame({"season": rng.choice([2019, 2020, 2021], n), "week": rng.integers(1, 18, n), "position": rng.choice(["RB", "WR"], n)})
    base["e_cur"] = rng.gamma(2.0, 3.0, n)
    better = base.assign(e_cand=base.e_cur * 0.90)                       # 10% better everywhere
    tiny = base.assign(e_cand=base.e_cur * 0.997)                        # too small
    lopsided = base.assign(e_cand=np.where(base.position == "RB", base.e_cur * 0.85, base.e_cur * 1.06))
    small = better.head(50)
    res = gate.decide([gate.assess(better, cfg), gate.assess(tiny, cfg), gate.assess(lopsided, cfg), gate.assess(small, cfg)], cfg)
    check(res[0]["accepted"], f"a 10% improvement everywhere is accepted ({res[0]['reasons'][0]})")
    check(not res[1]["accepted"] and any("below" in r for r in res[1]["reasons"]), "a 0.3% improvement fails the minimum gain")
    check(not res[2]["accepted"] and any("worse" in r for r in res[2]["reasons"]), "a candidate that hurts WR by 6% is rejected on the segment rule even though it wins overall")
    check(not res[3]["accepted"] and any("paired rows" in r for r in res[3]["reasons"]), "50 rows fail the sample requirement")
    check(res[1]["p_adj"] >= res[1]["p_raw"], "Holm adjustment never lowers a p-value")
    noise = base.assign(e_cand=base.e_cur * rng.uniform(0.9, 1.1, n))
    a = gate.assess(noise, cfg); a = gate.decide([a] * 5, cfg)[0]
    check(not a["accepted"], "noise around the current model is not promoted")

    # ---- ledger: kickoff rule, revisions, actuals, corrections
    tmp = tempfile.mkdtemp()
    fpath, apath = os.path.join(tmp, "f.ndjson"), os.path.join(tmp, "a.ndjson")
    sched = pd.DataFrame([dict(game_id="2026_02_A_B", season=2026, week=2, game_type="REG", home_team="B", away_team="A",
                               gameday=pd.Timestamp("2026-09-20"), gametime="13:00", home_score=np.nan, away_score=np.nan)])
    fantasy = {"season": 2026, "week": 2, "players": [
        {"player_key": "p1", "name": "One", "position": "RB", "team": "A", "opponent": "B", "proj_pts": 12.0, "p_play": 1.0, "availability": {"status": "ok"}, "range": {"full_ppr": {"p10": 4, "p90": 22}}, "proj": {}, "naive_pts": 11.0, "exp_pts": 12.0},
        {"player_key": "p2", "name": "Two", "position": "WR", "team": "B", "opponent": "A", "proj_pts": 9.0, "p_play": 0.6, "availability": {"status": "questionable"}, "range": {"full_ppr": {"p10": 2, "p90": 18}}, "proj": {}, "naive_pts": 8.0, "exp_pts": 5.4}]}
    n1, late1 = ledger.record_forecasts(fantasy, sched, "v1", "d1", now="2026-09-18T12:00:00Z", path=fpath, log=lambda *a: None)
    n2, _ = ledger.record_forecasts(fantasy, sched, "v1", "d1", now="2026-09-19T12:00:00Z", path=fpath, log=lambda *a: None)
    fantasy["players"][0]["proj_pts"] = 14.0
    n3, _ = ledger.record_forecasts(fantasy, sched, "v1", "d2", now="2026-09-19T18:00:00Z", path=fpath, log=lambda *a: None)
    n4, late4 = ledger.record_forecasts(fantasy, sched, "v1", "d3", now="2026-09-20T18:00:00Z", path=fpath, log=lambda *a: None)
    check(n1 == 2 and n2 == 0 and n3 == 1 and n4 == 0 and late4 == 2, f"ledger: 2 first revisions, none when unchanged, 1 when a number moved, none after kickoff ({n1},{n2},{n3},{n4},{late4})")
    lk = ledger.locked(fpath)
    check(lk[(2026, 2, "p1")]["revision"] == 2 and lk[(2026, 2, "p1")]["proj_pts"] == 14.0, "the locked forecast is the last pre-kickoff revision")
    sched.loc[0, ["home_score", "away_score"]] = [24, 20]
    plyr = pd.DataFrame([dict(game_id="2026_02_A_B", season=2026, week=2, season_type="REG", player_id="p1", fantasy_points_ppr=17.3, rushing_yards=90.0)])
    g1, c1 = ledger.record_actuals(plyr, sched, 2026, path=apath, fpath=fpath, now="2026-09-21T09:00:00Z", log=lambda *a: None)
    gr = ledger.graded(fpath, apath)
    check(g1 == 2 and c1 == 0, "both forecasts graded once the game is final and the box score is in")
    p2 = gr[gr.player_key == "p2"].iloc[0]
    check(bool(p2.played) is False and p2.act_pts == 0.0, "a player with no box-score row in a finished game is graded as did-not-play, not silently zero")
    plyr.loc[0, "fantasy_points_ppr"] = 18.1
    g2, c2 = ledger.record_actuals(plyr, sched, 2026, path=apath, fpath=fpath, now="2026-09-23T09:00:00Z", log=lambda *a: None)
    gr2 = ledger.graded(fpath, apath)
    check(g2 == 1 and c2 == 1 and float(gr2[gr2.player_key == "p1"].act_pts.iloc[0]) == 18.1 and bool(gr2[gr2.player_key == "p1"].corrected.iloc[0]),
          "a changed box score is appended as a correction and the latest value wins")
    sched2 = sched.copy(); sched2.loc[0, ["home_score", "away_score"]] = [np.nan, np.nan]
    check(ledger.record_actuals(plyr, sched2, 2026, path=apath, fpath=fpath, log=lambda *a: None)[0] == 0, "an unfinished game is never graded")

    # ---- start/sit strength from the record
    model = {"startsit_holdout": {"1-3 pts": {"higher_scored_more": 0.60, "n": 1000}, "3-5 pts": {"higher_scored_more": 0.70, "n": 1000}, "5+ pts": {"higher_scored_more": 0.82, "n": 1000}}}
    A = {"player_key": "a", "name": "A", "exp_pts": 18.0, "availability": {"status": "ok"}, "drivers": {"form": 3.0, "usage": 1.0}}
    B = {"player_key": "b", "name": "B", "exp_pts": 12.0, "availability": {"status": "ok"}, "drivers": {"form": 0.5, "usage": 1.0}}
    C = {"player_key": "c", "name": "C", "exp_pts": 17.5, "availability": {"status": "ok"}, "drivers": {}}
    check(startsit.compare([A, B], model)["strength"] == "start", "a 6-point gap is a start call")
    check(startsit.compare([A, C], model)["strength"] == "toss-up", "a half-point gap is a toss-up")
    r = startsit.compare([A, dict(B, exp_pts=15.5)], model)
    check(r["strength"] == "toss-up" and "only 60%" in r["verdict"], "a 2.5-point gap at 60% is offered as a toss-up, not a call")
    check(startsit.compare([dict(A, availability={"status": "out"}), B], model)["strength"] == "availability", "an Out player loses by default")

    # ---- registry: promote / rollback
    rpath = os.path.join(tmp, "registry.json")
    class RP: RIDGE_ALPHA = 5.0; EXTRA_FEATS = {"a": ["x"]}; ADJ_SHARES = {}
    reg = registry.init(RP, path=rpath)
    v1 = reg["active"]
    v = registry.add_candidate(reg, {"alpha": 2.0, "extra_feats": {}, "adj_shares": {}, "script_feats": False}, v1, {"rel_gain": 0.02}, "candidate", "test", "exp-1", path=rpath)
    registry.promote(reg, v["id"], "test promotion", path=rpath)
    check(registry.load(rpath)["active"] == v["id"] and registry.load(rpath)["previous"] == v1, "promotion makes the candidate active and remembers the previous version")
    registry.apply(RP, path=rpath, log=lambda *a: None)
    check(RP.RIDGE_ALPHA == 2.0 and RP.EXTRA_FEATS == {}, "apply() points the pipeline at the active parameters")
    registry.rollback(registry.load(rpath), "test", path=rpath)
    registry.apply(RP, path=rpath, log=lambda *a: None)
    check(registry.load(rpath)["active"] == v1 and RP.RIDGE_ALPHA == 5.0 and RP.EXTRA_FEATS == {"a": ["x"]}, "rollback restores the previous version and its parameters")
    shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("all checks passed" if not FAILS else f"{len(FAILS)} FAILED"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
