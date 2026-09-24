"""
Model Grades v2 + availability impact v2: the regression suite for the failures that
prompted the overhaul. Run: python3 test_grades_impact.py  (needs data/ populated).

Covers, at minimum:
  * offensive linemen actually receive grades (the v1 failure: rosters carry pfr_id for
    0% of OL, so the snap crosswalk silently dropped the whole position group)
  * every ungraded player carries a SPECIFIC reason, never a bare null
  * grades are labelled as ours (model), never as PFF/Madden/ESPN
  * confidence follows sample size and OL never claims HIGH
  * a starting QB ruled OUT is CRITICAL even when the depth chart has demoted him
  * game-membership validation suppresses rows from teams not in the matchup
  * availability rows come out sorted by impact, not alphabetically
"""
import json
import sys

import pandas as pd

import player_grades
from live import impact, context as live_context


PASS = []


def check(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(("ok " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    return cond


def load(pat, years):
    return pd.concat([pd.read_parquet(pat.format(y=y)) for y in years], ignore_index=True)


def test_grades():
    yrs = [2024, 2025, 2026]
    plyr = load("data/stats_player/stats_player_week_{y}.parquet", yrs)
    snap = load("data/snap_counts/snap_counts_{y}.parquet", yrs)
    rost = load("data/rosters/roster_{y}.parquet", yrs)
    team = load("data/stats_team/stats_team_week_{y}.parquet", yrs)
    depth = pd.read_parquet("data/depth/depth_charts_2026.parquet")
    depth["dt"] = pd.to_datetime(depth["dt"], utc=True, errors="coerce")
    pfrd = player_grades.fetch_pfr_def("data", yrs)
    import os
    pfrp = load("data/pfr/pass_{y}.parquet", [y for y in (2025, 2026) if os.path.exists(f"data/pfr/pass_{y}.parquet")])
    pbp = pd.concat([pd.read_parquet(f"data/pbp/play_by_play_{y}.parquet",
                                     columns=["season", "week", "season_type", "penalty", "penalty_player_id"])
                     for y in (2025, 2026)], ignore_index=True)
    rows, meta = player_grades.compute(plyr, 2026, 3, snap=snap, rost=rost, pfr_def=pfrd,
                                       pfr_pass=pfrp, pbp=pbp, team=team, depth=depth)
    by_pool = {}
    for r in rows.values():
        by_pool.setdefault(r["group"], []).append(r)
    graded = lambda p: [r for r in by_pool.get(p, []) if r["grade"] is not None]

    check("OL players receive grades", len(graded("OL")) >= 100, f"got {len(graded('OL'))}")
    check("CB players receive grades", len(graded("CB")) >= 60)
    check("S players receive grades", len(graded("S")) >= 40)
    check("DL players receive grades", len(graded("DL")) >= 100)
    check("K players receive grades", len(graded("K")) >= 20)
    check("QB/RB/WR/TE still grade", all(len(graded(p)) > 30 for p in ("QB", "RB", "WR")))
    ung = [r for r in rows.values() if r["grade"] is None]
    check("every ungraded row has a specific reason", all(r.get("reason") for r in ung))
    check("no reason is a bare N/A", all("n/a" not in str(r.get("reason", "")).lower() for r in ung))
    all_rows = list(rows.values())
    check("no grade is labelled PFF/Madden/ESPN",
          all(str(r.get("source")) == "model" for r in all_rows if r["grade"] is not None)
          and "not a pff" in meta["grade_source_label"].lower())
    g = graded("OL")
    check("OL confidence is never HIGH", all(r["confidence"] != "HIGH" for r in g))
    check("graded rows carry confidence + sample", all(r["confidence"] and r["n"] for r in g))
    hi = [r for r in graded("CB") if r["confidence"] == "HIGH"]
    check("HIGH confidence requires 8+ games", all(r["n"] >= 8 for r in hi))
    check("grades are 0-100 ints", all(isinstance(r["grade"], int) and 0 <= r["grade"] <= 100
                                       for r in all_rows if r["grade"] is not None))
    check("bands accompany grades", all(r.get("band") for r in all_rows if r["grade"] is not None))
    p_rows = by_pool.get("P", [])
    check("punters say WHY they are not graded",
          all(r["grade"] is None and "punting" in str(r["reason"]).lower() for r in p_rows) if p_rows else True)
    check("meta names its sources", meta["sources"]["pfr_def"] is not None and meta["sources"]["snap_counts"] is not None)


def test_impact():
    # the shipped failure: chart demoted the injured starter, v1 said LOW
    r = impact.score({"position": "QB", "normalized_status": "OUT", "depth_order": 2, "is_projected_starter": True})
    check("demoted starting QB OUT is CRITICAL", r["tier"] == "CRITICAL")
    r = impact.score({"position": "QB", "normalized_status": "OUT", "depth_order": 1})
    check("starting QB OUT is CRITICAL", r["tier"] == "CRITICAL")
    r = impact.score({"position": "QB", "normalized_status": "DOUBTFUL", "depth_order": 1})
    check("starting QB DOUBTFUL >= HIGH", r["tier"] in ("HIGH", "CRITICAL"))
    r = impact.score({"position": "QB", "normalized_status": "QUESTIONABLE", "depth_order": 1})
    check("starting QB QUESTIONABLE >= MODERATE", r["tier"] in ("MODERATE", "HIGH", "CRITICAL"))
    r = impact.score({"position": "QB", "normalized_status": "OUT", "depth_order": None})
    check("unknown-depth QB OUT fails UP (>= HIGH)", r["tier"] in ("HIGH", "CRITICAL"))
    r = impact.score({"position": "CB", "normalized_status": "LIMITED_PRACTICE", "depth_order": 3})
    check("depth CB limited stays LOW", r["tier"] == "LOW")
    r = impact.score({"position": "WR", "normalized_status": "OUT", "depth_order": 1, "tgt_share": 0.28})
    check("WR1 (28% targets) OUT >= HIGH", r["tier"] in ("HIGH", "CRITICAL"))
    r = impact.score({"position": "LT", "normalized_status": "OUT", "depth_order": 1})
    check("starting LT OUT >= MODERATE", r["tier"] in ("MODERATE", "HIGH", "CRITICAL"))
    lo = impact.score({"position": "RB", "normalized_status": "OUT", "depth_order": 1, "replacement_gap": -0.05})
    hi = impact.score({"position": "RB", "normalized_status": "OUT", "depth_order": 1, "replacement_gap": 0.30})
    check("replacement gap moves the score", lo["score"] < hi["score"])
    check("score is numeric and explained", 0 <= r["score"] <= 100 and r["reasons"])


def test_game_scoping():
    # a player attached to a game whose teams he does not belong to must be suppressed
    g = {"game_id": "2026_03_A_B", "home_team": "B", "away_team": "A", "home_qb": None, "away_qb": None}
    state = {"players": {
        "p1": {"internal_player_id": "p1", "player_name": "Right Player", "team": "A", "position": "QB",
               "game_id": "2026_03_A_B", "normalized_status": "OUT", "depth_order": 1, "source": "espn"},
        "p2": {"internal_player_id": "p2", "player_name": "Wrong Team Guy", "team": "DAL", "position": "WR",
               "game_id": "2026_03_A_B", "normalized_status": "OUT", "depth_order": 1, "source": "espn"},
        "p3": {"internal_player_id": "p3", "player_name": "Other Game Guy", "team": "A", "position": "RB",
               "game_id": "2026_03_X_Y", "normalized_status": "OUT", "depth_order": 1, "source": "espn"},
        "p4": {"internal_player_id": "p4", "player_name": "Aaa Depth Corner", "team": "B", "position": "CB",
               "game_id": "2026_03_A_B", "normalized_status": "QUESTIONABLE", "depth_order": 3, "source": "espn"},
    }, "kickoffs": {}, "usage": {}}
    cx = live_context.game_context(g, state)
    names = [p["player"] for p in cx["injuryImpact"]["players"]]
    check("player from a non-participating team is suppressed", "Wrong Team Guy" not in names)
    check("player attached to another game is suppressed", "Other Game Guy" not in names)
    check("valid player from a participating team shows", "Right Player" in names)
    check("low-impact questionable depth player is not card noise", "Aaa Depth Corner" not in names)
    rows = cx["injuryImpact"]["players"]
    check("rows sorted by impact, not alphabet", rows and rows[0]["player"] == "Right Player")
    check("card tier is the top driver, never an average", cx["injuryImpact"]["tier"] == "CRITICAL")
    check("drivenBy names the driver", (cx["injuryImpact"]["drivenBy"] or {}).get("player") == "Right Player")
    check("keyAvailability surfaces the critical row",
          any(k["player"] == "Right Player" for k in cx["injuryImpact"]["keyAvailability"]))
    check("row carries role context", rows[0]["role"] == "STARTER")


def test_payload_consistency():
    try:
        p = json.load(open("payload.json"))
    except Exception:
        print("skip payload checks (payload.json not built)")
        return
    G = (p.get("grades") or {}).get("rows") or {}
    ol = [g for g in G.values() if g.get("group") == "OL"]
    check("payload carries graded OL", any(g.get("grade") is not None for g in ol), f"{len(ol)} OL rows")
    for g in p.get("games") or []:
        I = (g.get("live") or {}).get("injuryImpact") or {}
        for r in I.get("players") or []:
            if not check(f"{g['game_id']}: {r['player']} belongs to the matchup",
                         r.get("team") in (g["home_team"], g["away_team"])):
                return
        sc = [r.get("score", 0) for r in I.get("players") or []]
        check(f"{g['game_id']}: availability sorted by impact", sc == sorted(sc, reverse=True))
        qb_out = [r for r in I.get("players") or [] if r.get("position") == "QB"
                  and r.get("role") == "STARTER" and r.get("status") in ("OUT", "INACTIVE", "IR")]
        for r in qb_out:
            check(f"{g['game_id']}: starting QB OUT tier", r.get("tier") == "CRITICAL")


if __name__ == "__main__":
    test_impact()
    test_game_scoping()
    test_grades()
    test_payload_consistency()
    bad = [n for n, ok in PASS if not ok]
    print(f"\n{len(PASS) - len(bad)}/{len(PASS)} checks pass" + (f"; FAILURES: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)
