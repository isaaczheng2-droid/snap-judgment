#!/usr/bin/env python3
"""
Checks for the scheme & coaching block: the staff file is complete and every photo carries a
licence, the season table ranks and counts correctly on synthetic games, and the style
sentences only speak when there is a number behind them.
"""
import json
import pandas as pd

import coaching

FAILS = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    staff = json.load(open("coaches.json"))
    teams = staff["teams"]
    check(len(teams) == 32, f"coaches.json covers 32 teams ({len(teams)})")
    check(all(all(r in t and t[r].get("name") for r in ("HC", "OC", "DC")) for t in teams.values()), "every team has HC, OC and DC with a name")
    photos = [t[r]["photo"] for t in teams.values() for r in ("HC", "OC", "DC") if t[r].get("photo")]
    check(all(p.get("url", "").startswith("https://upload.wikimedia.org/") and p.get("license") and p.get("page") for p in photos),
          f"all {len(photos)} photos are Wikimedia Commons files with a licence and a source page")
    check(all(t["HC"].get("since") and t["HC"].get("career") for t in teams.values()), "every head coach has a start year and a career record")
    check(all(len(t[r].get("bio", "")) >= 40 for t in teams.values() for r in ("HC", "OC", "DC")), "every coach has a bio")

    # synthetic per-game table: 3 teams, 2 games each
    sched = pd.DataFrame([dict(game_id=g, season=2026, week=w, game_type="REG") for g, w in [("g1", 1), ("g2", 1), ("g3", 2), ("g4", 2)]])
    pg = pd.DataFrame([
        dict(game_id="g1", team="A", edp_pass=0.6, pa_rate=0.3, motion=0.5, press_allow=0.2, go4_go=1, go4_n=2, press_made=0.3, blitz_made=0.2, sim_rate=0.05),
        dict(game_id="g3", team="A", edp_pass=0.4, pa_rate=0.1, motion=0.5, press_allow=0.2, go4_go=0, go4_n=1, press_made=0.3, blitz_made=0.2, sim_rate=0.05),
        dict(game_id="g1", team="B", edp_pass=0.5, pa_rate=None, motion=0.6, press_allow=0.3, go4_go=0, go4_n=0, press_made=0.2, blitz_made=0.3, sim_rate=0.02),
        dict(game_id="g2", team="C", edp_pass=0.3, pa_rate=0.2, motion=0.4, press_allow=0.1, go4_go=2, go4_n=2, press_made=0.4, blitz_made=0.1, sim_rate=0.10),
    ])
    rows = coaching._season_table(pg, sched, 2026)
    check(rows["A"]["games"] == 2 and abs(rows["A"]["edp_pass"]["value"] - 0.5) < 1e-9, "season table averages a team's games")
    check(rows["A"]["go4"]["go"] == 1 and rows["A"]["go4"]["n"] == 3, "4th-down decisions are summed as counts, not averaged rates")
    check(rows["B"]["pa_rate"]["n"] == 0 and rows["B"]["pa_rate"]["value"] is None and "rank" not in rows["B"]["pa_rate"], "a team with no charting for a metric gets no value and no rank, never a fill")
    check(rows["C"]["press_made"]["rank"] == 1 and rows["C"]["press_made"]["of"] == 3 and rows["B"]["press_made"]["rank"] == 3, "ranks run 1 = highest rate over the teams that have the metric")
    check(rows["A"]["pa_rate"]["of"] == 2, "the rank denominator excludes teams without the metric")

    s_cur = coaching.style_sentences(rows["A"], None, 2)
    check(s_cur == [], "with fewer than MIN_GAMES games and no previous season there are no sentences")
    s_prev = coaching.style_sentences(rows["A"], rows["C"], 2)
    check(all("(last season)" in x for x in s_prev) and len(s_prev) >= 4, "a young season falls back to last season and says so in every sentence")
    s_full = coaching.style_sentences(rows["A"], None, 3)
    check(any("50% of neutral" in x and "(this season)" in x for x in s_full), "sentences carry the number and the basis")
    check(not any("motion" in x.lower() for x in s_full) or rows["A"]["motion"]["rank"] in (1, 3), "motion is only mentioned at the extremes")

    recs = coaching.coach_records(pd.DataFrame([
        dict(season=2025, week=1, game_type="REG", home_team="A", away_team="B", home_coach="X", away_coach="Y", home_score=20, away_score=10),
        dict(season=2026, week=1, game_type="REG", home_team="B", away_team="A", home_coach="Y", away_coach="X", home_score=7, away_score=7),
        dict(season=2026, week=2, game_type="REG", home_team="A", away_team="B", home_coach="X", away_coach="Y", home_score=None, away_score=None),
    ]), 2026)
    check(recs[("A", "X")] == {"w": 1, "l": 0, "t": 1, "seasons": [2025, 2026]}, "head-coach record with the team counts finished games only, ties included")

    print("\n" + ("all checks passed" if not FAILS else f"{len(FAILS)} FAILED"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
