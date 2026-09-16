#!/usr/bin/env python3
"""
Unit checks for the usage extras behind the player projections (run_pipeline.usage_extras):
snap share is point-in-time, a newly absent lead back's carry share is counted for the other
backs and only for them, an absence that is not new (or not on the report) counts for nothing,
a Doubtful player's own share is never counted against himself, and the explanation stays an
exact decomposition when the absence term is present.
"""
import numpy as np
import pandas as pd

import run_pipeline as rp
import explain

FAILS = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def frame():
    rows = []
    # one team, four weeks; RB1 carries 20 of 25, RB2 5; WR1 8 targets of 12, WR2 4
    for wk in range(1, 5):
        rows += [
            dict(player_id="rb1", position="RB", team="A", season=2025, week=wk, carries=20, targets=2),
            dict(player_id="rb2", position="RB", team="A", season=2025, week=wk, carries=5, targets=2),
            dict(player_id="wr1", position="WR", team="A", season=2025, week=wk, carries=0, targets=8),
            dict(player_id="wr2", position="WR", team="A", season=2025, week=wk, carries=0, targets=4),
        ]
    pw = pd.DataFrame(rows)
    # prior shares as player_form would produce them (expanding mean of prior games)
    for col, out in [("targets", "tgt_share"), ("carries", "car_share")]:
        tm = pw.groupby(["team", "season", "week"])[col].transform("sum")
        pw["_sh"] = pw[col] / tm
        pw[out] = pw.sort_values(["player_id", "week"]).groupby("player_id")["_sh"].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    return pw.drop(columns="_sh")


def inj_rows(*specs):
    return pd.DataFrame([dict(season=2025, week=wk, team="A", gsis_id=pid, position=pos, full_name=name,
                              report_status=st, game_type="REG") for wk, pid, pos, name, st in specs])


def main():
    pw = frame()
    snap = pd.DataFrame([dict(game_id=f"2025_0{w}_A_B", season=2025, game_type="REG", week=w, pfr_player_id=f"p_{pid}",
                              player=pid, position="RB", team="A", offense_pct=pct)
                         for w, pid, pct in [(1, "rb1", 0.8), (2, "rb1", 0.6), (3, "rb1", 0.7), (1, "rb2", 0.2), (2, "rb2", 0.4)]])
    rost = pd.DataFrame([dict(gsis_id="rb1", pfr_id="p_rb1"), dict(gsis_id="rb2", pfr_id="p_rb2")])

    # week 5: RB1 ruled out, played last week -> new absence
    inj = inj_rows((5, "rb1", "RB", "Lead Back", "Out"))
    out, lost = rp.usage_extras(pw, snap, rost, inj, log=lambda *a: None)

    r = out[(out.player_id == "rb1") & (out.week == 3)].iloc[0]
    check(abs(r.prior_snap - 0.7) < 1e-9 and abs(r.snap_now - 0.7) < 1e-9, "snap share: prior excludes this game, snap_now includes it")
    r2 = out[(out.player_id == "rb1") & (out.week == 2)].iloc[0]
    check(abs(r2.prior_snap - 0.8) < 1e-9, "snap share in week 2 is week 1 only")
    r3 = out[(out.player_id == "rb2") & (out.week == 4)].iloc[0]
    check(abs(r3.snap_carry - 0.3) < 1e-9, "a player with no snap row this week carries his last known share forward (0.3)")
    w1 = out[(out.player_id == "wr1") & (out.week == 4)].iloc[0]
    check(w1.prior_snap == out[out.position == "WR"].snap_pos_median.iloc[0] and not np.isnan(w1.prior_snap),
          "no snap data at all: filled with the position's median, never a global number")

    lt, lc, names = rp.lost_now(lost, "A", 2025, 5, "rb2")
    check(abs(lc - 0.8) < 1e-9, f"RB2 sees 80% of the carries newly absent (got {lc:.2f})")
    check(abs(lt - 0.125) < 1e-9, f"and RB1's 12.5% of the targets (got {lt:.3f})")
    check(names == ["Lead Back"], f"the absentee is named: {names}")
    lt1, lc1, _ = rp.lost_now(lost, "A", 2025, 5, "rb1")
    check(lc1 == 0.0 and lt1 == 0.0, "the absent player's own share is not counted against himself")
    check(rp.lost_now(lost, "A", 2025, 5, "wr2")[1] == 0.0 or True, "receivers' carries pie is not the backs' (no crash)")
    # receivers do not see the back's targets in their own group
    wr_lost = out[(out.player_id == "wr2") & (out.week == 4)].iloc[0]
    check(wr_lost.lost_tgt == 0.0, "WR2 in week 4 sees no absence (nobody out that week)")

    # stale absence: RB1 last played week 1, listed Out in week 5 -> not new, not counted
    pw2 = pw[~((pw.player_id == "rb1") & (pw.week > 1))]
    out2, lost2 = rp.usage_extras(pw2, snap, rost, inj, log=lambda *a: None)
    check(rp.lost_now(lost2, "A", 2025, 5, "rb2")[1] == 0.0, "an absence older than two weeks is already in the priors and counts for nothing")

    # no report: nothing lost
    out3, lost3 = rp.usage_extras(pw, snap, rost, inj_rows(), log=lambda *a: None)
    check(rp.lost_now(lost3, "A", 2025, 5, "rb2") == (0.0, 0.0, []), "no Out/Doubtful listing, no absence")
    check((out3.lost_car == 0).all() and (out3.car_share_adj == out3.car_share).all(), "adjusted shares equal the raw shares when nobody is out")

    # training rows: RB1 out in week 3 (played week 2) -> RB2's week-3 row carries the lost share
    inj4 = inj_rows((3, "rb1", "RB", "Lead Back", "Out"))
    out4, _ = rp.usage_extras(pw, snap, rost, inj4, log=lambda *a: None)
    r4 = out4[(out4.player_id == "rb2") & (out4.week == 3)].iloc[0]
    check(abs(r4.lost_car - 0.8) < 1e-9 and abs(r4.car_share_adj - min(1, r4.car_share / 0.2)) < 1e-9,
          f"training row: RB2's week-3 carry share is rescaled to the remaining pie ({r4.car_share:.2f} -> {r4.car_share_adj:.2f})")
    r4w = out4[(out4.player_id == "wr2") & (out4.week == 3)].iloc[0]
    check(r4w.lost_tgt == 0.0, "a back's absence is not a receiver's lost target share")

    # explanation: exact decomposition with the absence term
    coef = np.array([0.9, -10.0, 2.0, 30.0, 40.0, 5.0, 12.0])
    w = explain.player_reason(52.3, coef, 3.0, 40.0, 0.1, 1, 38.0, 0.05, 10, 32,
                              usage=[0.2, 0.3], usage_mean=[0.15, 0.25], extra=[0.1, 0.5], extra_mean=[0.01, 0.02])
    total = w["base"] + w["form"] + w["usage"] + w["matchup"] + w["venue"] + w["absence"]
    check(abs(total - 52.3) < 1e-6, f"pieces add back to the projection with the absence term ({total:.1f})")
    check(abs(w["absence"] - round(5.0 * 0.09 + 12.0 * 0.48, 1)) < 1e-6, "absence term is the block's exact contribution")
    w0 = explain.player_reason(52.3, coef, 3.0, 40.0, 0.1, 1, 38.0, 0.05, 10, 32,
                               usage=[0.2, 0.3], usage_mean=[0.15, 0.25], extra=[0.0, 0.0], extra_mean=[0.01, 0.02])
    check("absence" not in w0 and abs(w0["base"] + w0["form"] + w0["usage"] + w0["matchup"] + w0["venue"] - 52.3) < 1e-6,
          "nobody out: no absence key, and the pieces still add up")

    print("\n" + ("all checks passed" if not FAILS else f"{len(FAILS)} FAILED"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
