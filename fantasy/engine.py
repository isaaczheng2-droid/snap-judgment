"""
The fantasy engine: which stats a fantasy projection needs, how a player's stat projections
become points and a range, and the flags a lineup decision has to see (bye, injury
designation, rookie, role change, competition for touches).

The stat projections themselves come from run_pipeline.player_projections, the same ridge
models the prop pages use, plus four fantasy-only targets (a back's receptions and
receiving touchdowns, a quarterback's rushing touchdowns and interceptions) that props do
not price but points do.
"""
import numpy as np
import pandas as pd

from . import scoring

# fantasy-only stat targets, same shape as run_pipeline.PTARGETS
EXTRA_TARGETS = {
    "rb_receptions":      dict(stat="receptions", pos=["RB"], vol="proj_carries", mn=5, opp="pass"),
    "rb_receiving_tds":   dict(stat="receiving_tds", pos=["RB"], vol="proj_carries", mn=5, opp="pass"),
    "qb_rushing_tds":     dict(stat="rushing_tds", pos=["QB"], vol="proj_attempts", mn=10, opp="rush"),
    "passing_interceptions": dict(stat="passing_interceptions", pos=["QB"], vol="proj_attempts", mn=10, opp="pass"),
}
# which projection keys make up each position's points
POS_KEYS = {
    "QB": ["passing_yards", "passing_tds", "passing_interceptions", "qb_rushing_yards", "qb_rushing_tds"],
    "RB": ["rushing_yards", "rushing_tds", "rb_receptions", "rb_receiving_yards", "rb_receiving_tds"],
    "WR": ["receptions", "receiving_yards", "receiving_tds"],
    "TE": ["receptions", "receiving_yards", "receiving_tds"],
}
# the opportunity a points projection rests on, for the start/sit card
OPPORTUNITY = {"QB": "proj_attempts", "RB": "proj_carries", "WR": "proj_targets", "TE": "proj_targets"}
PRIMARY_VOL = {"QB": ("proj_attempts", 10), "RB": ("proj_carries", 5), "WR": ("proj_targets", 3), "TE": ("proj_targets", 3)}
ROLE_SHIFT = 0.06          # change in target/carry share vs last season that counts as a role change
ROLE_MIN_GAMES = 3         # games this season before a share difference is called a role change


def all_targets(base_targets):
    t = dict(base_targets)
    t.update(EXTRA_TARGETS)
    return t


def points_for(rec, settings):
    """Points for one projection record (dict with projection keys) under settings."""
    pos = rec.get("position")
    return scoring.points({k: rec.get(k) for k in POS_KEYS.get(pos, [])}, settings, position=pos)


def range_for(pos, proj, model):
    """
    (p10, p25, p75, p90) around a points projection from the fitted residual spread for the
    position: residual scale = a + b * proj, quantiles of the standardised residual from the
    walk-forward. Returns None when the position has no fitted spread.
    """
    m = (model or {}).get("residuals", {}).get(pos)
    if not m or proj is None:
        return None
    scale = max(m["a"] + m["b"] * float(proj), 0.5)
    q = m["q"]
    return {k: round(max(float(proj) + scale * q[k], 0.0), 1) for k in ("p10", "p25", "p75", "p90")}


def availability(rec, inj_map, bye_teams, play_rates):
    """
    What the lineup decision needs to know about whether he plays. `play_rates` is the
    measured share of players with each designation who actually appeared (fantasy/backtest.py).
    """
    team = rec.get("team")
    if team in bye_teams:
        return {"status": "bye", "label": "Bye week", "p_play": 0.0, "note": "No game this week."}
    st = (inj_map or {}).get(rec.get("player_key")) or {}
    level = st.get("level")
    if level in ("out", "doubtful", "questionable"):
        pr = (play_rates or {}).get(level, {}).get(rec.get("position")) or (play_rates or {}).get(level, {}).get("ALL")
        p = float(pr["rate"]) if pr else {"out": 0.0, "doubtful": 0.1, "questionable": 0.75}[level]
        n = int(pr["n"]) if pr else 0
        return {"status": level, "label": st.get("label", level.title()), "p_play": round(p, 3),
                "why": st.get("why"), "updated": st.get("updated"),
                "note": f"{level.title()} players at this position have played {p:.0%} of the time since 2019 ({n} cases)." if n
                        else f"{level.title()}: historical play rate not measured for this position; a conservative default is shown."}
    if level in ("limited", "rest"):
        return {"status": level, "label": st.get("label", level.title()), "p_play": 1.0, "why": st.get("why"), "note": "Listed on the report without a game status."}
    return {"status": "ok", "label": "No designation", "p_play": 1.0}


def role_flags(rec, latest_row, prev_season_share, rookie_ids, new_team_ids):
    """Rookie, changed team, changed role (share moved vs last season), newly absent teammate."""
    flags = []
    pid = rec.get("player_key")
    if pid in rookie_ids:
        flags.append({"kind": "rookie", "text": "Rookie: no NFL history; the projection leans on the position average and his usage so far."})
    if pid in new_team_ids:
        flags.append({"kind": "new_team", "text": "New team this season; last season's numbers came in a different offence."})
    pos = rec.get("position")
    share_col = "car_share" if pos == "RB" else "tgt_share"
    now = None
    if latest_row is not None:
        now = latest_row.get(f"{share_col}_now")
        if now is None or pd.isna(now):
            now = latest_row.get(share_col)
    prev = prev_season_share.get(pid)
    games = 0
    if latest_row is not None:
        gp = latest_row.get("gp_prior")
        games = (0 if gp is None or pd.isna(gp) else int(gp)) + 1
    # one game is not a role; the flag needs three games of this season behind the share
    if games >= ROLE_MIN_GAMES and now is not None and prev is not None and not pd.isna(now) and not pd.isna(prev) and abs(float(now) - float(prev)) >= ROLE_SHIFT:
        d = float(now) - float(prev)
        flags.append({"kind": "role_change", "text": f"Role changed: his share of the team's {'carries' if pos == 'RB' else 'targets'} is {abs(d):.0%} {'higher' if d > 0 else 'lower'} than last season."})
    ab = (rec.get("absence") or {})
    for st, a in ab.items():
        names = ", ".join(a.get("names", [])[:2])
        if a.get("lost_car"):
            flags.append({"kind": "teammate_out", "text": f"{names} out: {a['lost_car']:.0%} of the backfield's carries are newly available."})
        elif a.get("lost_tgt"):
            flags.append({"kind": "teammate_out", "text": f"{names} out: {a['lost_tgt']:.0%} of the group's targets are newly available."})
    return flags


def competition(rec, team_rows):
    """Who else on the team competes for the same touches, with their prior share."""
    pos = rec.get("position")
    share_col = "car_share" if pos == "RB" else "tgt_share"
    same = [r for r in team_rows if r.get("player_key") != rec.get("player_key")
            and (r.get("position") == pos or (pos != "RB" and r.get("position") in ("WR", "TE")))]
    out = sorted([{"name": r.get("name") or r.get("player_display_name"), "position": r.get("position"), "share": r.get(share_col)}
                  for r in same if r.get(share_col) is not None], key=lambda x: -(x["share"] or 0))[:4]
    return out
