#!/usr/bin/env python3
"""
Point-in-time scheme identity per team-game, derived from weekly team box scores.

Everything here is built the same way the EPA ratings are: an exponentially-weighted
mean of games STRICTLY BEFORE the one being described, shrunk toward the team's
previous-season mean and then the league mean while the sample is small. A row for
week 7 knows nothing about week 7.

The metrics are chosen to describe how a staff plays, not how well:
  pass_rate   dropbacks / all offensive plays        -- pass-first vs run-first
  adot        air yards per attempt                  -- vertical vs short/quick
  pace        offensive plays per game               -- tempo
  yac_share   YAC / passing yards                    -- scheme-manufactured yards
  sack_rate   sacks taken / dropback                 -- protection + time to throw
  pressure    (sacks + QB hits) / dropbacks faced    -- pass-rush identity
  takeaway    INTs + opponent fumbles recovered      -- turnover-hunting defense
"""
import numpy as np
import pandas as pd

SCHEME = ["pass_rate", "adot", "pace", "yac_share", "sack_rate", "pressure", "takeaway"]
EW_SPAN, K = 8, 4.0     # longer span / slower shrink than the EPA ratings: style moves less than form


def _per_game(team, sched):
    """One row per team-game with the raw style rates for that single game."""
    tw = team[team.season_type == "REG"].copy()
    for c in ["attempts", "sacks_suffered", "carries", "passing_air_yards", "passing_yards",
              "passing_yards_after_catch", "def_sacks", "def_qb_hits", "def_interceptions",
              "fumble_recovery_opp"]:
        if c not in tw.columns:
            tw[c] = 0.0
        tw[c] = tw[c].fillna(0)

    tw["dropbacks"] = tw.attempts + tw.sacks_suffered
    tw["off_plays"] = tw.dropbacks + tw.carries

    o = tw[["season", "week", "team", "opponent_team", "game_id", "dropbacks", "off_plays",
            "attempts", "sacks_suffered", "passing_air_yards", "passing_yards",
            "passing_yards_after_catch", "def_sacks", "def_qb_hits", "def_interceptions",
            "fumble_recovery_opp"]].copy()

    # dropbacks faced comes from the opponent's own row
    faced = o[["season", "week", "game_id", "team", "dropbacks"]].rename(
        columns={"team": "opponent_team", "dropbacks": "db_faced"})
    o = o.merge(faced, on=["season", "week", "game_id", "opponent_team"], how="left")

    o["pass_rate"] = o.dropbacks / o.off_plays.replace(0, np.nan)
    o["adot"] = o.passing_air_yards / o.attempts.replace(0, np.nan)
    o["pace"] = o.off_plays
    o["yac_share"] = o.passing_yards_after_catch / o.passing_yards.replace(0, np.nan)
    o["sack_rate"] = o.sacks_suffered / o.dropbacks.replace(0, np.nan)
    o["pressure"] = (o.def_sacks + o.def_qb_hits) / o.db_faced.replace(0, np.nan)
    o["takeaway"] = o.def_interceptions + o.fumble_recovery_opp

    gd = sched[["game_id", "gameday"]].drop_duplicates("game_id")
    return o.merge(gd, on="game_id", how="left")


def team_scheme(team, sched, cur, target_week):
    """Point-in-time scheme ratings, including a synthetic row for the unplayed week."""
    m = _per_game(team, sched)
    lg = m[SCHEME].mean()
    m = m.sort_values(["team", "season", "gameday"]).reset_index(drop=True)
    m["gp_prior"] = m.groupby(["team", "season"]).cumcount()

    prev = m.groupby(["team", "season"])[SCHEME].mean().reset_index()
    prev = prev.rename(columns={x: f"prev_{x}" for x in SCHEME})
    prev["season"] += 1

    for x in SCHEME:
        sh = m.groupby(["team", "season"])[x].shift(1)
        m[f"ew_{x}"] = sh.groupby([m.team, m.season]).transform(
            lambda s: s.ewm(span=EW_SPAN, min_periods=1).mean())
    m = m.merge(prev, on=["team", "season"], how="left")
    for x in SCHEME:
        m[f"prev_{x}"] = m[f"prev_{x}"].fillna(lg[x])
        m[f"ew_{x}"] = m[f"ew_{x}"].fillna(m[f"prev_{x}"])
        m[f"pr_{x}"] = 0.5 * lg[x] + 0.5 * m[f"prev_{x}"]
    w = m.gp_prior / (m.gp_prior + K)
    for x in SCHEME:
        m[f"sch_{x}"] = w * m[f"ew_{x}"] + (1 - w) * m[f"pr_{x}"]

    out = m[["team", "season", "week", "game_id"] + [f"sch_{x}" for x in SCHEME]].copy()

    # ---- the upcoming week has no box score yet, so build its row the same way by hand ----
    played = m[m.season == cur]
    gp = played.groupby("team").size().to_dict()
    up = sched[(sched.season == cur) & (sched.week == target_week) & (sched.game_type == "REG")]
    teams = pd.unique(pd.concat([up.home_team, up.away_team]))
    prev_cur = prev[prev.season == cur].set_index("team")

    rows = []
    for t in teams:
        n = gp.get(t, 0)
        tp = played[played.team == t].sort_values("gameday")
        row = {"team": t, "season": cur, "week": target_week}
        for x in SCHEME:
            p = prev_cur[f"prev_{x}"].get(t, lg[x])
            if pd.isna(p):
                p = lg[x]
            prior = 0.5 * lg[x] + 0.5 * p
            ew = tp[x].ewm(span=EW_SPAN, min_periods=1).mean().iloc[-1] if n else prior
            if pd.isna(ew):
                ew = prior
            ww = n / (n + K)
            row[f"sch_{x}"] = ww * ew + (1 - ww) * prior
        g = up[(up.home_team == t) | (up.away_team == t)]
        row["game_id"] = g.game_id.values[0] if len(g) else None
        rows.append(row)

    full = pd.concat([out, pd.DataFrame(rows)], ignore_index=True)

    # league percentile within season-week, so "78th percentile pass rate" is sayable
    for x in SCHEME:
        full[f"pk_{x}"] = full.groupby(["season", "week"])[f"sch_{x}"].rank(pct=True)
    return full, lg


SCHEME_FEATS = ["pass_rate_diff", "adot_diff", "pace_diff",
                "press_edge_home", "prot_edge_home", "takeaway_diff"]
WEATHER_FEATS = ["wind_f", "temp_f", "is_indoor", "wind_x_pass"]


def add_scheme_cols(df, scheme):
    """Attach both teams' scheme ratings to a game frame and derive the matchup features."""
    sc = [f"sch_{x}" for x in SCHEME] + [f"pk_{x}" for x in SCHEME]
    for side in ["home", "away"]:
        s = scheme[["game_id", "team"] + sc].rename(
            columns={"team": f"{side}_team", **{c: f"{side}_{c}" for c in sc}})
        df = df.merge(s, on=["game_id", f"{side}_team"], how="left")

    df["pass_rate_diff"] = df.home_sch_pass_rate - df.away_sch_pass_rate
    df["adot_diff"] = df.home_sch_adot - df.away_sch_adot
    df["pace_diff"] = df.home_sch_pace - df.away_sch_pace
    # home's pass rush vs away's protection, minus the mirror image
    df["press_edge_home"] = ((df.home_sch_pressure - df.away_sch_sack_rate)
                             - (df.away_sch_pressure - df.home_sch_sack_rate))
    df["prot_edge_home"] = df.away_sch_sack_rate - df.home_sch_sack_rate
    df["takeaway_diff"] = df.home_sch_takeaway - df.away_sch_takeaway
    # wind should matter more to two pass-heavy teams than to two run-heavy ones
    df["wind_x_pass"] = df.wind_f * (df.home_sch_pass_rate + df.away_sch_pass_rate) / 2
    return df
