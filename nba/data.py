"""
NBA data layer: fetch, normalise, and never turn a missing number into a zero.

Source: the SportsDataverse data releases (ESPN-sourced), which are plain GitHub release
assets and therefore reachable from the dev container, the Actions runner and the browser.
One file per season and table, seasons named by their END year (2026 = the 2025-26 season).

Tables produced (parquet under nba/data/, one row per ...):
  games         ... game: espn game_id, season, phase, tipoff_utc, home/away team ids and
                    abbreviations, scores when final, periods played (OT = periods > 4),
                    venue, neutral flag, status.
  team_games    ... team-game: box totals, possessions (estimated), pace, rest days,
                    back-to-back, opponent, result.
  player_games  ... player-game: minutes and counting stats; a player listed on the game
                    roster who did not play keeps NULL minutes and NULL stats, never zero,
                    with the listed reason (injury text, "COACH'S DECISION", inactive).
  players       ... player: latest team, position, jersey, headshot URL (ESPN), first/last
                    seen, and the game-roster identity fields.

Identity: ESPN athlete_id is the stable player key throughout the NBA section (an integer
string). stats.nba.com person ids are a separate namespace; the crosswalk is filled by the
runner-side collector (nba/collect.py) when it can reach stats.nba.com and is NOT assumed here.

Timestamps: `game_date_time` from ESPN is the scheduled tipoff in UTC. Box scores carry no
"available at" time in the source; the collector stamps `ingested_at` when it writes, so a
historical reconstruction can tell "the game happened at T" from "we had the box score at T2".
"""
import io
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "data")
RELEASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
FILES = {
    "player_box": "espn_nba_player_boxscores/player_box_{s}.rds",
    "team_box": "espn_nba_team_boxscores/team_box_{s}.rds",
    "schedule": "espn_nba_schedules/nba_schedule_{s}.rds",
    "game_rosters": "espn_nba_game_rosters/game_rosters_{s}.rds",
}
PHASE = {1: "preseason", 2: "regular", 3: "postseason", 5: "play-in"}
STATS = ["field_goals_made", "field_goals_attempted", "three_point_field_goals_made",
         "three_point_field_goals_attempted", "free_throws_made", "free_throws_attempted",
         "offensive_rebounds", "defensive_rebounds", "rebounds", "assists", "steals", "blocks",
         "turnovers", "fouls", "points"]
SHORT = {"field_goals_made": "fgm", "field_goals_attempted": "fga", "three_point_field_goals_made": "fg3m",
         "three_point_field_goals_attempted": "fg3a", "free_throws_made": "ftm", "free_throws_attempted": "fta",
         "offensive_rebounds": "oreb", "defensive_rebounds": "dreb", "rebounds": "reb", "assists": "ast",
         "steals": "stl", "blocks": "blk", "turnovers": "tov", "fouls": "pf", "points": "pts"}


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- fetch
def fetch(table, season, refresh=False, timeout=240):
    """Download one release asset into the cache (skipped if present unless refresh)."""
    os.makedirs(CACHE, exist_ok=True)
    name = FILES[table].format(s=season)
    dest = os.path.join(CACHE, os.path.basename(name))
    if os.path.exists(dest) and os.path.getsize(dest) > 1000 and not refresh:
        return dest
    url = f"{RELEASE}/{name}"
    tmp = dest + ".new"
    r = subprocess.run(["curl", "-sSL", "--max-time", str(timeout), "-o", tmp, "-w", "%{http_code}", url],
                       capture_output=True, text=True, timeout=timeout + 15)
    code = r.stdout.strip()[-3:]
    if code == "200" and os.path.exists(tmp) and os.path.getsize(tmp) > 1000:
        os.replace(tmp, dest)
        return dest
    if os.path.exists(tmp):
        os.remove(tmp)
    if os.path.exists(dest):
        log(f"  fetch {name}: HTTP {code}; keeping the cached copy")
        return dest
    log(f"  fetch {name}: HTTP {code}; not available")
    return None


def read_rds(path):
    import pyreadr
    return pyreadr.read_r(path)[None]


# --------------------------------------------------------------------------- normalise
def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _seasons(first, last):
    return list(range(first, last + 1))


def build(first=2022, last=2027, refresh=False):
    """Fetch and normalise every season in [first, last] (end-year naming). Returns dict of frames."""
    os.makedirs(OUT, exist_ok=True)
    games, tgames, pgames, rosters = [], [], [], []
    stamps = {}
    for s in _seasons(first, last):
        p = fetch("schedule", s, refresh)
        if p:
            sc = read_rds(p)
            stamps[f"schedule_{s}"] = datetime.fromtimestamp(os.path.getmtime(p), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            games.append(_norm_schedule(sc, s))
        for table, sink, fn in (("team_box", tgames, _norm_team_box), ("player_box", pgames, _norm_player_box),
                                ("game_rosters", rosters, _norm_rosters)):
            p = fetch(table, s, refresh)
            if p:
                df = read_rds(p)
                stamps[f"{table}_{s}"] = datetime.fromtimestamp(os.path.getmtime(p), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                sink.append(fn(df, s))
    G = pd.concat(games, ignore_index=True).drop_duplicates("game_id")
    keep = set(G.game_id.dropna().astype(int))
    T = pd.concat(tgames, ignore_index=True) if tgames else pd.DataFrame()
    P = pd.concat(pgames, ignore_index=True) if pgames else pd.DataFrame()
    R = pd.concat(rosters, ignore_index=True) if rosters else pd.DataFrame()
    T, P, R = (x[x.game_id.isin(keep)] if not x.empty else x for x in (T, P, R))
    T = _team_context(T, G)
    P = P.merge(G[["game_id", "tipoff_utc", "phase", "periods"]], on="game_id", how="left")
    players = _players(P, R)
    ingested = now_iso()
    for df in (G, T, P):
        df["ingested_at"] = ingested
    G.to_parquet(os.path.join(OUT, "games.parquet"), index=False)
    T.to_parquet(os.path.join(OUT, "team_games.parquet"), index=False)
    P.to_parquet(os.path.join(OUT, "player_games.parquet"), index=False)
    players.to_parquet(os.path.join(OUT, "players.parquet"), index=False)
    json.dump({"ingested_at": ingested, "source_files": stamps, "seasons": _seasons(first, last)},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    log(f"nba data: {len(G)} games, {len(T)} team-games, {len(P)} player-games, {len(players)} players")
    return {"games": G, "team_games": T, "player_games": P, "players": players}


def _norm_schedule(sc, season):
    d = pd.DataFrame({
        "game_id": _num(sc["id"]).astype("Int64"),
        "season": season,
        "phase": sc["season_type"].map(PHASE).fillna("other") if "season_type" in sc else "regular",
        "phase_label": sc.get("type_abbreviation"),
        "tipoff_utc": pd.to_datetime(sc["date"], utc=True, errors="coerce"),
        "home_id": _num(sc["home_id"]).astype("Int64"), "away_id": _num(sc["away_id"]).astype("Int64"),
        "home": sc["home_abbreviation"], "away": sc["away_abbreviation"],
        "home_name": sc.get("home_display_name"), "away_name": sc.get("away_display_name"),
        "home_score": _num(sc["home_score"]), "away_score": _num(sc["away_score"]),
        "status": sc["status_type_name"],
        "final": sc["status_type_completed"].astype(bool) if "status_type_completed" in sc else sc["status_type_name"].eq("STATUS_FINAL"),
        "periods": _num(sc.get("status_period")),
        "venue": sc.get("venue_full_name"), "city": sc.get("venue_address_city"),
        "neutral": sc.get("neutral_site").astype(bool) if "neutral_site" in sc else False,
        "attendance": _num(sc.get("attendance")),
    })
    # a scheduled game has no score; an ESPN 0-0 on a postponed row must not read as a result
    d.loc[~d.final, ["home_score", "away_score"]] = np.nan
    d["cup_final"] = d["phase_label"].eq("CC")     # NBA Cup final: real game, excluded from standings
    # all-star weekend rows share season_type 2 in the ESPN export; they are not games we model
    d = d[(d.phase != "other") & ~d["phase_label"].eq("ALLSTAR")]
    return d


def _norm_team_box(tb, season):
    d = pd.DataFrame({
        "game_id": _num(tb["game_id"]).astype("Int64"), "season": season,
        "team_id": _num(tb["team_id"]).astype("Int64"), "team": tb["team_abbreviation"],
        "home": tb["team_home_away"].eq("home"),
        "opp_id": _num(tb["opponent_team_id"]).astype("Int64"),
        "pts": _num(tb["team_score"]), "opp_pts": _num(tb["opponent_team_score"]),
        "won": tb["team_winner"].astype(bool),
        "fgm": _num(tb["field_goals_made"]), "fga": _num(tb["field_goals_attempted"]),
        "fg3m": _num(tb["three_point_field_goals_made"]), "fg3a": _num(tb["three_point_field_goals_attempted"]),
        "ftm": _num(tb["free_throws_made"]), "fta": _num(tb["free_throws_attempted"]),
        "oreb": _num(tb["offensive_rebounds"]), "dreb": _num(tb["defensive_rebounds"]),
        "reb": _num(tb["total_rebounds"]), "ast": _num(tb["assists"]), "stl": _num(tb["steals"]),
        "blk": _num(tb["blocks"]), "tov": _num(tb["total_turnovers"]), "pf": _num(tb["fouls"]),
        "game_date": pd.to_datetime(tb["game_date"]).dt.date.astype(str),
    })
    return d


def _norm_player_box(pb, season):
    d = pd.DataFrame({
        "game_id": _num(pb["game_id"]).astype("Int64"), "season": season,
        "player_id": pb["athlete_id"].astype(str), "player": pb["athlete_display_name"],
        "team_id": _num(pb["team_id"]).astype("Int64"), "team": pb["team_abbreviation"],
        "opp_id": _num(pb["opponent_team_id"]).astype("Int64"), "opp": pb["opponent_team_abbreviation"],
        "home": pb["home_away"].eq("home"),
        "position": pb["athlete_position_abbreviation"], "jersey": pb["athlete_jersey"],
        "starter": pb["starter"].astype(bool),
        "played": pb["minutes"].notna() & (_num(pb["minutes"]) > 0),
        "dnp": pb["did_not_play"].astype(bool), "active": pb["active"].astype(bool),
        "reason": pb["reason"], "ejected": pb["ejected"].astype(bool),
        "min": _num(pb["minutes"]),
        "headshot": pb["athlete_headshot_href"],
        "game_date": pd.to_datetime(pb["game_date"]).dt.date.astype(str),
        "team_pts": _num(pb["team_score"]), "opp_pts": _num(pb["opponent_team_score"]),
    })
    for k, v in SHORT.items():
        d[v] = _num(pb[k])
    # Missing must stay missing: a row with no minutes is a player who did not play (or an
    # inactive), and every stat on it is NULL, whatever ESPN's export put there.
    idle = ~d["played"]
    d.loc[idle, ["min"] + list(SHORT.values())] = np.nan
    d["plus_minus"] = _num(pb["plus_minus"].astype(str).str.replace("+", "", regex=False))
    d.loc[idle, "plus_minus"] = np.nan
    return d


def _norm_rosters(gr, season):
    return pd.DataFrame({
        "game_id": _num(gr["game_id"]).astype("Int64"), "season": season,
        "player_id": gr["athlete_id"].astype(str), "team_id": _num(gr["team_id"]).astype("Int64"),
        "team": gr["team_abbreviation"], "first_name": gr.get("athlete_first_name"), "last_name": gr.get("athlete_last_name"),
        "position": gr.get("athlete_position"), "jersey": gr.get("athlete_jersey"),
        "headshot": gr.get("athlete_headshot"), "starter": gr["starter"].astype(bool),
        "dnp": gr["did_not_play"].astype(bool), "active": gr["active"].astype(bool), "reason": gr.get("reason"),
    })


def possessions(df):
    """Standard box-score estimate: FGA - OREB + TOV + 0.44 * FTA (team totals)."""
    return df["fga"] - df["oreb"] + df["tov"] + 0.44 * df["fta"]


def _team_context(T, G):
    if T.empty:
        return T
    T = T.merge(G[["game_id", "tipoff_utc", "phase", "periods", "neutral"]], on="game_id", how="left")
    T["poss"] = possessions(T)
    # each side gets the game's possessions as the mean of both estimates (they differ by a
    # possession or two on rebounds and end-of-quarter heaves)
    both = T.groupby("game_id")["poss"].transform("mean")
    T["poss_game"] = both
    T["minutes_game"] = 48 + 5 * (T["periods"].fillna(4).clip(lower=4) - 4)
    T["pace"] = T["poss_game"] * 48 / T["minutes_game"]          # possessions per 48
    T["ortg"] = 100 * T["pts"] / T["poss_game"]
    T["drtg"] = 100 * T["opp_pts"] / T["poss_game"]
    T = T.sort_values(["team_id", "tipoff_utc"]).reset_index(drop=True)
    prev = T.groupby(["team_id", "season"])["tipoff_utc"].shift(1)
    T["rest_days"] = (T["tipoff_utc"] - prev).dt.total_seconds() / 86400
    T["b2b"] = T["rest_days"].between(0.5, 1.5)
    T["games_last7"] = T.groupby(["team_id", "season"])["tipoff_utc"].transform(
        lambda s: pd.Series([((s > t - pd.Timedelta(days=7)) & (s < t)).sum() for t in s], index=s.index))
    return T


def _players(P, R):
    if P.empty:
        return pd.DataFrame()
    P = P.sort_values("tipoff_utc")
    last = P.groupby("player_id").tail(1)
    first = P.groupby("player_id")["season"].min().rename("first_season")
    gp = P[P.played].groupby("player_id").size().rename("games_played")
    out = last[["player_id", "player", "team_id", "team", "position", "jersey", "headshot", "season"]].rename(
        columns={"season": "last_season", "team": "last_team", "team_id": "last_team_id"})
    out = out.merge(first, on="player_id", how="left").merge(gp, on="player_id", how="left")
    out["games_played"] = out["games_played"].fillna(0).astype(int)
    if not R.empty:
        nm = R.sort_values("game_id").groupby("player_id").tail(1)[["player_id", "first_name", "last_name"]]
        out = out.merge(nm, on="player_id", how="left")
    return out.reset_index(drop=True)


def load():
    """The normalised tables from disk (build() first)."""
    return {k: pd.read_parquet(os.path.join(OUT, f"{k}.parquet")) for k in ("games", "team_games", "player_games", "players")}


DICTIONARY = {
    "games": {"game_id": "ESPN game id (int); stable key across every table", "season": "season by end year (2026 = 2025-26)",
              "phase": "preseason / regular / play-in / postseason", "tipoff_utc": "scheduled tipoff, UTC",
              "home/away": "ESPN team abbreviation; home_id/away_id are ESPN team ids", "home_score/away_score": "final score; NULL until final",
              "final": "True once ESPN marks the game completed", "periods": "periods played; 5+ means overtime",
              "neutral": "neutral-site flag from ESPN", "ingested_at": "when this table was written (UTC)"},
    "team_games": {"poss": "this side's box-score possession estimate FGA-OREB+TOV+0.44FTA", "poss_game": "mean of both sides' estimates",
                   "pace": "possessions per 48 minutes", "ortg/drtg": "points scored/allowed per 100 possessions",
                   "rest_days": "days since this team's previous game in the season; NULL for the first",
                   "b2b": "second night of a back-to-back", "games_last7": "games in the previous 7 days"},
    "player_games": {"player_id": "ESPN athlete id (string); the NBA section's stable player key", "min": "minutes played; NULL when the player did not play",
                     "played": "minutes recorded and > 0", "dnp": "ESPN did-not-play flag", "active": "on the active roster for the game",
                     "reason": "ESPN reason text for a DNP (injury text, COACH'S DECISION, ...)",
                     "pts/reb/ast/...": "counting stats; NULL when the player did not play, never 0",
                     "fgm/fga/fg3m/fg3a/ftm/fta": "shooting makes and attempts", "plus_minus": "plus/minus; NULL when not played"},
    "players": {"last_team": "team on the player's most recent box score", "games_played": "games with minutes > 0 across the loaded seasons",
                "headshot": "ESPN headshot URL (hotlinked, not cached; shown with an initials fallback)"},
}


if __name__ == "__main__":
    build()
