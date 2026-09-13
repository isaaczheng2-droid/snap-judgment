"""
Game-day windows: notice when a slate segment has finished and ask for a full rebuild.

An NFL Sunday is three segments (early, late, night) plus Thursday and Monday games. nflverse
posts a final score to games.csv within minutes of the whistle, and the ratings, the
tracker's grades and the remaining games' predictions all depend on it. The hourly rebuild
would pick that up eventually; this notices it on the next 15-minute poll instead.

Two triggers, both change-only and both recorded in the cache so nothing fires twice:

  window finished   every game in a kickoff window (kickoffs within WINDOW_SPAN of each
                    other) has a final score -> "Window finished: Sunday 1:00 PM ET games (6 final)"
  box scores landed the current season's stats_player_week file changed upstream while
                    this week has finals -> "Box scores updated" (grades the tracker and the
                    paper test; nflverse rebuilds it hours after the scores)

Neither invents a result: a game is final only when games.csv says so.
"""
import os
import subprocess
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from . import store
from .weather import kickoff_utc, _iso

SCHEDULES = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
REL = "https://github.com/nflverse/nflverse-data/releases/download"
WINDOW_SPAN = timedelta(minutes=90)
ET = ZoneInfo("America/New_York")


def fresh_schedule(datadir="data", log=print):
    """games.csv, refetched (small, changes minutes after each final); cached copy on failure."""
    os.makedirs(datadir, exist_ok=True)
    p = os.path.join(datadir, "games.csv")
    try:
        r = subprocess.run(["curl", "-sSL", "--max-time", "40", "-o", p + ".new", SCHEDULES], capture_output=True, timeout=60)
        if r.returncode == 0 and os.path.getsize(p + ".new") > 100_000:
            pd.read_csv(p + ".new", usecols=["game_id"])
            os.replace(p + ".new", p)
    except Exception as e:
        log(f"  windows: games.csv refetch failed ({e}); using the cached copy")
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def group(sched, season, week):
    """Kickoff windows for one week: games whose kickoffs fall within WINDOW_SPAN of each other."""
    g = sched[(sched.season == season) & (sched.week == week) & (sched.game_type == "REG")].copy()
    rows = []
    for r in g.itertuples():
        k = kickoff_utc(r.gameday, r.gametime)
        if k is None:
            continue
        rows.append({"game_id": r.game_id, "kick": k, "final": pd.notna(r.home_score) and pd.notna(r.away_score),
                     "home": r.home_team, "away": r.away_team})
    rows.sort(key=lambda x: x["kick"])
    windows, cur = [], None
    for r in rows:
        if cur is None or r["kick"] - cur["last"] > WINDOW_SPAN:
            cur = {"kick": r["kick"], "last": r["kick"], "games": []}
            windows.append(cur)
        cur["games"].append(r)
        cur["last"] = r["kick"]
    out = []
    for w in windows:
        local = w["kick"].astimezone(ET)
        label = f"{local.strftime('%A')} {local.strftime('%I:%M %p').lstrip('0')} ET game{'s' if len(w['games']) > 1 else ''}"
        out.append({"key": w["kick"].strftime("%Y-%m-%dT%H:%M"), "label": label, "kickoff_utc": _iso(w["kick"]),
                    "game_ids": [x["game_id"] for x in w["games"]], "n": len(w["games"]),
                    "n_final": sum(1 for x in w["games"] if x["final"]),
                    "all_final": all(x["final"] for x in w["games"])})
    return out


def _stats_signature(season):
    """ETag + length of the current season's weekly box-score file, from a HEAD request."""
    url = f"{REL}/stats_player/stats_player_week_{season}.parquet"
    try:
        r = subprocess.run(["curl", "-sSIL", "--max-time", "30", url], capture_output=True, text=True, timeout=45)
        sig = []
        for ln in r.stdout.splitlines():
            k, _, v = ln.partition(":")
            if k.strip().lower() in ("etag", "content-length"):
                sig.append(v.strip())
        return "|".join(sig[-2:]) if sig else None
    except Exception:
        return None


def check(state, season, week, now=None, datadir="data", log=print):
    """
    -> (reasons, windows). Reasons is a list of strings for refresh_request.json; empty when
    nothing new finished. Marks what it reported in state['windows_done'] / state['stats_sig'].
    """
    sched = fresh_schedule(datadir, log)
    if sched is None:
        return [], []
    wins = group(sched, season, week)
    bootstrap = "windows_done" not in state          # first run: record what is already final, report nothing
    done = state.setdefault("windows_done", {})
    reasons = []
    for w in wins:
        if w["all_final"] and w["key"] not in done:
            done[w["key"]] = {"at": store.now_iso(), "label": w["label"], "n": w["n"], "bootstrap": bootstrap}
            if not bootstrap:
                reasons.append(f"Window finished: {w['label']} ({w['n']} final)")
    # box scores: change-only, and only once this week has something to grade
    any_final = any(w["n_final"] for w in wins)
    sig = _stats_signature(season)
    prev = state.get("stats_sig")
    if sig and prev and sig != prev and any_final:
        reasons.append("Box scores updated: player results and the paper test can be graded")
    if sig:
        state["stats_sig"] = sig
    state["windows"] = [{k: w[k] for k in ("label", "kickoff_utc", "n", "n_final", "all_final")} for w in wins]
    if reasons:
        log("  windows: " + " | ".join(reasons))
    return reasons, wins
