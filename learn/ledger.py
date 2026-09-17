"""
The forecast ledger: what was predicted before kickoff, and what happened.

Two append-only files under learn/data/:

  forecasts.ndjson   one row per player-week revision, written only before kickoff. A later
                     run that changes the number (injury news, new odds) appends a new
                     revision; nothing is ever edited. The last revision before kickoff is
                     the forecast that gets graded.
  actuals.ndjson     one row per graded player-week. If a later stats file changes the
                     result, a new row is appended with `correction_of` pointing at the old
                     one; the latest row wins and the correction stays visible.

A player who did not appear in a finished game is recorded with played=false and the points
the box score gives him (zero) — explicitly, as a did-not-play, never as a silently filled
gap. A game whose box score has not landed is left ungraded.
"""
import json
import os
from datetime import datetime, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "data")
FORECASTS = os.path.join(STORE, "forecasts.ndjson")
ACTUALS = os.path.join(STORE, "actuals.ndjson")
STATS = ["passing_yards", "passing_tds", "passing_interceptions", "rushing_yards", "rushing_tds",
         "receptions", "receiving_yards", "receiving_tds", "targets", "carries", "attempts", "fantasy_points_ppr"]


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def _append(path, rows):
    if not rows:
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return len(rows)


def kickoffs(sched, season, week):
    """game_id -> kickoff UTC ISO for a week, from the schedule's gameday + gametime (ET)."""
    from live.weather import kickoff_utc, _iso
    out = {}
    g = sched[(sched.season == season) & (sched.week == week) & (sched.game_type == "REG")]
    for r in g.itertuples():
        k = kickoff_utc(r.gameday, r.gametime)
        out[r.game_id] = _iso(k) if k is not None else None
    teams = {}
    for r in g.itertuples():
        teams[r.home_team] = r.game_id
        teams[r.away_team] = r.game_id
    return out, teams


def record_forecasts(fantasy, sched, model_version, data_version, now=None, path=FORECASTS, log=print):
    """
    Append a revision for every fantasy player whose game has not kicked off and whose
    forecast changed since the last recorded revision. Returns (written, skipped_after_kickoff).
    """
    if not fantasy or not fantasy.get("players"):
        return 0, 0
    season, week = int(fantasy["season"]), int(fantasy["week"])
    now = now or now_iso()
    koff, team_game = kickoffs(sched, season, week)
    existing = {}
    for r in _read(path):
        if r["season"] == season and r["week"] == week:
            existing[r["player_key"]] = r          # last revision wins (file is chronological)
    rows, late = [], 0
    for p in fantasy["players"]:
        gid = team_game.get(p["team"])
        k = koff.get(gid)
        if k and now >= k:
            late += 1
            continue                               # after kickoff: never recorded, never revised
        prev = existing.get(p["player_key"])
        sig = (p.get("proj_pts"), p.get("p_play"), (p.get("availability") or {}).get("status"))
        if prev and (prev.get("proj_pts"), prev.get("p_play"), prev.get("status")) == sig:
            continue
        rev = (prev["revision"] + 1) if prev else 1
        rows.append({"id": f"{season}_{week:02d}_{p['player_key']}_r{rev}", "season": season, "week": week,
                     "player_key": p["player_key"], "name": p["name"], "position": p["position"], "team": p["team"],
                     "opponent": p.get("opponent"), "game_id": gid, "kickoff_utc": k, "created_at": now, "revision": rev,
                     "model_version": model_version, "data_version": data_version,
                     "proj": p.get("proj"), "proj_pts": p.get("proj_pts"), "naive_pts": p.get("naive_pts"),
                     "range": (p.get("range") or {}).get("full_ppr"), "p_play": p.get("p_play"), "exp_pts": p.get("exp_pts"),
                     "status": (p.get("availability") or {}).get("status"),
                     "inputs": dict(p.get("opportunity") or {}, depth_rank=p.get("depth_rank"),
                                    flags=[f["kind"] for f in (p.get("flags") or [])])})
    n = _append(path, rows)
    log(f"  ledger: {n} forecast revision(s) recorded for week {week}; {late} skipped (kickoff passed); {len(existing)} already on file")
    return n, late


def locked(path=FORECASTS):
    """The forecast that counts per (season, week, player): the last revision before kickoff."""
    best = {}
    for r in _read(path):
        if r.get("kickoff_utc") and r["created_at"] >= r["kickoff_utc"]:
            continue
        key = (r["season"], r["week"], r["player_key"])
        if key not in best or r["revision"] > best[key]["revision"]:
            best[key] = r
    return best


def record_actuals(plyr, sched, season, source_digest=None, path=ACTUALS, fpath=FORECASTS, now=None, log=print):
    """
    Grade every locked forecast whose game is final and whose box score has landed. A changed
    actual for an already-graded row is appended as a correction. Returns (graded, corrected).
    """
    now = now or now_iso()
    fc = locked(fpath)
    if not fc:
        return 0, 0
    reg = sched[(sched.season == season) & (sched.game_type == "REG")]
    final = set(reg[reg.home_score.notna()].game_id)
    st = plyr[(plyr.season == season) & (plyr.season_type == "REG")]
    scored_games = set(st.game_id.dropna())
    idx = {(r.game_id, r.player_id): r for r in st.itertuples(index=False)}
    latest = {}
    for r in _read(path):
        latest[(r["season"], r["week"], r["player_key"])] = r
    rows, corrected = [], 0
    for key, f in fc.items():
        gid = f.get("game_id")
        if not gid or gid not in final or gid not in scored_games:
            continue                               # not final, or box score not posted: stays ungraded
        row = idx.get((gid, f["player_key"]))
        if row is None:
            act = {"played": False, "act_pts": 0.0, "stats": None, "note": "no box-score row in a finished game: did not play"}
        else:
            d = row._asdict()
            stats = {k: (None if d.get(k) is None or pd.isna(d.get(k)) else float(d.get(k))) for k in STATS if k in d}
            act = {"played": True, "act_pts": float(stats.get("fantasy_points_ppr") or 0.0), "stats": stats, "note": None}
        prev = latest.get(key)
        if prev and prev["played"] == act["played"] and abs(float(prev["act_pts"]) - act["act_pts"]) < 1e-6:
            continue
        rid = f"{key[0]}_{key[1]:02d}_{key[2]}_a{(prev['seq'] + 1) if prev else 1}"
        rec = {"id": rid, "seq": (prev["seq"] + 1) if prev else 1, "season": key[0], "week": key[1], "player_key": key[2],
               "game_id": gid, "forecast_id": f["id"], "recorded_at": now, "source": "nflverse stats_player_week",
               "source_digest": source_digest, "correction_of": prev["id"] if prev else None, **act}
        if prev:
            corrected += 1
        rows.append(rec)
    n = _append(path, rows)
    log(f"  ledger: {n} actual(s) recorded ({corrected} correction(s)); {sum(1 for f in fc.values() if f.get('game_id') in final and f.get('game_id') not in scored_games)} finished game rows waiting for a box score")
    return n, corrected


def graded(fpath=FORECASTS, apath=ACTUALS):
    """One row per graded forecast: the locked forecast joined to the latest actual."""
    fc = locked(fpath)
    latest = {}
    for r in _read(apath):
        latest[(r["season"], r["week"], r["player_key"])] = r
    rows = []
    for key, f in fc.items():
        a = latest.get(key)
        if not a:
            continue
        rows.append({"season": key[0], "week": key[1], "player_key": key[2], "name": f["name"], "position": f["position"],
                     "team": f["team"], "opponent": f.get("opponent"), "proj_pts": f.get("proj_pts"), "naive_pts": f.get("naive_pts"),
                     "exp_pts": f.get("exp_pts"), "p_play": f.get("p_play"), "status": f.get("status"), "range": f.get("range"),
                     "model_version": f.get("model_version"), "data_version": f.get("data_version"), "revision": f["revision"],
                     "act_pts": a["act_pts"], "played": a["played"], "corrected": a.get("correction_of") is not None,
                     "flags": (f.get("inputs") or {}).get("flags", [])})
    return pd.DataFrame(rows)


def pending(fpath=FORECASTS, apath=ACTUALS):
    fc = locked(fpath)
    have = {(r["season"], r["week"], r["player_key"]) for r in _read(apath)}
    return [f for k, f in fc.items() if k not in have]
