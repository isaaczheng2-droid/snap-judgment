"""
Data manifest: what the pipeline read, from where, and when. Every run appends one record to
learn/data/manifests.ndjson; the `data_version` (hash of the file digests) travels with every
forecast in the ledger so a number can always be traced to the files that produced it.
"""
import glob
import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "data")
PATH = os.path.join(STORE, "manifests.ndjson")
REL = "https://github.com/nflverse/nflverse-data/releases/download"
BIG = 40 * 1024 * 1024        # files above this are identified by size + mtime, not a full digest


def _digest(path):
    size = os.path.getsize(path)
    if size > BIG:
        return f"size:{size}:mtime:{int(os.path.getmtime(path))}"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _extent(path):
    """Rows and the latest (season, week) inside a parquet/csv file, when it has them."""
    try:
        if path.endswith(".parquet"):
            d = pd.read_parquet(path)
        elif path.endswith(".csv"):
            d = pd.read_csv(path, low_memory=False)
        else:
            return {}
        out = {"rows": int(len(d))}
        if "season" in d.columns and "week" in d.columns and len(d):
            last = d[d.season == d.season.max()]
            wk = last.week.max()
            out["max_season"] = int(d.season.max())
            out["max_week"] = int(wk) if pd.notna(wk) else None
            if "home_score" in d.columns:      # schedule: the latest week with a result
                done = d[d.home_score.notna() & (d.season == d.season.max())]
                out["max_week_final"] = int(done.week.max()) if len(done) else None
        return out
    except Exception as e:
        return {"error": str(e)[:80]}


def sources(datadir, cur):
    """The files a run depends on, with their upstream. Missing files are listed as missing."""
    s = [
        ("schedule", f"{datadir}/games.csv", "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"),
        ("player_stats", f"{datadir}/stats_player/stats_player_week_{cur}.parquet", f"{REL}/stats_player/stats_player_week_{cur}.parquet"),
        ("team_stats", f"{datadir}/stats_team/stats_team_week_{cur}.parquet", f"{REL}/stats_team/stats_team_week_{cur}.parquet"),
        ("injuries", f"{datadir}/injuries/injuries_{cur}.parquet", f"{REL}/injuries/injuries_{cur}.parquet"),
        ("snap_counts", f"{datadir}/snap_counts/snap_counts_{cur}.parquet", f"{REL}/snap_counts/snap_counts_{cur}.parquet"),
        ("rosters", f"{datadir}/rosters/roster_{cur}.parquet", f"{REL}/rosters/roster_{cur}.parquet"),
        ("depth_charts", f"{datadir}/depth/depth_charts_{cur}.parquet", f"{REL}/depth_charts/depth_charts_{cur}.parquet"),
        ("play_by_play", f"{datadir}/pbp/play_by_play_{cur}.parquet", f"{REL}/pbp/play_by_play_{cur}.parquet"),
        ("odds_cache", "live/data/odds_cache.json", "The Odds API (FanDuel), via odds_api.py"),
        ("live_injuries", "live/data/injury_snapshots.ndjson", "ESPN + nflverse via run_live.py"),
        ("live_weather", "live/data/weather_snapshots.ndjson", "Open-Meteo + NWS via run_live.py"),
        ("prop_lines", "live/data/prop_lines.ndjson", "The Odds API line history via odds_api.py"),
    ]
    out = []
    for name, path, upstream in s:
        rec = {"source": name, "path": path, "upstream": upstream}
        if os.path.exists(path):
            rec.update({"present": True, "bytes": os.path.getsize(path), "digest": _digest(path),
                        "modified": datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
            rec.update(_extent(path))
        else:
            rec["present"] = False
        out.append(rec)
    return out


def snapshot(datadir, cur, write=True):
    srcs = sources(datadir, cur)
    key = "|".join(f"{r['source']}:{r.get('digest', 'missing')}" for r in srcs)
    rec = {"at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "season": int(cur),
           "data_version": hashlib.sha256(key.encode()).hexdigest()[:12], "sources": srcs}
    if write:
        os.makedirs(STORE, exist_ok=True)
        with open(PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def latest(path=PATH):
    try:
        last = None
        with open(path) as f:
            for line in f:
                if line.strip():
                    last = line
        return json.loads(last) if last else None
    except Exception:
        return None
