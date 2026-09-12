"""
Append-only tables as NDJSON files under live/data/.

Why files and not a database: the site's only compute is a GitHub Actions job and its only
durable storage is the repository. One JSON object per line diffs cleanly in git, survives a
crashed run (a partial last line is skipped, never fatal), and is queryable with DuckDB or
pandas when the research questions come ("how did forecasts move before games", "how did
probabilities evolve"). Rows are never edited in place; state is derived by taking the
latest row per key. `cache.json` holds only the small operational cache; the rest is rebuilt from the tables.

Every row gets `system_received_at` stamped here, so no caller can forget it.
"""
import json
import os
import time
import uuid
from datetime import datetime, timezone

ROOT = os.environ.get("SJ_LIVE_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TABLES = [
    "player_status",          # one row per (player, game, source, snapshot)
    "player_status_history",  # one row per meaningful status transition
    "player_practice_reports",# practice participation per player-date
    "depth_charts",           # depth chart snapshots per team
    "weather_forecasts",      # one row per (game, provider, fetch)
    "weather_alerts",         # active NWS alerts touching a stadium
    "weather_observed",       # post-game observed conditions
    "live_events",            # the event log
    "prediction_versions",    # every published prediction, never overwritten
    "prediction_change_log",  # field-level diffs between versions
    "api_sync_log",           # every provider call
    "source_conflicts",       # where two sources disagreed and how it was resolved
    "data_quality",           # validation failures, kept rather than dropped silently
]


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_id(prefix):
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def path(table):
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}")
    os.makedirs(ROOT, exist_ok=True)
    return os.path.join(ROOT, f"{table}.ndjson")


def append(table, rows):
    """Append rows (dict or list of dicts). Returns the number written."""
    if isinstance(rows, dict):
        rows = [rows]
    rows = [r for r in rows if r]
    if not rows:
        return 0
    stamp = now_iso()
    with open(path(table), "a") as f:
        for r in rows:
            r.setdefault("system_received_at", stamp)
            f.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")
    return len(rows)


def read(table, limit=None, where=None):
    """Read rows oldest-first. A torn final line (crashed writer) is skipped, not fatal."""
    p = path(table)
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if where is None or where(r):
                out.append(r)
    return out[-limit:] if limit else out


def latest(table, key, where=None):
    """Latest row per key. `key` is a field name or a function of the row."""
    kf = key if callable(key) else (lambda r: r.get(key))
    cur = {}
    for r in read(table, where=where):
        cur[kf(r)] = r
    return cur


def tail(table, n=50, where=None):
    rows = read(table, where=where)
    return rows[-n:][::-1]


# ---------------------------------------------------------------- state
# Everything large is derived from the append-only tables on demand, so the only file that
# changes every run is a small cache (grid metadata, sync times, alerts). That keeps the
# repository's history growing by what actually changed, not by a snapshot per poll.
CACHE_PATH = os.path.join(ROOT, "cache.json")
CACHE_KEYS = ("nws_grid", "last_sync", "last_poll", "sources", "kickoffs", "depth", "alerts", "snapshot_id", "conflicts_seen")


def load_state():
    try:
        return json.load(open(CACHE_PATH))
    except Exception:
        return {}


def save_state(state):
    os.makedirs(ROOT, exist_ok=True)
    slim = {k: state.get(k) for k in CACHE_KEYS if state.get(k) is not None}
    slim["saved_at"] = now_iso()
    tmp = CACHE_PATH + ".tmp"
    json.dump(slim, open(tmp, "w"), separators=(",", ":"), default=str)
    os.replace(tmp, CACHE_PATH)


def hydrate(state, season, week, game_ids):
    """Rebuild the derived parts of the state (players, raw, forecasts) from the tables."""
    from . import normalize
    cur = lambda r: str(r.get("season")) == str(season) and str(r.get("week")) == str(week)
    raw_rows = latest("player_status", lambda r: f"{r.get('internal_player_id')}|{r.get('source')}", where=cur)
    by_pid = {}
    for r in raw_rows.values():
        by_pid.setdefault(r.get("internal_player_id"), []).append(r)
    merged, _ = normalize.merge_sources([r for rs in by_pid.values() for r in rs])
    state["players"] = merged
    state["raw"] = {k: {f: r.get(f) for f in ("normalized_status", "practice_status", "estimated_return_date", "injury_body_part",
                                              "game_status", "depth_order", "injury_description")} for k, r in raw_rows.items()}
    gids = set(game_ids or [])
    fc = latest("weather_forecasts", lambda r: (r.get("game_id"), r.get("provider")), where=lambda r: r.get("game_id") in gids)
    forecasts = {}
    for (gid, prov), r in fc.items():
        forecasts.setdefault(gid, {})[prov] = {k: v for k, v in r.items() if k != "hourly"}
    state["forecasts"] = forecasts
    state.setdefault("alerts", {})
    return state


def quality(kind, detail, **fields):
    """Record a validation failure instead of dropping it on the floor."""
    append("data_quality", {"kind": kind, "detail": detail, **fields})
