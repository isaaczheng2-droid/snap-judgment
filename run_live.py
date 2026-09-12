#!/usr/bin/env python3
"""
The light poll: keep the live layer current without retraining anything.

    sources -> ingest -> validate -> store -> detect changes -> evaluate impact
            -> (request a model refresh if justified) -> update payload live context
            -> splice into the page

Runs in minutes, costs nothing, and never touches a prediction. The only thing it can do to
the prediction engine is leave a refresh request behind; the workflow then runs the full
pipeline, which records a new prediction version with the events as its reason.

    python3 run_live.py                       # poll whatever is due
    python3 run_live.py --force               # poll every game on the slate
    SJ_ESPN_FIXTURE=... SJ_WX_FIXTURES=dir    # offline, for tests and the dev machines
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

import espn_injuries
from live import store, teams, identity, normalize, stadiums, weather, events, impact, context

REL = "https://github.com/nflverse/nflverse-data/releases/download"

# adaptive cadence: how often a game is polled, by hours to kickoff
CADENCE = [(1.5, 0.25), (6, 1.0), (24, 2.0), (72, 6.0), (24 * 7, 12.0), (None, 24.0)]
KEY_SLOTS = {"QB", "RB", "WR", "TE", "LT", "RT", "LG", "RG", "C", "K"}


def log(*a):
    print(*a, flush=True)


def _clean(v):
    """pandas hands back NaN for missing strings; the store wants None, never 'nan'."""
    if v is None:
        return None
    try:
        if isinstance(v, float) and v != v:
            return None
    except Exception:
        pass
    s = str(v).strip()
    return None if s in ("", "nan", "None", "NaT") else s


def cadence_hours(h_to_kick):
    for limit, every in CADENCE:
        if limit is None or h_to_kick <= limit:
            return every
    return 24.0


# ------------------------------------------------------------------ inputs
def fetch_parquet(name, datadir):
    """nflverse release asset, with the cached copy as fallback. Never raises."""
    local = os.path.join(datadir, os.path.basename(name))
    os.makedirs(datadir, exist_ok=True)
    url = f"{REL}/{name}"
    try:
        r = subprocess.run(["curl", "-sSL", "--max-time", "60", "-o", local + ".new", url], capture_output=True, timeout=80)
        if r.returncode == 0 and os.path.getsize(local + ".new") > 1000:
            pd.read_parquet(local + ".new")
            os.replace(local + ".new", local)
    except Exception:
        pass
    try:
        return pd.read_parquet(local)
    except Exception:
        return None


def slate(payload):
    games = []
    for g in payload.get("games") or []:
        games.append({"game_id": g["game_id"], "home_team": g["home_team"], "away_team": g["away_team"],
                      "gameday_iso": g.get("gameday_iso"), "kickoff": g.get("kickoff"), "roof": g.get("roof"),
                      "stadium_id": g.get("stadium_id")})
    return games


# ------------------------------------------------------------------ players
def ingest_players(games, season, week, rost, inj_df, depth_df, espn_fixture, state):
    """-> (rows, sync_summary). Sources merged per player; conflicts recorded."""
    xw = identity.Crosswalk.from_roster(rost, season) if rost is not None else identity.Crosswalk([])
    xw.write()
    team_game = {}
    for g in games:
        team_game[g["home_team"]] = g; team_game[g["away_team"]] = g
    now = datetime.now(timezone.utc)
    summary = {"espn": {"status": "skipped"}, "nflverse": {"status": "skipped"}}

    # depth charts: latest snapshot per team, order per slot
    depth_order, depth_slots, names = {}, {}, {}
    if depth_df is not None and len(depth_df):
        d = depth_df.copy()
        d = d[d.dt == d.groupby("team").dt.transform("max")]
        for r in d.sort_values(["team", "pos_abb", "pos_rank"]).itertuples():
            if not isinstance(r.gsis_id, str):
                continue
            slot = str(r.pos_abb or "").upper()
            depth_order[r.gsis_id] = min(int(r.pos_rank), depth_order.get(r.gsis_id, 99))
            names[r.gsis_id] = r.player_name
            if slot in KEY_SLOTS:
                depth_slots.setdefault((teams.to_canonical(r.team) or r.team, slot), []).append(r.gsis_id)

    rows = []
    # --- ESPN (primary): timestamped, includes in-game injuries
    raw = espn_injuries.parse_psv(espn_fixture) if espn_fixture else espn_injuries.parse(espn_injuries.fetch())
    if raw is not None and len(raw):
        n = 0
        for r in raw.itertuples():
            team = teams.to_canonical(r.team, "espn")
            g = team_game.get(team)
            pid, how = xw.resolve("espn", r.espn_id, r.full_name, team)
            st = normalize.norm_status(r.espn_status, getattr(r, "injury_type", None))
            kick = (state.get("kickoffs") or {}).get(g["game_id"]) if g else None
            gameday = False
            if kick:
                h = context._hours_to(kick, now)
                gameday = h is not None and -4 <= h <= 2
            rows.append(normalize.record(
                internal_player_id=pid, player_name=r.full_name, team=team, position=r.position, game_id=g["game_id"] if g else None,
                season=season, week=week, original_status=r.espn_status, normalized_status=st or "UNKNOWN",
                injury_body_part=(None if str(getattr(r, "injury_type", "") or "").lower() in normalize.SCRATCH_TYPES else _clean(getattr(r, "injury_type", None))),
                injury_description=_clean(getattr(r, "note", None)), estimated_return_date=_clean(getattr(r, "return_date", None)),
                game_status=st if (gameday and st in ("ACTIVE", "INACTIVE")) else None,
                depth_order=depth_order.get(pid), source="espn",
                source_updated_at=(r.updated.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(r.updated) else None),
                last_verified_at=store.now_iso(), snapshot_id=state.get("snapshot_id")))
            n += 1
        summary["espn"] = {"status": "ok", "records": n, "at": store.now_iso()}
    else:
        summary["espn"] = {"status": "error", "error": "no rows"}

    # --- nflverse official report (fallback + practice participation)
    if inj_df is not None and len(inj_df):
        i = inj_df[(inj_df.season == season) & (inj_df.week == week)] if "week" in inj_df.columns else inj_df.iloc[0:0]
        n = 0
        for r in i.itertuples():
            team = teams.to_canonical(r.team)
            g = team_game.get(team)
            pid = r.gsis_id if isinstance(r.gsis_id, str) else xw.resolve(None, None, r.full_name, team)[0]
            st = normalize.norm_status(r.report_status)
            pr = normalize.norm_practice(getattr(r, "practice_status", None))
            rows.append(normalize.record(
                internal_player_id=pid, player_name=r.full_name, team=team, position=r.position, game_id=g["game_id"] if g else None,
                season=season, week=week, original_status=r.report_status if isinstance(r.report_status, str) else None,
                normalized_status=st or (pr or "UNKNOWN"), injury_body_part=_clean(getattr(r, "report_primary_injury", None)) or _clean(getattr(r, "practice_primary_injury", None)),
                original_practice=getattr(r, "practice_status", None) if isinstance(getattr(r, "practice_status", None), str) else None,
                practice_status=pr, depth_order=depth_order.get(pid), source="nflverse:injuries",
                source_updated_at=(str(getattr(r, "date_modified")) if isinstance(getattr(r, "date_modified", None), str) else None),
                last_verified_at=store.now_iso(), snapshot_id=state.get("snapshot_id")))
            n += 1
        summary["nflverse"] = {"status": "ok", "records": n, "at": store.now_iso()}
    else:
        summary["nflverse"] = {"status": "error", "error": "no rows for this week"}

    # roster status for anyone on IR/PUP/suspended who is on neither report
    if rost is not None:
        rs = rost[rost.season == season] if "season" in rost.columns else rost
        listed = {r["internal_player_id"] for r in rows}
        for r in rs[rs.status.isin(["RES", "PUP", "NON", "SUS"])].itertuples():
            if r.gsis_id in listed or not isinstance(r.gsis_id, str):
                continue
            team = teams.to_canonical(r.team)
            g = team_game.get(team)
            if not g:
                continue
            rows.append(normalize.record(internal_player_id=r.gsis_id, player_name=r.full_name, team=team, position=r.position,
                                         game_id=g["game_id"], season=season, week=week, original_status=r.status,
                                         normalized_status=normalize.norm_roster(r.status), roster_status=normalize.norm_roster(r.status),
                                         depth_order=depth_order.get(r.gsis_id), source="nflverse:rosters",
                                         last_verified_at=store.now_iso(), snapshot_id=state.get("snapshot_id")))

    ok, problems = normalize.validate(rows, xw, [g["game_id"] for g in games], now, previous=state.get("players") or {})
    if problems:
        log(f"  validation: {len(problems)} row(s) rejected or flagged ({', '.join(sorted({p[0] for p in problems}))})")
    merged, conflicts = normalize.merge_sources(ok)
    seen = set(state.get("conflicts_seen") or [])
    fresh = [c for c in conflicts if f"{c['player_id']}|{c['sources']['espn']}|{c['sources']['nflverse:injuries']}" not in seen]
    if fresh:
        store.append("source_conflicts", fresh)
    state["conflicts_seen"] = sorted(seen | {f"{c['player_id']}|{c['sources']['espn']}|{c['sources']['nflverse:injuries']}" for c in conflicts})[-500:]
    return ok, merged, depth_slots, names, summary


# ------------------------------------------------------------------ weather
def ingest_weather(games, state, force, fixtures_dir):
    now = datetime.now(timezone.utc)
    forecasts = state.setdefault("forecasts", {})
    alerts = state.setdefault("alerts", {})
    last_poll = state.setdefault("last_poll", {})
    new_events = []
    polled = 0
    for g in games:
        stadium = stadiums.get(g.get("stadium_id")) or stadiums.for_team(g["home_team"])
        if not stadium:
            store.quality("unknown_stadium", f"{g['game_id']}: no stadium row", game_id=g["game_id"]); continue
        kick = weather.kickoff_utc(g.get("gameday_iso"), g.get("kickoff"))
        if not kick:
            continue
        state.setdefault("kickoffs", {})[g["game_id"]] = weather._iso(kick)
        h = (kick - now).total_seconds() / 3600
        if h < -5:
            continue                                     # played; observed weather is a separate job
        roof = stadiums.roof_status(stadium, g.get("roof"))
        if roof["outdoor_weather_applies"] is False:
            continue                                     # fixed dome or reported closed: no forecast needed
        due = cadence_hours(h)
        lp = last_poll.get(g["game_id"])
        if not force and lp and (context._age_h(lp, now) or 0) < due:
            continue
        fx = None
        if fixtures_dir:
            fx = {"openmeteo": _fx(fixtures_dir, "openmeteo"), "nws_points": _fx(fixtures_dir, "nws_points"),
                  "nws_hourly": _fx(fixtures_dir, "nws_hourly"), "nws_alerts": _fx(fixtures_dir, "nws_alerts")}
        res = weather.fetch_game(g, stadium, fixtures=fx, log=log, state=state)
        polled += 1
        last_poll[g["game_id"]] = store.now_iso()
        prev = forecasts.get(g["game_id"], {})
        for prov, fc in res["forecasts"].items():
            store.append("weather_forecasts", {"game_id": g["game_id"], "stadium_id": stadium["stadium_id"], "kickoff_utc": res["kickoff_utc"],
                                               **{k: v for k, v in fc.items() if k != "hourly"}, "hourly": fc.get("hourly")})
            if prov == "openmeteo" or "openmeteo" not in res["forecasts"]:
                new_events += events.weather_changes(g["game_id"], prev.get(prov), fc, alerts.get(g["game_id"]), res["alerts"], prov)
        if res["alerts"]:
            store.append("weather_alerts", res["alerts"])
        forecasts[g["game_id"]] = {**prev, **{p: {k: v for k, v in fc.items() if k != "hourly"} for p, fc in res["forecasts"].items()}}
        alerts[g["game_id"]] = res["alerts"]
        if res["errors"]:
            log(f"  weather {g['game_id']}: " + "; ".join(res["errors"]))
    if polled:
        state.setdefault("last_sync", {})["weather"] = store.now_iso()
    return new_events, polled


def _fx(d, key):
    m = sorted(glob.glob(os.path.join(d, f"{key}*.json")))
    return m[-1] if m else None


# ------------------------------------------------------------------ admin page
def write_admin(state, summary, out_path):
    syncs = store.tail("api_sync_log", 40)
    evs = events.read_events(limit=25)
    q = store.tail("data_quality", 15)
    by_prov = {}
    for s in store.read("api_sync_log", limit=400):
        b = by_prov.setdefault(s.get("provider"), {"calls": 0, "errors": 0, "latency": [], "last_ok": None})
        b["calls"] += 1
        if s.get("status") != "ok": b["errors"] += 1
        else: b["last_ok"] = s.get("completed_at")
        if s.get("latency_ms") is not None: b["latency"].append(s["latency_ms"])
    row = lambda cells: "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="robots" content="noindex"><title>Snap Judgment · live data diagnostics</title>
<style>body{{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#111;background:#fafafa}}table{{border-collapse:collapse;width:100%;margin:8px 0 22px;background:#fff}}td,th{{border-bottom:1px solid #e5e5e5;padding:6px 8px;text-align:left;font-size:13px;vertical-align:top}}th{{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#666}}h1{{font-size:20px}}h2{{font-size:15px;margin:22px 0 6px}}.ok{{color:#0b7a45}}.err{{color:#c0271a}}code{{font-size:12px}}</style></head><body>
<h1>Live data diagnostics</h1><p>Generated {store.now_iso()} · games monitored {len(state.get('kickoffs') or {})} · last injuries sync {(state.get('last_sync') or {}).get('injuries') or '—'} · last weather sync {(state.get('last_sync') or {}).get('weather') or '—'}</p>
<h2>Providers (last 400 calls)</h2><table><tr><th>Provider</th><th>Calls</th><th>Errors</th><th>Median latency</th><th>Last OK</th></tr>
{''.join(row([p, b['calls'], f"<span class='{'err' if b['errors'] else 'ok'}'>{b['errors']}</span>", (str(sorted(b['latency'])[len(b['latency'])//2]) + ' ms') if b['latency'] else '—', b['last_ok'] or '—']) for p, b in by_prov.items())}</table>
<h2>This run</h2><pre><code>{json.dumps(summary, indent=1)}</code></pre>
<h2>Recent calls</h2><table><tr><th>When</th><th>Provider</th><th>Endpoint</th><th>Status</th><th>HTTP</th><th>ms</th><th>Retries</th><th>Error</th></tr>
{''.join(row([s.get('completed_at'), s.get('provider'), s.get('endpoint'), f"<span class='{'ok' if s.get('status')=='ok' else 'err'}'>{s.get('status')}</span>", s.get('http_status'), s.get('latency_ms'), s.get('retry_count'), (s.get('error') or '')[:80]]) for s in syncs)}</table>
<h2>Recent events</h2><table><tr><th>At</th><th>Type</th><th>Severity</th><th>Game</th><th>Detail</th><th>Recalculated</th></tr>
{''.join(row([e.get('occurred_at'), e.get('event_type'), e.get('severity'), e.get('game_id') or e.get('team_id') or '', e.get('detail'), '✓' if e.get('prediction_recalculated') else '']) for e in evs)}</table>
<h2>Data quality flags</h2><table><tr><th>At</th><th>Kind</th><th>Detail</th></tr>{''.join(row([x.get('system_received_at'), x.get('kind'), x.get('detail')]) for x in q)}</table>
<p style="color:#888;font-size:12px">Internal diagnostics. Not linked from the site. Weather data by Open-Meteo.com (CC BY 4.0) and the National Weather Service; player status from ESPN and the official NFL report via nflverse.</p></body></html>"""
    open(out_path, "w").write(html)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-weather", action="store_true")
    ap.add_argument("--no-players", action="store_true")
    a = ap.parse_args()

    payload_path = os.path.join(a.outdir, "payload.json")
    payload = json.load(open(payload_path))
    season, week = int(payload["season"]), int(payload["week"])
    games = slate(payload)
    if not games:
        log("no games on the slate; nothing to monitor"); return 0
    state = store.load_state()
    store.hydrate(state, season, week, [g["game_id"] for g in games])
    state["snapshot_id"] = store.new_id("snap")
    summary = {"run_started": store.now_iso(), "season": season, "week": week, "games": len(games)}
    new_events = []

    # kickoffs first: everything downstream keys on hours-to-kickoff
    for g in games:
        k = weather.kickoff_utc(g.get("gameday_iso"), g.get("kickoff"))
        if k:
            state.setdefault("kickoffs", {})[g["game_id"]] = weather._iso(k)

    if not a.no_players:
        rost = fetch_parquet(f"rosters/roster_{season}.parquet", a.datadir)
        inj = fetch_parquet(f"injuries/injuries_{season}.parquet", a.datadir)
        depth = fetch_parquet(f"depth_charts/depth_charts_{season}.parquet", a.datadir)
        fixture = os.environ.get("SJ_ESPN_FIXTURE") or None
        rows, merged, depth_slots, names, s = ingest_players(games, season, week, rost, inj, depth, fixture, state)
        summary["players"] = s
        # the table records observations that CHANGED something; an identical re-read only
        # bumps last_verified_at in the state cache. Every status change is therefore in the
        # table exactly once, and the table does not grow by the roster every poll.
        prev_raw = state.get("raw") or {}
        DIFF = ("normalized_status", "practice_status", "estimated_return_date", "injury_body_part", "game_status", "depth_order", "injury_description")
        changed = [r for r in rows if f"{r['internal_player_id']}|{r['source']}" not in prev_raw
                   or any(prev_raw[f"{r['internal_player_id']}|{r['source']}"].get(k) != r.get(k) for k in DIFF)]
        store.append("player_status", changed)
        state["raw"] = {**prev_raw, **{f"{r['internal_player_id']}|{r['source']}": {k: r.get(k) for k in DIFF} for r in rows}}
        summary["players"]["changed_rows"] = len(changed)
        # usage for impact tiers, from the payload's game logs (last game's shares)
        usage = {}
        for p in payload.get("players") or []:
            lg = (p.get("log") or [{}])[-1]
            usage[p["player_key"]] = {k: lg.get(k) for k in ("tgt_share", "car_share") if lg.get(k) is not None}
            usage[p["player_key"]]["is_starting_qb"] = p.get("position") == "QB"
        state["usage"] = usage
        prev_players = state.get("players") or {}
        meta = {pid: {**usage.get(pid, {}), "depth_order": (r.get("depth_order"))} for pid, r in merged.items()}
        ev, hist = events.player_changes(prev_players, merged, meta)
        prev_depth = {tuple(k.split("|")): v for k, v in (state.get("depth") or {}).items()}
        dev = events.depth_changes(prev_depth, depth_slots, names) if prev_depth else []
        new_events += ev + dev
        store.append("player_status_history", hist)
        SLIM = ("internal_player_id", "player_name", "team", "position", "game_id", "original_status", "normalized_status",
                "injury_body_part", "practice_status", "estimated_return_date", "game_status", "depth_order", "source",
                "source_updated_at", "last_verified_at", "roster_status")
        state["players"] = {**prev_players, **{pid: {k: v.get(k) for k in SLIM} for pid, v in merged.items()}}
        state["depth"] = {"|".join(k): v for k, v in depth_slots.items()}
        state.setdefault("last_sync", {})["injuries"] = store.now_iso() if s.get("espn", {}).get("status") == "ok" or s.get("nflverse", {}).get("status") == "ok" else state.get("last_sync", {}).get("injuries")
        state.setdefault("sources", {})["injuries"] = "espn" if s.get("espn", {}).get("status") == "ok" else "nflverse:injuries"
        log(f"  players: {len(rows)} rows, {len(merged)} players, {len(ev)} status event(s), {len(dev)} depth event(s)")

    if not a.no_weather:
        wev, polled = ingest_weather(games, state, a.force, os.environ.get("SJ_WX_FIXTURES") or None)
        new_events += wev
        summary["weather"] = {"polled": polled, "events": len(wev)}
        log(f"  weather: {polled} game(s) polled, {len(wev)} change event(s)")

    if new_events:
        events.record(new_events)
        crit = [e for e in new_events if impact.wants_refresh(e)]
        if crit:
            req = {"requested_at": store.now_iso(), "reasons": [e["detail"] for e in crit][:10],
                   "game_ids": sorted({e.get("game_id") for e in crit if e.get("game_id")}), "event_ids": [e["event_id"] for e in crit]}
            json.dump(req, open(os.path.join(store.ROOT, "refresh_request.json"), "w"), indent=1)
            log(f"  REFRESH REQUESTED: {len(crit)} event(s): " + " | ".join(req["reasons"][:3]))
        summary["events"] = {"total": len(new_events), "refresh": len(crit)}

    # ---- frontend objects
    for g in payload["games"]:
        g["live"] = context.game_context(g, state)
    payload["live_meta"] = context.live_meta(state, summary)
    json.dump(payload, open(payload_path, "w"), separators=(",", ":"), default=str)
    store.save_state(state)
    write_admin(state, summary, os.path.join(a.outdir, "admin.html"))
    log(f"wrote {payload_path} with live context for {len(games)} games")
    return 0


if __name__ == "__main__":
    sys.exit(main())
