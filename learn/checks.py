"""
Foundations: is the data fresh, do the identities line up, is anything missing that a
forecast silently depends on. Produces the `health` block: per-source freshness with a
plain verdict, identity-matching coverage, missing-data checks, and alerts. Nothing here
fills a gap with a zero; a missing input is reported as missing.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# how old a source may be before it is called stale (hours); in-season, during the week
MAX_AGE_H = {"schedule": 30, "player_stats": 96, "team_stats": 96, "injuries": 48, "snap_counts": 120,
             "rosters": 168, "depth_charts": 96, "play_by_play": 168, "odds_cache": 12,
             "live_injuries": 24, "live_weather": 24, "prop_lines": 72}
REQUIRED = {"schedule", "player_stats", "injuries", "rosters", "depth_charts", "snap_counts"}   # snap counts feed the injury features AND snap shares


def _hours_since(iso):
    try:
        t = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds() / 3600
    except Exception:
        return None


def freshness(manifest, target_week):
    out, alerts = [], []
    for s in manifest.get("sources", []):
        rec = {"source": s["source"], "present": s.get("present", False), "upstream": s.get("upstream")}
        if not s.get("present"):
            rec["verdict"] = "missing"
            if s["source"] in REQUIRED:
                alerts.append({"severity": "high", "text": f"{s['source']} is missing; forecasts that depend on it are not trustworthy."})
            else:
                alerts.append({"severity": "low", "text": f"{s['source']} not on file (optional)."})
            out.append(rec)
            continue
        age = _hours_since(s.get("modified"))
        rec.update({"modified": s.get("modified"), "age_hours": None if age is None else round(age, 1), "rows": s.get("rows"),
                    "max_week": s.get("max_week"), "digest": s.get("digest")})
        lim = MAX_AGE_H.get(s["source"], 168)
        stale = age is not None and age > lim
        rec["verdict"] = "stale" if stale else "fresh"
        if stale:
            alerts.append({"severity": "high" if s["source"] in REQUIRED else "medium",
                           "text": f"{s['source']} is {age / 24:.1f} days old (limit {lim / 24:.1f})."})
        # coverage against the week we are forecasting
        if s["source"] in ("player_stats", "snap_counts") and s.get("max_week") is not None and target_week:
            rec["weeks_behind"] = int(target_week) - 1 - int(s["max_week"])
            if rec["weeks_behind"] > 0:
                alerts.append({"severity": "medium", "text": f"{s['source']} ends at week {s['max_week']}; week {int(target_week) - 1} results are not in yet, so form and grading lag a week."})
        if s["source"] == "injuries" and s.get("max_week") is not None and target_week and int(s["max_week"]) < int(target_week):
            rec["this_week_listed"] = False
            alerts.append({"severity": "medium", "text": f"No injury report rows for week {int(target_week)} yet; designations come from the live overlay only."})
        out.append(rec)
    return out, alerts


def identity(pw, snap, rost, espn_stats=None):
    """How well the id systems line up: snap counts (pfr ids) to gsis, ESPN names to gsis."""
    out = {}
    if len(snap) and len(rost):
        s = snap[snap.game_type == "REG"]
        cw = rost.dropna(subset=["gsis_id", "pfr_id"])[["gsis_id", "pfr_id"]].drop_duplicates("pfr_id")
        mapped = s.pfr_player_id.isin(set(cw.pfr_id))
        out["snap_rows_mapped_to_gsis"] = {"rate": round(float(mapped.mean()), 4), "n": int(len(s))}
        cur = s[s.season == s.season.max()]
        if len(cur):
            out["snap_rows_mapped_current_season"] = {"rate": round(float(cur.pfr_player_id.isin(set(cw.pfr_id)).mean()), 4), "n": int(len(cur))}
    if pw is not None and "prior_snap" in pw.columns:
        cur = pw[pw.season == pw.season.max()]
        out["players_with_snap_history"] = {"rate": round(float(cur.snap_now.notna().mean()), 4) if "snap_now" in cur.columns and len(cur) else None, "n": int(len(cur))}
    if espn_stats:
        out["espn_to_gsis"] = espn_stats
    return out


def missing(fantasy, inj_map, depth, cur):
    """Checks on the forecast set itself."""
    out, alerts = {}, []
    players = (fantasy or {}).get("players", [])
    n = len(players)
    if not n:
        alerts.append({"severity": "high", "text": "No fantasy projections were produced."})
        return {"n_players": 0}, alerts
    no_vol = sum(1 for p in players if (p.get("opportunity") or {}).get("volume") is None)
    no_snap = sum(1 for p in players if (p.get("opportunity") or {}).get("snap_share") is None)
    no_range = sum(1 for p in players if not (p.get("range") or {}).get("full_ppr"))
    out.update({"n_players": n, "missing_volume": no_vol, "missing_snap_share": no_snap, "missing_range": no_range,
                "by_position": {pos: sum(1 for p in players if p["position"] == pos) for pos in ("QB", "RB", "WR", "TE")},
                "teams_covered": len({p["team"] for p in players})})
    if no_range:
        alerts.append({"severity": "medium", "text": f"{no_range} projections have no fitted range (fantasy_model.json missing or position not fitted)."})
    if no_snap / n > 0.25:
        alerts.append({"severity": "low", "text": f"{no_snap} of {n} players have no snap-share history; their usage input is a position median."})
    if out["teams_covered"] < 28 and not (fantasy or {}).get("byes"):
        alerts.append({"severity": "medium", "text": f"Only {out['teams_covered']} teams have projected players."})
    if len(depth):
        d = depth[depth.dt == depth.dt.max()]
        out["depth_chart_teams"] = int(d.team.nunique())
        if out["depth_chart_teams"] < 32:
            alerts.append({"severity": "medium", "text": f"Depth charts cover {out['depth_chart_teams']} of 32 teams."})
    return out, alerts


def build(manifest, target_week, pw, snap, rost, fantasy, inj_map, depth, cur, ledger_stats=None, espn_stats=None):
    fr, a1 = freshness(manifest, target_week)
    ms, a2 = missing(fantasy, inj_map, depth, cur)
    alerts = a1 + a2
    sev = {"high": 0, "medium": 1, "low": 2}
    alerts.sort(key=lambda x: sev.get(x["severity"], 9))
    return {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "data_version": manifest.get("data_version"),
            "sources": fr, "identity": identity(pw, snap, rost, espn_stats), "checks": ms, "ledger": ledger_stats or {},
            "alerts": alerts, "ok": not any(a["severity"] == "high" for a in alerts),
            "policy": "Missing inputs are reported, never zero-filled. A finished game whose box score has not landed stays ungraded; a player with no box-score row in a finished game is graded as did-not-play, explicitly."}
