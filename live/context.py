"""
Clean objects for the frontend. Vendor shapes stop here; the page only ever sees these.

  game_live_context = {
    injuryImpact, weather, alerts, recentEvents, versions, projectionChanged,
    lastUpdated, freshness
  }
"""
from datetime import datetime, timezone

from . import store, stadiums, weather as wx, events as ev, versions as pv, impact, normalize

# how old a source may be before the page must say so, by hours to kickoff
STALE_HOURS = {"injuries": [(6, 0.75), (24, 3), (72, 8), (None, 24)],
               "weather":  [(6, 1.5), (24, 3), (72, 8), (None, 24)]}


def _hours_to(kick_iso, now):
    try:
        k = datetime.fromisoformat(kick_iso.replace("Z", "+00:00"))
        return (k - now).total_seconds() / 3600
    except Exception:
        return None


def _age_h(iso, now):
    try:
        return (now - datetime.fromisoformat(iso.replace("Z", "+00:00"))).total_seconds() / 3600
    except Exception:
        return None


def _stale(kind, last_iso, kick_iso, now):
    age = _age_h(last_iso, now) if last_iso else None
    h = _hours_to(kick_iso, now) if kick_iso else None
    if h is not None and h < -4:            # game is over; nothing is expected to refresh
        return {"stale": False, "expected_within_h": None, "age_h": age, "final": True}
    for limit, allowed in STALE_HOURS[kind]:
        if h is None or limit is None or h <= limit:
            return {"stale": age is None or age > allowed, "expected_within_h": allowed, "age_h": None if age is None else round(age, 2)}
    return {"stale": True, "expected_within_h": None, "age_h": age}


def game_context(g, state, now=None):
    now = now or datetime.now(timezone.utc)
    gid = g["game_id"]
    st = state or {}
    kick = (st.get("kickoffs") or {}).get(gid)
    # ---- injuries: who is unavailable and how much it matters
    players = [r for r in (st.get("players") or {}).values() if r.get("game_id") == gid]
    unavailable = [r for r in players if r.get("normalized_status") in normalize.UNAVAILABLE | {"DOUBTFUL"}
                   and not (r.get("source") == "nflverse:rosters" and r.get("depth_order") != 1)]
    rows = []
    worst = "LOW"
    for r in sorted(unavailable, key=lambda r: r.get("player_name") or ""):
        tier, why = impact.classify({**r, **((st.get("usage") or {}).get(r["internal_player_id"]) or {})})
        if impact.TIERS.index(tier) > impact.TIERS.index(worst):
            worst = tier
        rows.append({"player": r.get("player_name"), "id": r.get("internal_player_id"), "team": r.get("team"), "position": r.get("position"),
                     "status": r.get("normalized_status"), "original": r.get("original_status"), "tier": tier, "why": why[:2],
                     "source": r.get("source"), "sourceUpdatedAt": r.get("source_updated_at")})
    inj_sync = (st.get("last_sync") or {}).get("injuries")
    # ---- weather
    stadium = stadiums.get(g.get("stadium_id")) or stadiums.for_team(g["home_team"])
    roof = stadiums.roof_status(stadium, g.get("roof"))
    fc = (st.get("forecasts") or {}).get(gid) or {}
    prim, sec = fc.get("openmeteo"), fc.get("nws")
    alerts = [a for a in (st.get("alerts") or {}).get(gid, []) if a.get("ends") is None or a["ends"] >= now.isoformat()]
    weather = {"applies": roof["outdoor_weather_applies"], "roof": roof, "stadium": stadium and {k: stadium[k] for k in ("stadium_id", "stadium_name", "timezone", "roof_type", "surface_type")},
               "provider": None, "summary": None, "impact": "NONE", "reasons": [], "effects": [], "uncertainty": {"flag": False, "notes": []},
               "secondary": None, "forecastCreatedAt": None, "fetchedAt": None, "attribution": None}
    src = prim or sec
    if roof["outdoor_weather_applies"] is False:
        weather["impact"] = "NONE"; weather["reasons"] = [roof["label"]]
    elif src:
        lvl, why, eff = wx.classify(src, alerts)
        unc, notes = wx.compare(prim, sec)
        weather.update(provider=src["provider"], summary={k: src.get(k) for k in ("kickoff_temperature", "kickoff_feels_like", "kickoff_wind", "kickoff_gust",
                       "kickoff_wind_direction", "max_window_wind", "max_window_gust", "precipitation_probability", "forecast_precipitation", "snowfall",
                       "humidity", "visibility_min_mi", "weather_code", "condition", "window_start", "window_end")},
                       impact=lvl if roof["outdoor_weather_applies"] else "NONE", reasons=why, effects=eff,
                       uncertainty={"flag": unc, "notes": notes},
                       secondary=(sec if prim else None) and {k: sec.get(k) for k in ("provider", "max_window_wind", "precipitation_probability", "kickoff_temperature", "forecast_created_at")},
                       forecastCreatedAt=src.get("forecast_created_at"), fetchedAt=src.get("fetched_at"),
                       attribution="Weather data by Open-Meteo.com (CC BY 4.0)" if src["provider"] == "openmeteo" else "National Weather Service")
        if roof["status"] == "pending":
            weather["reasons"] = ["Roof status pending; outdoor forecast shown for reference"] + weather["reasons"]
    # ---- events and versions
    recent = ev.read_events(limit=12, game_id=gid)
    team_ev = [e for e in ev.read_events(limit=40) if e.get("game_id") is None and e.get("team_id") in (g["home_team"], g["away_team"])]
    recent = sorted(recent + team_ev, key=lambda e: e.get("occurred_at") or "", reverse=True)[:12]
    vers = pv.history(gid, 6)
    changed = bool(vers and vers[0].get("previous_version_id") and (_age_h(vers[0].get("created_at"), now) or 99) < 48)
    wsync = (fc.get("openmeteo") or fc.get("nws") or {}).get("fetched_at")
    return {
        "injuryImpact": {"tier": worst if rows else "NONE", "players": rows, "count": len(rows)},
        "weather": weather,
        "alerts": [{k: a.get(k) for k in ("event", "severity", "headline", "onset", "ends", "sender")} for a in alerts],
        "recentEvents": [{"id": e["event_id"], "type": e["event_type"], "severity": e.get("severity"), "detail": e.get("detail"),
                          "previous": e.get("previous_value"), "new": e.get("new_value"), "at": e.get("occurred_at"),
                          "source": e.get("source"), "recalculated": e.get("prediction_recalculated", False), "player": e.get("player_name")} for e in recent],
        "versions": [{"id": v["version_id"], "at": v["created_at"], "pHome": v.get("p_home"), "margin": v.get("margin"),
                      "homeScore": v.get("home_score"), "awayScore": v.get("away_score"), "reason": v.get("reason"),
                      "change": v.get("change"), "modelVersion": v.get("model_version"), "dataVersion": v.get("data_version")} for v in vers],
        "projectionChanged": changed,
        "lastUpdated": max([x for x in (inj_sync, wsync, (recent[0].get("received_at") if recent else None)) if x] or [None]),
        "freshness": {"injuries": {"source": (st.get("sources") or {}).get("injuries", "espn"), "updatedAt": inj_sync, **_stale("injuries", inj_sync, kick, now)},
                      "weather": {"source": weather["provider"], "updatedAt": wsync, **_stale("weather", wsync, kick, now)} if weather["applies"] is not False else {"source": None, "updatedAt": None, "stale": False, "notApplicable": True}},
    }


def live_meta(state, sync_summary, now=None):
    now = now or datetime.now(timezone.utc)
    st = state or {}
    return {"syncedAt": store.now_iso(), "sources": sync_summary,
            "gamesMonitored": len(st.get("kickoffs") or {}),
            "injuriesStale": _stale("injuries", (st.get("last_sync") or {}).get("injuries"), None, now)["stale"],
            "weatherStale": _stale("weather", (st.get("last_sync") or {}).get("weather"), None, now)["stale"],
            "attribution": "Weather data by Open-Meteo.com (CC BY 4.0); alerts and validation from the National Weather Service; player status from ESPN and the official NFL report via nflverse."}
