"""
Change detection and the live event log.

An event is written only for a MEANINGFUL change: a normalized status moving between
categories, a practice level moving, a starter leaving or joining the depth chart, an
estimated return date moving, a forecast crossing a tolerance, an NWS alert appearing.
Re-fetching the same report produces nothing. The previous value travels with every event
so the reader can see what changed, not only what it is now.

Severity follows the brief:
  CRITICAL  starting QB status change, official inactive, severe weather warning
  HIGH      starting skill player out, major OL injury, large weather change
  MEDIUM    practice participation change, depth chart change, moderate forecast movement
  LOW       minor metadata (return date, description)
"""
from . import store, normalize, impact, weather

SEV = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _event(event_type, game_id=None, player_id=None, team=None, severity="LOW", previous=None, new=None,
           detail=None, source=None, occurred_at=None, **extra):
    return {"event_id": store.new_id("ev"), "event_type": event_type, "game_id": game_id, "player_id": player_id,
            "team_id": team, "severity": severity, "previous_value": previous, "new_value": new, "detail": detail,
            "source": source, "occurred_at": occurred_at or store.now_iso(), "received_at": store.now_iso(),
            "processed": False, "prediction_recalculated": False, **extra}


def _tier_to_sev(tier):
    return {"CRITICAL": "CRITICAL", "HIGH": "HIGH", "MODERATE": "MEDIUM", "LOW": "LOW"}.get(tier, "LOW")


# ------------------------------------------------------------------ players
def player_changes(previous, current, meta=None):
    """
    previous / current: {internal_player_id: player_status row}. meta: extra per-player facts
    for impact (depth_order, snap_share, tgt_share, car_share, is_starting_qb).
    -> (events, history_rows)
    """
    events, hist = [], []
    meta = meta or {}
    bootstrap = not previous            # first run ever: record baselines, announce nothing
    NEWSWORTHY = normalize.UNAVAILABLE | {"DOUBTFUL", "QUESTIONABLE"}
    for pid, cur in current.items():
        prev = previous.get(pid)
        if prev is None:
            # first sight of this player. A baseline goes into history; an event only if the
            # status itself is news (a fresh Out or Questionable), never for "listed as full".
            if cur.get("normalized_status") not in ("UNKNOWN", None):
                hist.append({"internal_player_id": pid, "game_id": cur.get("game_id"), "previous_status": None,
                             "new_status": cur.get("normalized_status"), "previous_practice": None, "new_practice": cur.get("practice_status"),
                             "timestamp": cur.get("source_updated_at") or store.now_iso(), "source": cur.get("source"), "baseline": True})
            if bootstrap or cur.get("normalized_status") not in NEWSWORTHY or cur.get("source") == "nflverse:rosters":
                continue
        name = cur.get("player_name") or pid
        info = {**(meta.get(pid) or {}), "position": cur.get("position"), "depth_order": cur.get("depth_order") or (meta.get(pid) or {}).get("depth_order")}
        tier, why = impact.classify(info)
        sev = _tier_to_sev(tier)
        ps, cs = (prev or {}).get("normalized_status"), cur.get("normalized_status")
        if ps != cs and not (ps is None and cs == "UNKNOWN"):
            hist.append({"internal_player_id": pid, "game_id": cur.get("game_id"), "previous_status": ps, "new_status": cs,
                         "previous_practice": (prev or {}).get("practice_status"), "new_practice": cur.get("practice_status"),
                         "timestamp": cur.get("source_updated_at") or store.now_iso(), "source": cur.get("source")})
            etype = "PLAYER_STATUS_CHANGE"
            s = sev
            if cs in ("INACTIVE", "ACTIVE") and cur.get("game_status") in ("INACTIVE", "ACTIVE"):
                etype, s = "GAMEDAY_INACTIVE" if cs == "INACTIVE" else "GAMEDAY_ACTIVE", ("CRITICAL" if cs == "INACTIVE" and tier != "LOW" else "HIGH" if cs == "INACTIVE" else sev)
            if info.get("position") == "QB" and info.get("is_starting_qb") and cs in normalize.UNAVAILABLE | {"DOUBTFUL", "QUESTIONABLE"}:
                etype, s = "STARTING_QB_CHANGE", "CRITICAL"
            # a move that makes a player MORE available is at most HIGH: it changes the picture,
            # but "cleared" rarely needs the urgency "ruled out" does
            if ps and normalize.SEVERITY_RANK.get(cs, 0) < normalize.SEVERITY_RANK.get(ps, 0) and s == "CRITICAL" and etype != "STARTING_QB_CHANGE":
                s = "HIGH"
            events.append(_event(etype, cur.get("game_id"), pid, cur.get("team"), s, ps, cs,
                                 f"{name}: {ps or 'no report'} → {cs}" + (f" ({cur.get('injury_body_part')})" if cur.get("injury_body_part") else ""),
                                 cur.get("source"), cur.get("source_updated_at"), impact_tier=tier, impact_reasons=why, player_name=name))
        pp, cp = (prev or {}).get("practice_status"), cur.get("practice_status")
        if prev is not None and cp and pp != cp:
            events.append(_event("PRACTICE_CHANGE", cur.get("game_id"), pid, cur.get("team"), "MEDIUM" if tier in ("HIGH", "CRITICAL") else "LOW",
                                 pp, cp, f"{name}: practice {pp or 'unknown'} → {cp}", cur.get("source"), cur.get("source_updated_at"),
                                 impact_tier=tier, player_name=name))
        pr, cr = (prev or {}).get("estimated_return_date"), cur.get("estimated_return_date")
        if prev and cr and pr != cr:
            events.append(_event("RETURN_DATE_CHANGE", cur.get("game_id"), pid, cur.get("team"), "LOW", pr, cr,
                                 f"{name}: estimated return {pr or 'unknown'} → {cr}", cur.get("source"), cur.get("source_updated_at"), player_name=name))
    return events, hist


def depth_changes(previous, current, names=None):
    """
    previous / current: {(team, position_slot): [player ids in depth order]}.
    -> events for starters removed, backups promoted.
    """
    events = []
    names = names or {}
    for key, cur in current.items():
        prev = previous.get(key)
        if prev is None or prev == cur:
            continue
        team, slot = key
        p0, c0 = (prev[0] if prev else None), (cur[0] if cur else None)
        if p0 != c0:
            sev = "CRITICAL" if slot == "QB" else "HIGH" if slot in ("RB", "WR", "TE", "LT", "RT", "T") else "MEDIUM"
            events.append(_event("DEPTH_CHART_CHANGE", None, c0, team, sev, p0, c0,
                                 f"{team} {slot}: {names.get(p0, p0)} → {names.get(c0, c0)} as the starter", "nflverse:depth_charts",
                                 detail_kind="starter_changed", slot=slot, previous_player=p0, player_name=names.get(c0, c0)))
        elif prev[1:] != cur[1:]:
            events.append(_event("DEPTH_CHART_CHANGE", None, None, team, "LOW", prev, cur, f"{team} {slot}: backup order changed",
                                 "nflverse:depth_charts", detail_kind="backup_order", slot=slot))
    return events


# ------------------------------------------------------------------ weather
def weather_changes(game_id, prev_summary, cur_summary, prev_alerts, cur_alerts, provider="openmeteo"):
    events = []
    for d in weather.diff(prev_summary, cur_summary):
        if d["field"] == "impact":
            worse = weather.IMPACT_ORDER.index(d["after"]) > weather.IMPACT_ORDER.index(d["before"])
            sev = "HIGH" if d["after"] == "HIGH" else "MEDIUM"
            events.append(_event("WEATHER_CHANGE_EVENT", game_id, None, None, sev, d["before"], d["after"],
                                 f"Weather impact {d['before']} → {d['after']}" + (" (worsening)" if worse else " (improving)"), provider,
                                 field="impact"))
        else:
            label = {"max_window_wind": "Wind", "max_window_gust": "Gusts", "precipitation_probability": "Rain chance",
                     "kickoff_temperature": "Temperature", "snowfall": "Snow"}[d["field"]]
            unit = {"max_window_wind": " mph", "max_window_gust": " mph", "precipitation_probability": "%", "kickoff_temperature": "°F", "snowfall": " in"}[d["field"]]
            big = d["field"] in ("max_window_wind", "max_window_gust") and abs((d["after"] or 0) - (d["before"] or 0)) >= 10
            events.append(_event("WEATHER_CHANGE_EVENT", game_id, None, None, "HIGH" if big else "MEDIUM", d["before"], d["after"],
                                 f"{label} forecast {_fmt(d['before'])}{unit} → {_fmt(d['after'])}{unit}", provider, field=d["field"]))
    seen = {a.get("alert_id") for a in prev_alerts or []}
    for a in cur_alerts or []:
        if a.get("alert_id") in seen:
            continue
        # NWS tags a Flood Watch "Severe" just like a Tornado Warning. A watch means conditions
        # are possible; a warning means they are happening or imminent. Only the latter is
        # CRITICAL; a severe watch is HIGH; advisories and statements are MEDIUM.
        strong = a.get("severity") in ("Extreme", "Severe")
        warning = "warning" in str(a.get("event") or "").lower()
        sev = "CRITICAL" if strong and warning else "HIGH" if strong else "MEDIUM"
        events.append(_event("SEVERE_WEATHER_ALERT", game_id, None, None, sev, None, a.get("event"),
                             a.get("headline") or a.get("event"), "nws", a.get("onset"), alert=a))
    return events


def _fmt(v):
    return "—" if v is None else (f"{v:.0f}" if isinstance(v, (int, float)) and abs(v) >= 10 else f"{v}")


# ------------------------------------------------------------------ log helpers
def record(events):
    return store.append("live_events", events)


def unprocessed(game_id=None):
    return [e for e in read_events(game_id=game_id) if not e.get("processed")]


def mark_processed(event_ids, recalculated):
    """Append processed markers; the log is append-only, so state is the latest marker per id."""
    rows = [{"event_id": eid, "marker": True, "processed": True, "prediction_recalculated": bool(recalculated)} for eid in event_ids]
    return store.append("live_events", rows)


def read_events(limit=None, game_id=None):
    """Events with their latest processed marker folded in, newest first."""
    rows = store.read("live_events")
    markers = {r["event_id"]: r for r in rows if r.get("marker")}
    out = []
    for r in rows:
        if r.get("marker"):
            continue
        m = markers.get(r["event_id"])
        if m:
            r = {**r, "processed": True, "prediction_recalculated": m.get("prediction_recalculated", False)}
        if game_id is None or r.get("game_id") == game_id:
            out.append(r)
    out.sort(key=lambda r: r.get("occurred_at") or "", reverse=True)
    return out[:limit] if limit else out
