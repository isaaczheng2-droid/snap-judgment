#!/usr/bin/env python3
"""
The live layer, end to end on fixtures. Every provider is blocked from the dev machines, so
these run on captured responses; the same code path runs live on the Actions runner.

What is being protected:
  - missing data is UNKNOWN, never HEALTHY; a healthy scratch is INACTIVE, not an injury
  - identity resolves by vendor id first and by name only when unambiguous
  - validation keeps flagged rows visible and drops only what is unusable
  - weather windows are the kickoff window, not the day; roofs are never assumed
  - change detection fires on meaningful transitions only, with the previous value attached
  - versions append and never overwrite; reasons come from events
  - stale data is labelled stale
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta

TMP = tempfile.mkdtemp()
os.environ["SJ_LIVE_DIR"] = TMP
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from live import store, teams, identity, normalize, stadiums, weather, events, impact, versions, context

FX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "fixtures")
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def main():
    print("\nTEAMS")
    check("ESPN LAR -> LA, WSH -> WAS", teams.to_canonical("LAR", "espn") == "LA" and teams.to_canonical("WSH", "espn") == "WAS")
    check("PFR RAM -> LA", teams.to_canonical("RAM", "pfr") == "LA")
    check("legacy OAK -> LV", teams.to_canonical("OAK") == "LV")
    check("garbage is None", teams.to_canonical("XYZ") is None)

    print("\nNORMALIZATION")
    check("Out is OUT", normalize.norm_status("Out") == "OUT")
    check("Out + Coach's Decision is INACTIVE, not an injury", normalize.norm_status("Out", "Coach's Decision") == "INACTIVE")
    check("Injured Reserve is IR", normalize.norm_status("Injured Reserve") == "IR")
    check("suspension type is SUSPENDED", normalize.norm_status("Out", "Suspension") == "SUSPENDED")
    check("unrecognised text is UNKNOWN, not healthy", normalize.norm_status("wibble") == "UNKNOWN")
    check("empty text is None (no claim)", normalize.norm_status("") is None)
    check("practice DNP", normalize.norm_practice("Did Not Participate In Practice") == "DID_NOT_PRACTICE")
    r = normalize.record(internal_player_id="00-1", player_name="A", team="LAR")
    check("record with no status is UNKNOWN and active is null", r["normalized_status"] == "UNKNOWN" and r["active"] is None and r["inactive"] is None)
    r = normalize.record(internal_player_id="00-1", player_name="A", team="KC", normalized_status="INACTIVE")
    check("INACTIVE sets inactive=true active=false", r["inactive"] is True and r["active"] is False)
    check("record canonicalises the team", r["team"] == "KC")
    check("every schema field present", set(normalize.FIELDS) <= set(r))

    print("\nIDENTITY")
    rost = pd.DataFrame([
        {"season": 2026, "gsis_id": "00-0001", "espn_id": "111", "pfr_id": None, "sportradar_id": None, "yahoo_id": None, "rotowire_id": None, "sleeper_id": None, "pff_id": None, "esb_id": None, "fantasy_data_id": None, "full_name": "Odell Beckham Jr.", "team": "MIA", "position": "WR", "status": "ACT"},
        {"season": 2026, "gsis_id": "00-0002", "espn_id": "222", "pfr_id": None, "sportradar_id": None, "yahoo_id": None, "rotowire_id": None, "sleeper_id": None, "pff_id": None, "esb_id": None, "fantasy_data_id": None, "full_name": "Ja'Marr Chase", "team": "CIN", "position": "WR", "status": "ACT"},
        {"season": 2026, "gsis_id": "00-0003", "espn_id": None, "pfr_id": None, "sportradar_id": None, "yahoo_id": None, "rotowire_id": None, "sleeper_id": None, "pff_id": None, "esb_id": None, "fantasy_data_id": None, "full_name": "Mike Williams", "team": "LAC", "position": "WR", "status": "ACT"},
        {"season": 2026, "gsis_id": "00-0004", "espn_id": None, "pfr_id": None, "sportradar_id": None, "yahoo_id": None, "rotowire_id": None, "sleeper_id": None, "pff_id": None, "esb_id": None, "fantasy_data_id": None, "full_name": "Mike Williams", "team": "LAC", "position": "DT", "status": "ACT"},
    ])
    xw = identity.Crosswalk.from_roster(rost, 2026)
    check("vendor id wins", xw.resolve("espn", "111", "Someone Else", "MIA") == ("00-0001", "espn_id"))
    check("suffix and punctuation ignored in name match", xw.resolve(None, None, "odell beckham", "MIA")[0] == "00-0001")
    check("apostrophe ignored", xw.resolve(None, None, "JaMarr Chase", "CIN")[0] == "00-0002")
    check("ambiguous name never resolves", xw.resolve(None, None, "Mike Williams", "LAC") == (None, "unresolved"))
    p = xw.write(); check("crosswalk written", os.path.exists(p))

    print("\nVALIDATION")
    now = datetime.now(timezone.utc)
    rows = [normalize.record(internal_player_id="00-0001", player_name="Odell Beckham Jr.", team="MIA", position="WR", game_id="G1", normalized_status="QUESTIONABLE", source="espn", source_updated_at=(now - timedelta(hours=2)).isoformat()),
            normalize.record(internal_player_id="00-0001", player_name="Odell Beckham Jr.", team="MIA", position="WR", game_id="G1", normalized_status="QUESTIONABLE", source="espn", source_updated_at=(now - timedelta(hours=2)).isoformat()),
            normalize.record(internal_player_id="00-0002", player_name="Ja'Marr Chase", team="MIA", position="WR", game_id="G1", normalized_status="OUT", source="espn"),
            normalize.record(internal_player_id=None, player_name="Ghost", team="MIA", game_id="G1", normalized_status="OUT", source="espn"),
            normalize.record(internal_player_id="00-0003", player_name="Mike Williams", team="LAC", game_id="G9", normalized_status="OUT", source="espn"),
            normalize.record(internal_player_id="00-0001", player_name="Odell Beckham Jr.", team="MIA", game_id="G1", normalized_status="OUT", source="espn", source_updated_at=(now + timedelta(days=2)).isoformat())]
    ok, probs = normalize.validate(rows, xw, ["G1"], now, previous={"00-0004": {"normalized_status": "IR"}})
    kinds = sorted(p[0] for p in probs)
    check("duplicate, team mismatch, unknown player, unknown game, future timestamp all caught",
          kinds == ["bad_timestamp" if False else "duplicate", "implausible_timestamp", "team_mismatch", "unknown_game", "unknown_player"], str(kinds))
    check("one clean row survives", len(ok) == 1)
    ok2, probs2 = normalize.validate([normalize.record(internal_player_id="00-0004", player_name="Mike Williams", team="LAC", game_id="G1", normalized_status="ACTIVE", source="espn")],
                                     None, ["G1"], now, previous={"00-0004": {"normalized_status": "IR"}})
    check("impossible IR -> ACTIVE is kept but flagged", len(ok2) == 1 and ok2[0].get("flag") == "impossible_transition" and probs2[0][0] == "impossible_transition")
    check("quality table has the failures", len(store.read("data_quality")) >= 5)

    print("\nSTADIUMS")
    kc = stadiums.get("KAN00")
    check("Arrowhead has stadium coordinates and tz", kc and abs(kc["latitude"] - 39.0489) < 0.002 and kc["timezone"] == "America/Chicago")
    check("for_team finds the shared SoFi row for both LA teams", stadiums.for_team("LA")["stadium_id"] == "LAX01" and stadiums.for_team("LAC")["stadium_id"] == "LAX01")
    check("dome -> indoor", stadiums.roof_status(stadiums.get("DET00"))["status"] == "indoor")
    check("retractable with no report -> pending, never assumed", stadiums.roof_status(stadiums.get("DAL00"))["status"] == "pending" and stadiums.roof_status(stadiums.get("DAL00"))["outdoor_weather_applies"] is None)
    check("retractable reported closed -> closed", stadiums.roof_status(stadiums.get("DAL00"), "closed")["outdoor_weather_applies"] is False)
    check("open-air -> open", stadiums.roof_status(kc)["status"] == "open")

    print("\nWEATHER")
    kick = weather.kickoff_utc("2026-09-14", "20:15")
    check("kickoff 8:15 PM ET -> 00:15 UTC next day", kick.strftime("%Y-%m-%dT%H:%M") == "2026-09-15T00:15", kick.isoformat())
    om = json.load(open(os.path.join(FX, "openmeteo_KAN00_2026-09-12.json")))
    rows = weather.parse_openmeteo(om, "America/Chicago")
    check("open-meteo hours parsed to UTC", len(rows) == 96 and rows[0]["time"] == "2026-09-12T05:00:00Z", rows[0]["time"])
    s = weather.summarize(rows, kick)
    check("window is kickoff-1h to +4h (6 hourly rows)", s and s["hours"] == 6, str(s and s["hours"]))
    i19 = om["hourly"]["time"].index("2026-09-14T19:00")        # 8:15 PM ET is 7:15 PM CDT; nearest hour
    check("kickoff wind/gust taken from the kickoff hour", s and s["kickoff_wind"] == om["hourly"]["wind_speed_10m"][i19] and s["kickoff_gust"] == om["hourly"]["wind_gusts_10m"][i19], f"{s and s['kickoff_wind']}/{s and s['kickoff_gust']}")
    check("max window gust is the max over 18:00-23:00 CDT", s and s["max_window_gust"] == max(om["hourly"]["wind_gusts_10m"][i19 - 1:i19 + 5]), str(s and s["max_window_gust"]))
    lvl, why, eff = weather.classify(s)
    check("17 mph sustained + 42 gusts -> HIGH with kicking listed", lvl == "HIGH" and "Kicking" in eff, f"{lvl} {why}")
    calm = dict(s, max_window_wind=5, max_window_gust=9, kickoff_wind=5, kickoff_gust=8, precipitation_probability=5, forecast_precipitation=0, snowfall=0, kickoff_temperature=60)
    check("calm evening -> NONE", weather.classify(calm)[0] == "NONE")
    check("59 -> 61 °F is not a change", weather.diff(dict(calm, kickoff_temperature=59), dict(calm, kickoff_temperature=61)) == [])
    d = weather.diff(dict(calm), dict(calm, max_window_wind=14))
    check("5 -> 14 mph wind IS a change (and crosses to LOW)", [x["field"] for x in d] == ["max_window_wind", "impact"], str(d))
    nws = json.load(open(os.path.join(FX, "nws_hourly_EAX_47_48.json")))
    nrows = weather.parse_nws_hourly(nws)
    ns = weather.summarize(nrows, kick)
    check("NWS hourly parsed ('12 mph' -> 12, S -> 180)", ns and ns["kickoff_wind"] == 12 and ns["kickoff_wind_direction"] == 180, str(ns and (ns["kickoff_wind"], ns["kickoff_wind_direction"])))
    unc0, _ = weather.compare(s, ns)
    check("a 3-4 mph wind difference is NOT uncertainty", unc0 is False)
    unc, notes = weather.compare(s, dict(ns, max_window_wind=26))
    check("a > 6 mph wind disagreement -> FORECAST UNCERTAINTY", unc and notes and "wind" in notes[0], str(notes))
    al = weather.parse_nws_alerts(json.load(open(os.path.join(FX, "nws_alerts_sample.json"))))
    check("alert parsed with severity and window", al and al[0]["event"] == "Wind Advisory" and al[0]["severity"] == "Moderate")
    check("no-alert response parses to empty", weather.parse_nws_alerts(json.load(open(os.path.join(FX, "nws_alerts_none.json")))) == [])
    lvl2, _, _ = weather.classify(calm, [{"event": "High Wind Warning", "severity": "Severe"}])
    check("severe NWS warning forces HIGH", lvl2 == "HIGH")
    g = {"game_id": "2026_01_DEN_KC", "gameday_iso": "2026-09-14", "kickoff": "20:15"}
    res = weather.fetch_game(g, kc, fixtures={"openmeteo": os.path.join(FX, "openmeteo_KAN00_2026-09-12.json"), "nws_points": os.path.join(FX, "nws_points_KAN00.json"),
                                              "nws_hourly": os.path.join(FX, "nws_hourly_EAX_47_48.json"), "nws_alerts": os.path.join(FX, "nws_alerts_none.json")}, log=lambda *a: None, state={})
    check("fetch_game returns both providers and no alerts", set(res["forecasts"]) == {"openmeteo", "nws"} and res["alerts"] == [] and not res["errors"], str(res["errors"]))
    check("every provider call is in the sync log", len(store.read("api_sync_log")) >= 4)
    check("sync log never carries a query string", all("?" not in r["url"] for r in store.read("api_sync_log")))

    print("\nIMPACT")
    check("starting QB -> CRITICAL", impact.classify({"position": "QB", "depth_order": 1, "is_starting_qb": True})[0] == "CRITICAL")
    check("backup QB -> LOW", impact.classify({"position": "QB", "depth_order": 2})[0] == "LOW")
    check("WR1 with 28% targets -> HIGH (never CRITICAL)", impact.classify({"position": "WR", "depth_order": 1, "tgt_share": 0.28})[0] == "HIGH")
    check("WR3 with 5% targets -> LOW", impact.classify({"position": "WR", "depth_order": 3, "tgt_share": 0.05})[0] == "LOW")
    check("punter -> LOW", impact.classify({"position": "P", "depth_order": 1})[0] == "LOW")

    print("\nEVENTS")
    prev = {"00-0001": {"normalized_status": "QUESTIONABLE", "practice_status": "LIMITED_PRACTICE", "player_name": "Odell Beckham Jr."}}
    cur = {"00-0001": normalize.record(internal_player_id="00-0001", player_name="Odell Beckham Jr.", team="MIA", position="WR", game_id="G1", normalized_status="OUT", practice_status="DID_NOT_PRACTICE", source="espn", source_updated_at="2026-09-13T15:34:00Z", injury_body_part="Ankle"),
           "00-0009": normalize.record(internal_player_id="00-0009", player_name="Some QB", team="MIA", position="QB", game_id="G1", normalized_status="OUT", source="espn")}
    ev, hist = events.player_changes(prev, cur, {"00-0001": {"depth_order": 1, "tgt_share": 0.26}, "00-0009": {"depth_order": 1, "is_starting_qb": True}})
    types = sorted(e["event_type"] for e in ev)
    check("QUESTIONABLE -> OUT and practice change produce events; QB out is a STARTING_QB_CHANGE", types == ["PLAYER_STATUS_CHANGE", "PRACTICE_CHANGE", "STARTING_QB_CHANGE"], str(types))
    sc = next(e for e in ev if e["event_type"] == "PLAYER_STATUS_CHANGE")
    check("event carries previous and new value and the body part", sc["previous_value"] == "QUESTIONABLE" and sc["new_value"] == "OUT" and "Ankle" in sc["detail"])
    check("WR1 out is HIGH severity, QB out is CRITICAL", sc["severity"] == "HIGH" and next(e for e in ev if e["event_type"] == "STARTING_QB_CHANGE")["severity"] == "CRITICAL")
    check("history row written for the transition", hist and hist[0]["previous_status"] == "QUESTIONABLE" and hist[0]["new_status"] == "OUT")
    same, _ = events.player_changes(cur, cur, {})
    check("re-reading the same report produces no events", same == [])
    unk, _ = events.player_changes({}, {"00-0007": normalize.record(internal_player_id="00-0007", player_name="X", team="KC")}, {})
    check("first sight of a player with no status is not an event", unk == [])
    dev = events.depth_changes({("KC", "QB"): ["00-A", "00-B"]}, {("KC", "QB"): ["00-B", "00-A"]}, {"00-A": "Starter", "00-B": "Backup"})
    check("starter replaced at QB -> CRITICAL depth event naming both", dev and dev[0]["severity"] == "CRITICAL" and "Starter" in dev[0]["detail"] and "Backup" in dev[0]["detail"])
    wev = events.weather_changes("G1", calm, dict(calm, max_window_wind=17, max_window_gust=30), [], al)
    from live import events as _ev
    _al = lambda ev, sv: _ev.weather_changes("g", {}, {}, [], [{"alert_id": "x", "event": ev, "severity": sv}], "nws")[0]["severity"]
    check("Tornado Warning CRITICAL, Flood Watch HIGH, Heat Advisory MEDIUM",
          _al("Tornado Warning", "Extreme") == "CRITICAL" and _al("Flood Watch", "Severe") == "HIGH" and _al("Heat Advisory", "Moderate") == "MEDIUM")
    check("wind 5 -> 17 gives a HIGH weather event plus an alert event", any(e["severity"] == "HIGH" and e["event_type"] == "WEATHER_CHANGE_EVENT" for e in wev) and any(e["event_type"] == "SEVERE_WEATHER_ALERT" for e in wev), str([(e["event_type"], e["severity"]) for e in wev]))
    check("refresh wanted for the QB event, not for the weather event", impact.wants_refresh(next(e for e in ev if e["event_type"] == "STARTING_QB_CHANGE")) and not any(impact.wants_refresh(e) for e in wev))
    events.record(ev + wev)
    check("events persisted and readable newest-first", len(events.read_events()) == len(ev) + len(wev) and not events.read_events()[0]["processed"])

    print("\nVERSIONS")
    payload = {"season": 2026, "week": 1, "games": [{"game_id": "G1", "home_team": "MIA", "away_team": "KC", "p_blend": 0.572, "p_model": 0.6, "p_market": 0.56, "margin_pred": 2.1, "predicted_home_score": 24.0, "predicted_away_score": 22.0}],
               "players": [{"player_key": "00-0001", "team": "MIA", "receiving_yards": 61.0}]}
    mv = versions.model_version({"max_depth": 3}, ["a", "b"], 0.2, 8, "2026-w1")
    v1, c1 = versions.record(payload, mv, "2026-09-12 10:00 UTC", log=lambda *a: None)
    check("first version is recorded as Initial prediction", len(v1) == 1 and v1[0]["reason"] == "Initial prediction" and c1 == [])
    payload2 = json.loads(json.dumps(payload)); payload2["games"][0]["p_blend"] = 0.489; payload2["games"][0]["margin_pred"] = -1.4; payload2["players"][0]["receiving_yards"] = 40.0
    # a new event arrives after the initial version: it must become the next version's reason
    later, _ = events.player_changes({"00-0009": {"normalized_status": "QUESTIONABLE"}}, {"00-0009": cur["00-0009"]}, {"00-0009": {"depth_order": 1, "is_starting_qb": True}})
    events.record(later)
    v2, c2 = versions.record(payload2, mv, "2026-09-12 11:38 UTC", default_reason="Scheduled refresh", log=lambda *a: None)
    check("changed prediction appends a version with the change and the event as reason",
          len(v2) == 1 and abs(v2[0]["change"]["p_home"] + 0.083) < 1e-6 and v2[0]["previous_version_id"] == v1[0]["version_id"] and "OUT" in v2[0]["reason"], str(v2 and v2[0]["reason"]))
    check("field-level change log covers probability, margin and the player", {c["field"] for c in c2} >= {"p_home", "margin", "player:00-0001:receiving_yards"}, str([c["field"] for c in c2]))
    check("both versions are retained", len(versions.history("G1")) == 2 and versions.history("G1")[0]["version_id"] == v2[0]["version_id"])
    check("events marked processed and recalculated", all(e["processed"] and e["prediction_recalculated"] for e in events.read_events(game_id="G1") if e["event_type"] != "WEATHER_CHANGE_EVENT"))
    v3, _ = versions.record(payload2, mv, "2026-09-12 12:00 UTC", log=lambda *a: None)
    check("an unchanged prediction adds no version", v3 == [])

    print("\nCONTEXT / FRESHNESS")
    state = {"kickoffs": {"G1": (now + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")},
             "players": cur, "forecasts": {"G1": {"openmeteo": {**s, "provider": "openmeteo", "fetched_at": (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"), "forecast_created_at": "x"}}},
             "alerts": {"G1": []}, "last_sync": {"injuries": (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")}, "sources": {"injuries": "espn"},
             "usage": {"00-0009": {"is_starting_qb": True}}}
    ctx = context.game_context({"game_id": "G1", "home_team": "KC", "away_team": "MIA", "stadium_id": "KAN00"}, state, now)
    check("injury impact lists the QB as CRITICAL", ctx["injuryImpact"]["tier"] == "CRITICAL" and any(p["position"] == "QB" for p in ctx["injuryImpact"]["players"]))
    check("weather applies at Arrowhead with HIGH impact", ctx["weather"]["applies"] is True and ctx["weather"]["impact"] == "HIGH")
    check("2h-old injuries 3h before kickoff are STALE (45 min allowed)", ctx["freshness"]["injuries"]["stale"] is True)
    check("2h-old weather 3h before kickoff is stale too (90 min allowed)", ctx["freshness"]["weather"]["stale"] is True)
    ctx2 = context.game_context({"game_id": "G1", "home_team": "DET", "away_team": "MIA", "stadium_id": "DET00"}, state, now)
    check("dome game: weather not applicable and never stale", ctx2["weather"]["applies"] is False and ctx2["freshness"]["weather"].get("notApplicable"))
    check("projection changed flag set after the second version", ctx["projectionChanged"] is True and ctx["versions"][0]["reason"])
    check("recent events newest first", [e["at"] for e in ctx["recentEvents"]] == sorted([e["at"] for e in ctx["recentEvents"]], reverse=True))

    print(f"\n{len(fails)} failed" if fails else "\nall checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
