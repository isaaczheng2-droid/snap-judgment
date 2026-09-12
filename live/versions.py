
"""
Prediction versioning. A published number is never replaced; a new version is appended
with the reason it exists, the model version that produced it and the data it saw.

`record(payload, model_version, ...)` is called once per pipeline run, after the payload is
built. For every game it compares the new prediction with the latest stored version and
writes a version row when anything material moved (win probability, predicted margin or
score, or a player projection), together with field-level change-log rows. Reasons come
from the live events that arrived since the previous version; when none did, the reason is
the scheduled refresh that ran.
"""
import hashlib
import json

from . import store, events

P_TOL = 0.005        # half a point of win probability
MARGIN_TOL = 0.25
PLAYER_TOL = 0.5     # yards / receptions
STAT_KEYS = ["passing_yards", "passing_tds", "qb_rushing_yards", "rushing_yards", "rushing_tds",
             "rb_receiving_yards", "receiving_yards", "receptions", "receiving_tds"]


def model_version(params, feats, blend_w, n_seeds, trained_through):
    blob = json.dumps({"params": params, "feats": list(feats), "blend_w": blend_w, "n_seeds": n_seeds,
                       "trained_through": str(trained_through)}, sort_keys=True, default=str)
    return "m_" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def latest_by_game():
    return store.latest("prediction_versions", "game_id")


def _players_for(payload, g):
    out = {}
    for p in payload.get("players") or []:
        if p.get("team") in (g["home_team"], g["away_team"]):
            out[p["player_key"]] = {k: p.get(k) for k in STAT_KEYS if p.get(k) is not None}
    return out


def _reason(game_events, default):
    hot = [e for e in game_events if e.get("severity") in ("CRITICAL", "HIGH")]
    src = hot or game_events
    if not src:
        return default, []
    src = sorted(src, key=lambda e: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(e.get("severity", "LOW")), reverse=True)
    return "; ".join(e.get("detail") or e.get("event_type") for e in src[:3]), [e["event_id"] for e in src]


def record(payload, mv, data_version, injury_snapshot_id=None, weather_snapshot_id=None, default_reason="Scheduled refresh: upstream data changed", log=print):
    prev = latest_by_game()
    pending = events.unprocessed()
    by_game = {}
    for e in pending:
        by_game.setdefault(e.get("game_id"), []).append(e)
    # depth-chart events carry a team, not a game: attach them to the team's game this week
    team_game = {}
    for g in payload.get("games") or []:
        team_game[g["home_team"]] = g["game_id"]; team_game[g["away_team"]] = g["game_id"]
    for e in by_game.pop(None, []):
        gid = team_game.get(e.get("team_id"))
        if gid:
            by_game.setdefault(gid, []).append(e)

    new_rows, changes, processed = [], [], []
    for g in payload.get("games") or []:
        gid = g["game_id"]
        cur = {"p_home": g.get("p_blend"), "p_model": g.get("p_model"), "p_market": g.get("p_market"),
               "margin": g.get("margin_pred"), "home_score": g.get("predicted_home_score"), "away_score": g.get("predicted_away_score"),
               "players": _players_for(payload, g)}
        old = prev.get(gid)
        diffs = []
        if old:
            for k, tol in (("p_home", P_TOL), ("margin", MARGIN_TOL), ("home_score", MARGIN_TOL), ("away_score", MARGIN_TOL)):
                a, b = old.get(k), cur.get(k)
                if a is not None and b is not None and abs(b - a) >= tol:
                    diffs.append({"field": k, "before": a, "after": b})
            for pk, stats in cur["players"].items():
                o = (old.get("players") or {}).get(pk) or {}
                for k, v in stats.items():
                    if o.get(k) is not None and abs(v - o[k]) >= PLAYER_TOL:
                        diffs.append({"field": f"player:{pk}:{k}", "before": o[k], "after": v})
        ge = by_game.get(gid, [])
        if old and not diffs:
            # nothing moved; events that arrived are processed without a recalculation credit
            processed += [(e["event_id"], False) for e in ge]
            continue
        if old:
            reason, eids = _reason(ge, default_reason)
        else:
            reason, eids = "Initial prediction", [e["event_id"] for e in ge]
        vid = store.new_id("pv")
        row = {"version_id": vid, "game_id": gid, "season": payload.get("season"), "week": payload.get("week"),
               "created_at": store.now_iso(), "model_version": mv, "data_version": data_version,
               "injury_snapshot_id": injury_snapshot_id, "weather_snapshot_id": weather_snapshot_id,
               "previous_version_id": old.get("version_id") if old else None,
               "reason": reason, "event_ids": eids,
               "change": {"p_home": round(cur["p_home"] - old["p_home"], 4) if old and old.get("p_home") is not None and cur["p_home"] is not None else None,
                          "margin": round(cur["margin"] - old["margin"], 2) if old and old.get("margin") is not None and cur["margin"] is not None else None,
                          "n_fields": len(diffs)},
               **cur}
        new_rows.append(row)
        for d in diffs:
            changes.append({"game_id": gid, "from_version": old.get("version_id") if old else None, "to_version": vid,
                            "field": d["field"], "before": d["before"], "after": d["after"], "reason": reason, "timestamp": row["created_at"]})
        processed += [(e["event_id"], True) for e in ge]
    store.append("prediction_versions", new_rows)
    store.append("prediction_change_log", changes)
    if processed:
        events.mark_processed([e for e, r in processed if r], True)
        events.mark_processed([e for e, r in processed if not r], False)
    log(f"  versions: {len(new_rows)} new prediction version(s), {len(changes)} field change(s), {len(processed)} event(s) processed")
    return new_rows, changes


def history(game_id, limit=6):
    rows = [r for r in store.read("prediction_versions") if r.get("game_id") == game_id]
    return rows[-limit:][::-1]
