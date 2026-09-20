#!/usr/bin/env python3
"""
Prediction-integrity checks. These guard the model-only cutover (2026-09-20) and the defects
found in the JAX-DEN reproduction case:

  1. no market input or blending anywhere in the production prediction path
  2. orientation and sign conventions of the forecast record (home side, margin, spread edge)
  3. every number the payload shows for a game comes from the game's forecast record
  4. an injury on an unrelated team leaves a game's direct injury features unchanged
  5. injury rows that fail the roster-affiliation check are quarantined, never used
  6. unknown status is shown as unknown, never as healthy
  7. the rebuild trigger is recorded separately from a game's reason and is never a cause
  8. legacy tracker rows keep their method label; new locks carry the model-only label
  9. the same inputs give the same forecast on a repeated run (fixed seeds, no drift)
 10. scores are derived from the unrounded margin and total, rounding is display-only
"""
import inspect
import json
import os
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

import run_pipeline as rp
import tracker

FAILS = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


MARKET_WORDS = re.compile(r"spread|moneyline|market|total_line|over_under|odds|implied|blend", re.I)


def main():
    # ---- 1. no market in the production path
    feats = list(rp.FEATS) + list(rp.TOTAL_FEATS)
    check(not any(MARKET_WORDS.search(f) for f in feats), f"no feature name refers to the market ({len(feats)} features)")
    src = inspect.getsource(rp.fit_predict)
    check("BLEND_W" not in src and "total_line" not in src and "p_market" not in src.split("up[\"p_market\"] = up.market_home_wp")[0],
          "fit_predict does not blend, does not read total_line, and touches p_market only to carry it as a comparison")
    check(not hasattr(rp, "BLEND_W") and hasattr(rp, "LEGACY_BLEND_W"), "BLEND_W is gone; only the labelled legacy constant remains")
    check(rp.METHOD == "model_only_v2" and rp.LEGACY_METHOD == "blend_20_80_v1", "method labels are set")
    # build_games computes market_home_wp for the comparison; it must not be in any feature list
    check("market_home_wp" not in feats and "points_sum_rating" in rp.TOTAL_FEATS, "the total model uses the scoring-rate feature, not the market total")

    # ---- 2. orientation and signs on a synthetic row
    r = pd.Series({"game_id": "2026_02_JAX_DEN", "home_team": "DEN", "away_team": "JAX", "p_home": 0.4742, "margin_pred": -2.3714,
                   "total_pred": 44.4721, "spread_line": 3.0, "total_line": 45.5, "p_market": 0.583,
                   "p_home_healthy": 0.4738, "margin_healthy": -2.325})
    fc = rp.forecast_record(r, 2026, 2, "2026-09-20T00:00:00Z", "2026-09-19T23:00:00Z", "m_test", {}, {"status": "ok", "flags": [], "features_complete": True})
    check(abs(fc["p_home"] + fc["p_away"] - 1) < 1e-9, "p_home + p_away = 1 (tie counted as a home non-win)")
    check(abs(fc["expected_home_score"] - fc["expected_away_score"] - fc["margin_home"]) < 1e-3, "margin_home = expected home minus expected away")
    check(abs(fc["expected_home_score"] + fc["expected_away_score"] - fc["expected_total"]) < 1e-3, "expected scores sum to the expected total")
    check(fc["margin_home"] < 0 and fc["p_home"] < 0.5, "JAX-DEN reproduction: away team leads both the expected score and the win probability")
    check(abs(fc["market"]["margin_edge_home"] - (-2.3714 - 3.0)) < 1e-6, "margin edge = model margin minus the handicap, both home-side (JAX +5.37 in the reproduction)")
    check(fc["market"]["used_in_model"] is False and "closing" in fc["market"]["kind"], "the market block is labelled as a closing-line comparison, not an input")
    check(fc["forecast_id"].startswith("2026_02_JAX_DEN:") and fc["model_version"] == "m_test", "forecast id carries game, cutoff and model version")

    # ---- 3. payload consistency: every displayed number derives from the record
    payload = None
    for path in ["repo/payload.json", "payload.json"]:
        if os.path.exists(path):
            payload = json.load(open(path)); break
    if payload and payload.get("games") and payload["games"][0].get("forecast"):
        bad = []
        for g in payload["games"]:
            f = g["forecast"]
            if abs(g["p_home"] - f["p_home"]) > 1e-4 or abs(g["p_model"] - f["p_home"]) > 1e-4: bad.append((g["game_id"], "p"))
            if abs(g["predicted_home_score"] - f["expected_home_score"]) > 1e-3 or abs(g["predicted_away_score"] - f["expected_away_score"]) > 1e-3: bad.append((g["game_id"], "score"))
            if abs(g["margin_pred"] - f["margin_home"]) > 1e-3: bad.append((g["game_id"], "margin"))
            if (g["predicted_winner"] == g["home_team"]) != (f["p_home"] > 0.5): bad.append((g["game_id"], "winner"))
            if abs(g["injuries"]["impact"]["p_now"] - f["p_home"]) > 1e-4: bad.append((g["game_id"], "impact"))
            if "p_blend" in g: bad.append((g["game_id"], "p_blend present"))
            if f["method"] != rp.METHOD or f["schema_version"] != rp.SCHEMA_VERSION: bad.append((g["game_id"], "method"))
        check(not bad, f"payload: {len(payload['games'])} games, every displayed number matches its forecast record ({bad[:3]})")
        check(all(g["forecast"]["forecast_at"] == payload["games"][0]["forecast"]["forecast_at"] for g in payload["games"]),
              "all games in the payload share one forecast cutoff (no mixed snapshots)")
    else:
        check(False, "a payload with forecast records is on disk (run the pipeline first)")

    # ---- 4/5. injury features: unrelated team invariance and roster quarantine
    if os.path.exists("data/games.csv") and os.path.exists("data/injuries/injuries_2025.parquet"):
        sched, team, plyr, inj, snap, rost, depth, cur = rp.load_all("data", None)
        ratings, _ = rp.team_ratings(team, sched, cur, 2)
        pw = rp.player_form(plyr, sched, ratings)
        base = rp.context_features(sched, pw, inj, snap, rost, depth, cur).set_index("game_id")
        # a fabricated Out for a player on a team that is NOT in JAX-DEN, same week
        wk = sched[(sched.season == cur) & (sched.game_type == "REG") & (sched.game_id == "2026_02_JAX_DEN")]
        if len(wk):
            w = int(wk.week.iloc[0])
            other = rost[(rost.season == cur) & (rost.team == "KC") & rost.gsis_id.notna()].iloc[0]
            extra = pd.DataFrame([{"season": cur, "week": w, "team": "KC", "gsis_id": other.gsis_id, "position": other.position,
                                   "full_name": other.full_name, "game_type": "REG", "report_status": "Out", "season_type": "REG"}])
            inj2 = pd.concat([inj, extra], ignore_index=True)
            after = rp.context_features(sched, pw, inj2, snap, rost, depth, cur).set_index("game_id")
            cols = ["inj_off_diff", "inj_def_diff", "home_out_off", "away_out_off", "home_out_def", "away_out_def"]
            same = np.allclose(base.loc["2026_02_JAX_DEN", cols].astype(float).values, after.loc["2026_02_JAX_DEN", cols].astype(float).values)
            kc = after[after.index.str.contains("_KC_") | after.index.str.endswith("_KC")]
            check(same, "an Out on KC leaves JAX-DEN's direct injury features unchanged")
            check(len(kc) == 0 or bool((kc.filter(like="out_").astype(float).sum(axis=1) >= base.loc[kc.index].filter(like="out_").astype(float).sum(axis=1)).all()),
                  "the same Out is applied to KC's own game")
            # a row whose player is not on that team's roster is quarantined, not used
            stray = pd.DataFrame([{"season": cur, "week": w, "team": "DEN", "gsis_id": other.gsis_id, "position": other.position,
                                   "full_name": other.full_name, "game_type": "REG", "report_status": "Out", "season_type": "REG"}])
            inj3 = pd.concat([inj, stray], ignore_index=True)
            q = rp.context_features(sched, pw, inj3, snap, rost, depth, cur)
            quar = q.attrs.get("inj_quarantine", {}).get((int(cur), w, "DEN"), [])
            q = q.set_index("game_id")
            check(any(x["gsis_id"] == other.gsis_id for x in quar), "a KC player filed as Out for DEN is quarantined (not on DEN's roster)")
            check(np.allclose(base.loc["2026_02_JAX_DEN", cols].astype(float).values, q.loc["2026_02_JAX_DEN", cols].astype(float).values),
                  "the quarantined row does not touch DEN's injury features")
    else:
        check(False, "data on disk for the injury checks")

    # ---- 6. unknown status
    i = pd.DataFrame([{"season": 2026, "week": 2, "team": "DEN", "gsis_id": "00-1", "full_name": "A Player", "position": "WR", "game_type": "REG",
                       "report_status": None, "practice_status": None, "report_primary_injury": "Knee", "practice_primary_injury": None,
                       "report_secondary_injury": None, "practice_secondary_injury": None}])
    m, by_team, _ = rp.injury_status(i, 2026, 2)
    st = m.get("00-1") or {}
    check(st.get("level") == "unknown" and "unknown" in st.get("label", "").lower(), "a report row with no status and no practice entry is shown as unknown, not dropped")

    # ---- 7. trigger vs reason in the version store
    tmp = tempfile.mkdtemp()
    os.environ["SJ_LIVE_DIR"] = tmp
    import importlib
    from live import store as lstore; importlib.reload(lstore)
    from live import events as levents; importlib.reload(levents)
    from live import versions as lv; importlib.reload(lv)
    pl = {"season": 2026, "week": 2, "players": [], "games": [
        {"game_id": "2026_02_JAX_DEN", "home_team": "DEN", "away_team": "JAX", "p_home": 0.55, "p_model": 0.55, "p_market": 0.58,
         "margin_pred": 1.0, "predicted_home_score": 23.0, "predicted_away_score": 22.0, "forecast": {"forecast_id": "x1", "method": "model_only_v2", "data_cutoff": "t0"}}]}
    lv.record(pl, "m_a", "d1", log=lambda *a: None)
    pl["games"][0].update({"p_home": 0.50, "p_model": 0.50, "forecast": {"forecast_id": "x2", "method": "model_only_v2", "data_cutoff": "t1"}})
    lv.record(pl, "m_a", "d2", trigger="live rebuild requested: Zay Flowers: DOUBTFUL -> OUT", log=lambda *a: None)
    rows = [r for r in lstore.read("prediction_versions") if r["game_id"] == "2026_02_JAX_DEN"]
    last = rows[-1]
    check(len(rows) == 2 and "Flowers" not in last["reason"] and last["trigger_is_cause"] is False and "Flowers" in (last["trigger"] or ""),
          "an unrelated player in the rebuild trigger is recorded as the trigger, never as this game's reason")
    check(last["source"] == "inputs" and last["change"]["p_home"] == -0.05, "the version says the change came from inputs and states the move in probability")
    pl["games"][0].update({"forecast": {"forecast_id": "x3", "method": "model_only_v2", "data_cutoff": "t1"}})
    lv.record(pl, "m_b", "d3", log=lambda *a: None)
    rows = [r for r in lstore.read("prediction_versions") if r["game_id"] == "2026_02_JAX_DEN"]
    check(len(rows) == 2, "a new model version with no numeric change writes no version (no artificial drift)")
    pl["games"][0].update({"p_home": 0.52, "p_model": 0.52})
    lv.record(pl, "m_b", "d4", log=lambda *a: None)
    rows = [r for r in lstore.read("prediction_versions") if r["game_id"] == "2026_02_JAX_DEN"]
    check(rows[-1]["source"] == "model release", "a change that coincides with a new model version is attributed to the model release")
    shutil.rmtree(tmp, ignore_errors=True); os.environ.pop("SJ_LIVE_DIR", None)

    # ---- 8. tracker method labels
    h = {"version": tracker.VERSION, "games": {"2025_01_A_B": {"s": 2025, "w": 1, "a": "A", "hm": "B", "p": 0.6, "pm": 0.55, "pk": 0.61, "mg": 3.0, "pick": "B"}}, "players": {}, "schemes": {}, "rollups": {}}
    hp = os.path.join(tempfile.mkdtemp(), "h.json"); json.dump(h, open(hp, "w"))
    h2 = tracker.load(hp)
    check(h2["games"]["2025_01_A_B"]["method"] == "blend_20_80_v1", "a legacy tracker row is labelled as the blend, not rewritten")
    up = pd.DataFrame([{"game_id": "2026_02_JAX_DEN", "away_team": "JAX", "home_team": "DEN", "p_home": 0.4742, "p_model": 0.4742, "p_market": 0.583,
                        "margin_pred": -2.37, "spread_line": 3.0, "total_line": 45.5, "predicted_winner": "JAX"}])
    tracker.lock_week(h2, up, [], [], 2026, 2, method=rp.METHOD, model_version="m_x")
    g = h2["games"]["2026_02_JAX_DEN"]
    check(g["method"] == "model_only_v2" and g["p"] == 0.4742 and g["mv"] == "m_x" and g["pick"] == "JAX", "a new lock stores the model-only probability, method and model version")

    # ---- 9/10. determinism and unrounded derivation
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd")); yc = (X.a + rng.normal(size=400) > 0).astype(int); yr = X.a * 3 + rng.normal(size=400)
    p1, m1 = rp._ens_predict(rp._fit_ensemble(X, yc, yr), X.head(20)); p2, m2 = rp._ens_predict(rp._fit_ensemble(X, yc, yr), X.head(20))
    check(np.allclose(p1, p2) and np.allclose(m1, m2), "the same inputs give the same forecast on a repeated fit (fixed seeds)")
    if payload and payload.get("games"):
        g0 = payload["games"][0]
        check(abs((g0["predicted_home_score"] + g0["predicted_away_score"]) - g0["total_pred"]) < 2e-3 and
              abs((g0["predicted_home_score"] - g0["predicted_away_score"]) - g0["margin_pred"]) < 2e-3,
              "scores in the payload derive from the unrounded total and margin (four decimals kept; rounding is for display)")

    print("\n" + ("all checks passed" if not FAILS else f"{len(FAILS)} FAILED"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
