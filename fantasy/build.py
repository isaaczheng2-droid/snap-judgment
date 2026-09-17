"""
Build the fantasy block of the payload: one record per QB/RB/WR/TE with a projection under
every preset, a validated range, availability, flags, opportunity, drivers and game script.
Called from run_pipeline.main after the prop projections; the stat models are the same.
"""
import json
import os

import numpy as np
import pandas as pd

from . import engine, scoring, script

WEIGHT_TERMS = ("form", "usage", "matchup", "venue", "absence")


def _model(datadir):
    for p in [os.path.join(datadir, "fantasy_model.json"), "fantasy_model.json",
              os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fantasy_model.json")]:
        try:
            return json.load(open(p))
        except Exception:
            continue
    return None


def _stat_weight(key, settings, pos):
    """Points per unit of a projection key under settings."""
    return scoring.points({key: 1.0}, settings, position=pos)


def build(rp, pw, ratings, sched, rost, depth, cur, target_week, inj_map, opp_sc, lost, datadir="data", log=print):
    model = _model(datadir) or {}
    smodel = script.load()
    targets = engine.all_targets(rp.PTARGETS)
    recs = rp.player_projections(pw, ratings, sched, rost, depth, cur, target_week, inj_map, opp_sc,
                                 lost=lost, targets=targets, depth_max_rank=3)
    recs = [r for r in recs if r.get("position") in engine.POS_KEYS]
    up = sched[(sched.season == cur) & (sched.week == target_week) & (sched.game_type == "REG")]
    playing = set(up.home_team) | set(up.away_team)
    all_teams = set(sched[(sched.season == cur) & (sched.game_type == "REG")].home_team)
    byes = sorted(all_teams - playing)
    # lines for the game script
    lines = {}
    for r in up.itertuples():
        for tm, own in ((r.home_team, r.spread_line), (r.away_team, -r.spread_line if pd.notna(r.spread_line) else np.nan)):
            lines[tm] = (own, r.total_line)
    scripts = {tm: script.scenario(smodel, sp, tot) for tm, (sp, tot) in lines.items()}
    # point-in-time inputs behind each candidate
    latest = pw.sort_values(["player_id", "gameday"]).groupby("player_id").tail(1).set_index("player_id")
    rookie_ids = set(rost[(rost.season == cur) & (rost.entry_year == cur)].gsis_id.dropna()) if "entry_year" in rost.columns else set()
    prev_team = pw[pw.season == cur - 1].sort_values("gameday").groupby("player_id").team.last().to_dict()
    cur_team = rost[rost.season == cur].dropna(subset=["gsis_id"]).drop_duplicates("gsis_id", keep="last").set_index("gsis_id").team.to_dict()
    new_team_ids = {pid for pid, t in cur_team.items() if pid in prev_team and prev_team[pid] != t}
    prev_share = {}
    if len(pw):
        ps = pw[pw.season == cur - 1].copy()
        for col, out in [("targets", "t"), ("carries", "c")]:
            tm = ps.groupby(["team", "season", "week"])[col].transform("sum")
            ps[f"_sh_{out}"] = (ps[col] / tm.replace(0, np.nan)).fillna(0)
        m = ps.groupby("player_id")[["_sh_t", "_sh_c"]].mean()
        for pid, r in m.iterrows():
            prev_share[pid] = float(r["_sh_c"]) if latest.position.get(pid) == "RB" else float(r["_sh_t"])
    play_rates = model.get("play_rates", {})
    presets = {k: v for k, v in scoring.PRESETS.items()}

    out = []
    by_team = {}
    for r in recs:
        pid = r["player_key"]; pos = r["position"]
        keys = engine.POS_KEYS[pos]
        proj = {k: r.get(k) for k in keys if r.get(k) is not None}
        if not proj:
            continue
        lr = latest.loc[pid] if pid in latest.index else None
        vol_key = engine.OPPORTUNITY[pos]
        pts = {name: scoring.points(proj, st, pos) for name, st in presets.items()}
        base = pts[scoring.DEFAULT]
        # the rolling-form baseline the backtest scores against, for the ledger
        naive = None
        if lr is not None:
            form = {k: lr.get(f"proj_{targets[k]['stat']}") for k in keys}
            if all(v is not None and not pd.isna(v) for v in form.values()):
                naive = scoring.points(form, presets[scoring.DEFAULT], pos)
        rng = {name: engine.range_for(pos, v, model) for name, v in pts.items()}
        # drivers in points: each stat's breakdown term times that stat's point weight
        drivers = {t: 0.0 for t in WEIGHT_TERMS}
        for k in keys:
            w = (r.get("why") or {}).get(k)
            if not w:
                continue
            wt = _stat_weight(k, presets[scoring.DEFAULT], pos)
            for t in WEIGHT_TERMS:
                if w.get(t) is not None:
                    drivers[t] += wt * float(w[t])
        drivers = {t: round(v, 2) for t, v in drivers.items()}
        avail = engine.availability(r, inj_map, set(byes), play_rates)
        flags = engine.role_flags(r, lr, prev_share, rookie_ids, new_team_ids)
        share_col = ("car_share" if pos == "RB" else "tgt_share")
        share_now = f"{share_col}_now" if lr is not None and f"{share_col}_now" in lr.index else share_col
        opp = {"volume_key": vol_key.replace("proj_", ""), "volume": None if lr is None or pd.isna(lr.get(vol_key)) else round(float(lr[vol_key]), 1),
               "share": None if lr is None or pd.isna(lr.get(share_now)) else round(float(lr[share_now]), 3),
               "snap_share": None if lr is None or pd.isna(lr.get("snap_carry")) else round(float(lr["snap_carry"]), 3),
               "games_this_season": None if lr is None else int(lr.get("gp_prior", 0)) + (1 if lr.get("season") == cur else 0)}
        rec = {"player_key": pid, "name": r["player_display_name"], "position": pos, "team": r["team"],
               "opponent": r.get("opponent_team"), "is_home": r.get("is_home"), "headshot": r.get("headshot"),
               "depth_rank": r.get("depth_rank"), "proj": {k: round(float(v), 2) for k, v in proj.items()},
               "pts": pts, "range": rng, "proj_pts": base, "naive_pts": naive,
               "p_play": avail["p_play"], "exp_pts": round(base * avail["p_play"], 2),
               "availability": avail, "flags": flags, "opportunity": opp, "drivers": drivers,
               "status": r.get("status"), "absence": r.get("absence")}
        out.append(rec)
        by_team.setdefault(r["team"], []).append(dict(rec, tgt_share=opp["share"] if pos != "RB" else None, car_share=opp["share"] if pos == "RB" else None))
    for rec in out:
        rec["competition"] = engine.competition(dict(rec, tgt_share=rec["opportunity"]["share"] if rec["position"] != "RB" else None,
                                                     car_share=rec["opportunity"]["share"] if rec["position"] == "RB" else None),
                                                by_team.get(rec["team"], []))
        rec["script"] = rec["team"] if scripts.get(rec["team"]) else None      # look up in fantasy.scripts
    out.sort(key=lambda x: -x["exp_pts"])
    val = {k: model.get(k) for k in ("coverage_holdout", "startsit_holdout", "version", "generated")}
    def _find(name):
        for cand in (os.path.join(datadir, name), name, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), name)):
            if os.path.exists(cand):
                return json.load(open(cand))
        raise FileNotFoundError(name)
    try:
        bt = _find("fantasy_backtest.json")
        val["holdout"] = bt.get("holdout", {}).get("by_position")
        val["holdout_all"] = bt.get("holdout", {}).get("all")
        val["rank_quality_holdout"] = bt.get("rank_quality_holdout")
        val["startsit_naive_holdout"] = bt.get("startsit_naive_holdout")
        val["circumstance_selection"] = bt.get("selection", {}).get("by_circumstance")
        val["notes"] = bt.get("notes")
    except Exception:
        pass
    try:
        st = _find("script_test.json")
        val["script_test"] = {k: {"rel_selection": v.get("rel_selection"), "rel_holdout": v.get("rel_holdout"), "side_flips": v.get("side_flips_vs_shipped")} for k, v in st.items()}
    except Exception:
        pass
    log(f"  fantasy: {len(out)} players ({sum(1 for x in out if x['availability']['status'] not in ('ok',))} with an availability flag), byes {byes}")
    return {"season": int(cur), "week": int(target_week), "settings_default": scoring.DEFAULT,
            "presets": {k: v.to_dict() for k, v in presets.items()}, "not_projected": scoring.NOT_PROJECTED,
            "lineup": scoring.lineup_slots(presets[scoring.DEFAULT]),
            "byes": byes, "players": out, "validation": val, "play_rates": play_rates,
            "scripts": {tm: (None if not s else {k: s[k] for k in ("summary", "caveat", "p_lead_late", "p_trail_late", "exp_plays", "exp_pass_rate", "exp_pass_att", "exp_rush_att", "n_similar")}) for tm, s in scripts.items()},
            "script_status": "context only: tested on the walk-forward harness (fantasy/script_test.py) and kept out of the number; see validation.script_test"}
