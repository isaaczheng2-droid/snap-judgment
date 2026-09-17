"""
Candidate models: a fixed menu of parameter and feature-set changes applied to the active
model, each run through the same walk-forward harness as the shipped model and compared
row by row. Nothing here invents a feature; the menu lives in learn/config.json.
"""
import copy

import numpy as np

from fantasy import backtest, engine, scoring
from fantasy import script as fscript
from fantasy import script_test


def params_for(spec, current):
    """The candidate's full parameter set: the current one with the spec's changes applied."""
    p = copy.deepcopy(current)
    if "alpha" in spec:
        p["alpha"] = float(spec["alpha"])
    for stat, feats in (spec.get("extra_feats_add") or {}).items():
        p["extra_feats"][stat] = sorted(set(p["extra_feats"].get(stat, [])) | set(feats))
    for stat in spec.get("extra_feats_drop") or []:
        p["extra_feats"].pop(stat, None)
    for stat in spec.get("adj_shares_drop") or []:
        p["adj_shares"].pop(stat, None)
    p["script_feats"] = bool(spec.get("script_feats", p.get("script_feats", False)))
    return p


def feature_fn(rp, params, targets):
    ef, ad = params["extra_feats"], params["adj_shares"]
    def fn(key):
        f, _, _ = rp.player_feature_set(key, True, targets=targets, extra_feats=ef, adj_shares=ad)
        if params.get("script_feats"):
            f = f + ["exp_pass_att", "exp_rush_att", "p_lead_late", "exp_targets", "exp_carries"]
        return f
    return fn


def run(rp, pw, sched, config, current_params, specs, seasons, log=print):
    """
    Walk-forward scored frames for the current model and each candidate on `seasons`
    (inclusive range). Returns {"current": frame, "<id>": frame, ...} and the params used.
    """
    targets = engine.all_targets(rp.PTARGETS)
    settings = scoring.PRESETS[scoring.DEFAULT]
    tests = list(range(int(seasons[0]), int(seasons[1]) + 1))
    need_script = any(s.get("script_feats") for s in specs) or current_params.get("script_feats")
    if need_script and "exp_pass_att" not in pw.columns:
        tg, rates = fscript.team_games(log=log)
        pw = script_test.add_script(pw, sched, tg, rates, tests)
    out, used = {}, {}
    cur_fn = feature_fn(rp, current_params, targets)
    rows = backtest.walk_forward(pw, targets, tests, alpha=current_params["alpha"], feature_fn=cur_fn, log=log)
    out["current"] = backtest.score_rows(rows, settings, targets)
    used["current"] = current_params
    for spec in specs:
        p = params_for(spec, current_params)
        fn = feature_fn(rp, p, targets)
        rows = backtest.walk_forward(pw, targets, tests, alpha=p["alpha"], feature_fn=fn, log=log)
        out[spec["id"]] = backtest.score_rows(rows, settings, targets)
        used[spec["id"]] = p
        log(f"  candidate {spec['id']}: {len(out[spec['id']])} rows, MAE {float((out[spec['id']].act_pts - out[spec['id']].proj_pts).abs().mean()):.3f}")
    return out, used
