"""
Game script: how a game is likely to unfold, from the line, and what that means for volume.

Built from play-by-play (2016 onward): for each team-game, the share of its offensive snaps
taken while leading big, leading, tied, trailing, trailing big, split into the first three
quarters and the fourth, plus how many plays it ran. Those shares are averaged by the
team's pregame spread and the game total, so a 7-point favourite in a 48-point game gets
the mix of scripts such teams have actually played, not a guess. Pass rate per state comes
from the same plays. The output per team is a scenario mix with probabilities, the expected
pass and rush attempts it implies, and the plain-words version for the page.

This is displayed as context. As a model input it is tested on the walk-forward harness
(fantasy/script_test.py) and only kept in the number if it beats the shipped model; the
spread and total alone were tested before and improved errors by under 1%.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STATES = ["trail9", "trail1_8", "tied", "lead1_8", "lead9"]
LABELS = {"trail9": "trailing by 9+", "trail1_8": "trailing by 1-8", "tied": "tied", "lead1_8": "leading by 1-8", "lead9": "leading by 9+"}
SPREAD_BINS = [-99, -7, -3, -0.5, 0.5, 3, 7, 99]   # own spread: + = favoured; a pick'em sits in its own bin
TOTAL_BINS = [0, 41, 47, 99]
PATH = "script_model.json"


def _state(diff):
    return pd.cut(diff, [-100, -9, -1, 0, 8, 100], labels=STATES).astype(str)


def team_games(pbp_glob="data/pbp/play_by_play_*.parquet", seasons=(2016, 2025), log=print):
    """
    Per team-game: spread, total, plays, pass plays and the share of plays in each (period,
    state) cell; plus per-season pass-play and play counts by cell. Everything a model needs,
    so a model for "seasons before S" is a filter and a sum (no leakage in the walk-forward).
    """
    cols = ["game_id", "season", "week", "posteam", "home_team", "qtr", "score_differential", "play_type",
            "pass", "rush", "spread_line", "total_line"]
    frames = []
    for f in sorted(glob.glob(pbp_glob)):
        try:
            d = pd.read_parquet(f, columns=cols)
        except Exception as e:
            log(f"  script: skip {f}: {e}")
            continue
        d = d[d.play_type.isin(["pass", "run"]) & d.season.between(*seasons)]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d = d.dropna(subset=["posteam", "score_differential", "spread_line", "total_line"])
    d["state"] = _state(d.score_differential)
    d["late"] = (d.qtr >= 4)
    d["is_home"] = d.posteam == d.home_team
    d["own_spread"] = np.where(d.is_home, d.spread_line, -d.spread_line)
    d["cell"] = np.where(d.late, "late:", "early:") + d.state
    rates = d.groupby(["season", "cell"]).agg(pass_plays=("pass", "sum"), plays=("pass", "size")).reset_index()
    tg = d.groupby(["game_id", "season", "posteam", "own_spread", "total_line"]).agg(plays=("pass", "size"), pass_plays=("pass", "sum")).reset_index()
    shares = d.groupby(["game_id", "posteam", "cell"]).size().unstack(fill_value=0)
    shares = shares.div(shares.sum(axis=1), axis=0).reset_index()
    tg = tg.merge(shares, on=["game_id", "posteam"])
    tg["sb"] = pd.cut(tg.own_spread, SPREAD_BINS, labels=False)
    tg["tb"] = pd.cut(tg.total_line, TOTAL_BINS, labels=False)
    return tg, rates


def model_from(tg, rates, seasons):
    """The scenario tables from the team-games of `seasons` only."""
    t = tg[tg.season.between(*seasons)]
    r = rates[rates.season.between(*seasons)].groupby("cell")[["pass_plays", "plays"]].sum()
    pass_rate = {c: round(float(v.pass_plays / v.plays), 4) for c, v in r.iterrows() if v.plays}
    cells = [c for c in t.columns if ":" in str(c)]
    table, by_spread = {}, {}
    for (sb, tb), g in t.groupby(["sb", "tb"]):
        if len(g) < 40:
            continue
        table[f"{int(sb)}:{int(tb)}"] = {"n": int(len(g)), "plays": round(float(g.plays.mean()), 2),
                                        "plays_sd": round(float(g.plays.std()), 2),
                                        "shares": {c: round(float(g[c].mean()), 4) for c in cells},
                                        "pass_rate_actual": round(float(g.pass_plays.sum() / g.plays.sum()), 4)}
    for sb, g in t.groupby("sb"):
        by_spread[str(int(sb))] = {"n": int(len(g)), "plays": round(float(g.plays.mean()), 2),
                                   "shares": {c: round(float(g[c].mean()), 4) for c in cells},
                                   "pass_rate_actual": round(float(g.pass_plays.sum() / g.plays.sum()), 4)}
    return {"seasons": [int(seasons[0]), int(seasons[1])], "n_team_games": int(len(t)), "pass_rate": pass_rate, "table": table,
            "by_spread": by_spread, "spread_bins": SPREAD_BINS, "total_bins": TOTAL_BINS,
            "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")}


def fit(pbp_glob="data/pbp/play_by_play_*.parquet", seasons=(2016, 2025), out=PATH, log=print):
    """The shipped model: all completed seasons. Written to script_model.json (root and data/)."""
    tg, rates = team_games(pbp_glob, seasons, log)
    model = model_from(tg, rates, seasons)
    if out:
        json.dump(model, open(out, "w"), indent=1)
        os.makedirs("data", exist_ok=True)
        json.dump(model, open(os.path.join("data", os.path.basename(out)), "w"), indent=1)
    log(f"  script: {model['n_team_games']} team-games, {len(model['table'])} spread x total cells")
    return model


def load(path=PATH):
    for p in [path, os.path.join("data", path), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)]:
        try:
            return json.load(open(p))
        except Exception:
            continue
    return None


def _bin(x, edges):
    for i in range(len(edges) - 1):
        if edges[i] < x <= edges[i + 1]:
            return i
    return 0 if x <= edges[0] else len(edges) - 2


def scenario(model, own_spread, total, team=None, opp=None):
    """
    The scenario mix for one team: probabilities of ending the fourth quarter in each state
    (from the share of late plays), expected plays, expected pass and rush attempts, and
    the sentence for the page. None when the model or the line is missing.
    """
    if not model or own_spread is None or total is None or pd.isna(own_spread) or pd.isna(total):
        return None
    sb, tb = _bin(float(own_spread), model["spread_bins"]), _bin(float(total), model["total_bins"])
    cell = model["table"].get(f"{sb}:{tb}") or model["by_spread"].get(str(sb))
    if not cell:
        return None
    shares, pr = cell["shares"], model["pass_rate"]
    late = {s: shares.get(f"late:{s}", 0.0) for s in STATES}
    late_tot = sum(late.values()) or 1.0
    late_p = {s: round(v / late_tot, 3) for s, v in late.items()}
    exp_pass_rate = sum(shares.get(k, 0.0) * pr.get(k, 0.6) for k in shares)
    plays = cell["plays"]
    lead = late_p["lead9"] + late_p["lead1_8"]
    trail = late_p["trail9"] + late_p["trail1_8"]
    main = max(late_p, key=late_p.get)
    words = (f"Most likely {LABELS[main]} late" if late_p[main] >= 0.4 else "No dominant script")
    tilt = ("run-heavy finish" if late_p["lead9"] >= 0.3 else "pass-heavy finish" if late_p["trail9"] >= 0.3 else "balanced finish")
    return {"own_spread": float(own_spread), "total": float(total), "n_similar": cell["n"],
            "p_lead_late": round(lead, 3), "p_trail_late": round(trail, 3), "late_states": late_p,
            "exp_plays": round(plays, 1), "exp_pass_rate": round(exp_pass_rate, 3),
            "exp_pass_att": round(plays * exp_pass_rate, 1), "exp_rush_att": round(plays * (1 - exp_pass_rate), 1),
            "summary": f"{words}: in similar games {lead:.0%} of fourth-quarter snaps came with a lead and {trail:.0%} trailing; {tilt}. "
                       f"About {plays:.0f} plays at a {exp_pass_rate:.0%} pass rate.",
            "caveat": "A likely lead is not a particular player's carries: possession count, the rotation and who takes the clock-killing runs are not settled by the script."}
