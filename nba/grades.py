"""
Snap Grade: a 0-100 DESCRIPTIVE summary of a player's recent production, one number per player
with its components, comparison group, timeframe and sample size attached.

What it is not: a prop probability, a projection, a Madden/2K/PFF/ESPN rating, or a proof of
predictive value. Until the chronological validation in validate() shows a grade predicts
next-game production better than a trailing average, every grade is labelled
"descriptive summary". The grade is a PERCENTILE among the comparison group (labelled as such);
the weighted rating it is built from is shown beside it under its own name.

NBA components (per position group G / F / C, current season, minimum 5 games played):
  production   per-36 box production (pts, reb, ast, stl, blk, tov, shooting efficiency via
               TS%), shrunk toward the position mean by games played
  role         minutes per game and start share
  form         last-10 production per 36 vs the season figure
  impact       plus-minus per 36 relative to the team's overall margin (a coarse on-court
               proxy; NOT a defence grade; steals/blocks are not used as a defence stand-in)
Defence is not graded until lineup on/off data is collected (nba/collect.py, pbpstats).

Every output row carries the explainability block the page shows on demand.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_VERSION = "snap_grade_nba_v1"
MIN_GAMES = 5
POS_GROUP = {"G": "Guards", "F": "Forwards", "C": "Centers"}
WEIGHTS = {"production": 0.5, "role": 0.2, "form": 0.15, "impact": 0.15}


def _pct(s):
    return s.rank(pct=True) * 100


def _per36(df):
    m = df["min"].sum()
    if not m:
        return None
    f = 36.0 / m
    ts_den = 2 * (df.fga.sum() + 0.44 * df.fta.sum())
    return {"pts": df.pts.sum() * f, "reb": df.reb.sum() * f, "ast": df.ast.sum() * f, "stl": df.stl.sum() * f,
            "blk": df.blk.sum() * f, "tov": df.tov.sum() * f, "ts": (df.pts.sum() / ts_den) if ts_den else np.nan,
            "pm": df.plus_minus.sum() * f if "plus_minus" in df else np.nan}


def production_score(p):
    """Box production per 36 as a single number; TS% enters relative to a 0.57 league reference."""
    return p["pts"] + 1.2 * p["reb"] + 1.5 * p["ast"] + 2.5 * (p["stl"] + p["blk"]) - 1.5 * p["tov"] + 25 * ((p["ts"] if pd.notna(p["ts"]) else 0.57) - 0.57)


def compute(P, T, season, asof=None):
    """P: player_games, T: team_games (for team margin). Returns list of grade rows for `season`
    using games up to `asof` (UTC timestamp) only."""
    Q = P[(P.season == season) & P.played]
    if asof is not None:
        Q = Q[Q.tipoff_utc <= asof]
    tm = T[T.season == season]
    if asof is not None:
        tm = tm[tm.tipoff_utc <= asof]
    team_margin = (tm.pts - tm.opp_pts).groupby(tm.team_id).mean().to_dict()
    rows = []
    for pid, g in Q.groupby("player_id"):
        g = g.sort_values("tipoff_utc")
        n = len(g)
        last = g.iloc[-1]
        pos = str(last.position or "G")[0]
        per = _per36(g)
        if per is None:
            continue
        last10 = _per36(g.tail(10))
        rows.append({"player_id": pid, "player": last.player, "team": last.team, "team_id": int(last.team_id), "pos": pos,
                     "games": n, "mpg": float(g["min"].mean()), "start_share": float(g.starter.mean()),
                     "prod": production_score(per), "prod10": production_score(last10) if last10 else np.nan,
                     "pm36": per["pm"], "team_margin": team_margin.get(last.team_id, 0.0), "per36": per,
                     "headshot": last.headshot, "last_game": last.tipoff_utc})
    if not rows:
        return []
    D = pd.DataFrame(rows)
    out = []
    for pos, grp in D.groupby("pos"):
        grp = grp.copy()
        elig = grp.games >= MIN_GAMES
        pool = grp[elig]
        if len(pool) < 10:
            for _, r in grp.iterrows():
                out.append(_row(r, None, {}, pos, "Not enough data: fewer than 10 graded players in the group", len(pool)))
            continue
        mean_prod = pool["prod"].mean()
        # shrink production toward the group mean by games played
        w = pool.games / (pool.games + 8)
        prod_s = w * pool["prod"] + (1 - w) * mean_prod
        comp = pd.DataFrame(index=pool.index)
        comp["production"] = _pct(prod_s)
        comp["role"] = _pct(0.7 * pool.mpg + 12 * pool.start_share)
        comp["form"] = _pct((pool.prod10 - pool["prod"]).fillna(0))
        comp["impact"] = _pct((pool.pm36 - pool.team_margin * 36 / 48).fillna(0))
        rating = sum(WEIGHTS[k] * comp[k] for k in WEIGHTS)
        grade = _pct(rating)
        for i, r in grp.iterrows():
            if i in pool.index:
                c = {k: {"percentile": round(float(comp.loc[i, k]), 1)} for k in WEIGHTS}
                c["production"]["value"] = round(float(prod_s.loc[i]), 1)
                c["production"]["unit"] = "box production per 36 (shrunk)"
                c["role"]["value"] = round(float(r.mpg), 1); c["role"]["unit"] = "minutes per game"
                c["form"]["value"] = round(float((r.prod10 - r["prod"]) if pd.notna(r.prod10) else 0), 1); c["form"]["unit"] = "last-10 minus season, per 36"
                c["impact"]["value"] = round(float(r.pm36 - r.team_margin * 36 / 48) if pd.notna(r.pm36) else 0, 1); c["impact"]["unit"] = "plus-minus per 36 vs team"
                out.append(_row(r, float(grade.loc[i]), c, pos, None, len(pool), rating=float(rating.loc[i])))
            else:
                out.append(_row(r, None, {}, pos, f"Not enough data: {int(r.games)} of {MIN_GAMES} games played", len(pool)))
    return out


META = {"grade_type": "percentile", "coverage": "box scores only (no lineup/on-off data yet)",
        "defence": "not graded: lineup on/off data not yet collected", "methodology_version": METHOD_VERSION,
        "components": {"production": "per-36 box production incl. TS%, shrunk to the position mean by games played",
                       "role": "minutes per game and start share", "form": "last-10 production per 36 minus the season figure",
                       "impact": "plus-minus per 36 relative to the team's margin (coarse on-court proxy, not a defence grade)"},
        "weights": WEIGHTS, "min_games": MIN_GAMES}


def _row(r, grade, comps, pos, reason, n_group, rating=None):
    """One compact row; the shared explanation strings live in META (payload 'grades.meta')."""
    return {"player_id": r.player_id, "player": r.player, "team": r.team, "team_id": r.team_id, "position": pos,
            "grade": None if grade is None else round(grade), "weighted_rating": None if rating is None else round(rating, 1),
            "label": "Not enough data" if grade is None else "descriptive summary", "reason": reason,
            "components": {k: {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "unit"} for k, v in comps.items()},
            "group": f"{POS_GROUP.get(pos, pos)}, {n_group} with {MIN_GAMES}+ games", "sample_size": int(r.games),
            "headshot": r.headshot, "per36": {k: (None if pd.isna(v) else round(float(v), 1)) for k, v in r.per36.items()}}


def validate(P, T, seasons=(2024, 2025, 2026), checkpoints=(20, 40, 60)):
    """Chronological check of predictive value: at game-N of each season, does the grade rank
    players' production over their NEXT 10 games better than their trailing per-36 alone?
    Reported as Spearman correlations; the page quotes it as-is."""
    from scipy.stats import spearmanr
    res = []
    for s in seasons:
        Q = P[(P.season == s) & P.played].sort_values("tipoff_utc")
        if Q.empty:
            continue
        dates = sorted(Q.tipoff_utc.unique())
        for cp in checkpoints:
            # checkpoint: the date by which the median team has played cp games
            per_team = Q.groupby("team_id").tipoff_utc.apply(lambda x: sorted(x.unique()))
            cut = np.median([t[min(cp, len(t)) - 1].value for t in per_team if len(t) >= cp]) if any(len(t) >= cp for t in per_team) else None
            if cut is None:
                continue
            cut = pd.Timestamp(cut, tz="UTC")
            grades = compute(P, T, s, asof=cut)
            gmap = {g["player_id"]: g for g in grades if g["grade"] is not None}
            after = Q[Q.tipoff_utc > cut]
            nxt = {}
            for pid, g in after.groupby("player_id"):
                g = g.head(10)
                if len(g) >= 5 and pid in gmap:
                    per = _per36(g)
                    nxt[pid] = production_score(per) * g["min"].mean() / 36   # per-game production over the next 10
            ids = list(nxt)
            if len(ids) < 50:
                continue
            y = np.array([nxt[i] for i in ids])
            gr = np.array([gmap[i]["grade"] for i in ids])
            tr = np.array([gmap[i]["per36"]["pts"] + 1.2 * gmap[i]["per36"]["reb"] + 1.5 * gmap[i]["per36"]["ast"] for i in ids])
            res.append({"season": s, "checkpoint_games": cp, "n": len(ids), "spearman_grade": round(float(spearmanr(gr, y)[0]), 3),
                        "spearman_trailing_per36": round(float(spearmanr(tr, y)[0]), 3)})
    return res


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    from nba import data
    d = data.load()
    P, T = d["player_games"], d["team_games"]
    P = P.merge(d["games"][["game_id"]], on="game_id")
    v = validate(P, T)
    for r in v:
        print(r)
    json.dump({"methodology_version": METHOD_VERSION, "validation": v,
               "reading": "Spearman rank correlation with per-game production over the next 10 games, at three checkpoints per season; the grade is labelled a descriptive summary regardless, since this checks ranking of production, not prop or fantasy outcomes"},
              open(os.path.join(HERE, "data", "grade_validation.json"), "w"), indent=1)
