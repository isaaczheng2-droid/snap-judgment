"""
Error measurement: the prospective record (forecasts saved before kickoff, graded after) and
the walk-forward frames the candidates are compared on. Every block carries its sample size;
a block with too few rows says so instead of quoting a number.
"""
import numpy as np
import pandas as pd

MIN_N = 30


def _block(g, played_only=False):
    if played_only:
        g = g[g.played == True]
    n = int(len(g))
    if n < MIN_N:
        return {"n": n, "note": f"fewer than {MIN_N} graded rows"}
    e = (g.act_pts - g.proj_pts).abs()
    out = {"n": n, "mae": round(float(e.mean()), 3), "bias": round(float((g.proj_pts - g.act_pts).mean()), 3)}
    nv = g.dropna(subset=["naive_pts"])
    if len(nv) >= MIN_N:
        en = (nv.act_pts - nv.naive_pts).abs()
        em = (nv.act_pts - nv.proj_pts).abs()
        out["naive_mae"] = round(float(en.mean()), 3)
        out["vs_naive"] = round(float(1 - em.mean() / en.mean()), 4) if en.mean() else None
        # paired bootstrap on the mean absolute-error difference
        d = (en - em).values
        rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(300)]
        out["vs_naive_ci95_pts"] = [round(float(np.quantile(bs, .025)), 3), round(float(np.quantile(bs, .975)), 3)]
    rr = g.dropna(subset=["range"]) if "range" in g.columns else g.iloc[0:0]
    rr = rr[rr.range.apply(lambda r: isinstance(r, dict) and "p10" in r)]
    if len(rr) >= MIN_N:
        lo = rr.range.apply(lambda r: r["p10"]); hi = rr.range.apply(lambda r: r["p90"])
        out["coverage_p10_p90"] = round(float(((rr.act_pts >= lo) & (rr.act_pts <= hi)).mean()), 4)
        out["coverage_n"] = int(len(rr))
    return out


def prospective(graded):
    """Metrics for the prospective ledger, overall and by position, week, status, played."""
    if graded is None or not len(graded):
        return {"n": 0, "note": "no graded forecasts yet"}
    g = graded.copy()
    out = {"n": int(len(g)), "weeks": sorted({int(w) for w in g.week.unique()}), "seasons": sorted({int(s) for s in g.season.unique()}),
           "all": _block(g), "played_only": _block(g, played_only=True),
           "did_not_play": int((g.played == False).sum()),
           "by_position": {p: _block(x) for p, x in g.groupby("position")},
           "by_week": {f"{int(s)}-{int(w):02d}": _block(x) for (s, w), x in g.groupby(["season", "week"])},
           "by_status": {str(s): _block(x) for s, x in g.groupby("status")},
           "corrections": int(g.corrected.sum()) if "corrected" in g.columns else 0,
           "by_model_version": {str(v): _block(x) for v, x in g.groupby("model_version")}}
    q = g[g.status == "questionable"]
    if len(q) >= 10:
        out["questionable_play_rate"] = {"n": int(len(q)), "played": round(float(q.played.mean()), 3),
                                         "p_play_claimed": round(float(q.p_play.mean()), 3)}
    return out


def paired(current, candidate, key=("player_id", "season", "week")):
    """
    Row-aligned absolute errors for two scored walk-forward frames. Returns a frame with
    e_cur, e_cand, season, position for the gate.
    """
    a = current.set_index(list(key))[["act_pts", "proj_pts", "position"]]
    b = candidate.set_index(list(key))[["proj_pts"]].rename(columns={"proj_pts": "cand_pts"})
    j = a.join(b, how="inner").reset_index()
    j["e_cur"] = (j.act_pts - j.proj_pts).abs()
    j["e_cand"] = (j.act_pts - j.cand_pts).abs()
    return j
