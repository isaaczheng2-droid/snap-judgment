#!/usr/bin/env python3
"""
Fantasy projections, scored the way a lineup decision is scored.

Walk-forward by season (test season S trained on seasons < S), same ridge stat models the
site runs, combined into points under the league settings. Selection seasons 2019-2024;
2025 is reported separately as the untouched holdout. Reported against three baselines a
reader could compute themselves:

  naive         points from the player's own rolling form (what he usually does)
  last_season   his average points per game last season, position average for newcomers
  position      the position's average every week

Metrics: MAE / RMSE per position and season; ranking quality (Spearman per week, top-N
precision); coverage of the published range; start/sit pair accuracy by projected gap; error
by circumstance (rookie, questionable, role changed). Also the play rate by injury
designation, which is what "questionable" is worth on the page.

Writes data/fantasy_backtest.json (the numbers), fantasy_model.json (the residual spread the
page draws ranges from, plus play rates) and data/fantasy_oof.parquet (row level, for the
learning cycle). Every number is reported whichever way it comes out.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_pipeline as rp                                   # noqa: E402
from fantasy import scoring, engine                         # noqa: E402

SEL = (2019, 2024)
HOLDOUT = 2025
GAP_BANDS = [(1, 3, "1-3 pts"), (3, 5, "3-5 pts"), (5, 99, "5+ pts")]
TOP_N = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}


# --------------------------------------------------------------------------- features
def annotate(pw, inj, rost, plyr):
    """Injury designation, rookie, role change and last-season points per row, all point-in-time."""
    pw = pw.copy()
    pw[["season", "week"]] = pw[["season", "week"]].astype("int64")
    if len(inj) and "report_status" in inj.columns:
        i = inj[inj.game_type == "REG"][["season", "week", "gsis_id", "report_status"]].dropna(subset=["gsis_id"]).copy()
        i[["season", "week"]] = i[["season", "week"]].astype("int64")
        i = i.drop_duplicates(["season", "week", "gsis_id"]).rename(columns={"gsis_id": "player_id", "report_status": "designation"})
        pw = pw.merge(i, on=["player_id", "season", "week"], how="left")
    else:
        pw["designation"] = None
    if len(rost) and "entry_year" in rost.columns:
        rk = rost.dropna(subset=["gsis_id"])[["season", "gsis_id", "entry_year"]].drop_duplicates(["season", "gsis_id"])
        rk = rk.rename(columns={"gsis_id": "player_id"})
        rk["rookie"] = rk.entry_year == rk.season
        pw = pw.merge(rk[["player_id", "season", "rookie"]], on=["player_id", "season"], how="left")
        pw["rookie"] = pw.rookie.fillna(False).astype(bool)
    else:
        pw["rookie"] = False
    # role change: this season's prior share vs last season's mean share
    for col, out in [("targets", "tgt"), ("carries", "car")]:
        tm = pw.groupby(["team", "season", "week"])[col].transform("sum")
        pw[f"_sh_{out}"] = (pw[col] / tm.replace(0, np.nan)).fillna(0)
    prev = pw.groupby(["player_id", "season"])[["_sh_tgt", "_sh_car"]].mean().reset_index()
    prev["season"] += 1
    prev = prev.rename(columns={"_sh_tgt": "prev_tgt_share", "_sh_car": "prev_car_share"})
    pw = pw.merge(prev, on=["player_id", "season"], how="left")
    share_now = np.where(pw.position == "RB", pw.car_share, pw.tgt_share)
    share_prev = np.where(pw.position == "RB", pw.prev_car_share, pw.prev_tgt_share)
    pw["role_change"] = (~pd.isna(share_prev)) & (np.abs(share_now - share_prev) >= engine.ROLE_SHIFT) & (pw.gp_prior >= engine.ROLE_MIN_GAMES)
    # last season's actual points per game (the baseline), from the box scores
    if "fantasy_points_ppr" in plyr.columns:
        ls = plyr[plyr.season_type == "REG"].groupby(["player_id", "season"]).fantasy_points_ppr.mean().reset_index()
        ls["season"] += 1
        ls = ls.rename(columns={"fantasy_points_ppr": "last_season_ppg"})
        pw = pw.merge(ls, on=["player_id", "season"], how="left")
    else:
        pw["last_season_ppg"] = np.nan
    return pw


def walk_forward(pw, targets, tests, alpha=5.0, feature_fn=None, log=print):
    """
    Out-of-sample stat projections for every eligible player-week in `tests`. Returns rows
    with proj_<key> per projection key, the actuals, and the baselines' inputs.
    """
    feature_fn = feature_fn or (lambda k: rp.player_feature_set(k, True, targets=targets)[0])
    frames = []
    for pos, keys in engine.POS_KEYS.items():
        vol, mn = engine.PRIMARY_VOL[pos]
        base = pw[(pw.position == pos) & (pw[vol] >= mn)].copy()
        if not len(base):
            continue
        for key in keys:
            cfg = targets[key]
            feats = feature_fn(key)
            need = feats + [cfg["stat"]]
            sub = pw[pw.position.isin(cfg["pos"]) & (pw[cfg["vol"]] >= cfg["mn"])].dropna(subset=need)
            col = f"fp_{key}"
            base[col] = np.nan
            for s in tests:
                tr = sub[sub.season < s]
                te_idx = base.index[(base.season == s)]
                te = base.loc[te_idx].dropna(subset=feats)
                if len(tr) < 300 or not len(te):
                    continue
                m = Ridge(alpha=alpha).fit(tr[feats], tr[cfg["stat"]])
                base.loc[te.index, col] = m.predict(te[feats])
        frames.append(base[base.season.isin(tests)])
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return rows


def score_rows(rows, settings, targets):
    """Projected, naive and actual points per row under the settings."""
    st = settings if isinstance(settings, scoring.LeagueSettings) else scoring.LeagueSettings.from_dict(settings)
    proj, naive, act = [], [], []
    for r in rows.itertuples(index=False):
        d = r._asdict()
        pos = d["position"]
        keys = engine.POS_KEYS.get(pos, [])
        p = {k: d.get(f"fp_{k}") for k in keys}
        if any(v is None or pd.isna(v) for v in p.values()):
            proj.append(np.nan); naive.append(np.nan); act.append(np.nan)
            continue
        proj.append(scoring.points(p, st, pos))
        naive.append(scoring.points({k: d.get(f"proj_{targets[k]['stat']}") for k in keys}, st, pos))
        act.append(scoring.actual_points(d, st, pos))
    out = rows.copy()
    out["proj_pts"], out["naive_pts"], out["act_pts_custom"] = proj, naive, act
    # under Full PPR the reference actual is nflverse's own number, which includes fumbles and 2pt
    out["act_pts"] = out["fantasy_points_ppr"] if (st.rec == 1.0 and st.pass_td == 4.0 and "fantasy_points_ppr" in out.columns) else out["act_pts_custom"]
    return out.dropna(subset=["proj_pts", "act_pts"])


def fit_residuals(scored, seasons):
    """Per position: |residual| scale a + b * proj (least squares on absolute residuals) and quantiles of the standardised residual."""
    out = {}
    sel = scored[scored.season.between(*seasons)]
    for pos, g in sel.groupby("position"):
        if len(g) < 300:
            continue
        res = (g.act_pts - g.proj_pts).values
        proj = g.proj_pts.values
        A = np.vstack([np.ones_like(proj), proj]).T
        coef, *_ = np.linalg.lstsq(A, np.abs(res), rcond=None)
        a, b = float(coef[0]), float(coef[1])
        scale = np.maximum(a + b * proj, 0.5)
        z = res / scale
        out[pos] = {"a": round(a, 4), "b": round(b, 4), "n": int(len(g)),
                    "q": {k: round(float(np.quantile(z, v)), 4) for k, v in [("p10", .1), ("p25", .25), ("p50", .5), ("p75", .75), ("p90", .9)]}}
    return out


def coverage(scored, resid, season):
    out = {}
    h = scored[scored.season == season]
    for pos, g in h.groupby("position"):
        m = resid.get(pos)
        if not m or len(g) < 50:
            continue
        scale = np.maximum(m["a"] + m["b"] * g.proj_pts.values, 0.5)
        lo10, hi90 = g.proj_pts + scale * m["q"]["p10"], g.proj_pts + scale * m["q"]["p90"]
        lo25, hi75 = g.proj_pts + scale * m["q"]["p25"], g.proj_pts + scale * m["q"]["p75"]
        out[pos] = {"n": int(len(g)),
                    "in_p10_p90": round(float(((g.act_pts >= np.maximum(lo10, 0)) & (g.act_pts <= hi90)).mean()), 4),
                    "in_p25_p75": round(float(((g.act_pts >= np.maximum(lo25, 0)) & (g.act_pts <= hi75)).mean()), 4)}
    return out


def rank_quality(scored, label_col="proj_pts"):
    """Spearman per (season, week, position) and top-N precision, averaged."""
    out = {}
    for pos, g in scored.groupby("position"):
        rhos, prec, n_weeks = [], [], 0
        for (s, w), gg in g.groupby(["season", "week"]):
            if len(gg) < 8:
                continue
            rho = sps.spearmanr(gg[label_col], gg.act_pts).correlation
            if not np.isnan(rho):
                rhos.append(rho)
            n = min(TOP_N[pos], max(3, len(gg) // 3))
            top_proj = set(gg.nlargest(n, label_col).player_id)
            top_act = set(gg.nlargest(n, "act_pts").player_id)
            prec.append(len(top_proj & top_act) / n)
            n_weeks += 1
        if n_weeks:
            out[pos] = {"weeks": n_weeks, "spearman": round(float(np.mean(rhos)), 4), "top_n": TOP_N[pos],
                        "top_n_precision": round(float(np.mean(prec)), 4)}
    return out


def startsit(scored, label_col="proj_pts", flex=False, max_pairs=400000):
    """
    Pairwise: when the projection prefers A over B by a gap, how often did A score more?
    Ties in actual points are excluded. Flex pairs mix RB/WR/TE.
    """
    rng = np.random.default_rng(0)
    out = {}
    groups = scored[scored.position.isin(["RB", "WR", "TE"])].groupby(["season", "week"]) if flex else scored.groupby(["season", "week", "position"])
    rec = {lab: [] for _, _, lab in GAP_BANDS}
    for _, g in groups:
        if len(g) < 4:
            continue
        p, a = g[label_col].values, g.act_pts.values
        i, j = np.triu_indices(len(g), k=1)
        gap = p[i] - p[j]
        sign = np.sign(gap)
        keep = (sign != 0) & (a[i] != a[j])
        hi_won = np.where(sign[keep] > 0, a[i][keep] > a[j][keep], a[j][keep] > a[i][keep])
        agap = np.abs(gap[keep])
        for lo, hi, lab in GAP_BANDS:
            m = (agap >= lo) & (agap < hi)
            if m.any():
                rec[lab].append(hi_won[m])
    for lab, parts in rec.items():
        if not parts:
            continue
        v = np.concatenate(parts)
        if len(v) > max_pairs:
            v = rng.choice(v, max_pairs, replace=False)
        n = int(len(v)); k = int(v.sum())
        lo, hi = sps.binomtest(k, n, 0.5).proportion_ci(0.95)
        out[lab] = {"n": n, "higher_scored_more": round(k / n, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
    return out


def by_segment(scored, seasons=None):
    """MAE by position, season, and circumstance, with naive and last-season baselines beside it."""
    d = scored if seasons is None else scored[scored.season.between(*seasons)]
    pos_mean = d.groupby("position").act_pts.transform("mean")
    d = d.assign(ls_pts=d.last_season_ppg.fillna(pos_mean), pos_pts=pos_mean)
    def block(g):
        n = int(len(g))
        if not n:
            return {"n": 0}
        e = (g.act_pts - g.proj_pts).abs()
        return {"n": n, "mae": round(float(e.mean()), 3), "rmse": round(float(np.sqrt(((g.act_pts - g.proj_pts) ** 2).mean())), 3),
                "bias": round(float((g.proj_pts - g.act_pts).mean()), 3),
                "naive_mae": round(float((g.act_pts - g.naive_pts).abs().mean()), 3),
                "last_season_mae": round(float((g.act_pts - g.ls_pts).abs().mean()), 3),
                "position_mae": round(float((g.act_pts - g.pos_pts).abs().mean()), 3),
                "vs_naive": round(float(1 - e.mean() / (g.act_pts - g.naive_pts).abs().mean()), 4) if n else None}
    out = {"all": block(d), "by_position": {p: block(g) for p, g in d.groupby("position")},
           "by_season": {int(s): block(g) for s, g in d.groupby("season")},
           "by_circumstance": {
               "rookie": block(d[d.rookie == True]),
               "questionable": block(d[d.designation == "Questionable"]),
               "role_change": block(d[d.role_change == True]),
               "no_flags": block(d[(d.rookie == False) & (d.role_change == False) & (d.designation.isna())]),
           }}
    return out


def play_rates(inj, plyr, seasons=(2019, 2025)):
    """Share of players with each designation who appeared in that week's box score, by position."""
    if not len(inj) or "report_status" not in inj.columns:
        return {}
    i = inj[(inj.game_type == "REG") & inj.report_status.isin(["Out", "Doubtful", "Questionable"]) & inj.season.between(*seasons)]
    i = i.dropna(subset=["gsis_id"])[["season", "week", "gsis_id", "position", "report_status"]].drop_duplicates(["season", "week", "gsis_id"])
    played = plyr[plyr.season_type == "REG"][["season", "week", "player_id"]].drop_duplicates()
    played["played"] = True
    m = i.merge(played, left_on=["season", "week", "gsis_id"], right_on=["season", "week", "player_id"], how="left")
    m["played"] = m.played.fillna(False)
    out = {}
    for st, g in m.groupby("report_status"):
        lvl = st.lower()
        out[lvl] = {"ALL": {"rate": round(float(g.played.mean()), 4), "n": int(len(g))}}
        for pos, gg in g[g.position.isin(["QB", "RB", "WR", "TE"])].groupby("position"):
            if len(gg) >= 30:
                out[lvl][pos] = {"rate": round(float(gg.played.mean()), 4), "n": int(len(gg))}
    return out


def evaluate(scored, resid=None, sel=SEL, holdout=HOLDOUT):
    """The full metric set for one scored frame; `resid` fitted on `sel` if not given."""
    resid = resid or fit_residuals(scored, sel)
    hold = scored[scored.season == holdout]
    return {
        "selection_seasons": list(sel), "holdout_season": holdout,
        "selection": by_segment(scored, sel),
        "holdout": by_segment(scored, (holdout, holdout)),
        "rank_quality_selection": rank_quality(scored[scored.season.between(*sel)]),
        "rank_quality_holdout": rank_quality(hold),
        "rank_quality_naive_holdout": rank_quality(hold, "naive_pts"),
        "startsit_selection": startsit(scored[scored.season.between(*sel)]),
        "startsit_holdout": startsit(hold),
        "startsit_naive_holdout": startsit(hold, "naive_pts"),
        "startsit_flex_holdout": startsit(hold, flex=True),
        "coverage_holdout": coverage(scored, resid, holdout),
        "residuals": resid,
    }


def build_rows(datadir="data", log=print):
    sched, team, plyr, inj, snap, rost, depth, cur = rp.load_all(datadir, None)
    reg = sched[(sched.season == cur) & (sched.game_type == "REG")]
    un = reg[reg.home_score.isna()]
    tw = int(un.week.min()) if len(un) else int(reg.week.max())
    ratings, _ = rp.team_ratings(team, sched, cur, tw)
    pw = rp.player_form(plyr, sched, ratings)
    pw, _ = rp.usage_extras(pw, snap, rost, inj, log=log)
    pw = annotate(pw, inj, rost, plyr)
    return pw, inj, plyr, cur


def main(datadir="data", write=True):
    pw, inj, plyr, cur = build_rows(datadir, log=rp.log)
    targets = engine.all_targets(rp.PTARGETS)
    tests = list(range(SEL[0], int(cur)))
    rows = walk_forward(pw, targets, tests, alpha=rp.RIDGE_ALPHA, log=rp.log)
    settings = scoring.PRESETS[scoring.DEFAULT]
    scored = score_rows(rows, settings, targets)
    res = evaluate(scored)
    pr = play_rates(inj, plyr)
    fingerprint = hashlib.sha256(json.dumps({"extra": rp.EXTRA_FEATS, "adj": rp.ADJ_SHARES, "alpha": rp.RIDGE_ALPHA}, sort_keys=True).encode()).hexdigest()[:12]
    res.update({"settings": settings.to_dict(), "n_rows": int(len(scored)), "model_fingerprint": fingerprint,
                "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                "notes": ["Rows are player-weeks where the player appeared; availability is handled separately through play rates by designation.",
                          "Actual points are nflverse fantasy_points_ppr, which include fumbles and two-point conversions the projection does not model.",
                          "2025 was not used to choose anything; selection seasons are 2019-2024."]})
    if write:
        json.dump(res, open(os.path.join(datadir, "fantasy_backtest.json"), "w"), indent=1)
        model = {"version": fingerprint, "generated": res["generated"], "settings": settings.to_dict(),
                 "residuals": res["residuals"], "play_rates": pr,
                 "coverage_holdout": res["coverage_holdout"], "startsit_holdout": res["startsit_holdout"]}
        for path in [os.path.join(datadir, "fantasy_model.json"), "fantasy_model.json"]:
            json.dump(model, open(path, "w"), indent=1)
        keep = ["player_id", "player_display_name", "position", "team", "season", "week", "proj_pts", "naive_pts", "act_pts",
                "designation", "rookie", "role_change", "last_season_ppg"] + [c for c in scored.columns if c.startswith("fp_")]
        scored[keep].to_parquet(os.path.join(datadir, "fantasy_oof.parquet"), index=False)
    # print the short version
    print(f"\nfantasy backtest: {len(scored)} player-weeks; selection {SEL}, holdout {HOLDOUT}")
    for name in ("selection", "holdout"):
        blk = res[name]
        print(f"  {name}: MAE {blk['all']['mae']} vs naive {blk['all']['naive_mae']} vs last season {blk['all']['last_season_mae']} vs position {blk['all']['position_mae']}")
        for pos, b in blk["by_position"].items():
            print(f"     {pos}: n={b['n']} MAE {b['mae']} naive {b['naive_mae']} ({b['vs_naive']:+.1%})")
    print("  rank quality (holdout):", {p: (v['spearman'], v['top_n_precision']) for p, v in res['rank_quality_holdout'].items()})
    print("  start/sit (holdout):", {k: (v['higher_scored_more'], v['n']) for k, v in res['startsit_holdout'].items()})
    print("  start/sit naive (holdout):", {k: (v['higher_scored_more'], v['n']) for k, v in res['startsit_naive_holdout'].items()})
    print("  coverage (holdout):", res["coverage_holdout"])
    print("  play rates:", {k: v.get('ALL') for k, v in pr.items()})
    return res


if __name__ == "__main__":
    main()
