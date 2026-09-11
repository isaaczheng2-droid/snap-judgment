#!/usr/bin/env python3
"""
Re-run the walk-forward audit with the scheme features included and emit the blocks of
the site's static `accuracy` object that the change invalidates. Writes acc_patch.json.

Nothing here is fitted on the season it scores: for test season S the model sees only
seasons < S. That is the only claim the Accuracy tab makes, and it has to stay true.
"""
import json
import numpy as np, pandas as pd, xgboost as xgb
from scipy import stats

import run_pipeline as rp
from adjusted_ratings import ADJ_FEATS, add_adjusted_cols, team_adjusted
from elo import ELO_FEATS, add_elo_cols
from scheme_features import team_scheme, add_scheme_cols, SCHEME_FEATS, DEF_FEATS, WEATHER_FEATS

P = dict(max_depth=3, n_estimators=150, learning_rate=0.05, subsample=0.8,
         colsample_bytree=0.8, reg_lambda=2.0)
RAW_EDGES = ["off_epa_diff", "def_epa_diff", "net_epa_edge_home",
             "pass_epa_edge_home", "rush_epa_edge_home"]
# the ladder is reconstructed from the RAW edges so the historical rows stay comparable to
# what was published before opponent adjustment; only the shipped row uses the new ratings
OLD = RAW_EDGES + ["points_diff_rating", "div_game", "rest_diff"] + rp.CTX_FEATS
MID = OLD + SCHEME_FEATS
NEW = MID + DEF_FEATS
TRACK = NEW + ELO_FEATS         # the ladder step before opponent adjustment
SHIPPED = rp.FEATS              # what actually runs: adjusted edges + everything above
TIERS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def walk(d, feats, tests, keep=False):
    ps, ms, ys, mk, sp, am, sn, frames = [], [], [], [], [], [], [], []
    for s in tests:
        tr, te = d[d.season < s], d[d.season == s]
        if len(tr) < 300 or not len(te):
            continue
        # same 8-seed ensemble the pipeline ships, so the audit describes the model that
        # actually runs rather than one lucky draw of it
        pp, mm = [], []
        for sd in range(rp.N_SEEDS):
            c = xgb.XGBClassifier(**P, eval_metric="logloss", random_state=sd).fit(tr[feats], tr.home_win)
            r = xgb.XGBRegressor(**P, random_state=sd).fit(tr[feats], tr.home_margin)
            pp.append(c.predict_proba(te[feats])[:, 1]); mm.append(r.predict(te[feats]))
        ps.append(np.mean(pp, axis=0)); ms.append(np.mean(mm, axis=0))
        ys.append(te.home_win.values); mk.append(te.market_home_wp.values)
        sp.append(te.spread_line.values); am.append(te.home_margin.values)
        sn.append(te.season.values)
        frames.append(te)
    j = np.concatenate
    out = (j(ps), j(ms), j(ys), j(mk), j(sp), j(am), j(sn))
    return out + (pd.concat(frames),) if keep else out


def bins(vals, y, edges, labels, other=None):
    """Shared binning for the calibration and confidence tables."""
    rows = []
    for i, lab in enumerate(labels):
        sel = (vals >= edges[i]) & (vals < edges[i + 1])
        if sel.sum() < 10:
            continue
        row = {"bin": lab, "n": int(sel.sum())}
        row["predicted"] = float(vals[sel].mean())
        row["actual"] = float(y[sel].mean())
        if other is not None:
            row["market"] = float(other[sel].mean())
        rows.append(row)
    return rows


def main():
    sched, team, plyr, inj, snap, rost, depth, cur = rp.load_all("data", None)
    reg = sched[(sched.season == cur) & (sched.game_type == "REG")]
    un = reg[reg.home_score.isna()]
    tw = int(un.week.min()) if len(un) else int(reg.week.max())

    ratings, _ = rp.team_ratings(team, sched, cur, tw)
    pw = rp.player_form(plyr, sched, ratings)
    ctx = rp.context_features(sched, pw, inj, snap, rost, depth, cur)
    df = rp.build_games(sched, ratings, ctx)
    scheme, _ = team_scheme(team, sched, cur, tw)
    df = add_scheme_cols(df, scheme)
    df = add_adjusted_cols(df, team_adjusted(team, sched, cur, tw))
    df = add_elo_cols(df)

    allf = sorted(set(SHIPPED + TRACK + NEW + MID + OLD + WEATHER_FEATS))
    d = df.dropna(subset=["home_win", "home_margin"] + allf)
    tests = list(range(2019, int(cur)))

    p, m, y, mk, sp, am, sn, te = walk(d, SHIPPED, tests, keep=True)
    pd_, md_, *_ = walk(d, NEW, tests)          # the set before Elo was added
    pt_, mt_, *_ = walk(d, TRACK, tests)        # Elo added, ratings still unadjusted
    pb, mb, *_ = walk(d, OLD, tests)
    pw_, mw_, *_ = walk(d, OLD + WEATHER_FEATS, tests)
    pm_, mm_, *_ = walk(d, MID, tests)
    p_elo = te.p_elo.values                     # rating system alone, no fitting at all
    wk = te.week.values

    n = len(y)
    blend = np.where(~np.isnan(mk), rp.BLEND_W * p + (1 - rp.BLEND_W) * mk, p)
    hit = (p > 0.5).astype(int) == y
    bhit = (blend > 0.5).astype(int) == y
    mhit = (mk > 0.5).astype(int) == y
    cov = np.where(am > sp, 1, np.where(am < sp, 0, -1))
    ok = cov >= 0
    ats_hit = (m > sp).astype(int)[ok] == cov[ok]

    def brier(x): return float(np.mean((x - y) ** 2))
    def ll(x): return float(-np.mean(y * np.log(np.clip(x, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - x, 1e-9, 1))))

    lo, hi = stats.binomtest(int(ats_hit.sum()), len(ats_hit), 0.5).proportion_ci(0.95)

    out = {
        "overall": {
            "n_games": int(n), "model_su": float(hit.mean()), "market_su": float(mhit.mean()),
            "blend_su": float(bhit.mean()), "model_wins": int(hit.sum()),
            "blend_wins": int(bhit.sum()), "market_wins": int(mhit.sum()),
            "home_win_rate": float(y.mean()), "always_home_su": float(y.mean()),
        },
        "baselines": [
            {"label": "Always pick the home team", "value": float(y.mean()), "kind": "baseline"},
            {"label": "This model, on its own", "value": float(hit.mean()), "kind": "model"},
            {"label": "Model blended with the line", "value": float(bhit.mean()), "kind": "blend"},
            {"label": "Always pick the Vegas favorite", "value": float(mhit.mean()), "kind": "market"},
        ],
        "ats": {
            "n": int(len(ats_hit)), "rate": float(ats_hit.mean()), "pushes": int((~ok).sum()),
            "breakeven": 0.5238, "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(stats.binomtest(int(ats_hit.sum()), len(ats_hit), 0.5).pvalue),
            "by_edge": [{"thr": t, "n": int(((np.abs(m - sp) >= t) & ok).sum()),
                         "rate": float((((m > sp).astype(int) == cov)[(np.abs(m - sp) >= t) & ok]).mean())}
                        for t in [1, 2, 3, 5]],
            "by_season": [{"season": int(s), "n": int((ok & (sn == s)).sum()),
                           "rate": float((((m > sp).astype(int) == cov)[ok & (sn == s)]).mean())}
                          for s in tests],
        },
        "margin": {"mae": float(np.abs(m - am).mean()),
                   "market_mae": float(np.abs(sp - am).mean())},
        "pick_tiers": [],
        "ab_test": [
            {"set": "base", "acc": 0.6097, "logloss": 0.6606, "brier": 0.2334, "mae": 10.642, "ats": 0.5044},
            {"set": "+ QB", "acc": 0.6178, "logloss": 0.657, "brier": 0.2319, "mae": 10.542, "ats": 0.5221},
            {"set": "+ QB + injuries (old)", "acc": 0.6124, "logloss": 0.6564, "brier": 0.2316,
             "mae": 10.541, "ats": 0.5039},
            {"set": "+ QB + injuries (fixed)", "acc": float(np.round(((pb > 0.5).astype(int) == y).mean(), 4)),
             "logloss": round(ll(pb), 4), "brier": round(brier(pb), 4),
             "mae": round(float(np.abs(mb - am).mean()), 3),
             "ats": round(float((((mb > sp).astype(int) == cov)[ok]).mean()), 4)},
            {"set": "+ weather", "acc": float(np.round(((pw_ > 0.5).astype(int) == y).mean(), 4)),
             "logloss": round(ll(pw_), 4), "brier": round(brier(pw_), 4),
             "mae": round(float(np.abs(mw_ - am).mean()), 3),
             "ats": round(float((((mw_ > sp).astype(int) == cov)[ok]).mean()), 4)},
            {"set": "+ offensive scheme", "acc": float(np.round(((pm_ > 0.5).astype(int) == y).mean(), 4)),
             "logloss": round(ll(pm_), 4), "brier": round(brier(pm_), 4),
             "mae": round(float(np.abs(mm_ - am).mean()), 3),
             "ats": round(float((((mm_ > sp).astype(int) == cov)[ok]).mean()), 4)},
            {"set": "+ defensive scheme", "acc": float(np.round(((pd_ > 0.5).astype(int) == y).mean(), 4)),
             "logloss": round(ll(pd_), 4), "brier": round(brier(pd_), 4),
             "mae": round(float(np.abs(md_ - am).mean()), 3),
             "ats": round(float((((md_ > sp).astype(int) == cov)[ok]).mean()), 4)},
            {"set": "+ track record", "acc": float(np.round(((pt_ > 0.5).astype(int) == y).mean(), 4)),
             "logloss": round(ll(pt_), 4), "brier": round(brier(pt_), 4),
             "mae": round(float(np.abs(mt_ - am).mean()), 3),
             "ats": round(float((((mt_ > sp).astype(int) == cov)[ok]).mean()), 4)},
            {"set": "+ opponent-adjusted ratings (shipped)", "acc": round(float(hit.mean()), 4),
             "logloss": round(ll(p), 4), "brier": round(brier(p), 4),
             "mae": round(float(np.abs(m - am).mean()), 3), "ats": round(float(ats_hit.mean()), 4)},
            # measured by test_pbp.py on these exact folds and this exact protocol; kept
            # here so a regeneration cannot quietly drop a negative result
            {"set": "+ play-by-play detail (rejected)", "acc": 0.6323, "logloss": 0.6472,
             "brier": 0.2266, "mae": 10.29, "ats": 0.5215},
            {"set": "track record ALONE, no features",
             "acc": float(np.round(((p_elo > 0.5).astype(int) == y).mean(), 4)),
             "logloss": round(ll(p_elo), 4), "brier": round(brier(p_elo), 4),
             "mae": None, "ats": None},
        ],
    }

    # ---- per-season, per-week, calibration, confidence ----
    out["by_season"] = [{"season": int(s), "n": int((sn == s).sum()),
                         "model": float(hit[sn == s].mean()),
                         "market": float(mhit[sn == s].mean()),
                         "blend": float(bhit[sn == s].mean())} for s in tests]

    out["by_week"] = []
    for lab, lo_w, hi_w in [("Weeks 1-4", 1, 4), ("Weeks 5-9", 5, 9),
                            ("Weeks 10-13", 10, 13), ("Weeks 14-18", 14, 18)]:
        s = (wk >= lo_w) & (wk <= hi_w)
        if s.sum() < 10:
            continue
        out["by_week"].append({"bin": lab, "n": int(s.sum()),
                               "model": float(hit[s].mean()), "market": float(mhit[s].mean())})

    out["calibration"] = bins(p, y, [0, .35, .45, .55, .65, .75, 1.01],
                              ["<35%", "35-45%", "45-55%", "55-65%", "65-75%", "75%+"])

    # confidence bands read on the favorite's side, whichever team that is
    conf = np.maximum(p, 1 - p)
    cy = (p > 0.5).astype(int) == y
    cm = (mk > 0.5).astype(int) == y
    out["confidence"] = []
    for lab, lo_c, hi_c in [("coin flip (50-55%)", .5, .55), ("lean (55-60%)", .55, .60),
                            ("moderate (60-65%)", .60, .65), ("confident (65-75%)", .65, .75),
                            ("strong (75%+)", .75, 1.01)]:
        s = (conf >= lo_c) & (conf < hi_c)
        if s.sum() < 10:
            continue
        out["confidence"].append({"bin": lab, "n": int(s.sum()),
                                  "model": float(cy[s].mean()), "market": float(cm[s].mean())})

    # ---- where the model and the market part ways ----
    same = (p > 0.5) == (mk > 0.5)
    dis = ~same
    out["agreement"] = {"n": int(same.sum()), "rate": round(float(hit[same].mean()), 4)}
    out["disagreements"] = {
        "n_disagreements": int(dis.sum()),
        "share_of_games": float(dis.mean()),
        "model_right": float(hit[dis].mean()),
        "market_right": float(mhit[dis].mean()),
    }
    mo, ko = int((hit & ~mhit).sum()), int((~hit & mhit).sum())
    chi2 = (abs(mo - ko) - 1) ** 2 / (mo + ko) if mo + ko else 0.0
    out["mcnemar"] = {"model_only_right": mo, "market_only_right": ko,
                      "chi2": round(float(chi2), 1),
                      "p": float(stats.chi2.sf(chi2, 1))}

    # ---- predicted vs actual season win totals, from the same walk-forward margins ----
    tr_ = pd.DataFrame({"season": sn, "home": te.home_team.values, "away": te.away_team.values,
                        "p": p, "y": y})
    exp_, act_ = {}, {}
    for _, r in tr_.iterrows():
        exp_[(r.season, r.home)] = exp_.get((r.season, r.home), 0) + r.p
        exp_[(r.season, r.away)] = exp_.get((r.season, r.away), 0) + (1 - r.p)
        act_[(r.season, r.home)] = act_.get((r.season, r.home), 0) + r.y
        act_[(r.season, r.away)] = act_.get((r.season, r.away), 0) + (1 - r.y)
    scat = [{"season": int(s), "team": t, "exp": round(float(v), 2), "act": int(act_[(s, t)])}
            for (s, t), v in exp_.items()]
    err = np.array([abs(x["exp"] - x["act"]) for x in scat])
    ev = np.array([x["exp"] for x in scat]); av = np.array([x["act"] for x in scat])
    out["team_records"] = {
        "team_seasons": len(scat), "wins_mae": float(err.mean()),
        "corr": float(np.corrcoef(ev, av)[0, 1]),
        "within_1_win": float((err <= 1).mean()), "within_2_wins": float((err <= 2).mean()),
    }
    out["team_record_scatter"] = sorted(scat, key=lambda x: (x["season"], x["team"]))
    out["biggest_record_misses"] = [
        {"season": x["season"], "team": x["team"], "exp": round(x["exp"], 1), "act": x["act"]}
        for x in sorted(scat, key=lambda x: -abs(x["exp"] - x["act"]))[:8]]

    # McNemar: scheme vs the previous feature set, on the same games
    bh = (pm_ > 0.5).astype(int) == y
    s_only, b_only = int((hit & ~bh).sum()), int((~hit & bh).sum())
    out["mcnemar_scheme"] = {
        "scheme_only_right": s_only, "base_only_right": b_only,
        "changed": s_only + b_only,
        "p": round(float(stats.binomtest(s_only, s_only + b_only, 0.5).pvalue), 3),
    }

    ns = len(tests)
    for t in TIERS:
        sel = np.maximum(blend, 1 - blend) >= t
        if sel.sum() < 20:
            continue
        picked_home = blend > 0.5
        right = (picked_home.astype(int) == y)[sel]
        out["pick_tiers"].append({
            "thr": t, "n": int(sel.sum()), "per_season": round(float(sel.sum() / ns), 1),
            "hit": float(right.mean()),
            "claimed": float(np.maximum(blend, 1 - blend)[sel].mean()),
        })

    # ---------------------------------------------------------------- backtest.json
    # The headline audit figures, written where the daily job can read them.
    #
    # These used to live as a hardcoded dict in run_pipeline.py with a comment saying to
    # regenerate it by hand when the model changed. It was not regenerated when the model
    # changed, and because merge_payload.py treats `backtest` as a fresh key, every hourly
    # run quietly overwrote the correct audited numbers with the stale constant. The site
    # spent 2026-09-11 telling the reader 64.3% in prose and 61.8% in its own data block,
    # twelve hours after the correct numbers were published.
    #
    # A number that is measured in one file and retyped in another will drift. This is the
    # only place these are computed, so this is the only place they are written.
    bt = {
        "n_games": int(n),
        "model_su": round(float(hit.mean()), 4),
        "market_su": round(float(mhit.mean()), 4),
        "blend_su": round(float(bhit.mean()), 4),
        "always_home": round(float(y.mean()), 4),
        "margin_mae": round(float(np.abs(m - am).mean()), 2),
        "market_margin_mae": round(float(np.abs(sp - am).mean()), 2),
        "ats": round(float(ats_hit.mean()), 4),
        "brier_model": round(brier(p), 4),
        # the market has no price on a handful of games; score it only where it has one,
        # against the matching outcomes rather than against all of them
        "brier_market": round(float(np.mean((mk[~np.isnan(mk)] - y[~np.isnan(mk)]) ** 2)), 4),
        "note": "opponent-adjusted ratings + QB + injuries + scheme + Elo",
        "blend_w": rp.BLEND_W,
        "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }
    json.dump(bt, open("data/backtest.json", "w"), indent=1)
    print("\nwrote data/backtest.json (the daily job reads this; nothing is retyped)")
    for k, v in bt.items():
        print(f"  {k:<20} {v}")

    json.dump(out, open("acc_patch.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "pick_tiers"}, indent=1))
    print("\npick tiers")
    for t in out["pick_tiers"]:
        print(f"  {t['thr']:.0%}+  n={t['n']:<5} {t['per_season']:>5}/season  "
              f"hit {t['hit']:.1%}  claimed {t['claimed']:.1%}")


if __name__ == "__main__":
    main()
