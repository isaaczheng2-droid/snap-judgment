#!/usr/bin/env python3
"""
Do the coaching / scheme factors from the "best player doesn't play" argument buy anything
at the game level, measured the same way every other feature was?

Factors (all point-in-time: an exponentially-weighted mean of a team's PRIOR games, shrunk
toward last season and the league while the sample is small, same recipe as scheme_features):

  pfr (2019+)   press_made   pressures generated / dropbacks faced         (PFR charting)
                press_allow  pressures allowed / dropbacks                 (PFR charting)
                blitz_made   blitzes sent / dropbacks faced
  pbp (2019+)   edp_pass     early-down pass rate, Q1-Q3, downs 1-2, |lead| <= 10
                go4          4th-and-<=2 go rate, midfield band, Q1-Q3, |lead| <= 14 (coach aggressiveness)
  ftn (2022+)   pa_rate      play-action share of dropbacks
                motion       pre-snap motion share of pass+run plays
                sim_rate     dropbacks faced with <=4 rushers AND >=1 blitzer (a creeper / sim pressure proxy)
  sched         coach_new    head coach differs from the team's last game of the previous season

Matchup features are home-minus-away differences plus one real pass-rush-vs-protection edge
built from measured pressure instead of sacks. Walk-forward 2019-2025 on identical folds,
8-seed ensemble, paired against the shipped feature set: accuracy, log loss, Brier, McNemar
on the picks and a paired bootstrap on per-game log loss.
"""
import glob
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

import run_pipeline as rp
from adjusted_ratings import add_adjusted_cols, team_adjusted
from elo import add_elo_cols
from scheme_features import team_scheme, add_scheme_cols

P = dict(max_depth=3, n_estimators=150, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0)
EW_SPAN, K = 8, 4.0
PFR = ["press_made", "press_allow", "blitz_made"]
PBP = ["edp_pass", "go4"]
FTN = ["pa_rate", "motion", "sim_rate"]
METRICS = PFR + PBP + FTN


def per_game_pbp(seasons):
    cols = ["game_id", "play_id", "posteam", "defteam", "down", "play_type", "qtr", "score_differential",
            "ydstogo", "yardline_100", "season", "week", "season_type"]
    rows = []
    for s in seasons:
        d = pd.read_parquet(f"data/pbp/play_by_play_{s}.parquet", columns=cols)
        d = d[(d.season_type == "REG") & d.posteam.notna()]
        try:
            f = pd.read_parquet(f"data/ftn/ftn_{s}.parquet",
                                columns=["nflverse_game_id", "nflverse_play_id", "is_play_action", "is_motion", "n_blitzers", "n_pass_rushers"])
            d = d.merge(f, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"], how="left")
        except FileNotFoundError:
            for c in ["is_play_action", "is_motion", "n_blitzers", "n_pass_rushers"]:
                d[c] = np.nan
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    live = d[d.play_type.isin(["pass", "run"])]
    ed = live[(live.down <= 2) & (live.qtr <= 3) & (live.score_differential.abs() <= 10)]
    edp = ed.groupby(["game_id", "posteam"]).apply(lambda g: (g.play_type == "pass").mean()).rename("edp_pass")
    fd = d[(d.down == 4) & (d.ydstogo <= 2) & d.yardline_100.between(35, 65) & (d.qtr <= 3) & (d.score_differential.abs() <= 14)
           & d.play_type.isin(["pass", "run", "punt", "field_goal"])]
    go4 = fd.groupby(["game_id", "posteam"]).apply(lambda g: g.play_type.isin(["pass", "run"]).mean()).rename("go4")
    go4n = fd.groupby(["game_id", "posteam"]).size().rename("go4_n")
    pa = live[live.play_type == "pass"].groupby(["game_id", "posteam"]).is_play_action.mean().rename("pa_rate")
    mo = live.groupby(["game_id", "posteam"]).is_motion.mean().rename("motion")
    ps = live[live.play_type == "pass"].copy()
    ps["sim"] = ((ps.n_pass_rushers <= 4) & (ps.n_blitzers >= 1)).astype(float).where(ps.n_pass_rushers.notna())
    sim = ps.groupby(["game_id", "defteam"]).sim.mean().rename("sim_rate")
    sim.index = sim.index.set_names(["game_id", "posteam"])
    out = pd.concat([edp, go4, go4n, pa, mo, sim], axis=1).reset_index().rename(columns={"posteam": "team"})
    return out


def per_game_pfr(seasons, team):
    rows = [pd.read_parquet(f"data/pfr/pass_{s}.parquet") for s in seasons]
    p = pd.concat(rows, ignore_index=True)
    p = p[p.game_type == "REG"]
    p["team"] = p.team.replace({"LAR": "LA", "WSH": "WAS"})
    p["opponent"] = p.opponent.replace({"LAR": "LA", "WSH": "WAS"})
    g = p.groupby(["game_id", "team", "opponent"])[["times_pressured", "times_blitzed"]].sum().reset_index()
    tw = team[team.season_type == "REG"][["game_id", "team", "attempts", "sacks_suffered"]].copy()
    tw["dropbacks"] = tw.attempts.fillna(0) + tw.sacks_suffered.fillna(0)
    g = g.merge(tw[["game_id", "team", "dropbacks"]], on=["game_id", "team"], how="left")
    g["press_allow"] = g.times_pressured / g.dropbacks.replace(0, np.nan)
    # what the defence did is on the opponent's offensive row
    d = g.rename(columns={"team": "opponent", "opponent": "team", "times_pressured": "made", "times_blitzed": "blz", "dropbacks": "db_faced"})
    d["press_made"] = d.made / d.db_faced.replace(0, np.nan)
    d["blitz_made"] = d.blz / d.db_faced.replace(0, np.nan)
    out = g[["game_id", "team", "press_allow"]].merge(d[["game_id", "team", "press_made", "blitz_made"]], on=["game_id", "team"], how="outer")
    return out


def point_in_time(m, metrics, sched):
    """EW mean of prior games, shrunk to last season then the league. Mirrors scheme_features."""
    gd = sched[["game_id", "season", "week", "gameday"]].drop_duplicates("game_id")
    m = m.merge(gd, on="game_id", how="inner")
    m = m.sort_values(["team", "season", "gameday"]).reset_index(drop=True)
    m["gp_prior"] = m.groupby(["team", "season"]).cumcount()
    lg = m.groupby("season")[metrics].transform("mean")          # league mean of that season (rates drift)
    prev = m.groupby(["team", "season"])[metrics].mean().reset_index()
    prev["season"] += 1
    prev = prev.rename(columns={x: f"prev_{x}" for x in metrics})
    m = m.merge(prev, on=["team", "season"], how="left")
    w = m.gp_prior / (m.gp_prior + K)
    for x in metrics:
        sh = m.groupby(["team", "season"])[x].shift(1)
        ew = sh.groupby([m.team, m.season]).transform(lambda s: s.ewm(span=EW_SPAN, min_periods=1).mean())
        pv = m[f"prev_{x}"].fillna(lg[x])
        prior = 0.5 * lg[x] + 0.5 * pv
        ew = ew.fillna(prior)
        m[f"c_{x}"] = w * ew + (1 - w) * prior
    return m[["game_id", "team", "season"] + [f"c_{x}" for x in metrics]]


def coach_change(sched):
    s = sched[sched.game_type == "REG"].sort_values(["season", "week"])
    rows = []
    for side in ["home", "away"]:
        rows.append(s[["game_id", "season", "week", f"{side}_team", f"{side}_coach"]].rename(columns={f"{side}_team": "team", f"{side}_coach": "coach"}))
    t = pd.concat(rows).sort_values(["team", "season", "week"])
    last = t.groupby(["team", "season"]).coach.last().reset_index()
    last["season"] += 1
    last = last.rename(columns={"coach": "prev_coach"})
    t = t.merge(last, on=["team", "season"], how="left")
    t["coach_new"] = ((t.coach != t.prev_coach) & t.prev_coach.notna()).astype(float)
    return t[["game_id", "team", "coach_new"]]


def add_cols(df, pit, cc):
    for side in ["home", "away"]:
        p = pit.drop(columns="season").rename(columns={"team": f"{side}_team", **{f"c_{x}": f"{side}_{x}" for x in METRICS}})
        df = df.merge(p, on=["game_id", f"{side}_team"], how="left")
        c = cc.rename(columns={"team": f"{side}_team", "coach_new": f"{side}_coach_new"})
        df = df.merge(c, on=["game_id", f"{side}_team"], how="left")
    # measured pass rush vs protection, both directions
    df["press_edge_real"] = (df.home_press_made - df.away_press_allow) - (df.away_press_made - df.home_press_allow)
    for x in METRICS:
        df[f"{x}_diff"] = df[f"home_{x}"] - df[f"away_{x}"]
    df["coach_new_diff"] = df.home_coach_new - df.away_coach_new
    return df


def walk(d, feats, tests):
    ps, ys, sn = [], [], []
    for s in tests:
        tr, te = d[d.season < s], d[d.season == s]
        if len(tr) < 300 or not len(te):
            continue
        pp = []
        for sd in range(rp.N_SEEDS):
            c = xgb.XGBClassifier(**P, eval_metric="logloss", random_state=sd).fit(tr[feats], tr.home_win)
            pp.append(c.predict_proba(te[feats])[:, 1])
        ps.append(np.mean(pp, axis=0)); ys.append(te.home_win.values); sn.append(te.season.values)
    return np.concatenate(ps), np.concatenate(ys), np.concatenate(sn)


def score(p, y):
    pc = np.clip(p, 1e-9, 1 - 1e-9)
    return {"acc": float(((p > 0.5).astype(int) == y).mean()),
            "logloss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
            "brier": float(np.mean((p - y) ** 2))}


def compare(p0, p1, y, rng, B=2000):
    pc0, pc1 = np.clip(p0, 1e-9, 1 - 1e-9), np.clip(p1, 1e-9, 1 - 1e-9)
    l0 = -(y * np.log(pc0) + (1 - y) * np.log(1 - pc0)); l1 = -(y * np.log(pc1) + (1 - y) * np.log(1 - pc1))
    diff = l0 - l1                                   # positive = candidate better
    n = len(y)
    bs = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(B)])
    h0, h1 = (p0 > 0.5).astype(int) == y, (p1 > 0.5).astype(int) == y
    b, c = int((h0 & ~h1).sum()), int((~h0 & h1).sum())
    mcn = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"logloss_gain": float(diff.mean()), "gain_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p_bootstrap": float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))),
            "picks_only_base": b, "picks_only_cand": c, "mcnemar_p": float(mcn)}


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

    seasons = list(range(2018, int(cur) + 1))
    pfr_seasons = [s for s in seasons if glob.glob(f"data/pfr/pass_{s}.parquet")]
    pg = per_game_pbp(seasons).merge(per_game_pfr(pfr_seasons, team), on=["game_id", "team"], how="outer")
    rp.log(f"per-game metric rows {len(pg)}; coverage " + ", ".join(f"{x} {pg[x].notna().mean():.0%}" for x in METRICS))
    pit = point_in_time(pg, METRICS, sched)
    df = add_cols(df, pit, coach_change(sched))

    base = rp.FEATS
    SETS = {
        "+ pressure (PFR): made, allowed, real rush-vs-protection edge": base + ["press_made_diff", "press_allow_diff", "press_edge_real"],
        "+ blitz rate": base + ["blitz_made_diff"],
        "+ early-down pass rate": base + ["edp_pass_diff"],
        "+ 4th-down aggressiveness": base + ["go4_diff"],
        "+ new head coach": base + ["home_coach_new", "away_coach_new"],
        "+ pressure + blitz + early-down pass + 4th down + coach": base + ["press_made_diff", "press_allow_diff", "press_edge_real", "blitz_made_diff", "edp_pass_diff", "go4_diff", "home_coach_new", "away_coach_new"],
    }
    FTN_SETS = {
        "+ play action (2022+)": base + ["pa_rate_diff"],
        "+ motion (2022+)": base + ["motion_diff"],
        "+ sim pressure proxy (2022+)": base + ["sim_rate_diff"],
        "+ all three FTN (2022+)": base + ["pa_rate_diff", "motion_diff", "sim_rate_diff"],
    }
    rng = np.random.default_rng(7)
    results = {}

    # candidate columns are left as NaN where a source is missing (xgboost treats NaN as
    # missing); the game set is the same 1,855 the published audit scores
    d = df.dropna(subset=["home_win"] + base)
    tests = list(range(2019, int(cur)))
    rp.log(f"main folds: {len(d)} games, tests {tests}")
    p0, y, sn = walk(d, base, tests)
    results["base (shipped)"] = score(p0, y) | {"n": int(len(y))}
    rp.log(f"  base: {results['base (shipped)']}")
    for name, feats in SETS.items():
        p1, y1, _ = walk(d, feats, tests)
        assert len(y1) == len(y)
        results[name] = score(p1, y) | compare(p0, p1, y, rng) | {"n": int(len(y))}
        r = results[name]
        rp.log(f"  {name}: acc {r['acc']:.4f} ({r['acc'] - results['base (shipped)']['acc']:+.4f}) logloss {r['logloss']:.4f} gain {r['logloss_gain']:+.5f} CI {r['gain_ci95'][0]:+.5f}..{r['gain_ci95'][1]:+.5f} p {r['p_bootstrap']:.3f} mcnemar {r['mcnemar_p']:.3f}")

    # FTN-era features: train from 2019 with missing values before 2022 (xgboost handles NaN),
    # score only the folds where every team has the feature: 2023+ (2022 is the first FTN season,
    # so week-1 2022 rows fall back to the league prior).
    need_f = sorted(set(sum(FTN_SETS.values(), [])) - set(base))
    df_f = df.copy()
    tests_f = [s for s in range(2023, int(cur))]
    d_f = df_f.dropna(subset=["home_win"] + base)
    rp.log(f"FTN folds: {len(d_f[d_f.season.isin(tests_f)])} test games, tests {tests_f}")
    p0f, yf, snf = walk(d_f, base, tests_f)
    results["base (shipped), 2023+ folds"] = score(p0f, yf) | {"n": int(len(yf))}
    rp.log(f"  base 2023+: {results['base (shipped), 2023+ folds']}")
    for name, feats in FTN_SETS.items():
        p1, y1, _ = walk(d_f, feats, tests_f)
        assert len(y1) == len(yf)
        results[name] = score(p1, yf) | compare(p0f, p1, yf, rng) | {"n": int(len(yf))}
        r = results[name]
        rp.log(f"  {name}: acc {r['acc']:.4f} ({r['acc'] - results['base (shipped), 2023+ folds']['acc']:+.4f}) logloss {r['logloss']:.4f} gain {r['logloss_gain']:+.5f} CI {r['gain_ci95'][0]:+.5f}..{r['gain_ci95'][1]:+.5f} p {r['p_bootstrap']:.3f} mcnemar {r['mcnemar_p']:.3f}")

    # how stable are the factors themselves? year-to-year correlation of a team's mean says
    # whether it is an identity (worth modelling) or noise
    pg2 = pg.merge(sched[["game_id", "season"]].drop_duplicates(), on="game_id")
    ty = pg2.groupby(["team", "season"])[METRICS].mean().reset_index()
    nxt = ty.copy(); nxt["season"] -= 1
    j = ty.merge(nxt, on=["team", "season"], suffixes=("", "_next"))
    stability = {x: float(j[[x, f"{x}_next"]].dropna().corr().iloc[0, 1]) for x in METRICS}
    results["_stability_year_to_year_r"] = stability
    rp.log("year-to-year r: " + ", ".join(f"{k} {v:.2f}" for k, v in stability.items()))
    # and how much of it the market already knows: correlation of each diff with the spread
    corr = {x: float(d[[f"{x}_diff", "spread_line"]].dropna().corr().iloc[0, 1]) for x in PFR + PBP}
    results["_corr_with_spread"] = corr
    rp.log("corr with spread: " + ", ".join(f"{k} {v:+.2f}" for k, v in corr.items()))

    json.dump(results, open("data/coaching_test.json", "w"), indent=1)
    rp.log("wrote data/coaching_test.json")


if __name__ == "__main__":
    main()
