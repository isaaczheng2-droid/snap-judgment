#!/usr/bin/env python3
"""
Snap Judgment — full daily pipeline. Self-contained: downloads everything it needs,
rebuilds ratings and projections, predicts the next unplayed week, scores the model
live against completed games of the current season, and writes payload.json.

Run:  python3 run_pipeline.py [--outdir .]
Deps: pandas numpy pyarrow scikit-learn xgboost  (pip install --break-system-packages)

Design notes that matter if you edit this:
  * nfl_data_py's own URL patterns are stale; the release-asset paths below are current.
  * habitatring.com (nfl_data_py's schedule source) is blocked here — schedules come
    from raw.githubusercontent.com instead.
  * Every feature is point-in-time: ratings for a game use only games before it.
"""
import argparse, io, json, os, subprocess, sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REL = "https://github.com/nflverse/nflverse-data/releases/download"
SCHEDULES = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
FIRST_SEASON = 2016
EW_SPAN, K = 6, 3.0          # in-season smoothing span; shrinkage speed toward the preseason prior

METRICS = ["g_off_epa_pp", "g_def_epa_pp_allowed", "g_off_pass_epa_pp", "g_off_rush_epa_pp",
           "g_def_pass_epa_pp_allowed", "g_def_rush_epa_pp_allowed", "points_scored", "points_allowed"]
BASE_FEATS = ["off_epa_diff", "def_epa_diff", "net_epa_edge_home", "pass_epa_edge_home",
              "rush_epa_edge_home", "points_diff_rating", "div_game", "rest_diff"]
CTX_FEATS = ["qb_epa_diff", "qb_rush_diff", "qb_change_diff", "inj_off_diff", "inj_def_diff"]
FEATS = BASE_FEATS + CTX_FEATS

# Walk-forward backtest results, 2019-2025. These describe the model design, not today's
# data, so they are constants; regenerate them if the feature set or hyperparameters change.
BACKTEST = {
    "n_games": 1855, "model_su": 0.6124, "market_su": 0.6663, "blend_su": 0.6544,
    "always_home": 0.5288, "margin_mae": 10.54, "market_margin_mae": 9.81,
    "ats": 0.5039, "brier_model": 0.2316, "brier_market": 0.2104,
    "note": "base + QB + injury feature set",
}


def log(*a):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}]", *a, flush=True)


def fetch(url, path):
    """curl is used rather than pandas' URL reader: GitHub release assets 302 to a
    signed host that urllib intermittently 502s on."""
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return True
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    r = subprocess.run(["curl", "-sSL", "--max-time", "90", "-o", path, url],
                       capture_output=True)
    ok = r.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 1024
    if not ok and os.path.exists(path):
        os.remove(path)
    return ok


def read_many(pattern, seasons, d):
    frames = []
    for y in seasons:
        p = f"{d}/{pattern.format(y=y)}"
        if fetch(f"{REL}/{pattern.format(y=y)}", p):
            try:
                frames.append(pd.read_parquet(p))
            except Exception as e:
                log(f"  skip {p}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- download
def load_all(d, seasons):
    log("downloading schedules")
    sched_path = f"{d}/games.csv"
    if os.path.exists(sched_path):
        os.remove(sched_path)                     # always refresh: lines, weather, QBs move daily
    subprocess.run(["curl", "-sSL", "--max-time", "90", "-o", sched_path, SCHEDULES], check=True)
    sched = pd.read_csv(sched_path, low_memory=False)
    sched["gameday"] = pd.to_datetime(sched["gameday"])

    cur = int(sched[sched.game_type == "REG"].season.max())
    log(f"current season in schedule: {cur}")

    log("downloading stats / rosters / injuries / snaps")
    # current-season files change during the week, so drop them before re-fetching
    for stale in [f"{d}/stats_team/stats_team_week_{cur}.parquet",
                  f"{d}/stats_player/stats_player_week_{cur}.parquet",
                  f"{d}/injuries/injuries_{cur}.parquet",
                  f"{d}/snaps/snap_counts_{cur}.parquet",
                  f"{d}/depth/depth_charts_{cur}.parquet",
                  f"{d}/rosters/roster_{cur}.parquet"]:
        if os.path.exists(stale):
            os.remove(stale)

    yrs = list(range(FIRST_SEASON, cur + 1))
    team = read_many("stats_team/stats_team_week_{y}.parquet", yrs, d)
    plyr = read_many("stats_player/stats_player_week_{y}.parquet", yrs, d)
    inj = read_many("injuries/injuries_{y}.parquet", yrs, d)
    snap = read_many("snaps/snap_counts_{y}.parquet", yrs, d)
    rost = read_many("rosters/roster_{y}.parquet", yrs, d)

    depth = pd.DataFrame()
    dp = f"{d}/depth/depth_charts_{cur}.parquet"
    if fetch(f"{REL}/depth_charts/depth_charts_{cur}.parquet", dp):
        depth = pd.read_parquet(dp)

    log(f"team {team.shape} player {plyr.shape} inj {inj.shape} snap {snap.shape} depth {depth.shape}")
    return sched, team, plyr, inj, snap, rost, depth, cur


# --------------------------------------------------------------------------- team ratings
def team_ratings(team, sched, cur, target_week):
    tw = team[team.season_type == "REG"].copy()
    tw["off_plays"] = tw.attempts.fillna(0) + tw.sacks_suffered.fillna(0) + tw.carries.fillna(0)
    tw["off_epa_total"] = tw.passing_epa.fillna(0) + tw.rushing_epa.fillna(0)
    tw["pass_plays"] = tw.attempts.fillna(0) + tw.sacks_suffered.fillna(0)
    tw["rush_plays"] = tw.carries.fillna(0)
    keep = tw[["season", "week", "team", "opponent_team", "game_id", "off_plays", "off_epa_total",
               "pass_plays", "passing_epa", "rush_plays", "rushing_epa"]].rename(
        columns={"passing_epa": "pass_epa_total", "rushing_epa": "rush_epa_total"})

    opp = keep.rename(columns={
        "team": "opponent_team", "opponent_team": "team",
        "off_plays": "def_plays_faced", "off_epa_total": "def_epa_allowed_total",
        "pass_plays": "def_pass_plays_faced", "pass_epa_total": "def_pass_epa_allowed_total",
        "rush_plays": "def_rush_plays_faced", "rush_epa_total": "def_rush_epa_allowed_total"})
    m = keep.merge(opp, on=["season", "week", "game_id", "team", "opponent_team"], how="left")

    h = sched[["game_id", "gameday", "home_team", "away_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "points_scored", "away_score": "points_allowed"})
    a = sched[["game_id", "gameday", "home_team", "away_team", "home_score", "away_score"]].rename(
        columns={"away_team": "team", "away_score": "points_scored", "home_score": "points_allowed"})
    pts = pd.concat([h[["game_id", "team", "points_scored", "points_allowed", "gameday"]],
                     a[["game_id", "team", "points_scored", "points_allowed", "gameday"]]])
    m = m.merge(pts, on=["game_id", "team"], how="left")

    m["g_off_epa_pp"] = m.off_epa_total / m.off_plays.replace(0, np.nan)
    m["g_def_epa_pp_allowed"] = m.def_epa_allowed_total / m.def_plays_faced.replace(0, np.nan)
    m["g_off_pass_epa_pp"] = m.pass_epa_total / m.pass_plays.replace(0, np.nan)
    m["g_off_rush_epa_pp"] = m.rush_epa_total / m.rush_plays.replace(0, np.nan)
    m["g_def_pass_epa_pp_allowed"] = m.def_pass_epa_allowed_total / m.def_pass_plays_faced.replace(0, np.nan)
    m["g_def_rush_epa_pp_allowed"] = m.def_rush_epa_allowed_total / m.def_rush_plays_faced.replace(0, np.nan)

    lg = m[METRICS].mean()
    m = m.sort_values(["team", "season", "gameday"]).reset_index(drop=True)
    m["gp_prior"] = m.groupby(["team", "season"]).cumcount()

    season_final = m.groupby(["team", "season"])[METRICS].mean().reset_index()
    season_final = season_final.rename(columns={x: f"prev_{x}" for x in METRICS})
    season_final["season"] += 1                       # available as next season's prior

    for x in METRICS:
        sh = m.groupby(["team", "season"])[x].shift(1)
        m[f"ewma_{x}"] = sh.groupby([m.team, m.season]).transform(
            lambda s: s.ewm(span=EW_SPAN, min_periods=1).mean())
    m = m.merge(season_final, on=["team", "season"], how="left")
    for x in METRICS:
        m[f"prev_{x}"] = m[f"prev_{x}"].fillna(lg[x])
        m[f"ewma_{x}"] = m[f"ewma_{x}"].fillna(m[f"prev_{x}"])
        m[f"prior_{x}"] = 0.5 * lg[x] + 0.5 * m[f"prev_{x}"]
    w = m.gp_prior / (m.gp_prior + K)
    for x in METRICS:
        m[f"rating_{x}"] = w * m[f"ewma_{x}"] + (1 - w) * m[f"prior_{x}"]

    ratings = m[["team", "season", "week", "game_id"] + [f"rating_{x}" for x in METRICS]].copy()

    # ---- ratings for the upcoming (unplayed) week, which has no stats row yet ----
    played = m[m.season == cur]
    gp = played.groupby("team").size().to_dict()
    upcoming = sched[(sched.season == cur) & (sched.week == target_week) & (sched.game_type == "REG")]
    teams = pd.unique(pd.concat([upcoming.home_team, upcoming.away_team]))

    prev_cur = season_final[season_final.season == cur].set_index("team")
    rows = []
    for t in teams:
        n = gp.get(t, 0)
        row = {"team": t, "season": cur, "week": target_week}
        tp = played[played.team == t].sort_values("gameday")
        for x in METRICS:
            prev = prev_cur[f"prev_{x}"].get(t, lg[x])
            if pd.isna(prev):
                prev = lg[x]
            prior = 0.5 * lg[x] + 0.5 * prev
            ew = tp[x].ewm(span=EW_SPAN, min_periods=1).mean().iloc[-1] if n else prior
            if pd.isna(ew):
                ew = prior
            ww = n / (n + K)
            row[f"rating_{x}"] = ww * ew + (1 - ww) * prior
        g = upcoming[(upcoming.home_team == t) | (upcoming.away_team == t)]
        row["game_id"] = g.game_id.values[0] if len(g) else None
        rows.append(row)
    return pd.concat([ratings, pd.DataFrame(rows)], ignore_index=True), lg


# --------------------------------------------------------------------------- player form
QB_STATS = ["attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
            "passing_epa", "carries", "rushing_yards"]
ALL_STATS = sorted(set(QB_STATS + ["rushing_tds", "rushing_epa", "targets", "receptions",
                                    "receiving_yards", "receiving_tds", "receiving_epa"]))


def player_form(plyr, sched, ratings):
    pw = plyr[plyr.season_type == "REG"].copy()
    pw = pw.merge(sched[["game_id", "gameday", "home_team", "away_team"]], on="game_id", how="left")
    pw["is_home"] = (pw.team == pw.home_team).astype(int)
    pw = pw.sort_values(["player_id", "gameday"]).reset_index(drop=True)
    pw["gp_prior"] = pw.groupby(["player_id", "season"]).cumcount()

    for s in ALL_STATS:
        if s not in pw.columns:
            pw[s] = np.nan
    pos_mean = pw.groupby("position")[ALL_STATS].mean()

    sf = pw.groupby(["player_id", "season"])[ALL_STATS].mean().reset_index()
    sf = sf.rename(columns={s: f"prev_{s}" for s in ALL_STATS})
    sf["season"] += 1
    for s in ALL_STATS:
        sh = pw.groupby(["player_id", "season"])[s].shift(1)
        pw[f"ewma_{s}"] = sh.groupby([pw.player_id, pw.season]).transform(
            lambda x: x.ewm(span=EW_SPAN, min_periods=1).mean())
    pw = pw.merge(sf, on=["player_id", "season"], how="left")
    for s in ALL_STATS:
        lgc = pw.position.map(lambda p: pos_mean[s].get(p, np.nan))
        pw[f"prev_{s}"] = pw[f"prev_{s}"].fillna(lgc)
        pw[f"ewma_{s}"] = pw[f"ewma_{s}"].fillna(pw[f"prev_{s}"])
        pw[f"prior_{s}"] = 0.5 * lgc + 0.5 * pw[f"prev_{s}"]
    w = pw.gp_prior / (pw.gp_prior + K)
    for s in ALL_STATS:
        pw[f"proj_{s}"] = w * pw[f"ewma_{s}"] + (1 - w) * pw[f"prior_{s}"]

    dcols = ["rating_g_def_pass_epa_pp_allowed", "rating_g_def_rush_epa_pp_allowed"]
    od = ratings[["game_id", "team"] + dcols].rename(
        columns={"team": "opponent_team", **{c: f"opp_{c}" for c in dcols}})
    pw = pw.merge(od, on=["game_id", "opponent_team"], how="left")
    return pw


# --------------------------------------------------------------------------- context
def context_features(sched, pw, inj, snap, rost, depth, cur):
    g = sched[(sched.game_type == "REG") & (sched.season >= FIRST_SEASON)].copy()

    if len(depth):
        d = depth[depth.dt == depth.dt.max()]
        qb1 = (d[(d.pos_abb == "QB") & (d.pos_rank == 1)].drop_duplicates("team")
               .set_index("team")["gsis_id"].to_dict())
        for side in ["home", "away"]:
            need = g[f"{side}_qb_id"].isna()
            g.loc[need, f"{side}_qb_id"] = g.loc[need, f"{side}_team"].map(qb1)

    QC = ["proj_passing_epa", "proj_rushing_yards"]
    qb = pw[pw.position == "QB"][["player_id", "season", "week"] + QC].drop_duplicates(
        ["player_id", "season", "week"])
    latest = pw[pw.position == "QB"].sort_values(["player_id", "gameday"]).groupby(
        "player_id").tail(1)[["player_id"] + QC]
    ren = {"proj_passing_epa": "qb_epa", "proj_rushing_yards": "qb_rush"}
    for side in ["home", "away"]:
        g = g.merge(qb.rename(columns={"player_id": f"{side}_qb_id",
                                        **{k: f"{side}_{v}" for k, v in ren.items()}}),
                    on=[f"{side}_qb_id", "season", "week"], how="left")
        g = g.merge(latest.rename(columns={"player_id": f"{side}_qb_id",
                                            **{k: f"{side}_{v}_l" for k, v in ren.items()}}),
                    on=f"{side}_qb_id", how="left")
        for v in ren.values():
            g[f"{side}_{v}"] = g[f"{side}_{v}"].fillna(g[f"{side}_{v}_l"])
            g.drop(columns=[f"{side}_{v}_l"], inplace=True)
    for v in ren.values():
        mu = pd.concat([g[f"home_{v}"], g[f"away_{v}"]]).mean()
        g[f"home_{v}"] = g[f"home_{v}"].fillna(mu)
        g[f"away_{v}"] = g[f"away_{v}"].fillna(mu)

    long = pd.concat([
        g[["game_id", "season", "gameday", "home_team", "home_qb_id"]].rename(
            columns={"home_team": "team", "home_qb_id": "qb"}),
        g[["game_id", "season", "gameday", "away_team", "away_qb_id"]].rename(
            columns={"away_team": "team", "away_qb_id": "qb"})]).sort_values(["team", "gameday"])
    long["prev"] = long.groupby(["team", "season"]).qb.shift(1)
    long["chg"] = ((long.prev.notna()) & (long.qb != long.prev)).astype(int)
    for side in ["home", "away"]:
        g = g.merge(long[["game_id", "team", "chg"]].rename(
            columns={"team": f"{side}_team", "chg": f"{side}_chg"}),
            on=["game_id", f"{side}_team"], how="left")
    g[["home_chg", "away_chg"]] = g[["home_chg", "away_chg"]].fillna(0)

    g["is_indoor"] = g.roof.isin(["dome", "closed"]).astype(int)
    g["wind_f"] = np.where(g.is_indoor == 1, 0.0, g.wind.fillna(g.wind.median()))
    g["temp_f"] = np.where(g.is_indoor == 1, 70.0, g.temp.fillna(g.temp.median()))

    # injuries weighted by the player's recent snap share
    if len(snap) and len(inj):
        s = snap[snap.game_type == "REG"].copy()
        cw = rost.dropna(subset=["gsis_id", "pfr_id"])[["gsis_id", "pfr_id"]].drop_duplicates("pfr_id")
        s = s.merge(cw, left_on="pfr_player_id", right_on="pfr_id", how="left").dropna(subset=["gsis_id"])
        s = s.sort_values(["gsis_id", "season", "week"])
        for c in ["offense_pct", "defense_pct"]:
            s[f"p_{c}"] = s.groupby(["gsis_id", "season"])[c].transform(lambda x: x.expanding().mean())
        usage = s[["gsis_id", "season", "week", "p_offense_pct", "p_defense_pct"]].copy()
        usage[["season", "week"]] = usage[["season", "week"]].astype("int64")
        usage = usage.sort_values("week")

        i = inj[(inj.game_type == "REG") & (inj.report_status == "Out")][
            ["season", "week", "team", "gsis_id"]].dropna(subset=["gsis_id"]).copy()
        i[["season", "week"]] = i[["season", "week"]].astype("int64")
        i = i.sort_values("week")
        # a player who is Out has no snap row that week, so reach back to his last appearance
        i = pd.merge_asof(i, usage, on="week", by=["gsis_id", "season"], direction="backward")
        i[["p_offense_pct", "p_defense_pct"]] = i[["p_offense_pct", "p_defense_pct"]].fillna(0)
        ti = i.groupby(["season", "week", "team"]).agg(
            out_off=("p_offense_pct", "sum"), out_def=("p_defense_pct", "sum"),
            n_out=("gsis_id", "size")).reset_index()
        for side in ["home", "away"]:
            g = g.merge(ti.rename(columns={"team": f"{side}_team", "out_off": f"{side}_out_off",
                                            "out_def": f"{side}_out_def", "n_out": f"{side}_n_out"}),
                        on=["season", "week", f"{side}_team"], how="left")
    for c in ["home_out_off", "home_out_def", "away_out_off", "away_out_def", "home_n_out", "away_n_out"]:
        if c not in g.columns:
            g[c] = 0.0
        g[c] = g[c].fillna(0)

    g["qb_epa_diff"] = g.home_qb_epa - g.away_qb_epa
    g["qb_rush_diff"] = g.home_qb_rush - g.away_qb_rush
    g["qb_change_diff"] = g.away_chg - g.home_chg
    g["inj_off_diff"] = g.away_out_off - g.home_out_off
    g["inj_def_diff"] = g.away_out_def - g.home_out_def

    # display name for the projected starter; the schedule leaves it blank for future
    # weeks, so fall back to the roster entry for whoever the depth chart named
    names = rost.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index(
        "gsis_id")["full_name"].to_dict()
    for side in ["home", "away"]:
        g[f"{side}_qb_disp"] = g[f"{side}_qb_name"].fillna(g[f"{side}_qb_id"].map(names))

    return g[["game_id"] + CTX_FEATS + ["wind_f", "temp_f", "is_indoor",
                                         "home_qb_disp", "away_qb_disp",
                                         "home_n_out", "away_n_out"]]


# --------------------------------------------------------------------------- assemble + model
def build_games(sched, ratings, ctx):
    df = sched[sched.game_type == "REG"].copy()
    rc = [f"rating_{x}" for x in METRICS]
    for side in ["home", "away"]:
        r = ratings[["game_id", "team"] + rc].rename(
            columns={"team": f"{side}_team", **{c: f"{side}_{c}" for c in rc}})
        df = df.merge(r, on=["game_id", f"{side}_team"], how="left")
    df = df.merge(ctx, on="game_id", how="left")

    df["off_epa_diff"] = df.home_rating_g_off_epa_pp - df.away_rating_g_off_epa_pp
    df["def_epa_diff"] = df.away_rating_g_def_epa_pp_allowed - df.home_rating_g_def_epa_pp_allowed
    df["net_epa_edge_home"] = ((df.home_rating_g_off_epa_pp - df.away_rating_g_def_epa_pp_allowed)
                               - (df.away_rating_g_off_epa_pp - df.home_rating_g_def_epa_pp_allowed))
    df["pass_epa_edge_home"] = ((df.home_rating_g_off_pass_epa_pp - df.away_rating_g_def_pass_epa_pp_allowed)
                                - (df.away_rating_g_off_pass_epa_pp - df.home_rating_g_def_pass_epa_pp_allowed))
    df["rush_epa_edge_home"] = ((df.home_rating_g_off_rush_epa_pp - df.away_rating_g_def_rush_epa_pp_allowed)
                                - (df.away_rating_g_off_rush_epa_pp - df.home_rating_g_def_rush_epa_pp_allowed))
    df["points_diff_rating"] = ((df.home_rating_points_scored - df.home_rating_points_allowed)
                                - (df.away_rating_points_scored - df.away_rating_points_allowed))
    df["rest_diff"] = df.home_rest - df.away_rest
    df["div_game"] = df.div_game.fillna(0)

    ml = lambda s: np.where(s < 0, -s / (-s + 100), 100 / (s + 100))
    hp, ap = ml(df.home_moneyline.astype(float)), ml(df.away_moneyline.astype(float))
    df["market_home_wp"] = hp / (hp + ap)
    df["home_win"] = np.where(df.home_score.isna(), np.nan, (df.home_score > df.away_score).astype(float))
    df["home_margin"] = df.home_score - df.away_score
    return df


def fit_predict(df, cur, target_week):
    import xgboost as xgb
    tr = df[(df.season < cur) | ((df.season == cur) & df.home_win.notna())]
    tr = tr.dropna(subset=["home_win", "home_margin"] + FEATS)
    log(f"training on {len(tr)} completed games")

    clf = xgb.XGBClassifier(max_depth=3, n_estimators=150, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=2.0, eval_metric="logloss")
    clf.fit(tr[FEATS], tr.home_win)
    reg = xgb.XGBRegressor(max_depth=3, n_estimators=150, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, reg_lambda=2.0)
    reg.fit(tr[FEATS], tr.home_margin)

    up = df[(df.season == cur) & (df.week == target_week)].dropna(subset=FEATS).copy()
    up["p_model"] = clf.predict_proba(up[FEATS])[:, 1]
    up["p_market"] = up.market_home_wp
    up["p_blend"] = np.where(up.p_market.notna(), 0.4 * up.p_model + 0.6 * up.p_market, up.p_model)
    up["margin_pred"] = reg.predict(up[FEATS])
    tot = up.total_line.fillna(45.0)
    up["predicted_home_score"] = (tot + up.margin_pred) / 2
    up["predicted_away_score"] = (tot - up.margin_pred) / 2
    up["predicted_winner"] = np.where(up.p_blend > 0.5, up.home_team, up.away_team)

    imp = pd.Series(clf.feature_importances_, index=FEATS).sort_values(ascending=False)

    # live out-of-sample scoring: a model that saw only prior seasons, graded on this one
    live = None
    done = df[(df.season == cur) & df.home_win.notna()].dropna(subset=FEATS)
    if len(done) >= 8:
        prior = df[df.season < cur].dropna(subset=["home_win", "home_margin"] + FEATS)
        c2 = xgb.XGBClassifier(max_depth=3, n_estimators=150, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, reg_lambda=2.0, eval_metric="logloss").fit(
            prior[FEATS], prior.home_win)
        r2 = xgb.XGBRegressor(max_depth=3, n_estimators=150, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, reg_lambda=2.0).fit(prior[FEATS], prior.home_margin)
        p = c2.predict_proba(done[FEATS])[:, 1]
        mg = r2.predict(done[FEATS])
        correct = ((p > 0.5).astype(int) == done.home_win.values)
        mk = ((done.market_home_wp > 0.5).astype(int) == done.home_win.values)
        sp = done.spread_line.values
        cov = np.where(done.home_margin.values > sp, 1, np.where(done.home_margin.values < sp, 0, -1))
        ok = cov >= 0
        live = {
            "season": int(cur), "n": int(len(done)),
            "model_su": float(correct.mean()), "market_su": float(mk.mean()),
            "margin_mae": float(np.abs(mg - done.home_margin.values).mean()),
            "ats": float(((mg > sp).astype(int)[ok] == cov[ok]).mean()) if ok.sum() else None,
            "games": [{"game_id": g, "away": aw, "home": hm, "p": float(pp), "pred_margin": float(mm),
                       "actual_margin": float(am), "correct": bool(cc)}
                      for g, aw, hm, pp, mm, am, cc in zip(
                          done.game_id, done.away_team, done.home_team, p, mg,
                          done.home_margin.values, correct)],
        }
    return up, imp, live


# --------------------------------------------------------------------------- player projections
PTARGETS = {
    "passing_yards": dict(stat="passing_yards", pos=["QB"], vol="proj_attempts", mn=10, opp="pass"),
    "passing_tds": dict(stat="passing_tds", pos=["QB"], vol="proj_attempts", mn=10, opp="pass"),
    "qb_rushing_yards": dict(stat="rushing_yards", pos=["QB"], vol="proj_attempts", mn=10, opp="rush"),
    "rushing_yards": dict(stat="rushing_yards", pos=["RB"], vol="proj_carries", mn=5, opp="rush"),
    "rushing_tds": dict(stat="rushing_tds", pos=["RB"], vol="proj_carries", mn=5, opp="rush"),
    "rb_receiving_yards": dict(stat="receiving_yards", pos=["RB"], vol="proj_carries", mn=5, opp="pass"),
    "receiving_yards": dict(stat="receiving_yards", pos=["WR", "TE"], vol="proj_targets", mn=3, opp="pass"),
    "receptions": dict(stat="receptions", pos=["WR", "TE"], vol="proj_targets", mn=3, opp="pass"),
    "receiving_tds": dict(stat="receiving_tds", pos=["WR", "TE"], vol="proj_targets", mn=3, opp="pass"),
}
OPPCOL = {"pass": "opp_rating_g_def_pass_epa_pp_allowed", "rush": "opp_rating_g_def_rush_epa_pp_allowed"}


def player_projections(pw, ratings, sched, rost, depth, cur, target_week):
    from sklearn.linear_model import Ridge
    up = sched[(sched.season == cur) & (sched.week == target_week) & (sched.game_type == "REG")]
    opp = {}
    for _, r in up.iterrows():
        opp[r.home_team] = (r.away_team, 1)
        opp[r.away_team] = (r.home_team, 0)
    dcols = ["rating_g_def_pass_epa_pp_allowed", "rating_g_def_rush_epa_pp_allowed"]
    wk_def = ratings[(ratings.season == cur) & (ratings.week == target_week)][["team"] + dcols]
    wk_def = wk_def.rename(columns={c: f"opp_{c}" for c in dcols}).set_index("team")

    latest = pw.sort_values(["player_id", "gameday"]).groupby("player_id").tail(1)
    active = rost[(rost.season == cur) & (rost.status == "ACT")][
        ["team", "gsis_id", "position", "full_name", "headshot_url"]].rename(
        columns={"gsis_id": "player_id", "full_name": "player_display_name"})
    starters = set()
    if len(depth):
        d = depth[depth.dt == depth.dt.max()]
        starters = set(d[d.pos_rank == 1].gsis_id.dropna())

    rows = []
    for out_col, cfg in PTARGETS.items():
        oc = OPPCOL[cfg["opp"]]
        pc = f"proj_{cfg['stat']}"
        sub = pw[pw.position.isin(cfg["pos"])].dropna(subset=[cfg["stat"], pc, oc, "is_home"])
        sub = sub[sub[cfg["vol"]] >= cfg["mn"]]
        if len(sub) < 200:
            continue
        f = [pc, oc, "is_home"]
        model = Ridge(alpha=5.0).fit(sub[f], sub[cfg["stat"]])

        cand = active[active.position.isin(cfg["pos"])].merge(
            latest[["player_id", pc, cfg["vol"]]], on="player_id", how="inner")
        cand = cand[(cand[cfg["vol"]] >= cfg["mn"]) & (cand.team.isin(opp))]
        if not len(cand):
            continue
        cand["opponent_team"] = cand.team.map(lambda t: opp[t][0])
        cand["is_home"] = cand.team.map(lambda t: opp[t][1])
        cand[oc] = cand.opponent_team.map(lambda t: wk_def[oc].get(t, np.nan))
        cand = cand.dropna(subset=f)
        cand["val"] = model.predict(cand[f])
        for _, r in cand.iterrows():
            rows.append({"player_id": r.player_id, "player_display_name": r.player_display_name,
                          "position": r.position, "team": r.team, "opponent_team": r.opponent_team,
                          "is_home": int(r.is_home), "stat": out_col, "v": float(r.val),
                          "headshot": r.headshot_url if isinstance(r.headshot_url, str) else None})
    if not rows:
        return []
    rdf = pd.DataFrame(rows)
    heads = rdf.dropna(subset=["headshot"]).drop_duplicates("player_id").set_index("player_id")["headshot"]
    pdf = rdf.pivot_table(
        index=["player_id", "player_display_name", "position", "team", "opponent_team", "is_home"],
        columns="stat", values="v").reset_index()
    pdf = pdf[pdf.player_id.isin(starters)] if starters else pdf
    keys = ("player_id", "player_display_name", "position", "team", "opponent_team", "is_home")
    for c in pdf.columns:
        if c not in keys:
            pdf[c] = pdf[c].round(1)
    pdf["headshot"] = pdf.player_id.map(heads)
    return pdf.replace({np.nan: None}).drop(columns=["player_id"]).to_dict(orient="records")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--datadir", default="data")
    a = ap.parse_args()
    os.makedirs(a.datadir, exist_ok=True)

    sched, team, plyr, inj, snap, rost, depth, cur = load_all(a.datadir, None)

    reg = sched[(sched.season == cur) & (sched.game_type == "REG")]
    unplayed = reg[reg.home_score.isna()]
    if not len(unplayed):
        log("season complete; nothing upcoming")
        target_week = int(reg.week.max())
    else:
        target_week = int(unplayed.week.min())
    log(f"target: {cur} week {target_week}")

    ratings, lg = team_ratings(team, sched, cur, target_week)
    pw = player_form(plyr, sched, ratings)
    ctx = context_features(sched, pw, inj, snap, rost, depth, cur)
    df = build_games(sched, ratings, ctx)
    up, imp, live = fit_predict(df, cur, target_week)
    players = player_projections(pw, ratings, sched, rost, depth, cur, target_week)

    up = up.copy()
    up["gameday_s"] = pd.to_datetime(up.gameday).dt.strftime("%a %b ") + \
                      pd.to_datetime(up.gameday).dt.day.astype(str)
    games_out = []
    for _, r in up.iterrows():
        games_out.append({
            "game_id": r.game_id, "gameday": r.gameday_s,
            "home_team": r.home_team, "away_team": r.away_team,
            "spread_line": None if pd.isna(r.spread_line) else float(r.spread_line),
            "total_line": None if pd.isna(r.total_line) else float(r.total_line),
            "p_model": float(r.p_model), "p_market": None if pd.isna(r.p_market) else float(r.p_market),
            "p_blend": float(r.p_blend), "margin_pred": float(r.margin_pred),
            "predicted_winner": r.predicted_winner,
            "predicted_home_score": float(r.predicted_home_score),
            "predicted_away_score": float(r.predicted_away_score),
            "home_qb": r.get("home_qb_disp"), "away_qb": r.get("away_qb_disp"),
            "wind": None if pd.isna(r.get("wind_f")) else float(r.get("wind_f")),
            "temp": None if pd.isna(r.get("temp_f")) else float(r.get("temp_f")),
            "indoor": int(r.get("is_indoor", 0)),
            "home_out": int(r.get("home_n_out", 0)), "away_out": int(r.get("away_n_out", 0)),
        })

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "season": cur, "week": target_week,
        "games": games_out,
        "players": players,
        "feature_importance": [{"feature": k, "importance": float(v)} for k, v in imp.items()],
        "backtest": BACKTEST,
        "live": live,
    }
    out = os.path.join(a.outdir, "payload.json")
    json.dump(payload, open(out, "w"), indent=1, default=str)
    log(f"wrote {out}: {len(games_out)} games, {len(players)} players"
        + (f", live {live['n']} scored" if live else ""))


if __name__ == "__main__":
    main()
