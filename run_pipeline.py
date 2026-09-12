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

import elo
import espn_injuries
from live import store as live_store, versions as live_versions, context as live_context
import explain
import odds_api
import prop_value
import tracker
from adjusted_ratings import ADJ_FEATS, add_adjusted_cols, team_adjusted
from elo import ELO_FEATS, add_elo_cols
from scheme_features import (SCHEME, SCHEME_FEATS, DEF_FEATS, add_scheme_cols,
                             team_scheme, _per_game as sf_per_game)

REL = "https://github.com/nflverse/nflverse-data/releases/download"
SCHEDULES = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
FIRST_SEASON = 2016
EW_SPAN, K = 6, 3.0          # in-season smoothing span; shrinkage speed toward the preseason prior

METRICS = ["g_off_epa_pp", "g_def_epa_pp_allowed", "g_off_pass_epa_pp", "g_off_rush_epa_pp",
           "g_def_pass_epa_pp_allowed", "g_def_rush_epa_pp_allowed", "points_scored", "points_allowed"]
# The five EPA edges here are OPPONENT-ADJUSTED (adjusted_ratings.py), not raw exponentially
# weighted averages. The raw versions are still computed — the quality table and the
# explanations quote them — but the model does not see them. Swapping rather than adding
# keeps the feature count at 24 and beat the raw version on every probability measure:
# Brier 0.2271 -> 0.2259, log loss 0.6486 -> 0.6461, margin MAE 10.32 -> 10.27, and blended
# 66.90% -> 67.01%. Adding both instead was better on picks alone but worse on Brier, worse
# blended, and cost five more features on 2,300 games.
BASE_FEATS = ADJ_FEATS + ["points_diff_rating", "div_game", "rest_diff"]
CTX_FEATS = ["qb_epa_diff", "qb_rush_diff", "qb_change_diff", "inj_off_diff", "inj_def_diff"]
# Positional value for the injury features. QB is zero deliberately — see context_features.
POS_WEIGHT = {"T": 1.35, "DE": 1.30, "CB": 1.25, "WR": 1.20, "DT": 1.00, "G": 0.85, "C": 0.85,
              "TE": 0.85, "S": 0.80, "LB": 0.70, "RB": 0.65, "FB": 0.25, "K": 0.30, "P": 0.20,
              "LS": 0.10, "QB": 0.0}
OFF_POS = {"T", "G", "C", "WR", "TE", "RB", "FB", "QB"}
# Both scheme blocks earn their place on calibration rather than on picks. Offensive scheme:
# Brier 0.2323 -> 0.2305, MAE 10.51 -> 10.45. Defensive scheme on top of that: Brier -> 0.2293,
# log loss 0.6544 -> 0.6512, MAE -> 10.43, while picks slip 62.3% -> 61.9% (McNemar p = 0.62).
# Neither improves the picks and the site says so plainly.
#
# The defensive block was first judged useless on a single-seed run. The 8-seed ensemble
# reversed that: single fits vary by about +/-0.4pp here, which is wider than the effect being
# measured, so that first read was measuring the seed. Anything evaluated on one fit is noise.
#
# Weather stays OUT: it changed the pick rate by 0.00% and made Brier worse. Displayed as
# context, never fed to the model.
#
# Elo is the single most valuable feature in the set and was the last one added, which is
# its own small lesson: every feature above is a description of HOW a team plays, and none
# of them had been checked against a rating system that only knows WHO BEAT WHOM. Alone,
# Elo outscores the whole block (63.3% vs 61.9%). Added to it: 61.94% -> 63.83% straight
# up, Brier 0.2299 -> 0.2253, margin MAE 10.42 -> 10.28. See elo.py for why that is not
# an argument for deleting everything else.
FEATS = BASE_FEATS + CTX_FEATS + SCHEME_FEATS + DEF_FEATS + ELO_FEATS

# Walk-forward backtest results, 2019-2025. These describe the model design rather than
# today's data, so they do not belong in the daily job's own computation — but they DO get
# republished by it, because merge_payload.py carries `backtest` through as a fresh key.
#
# They used to be a hardcoded dict here with a comment asking whoever changed the model to
# remember to retype them. Nobody did, and the failure was silent and self-reversing: the
# audit would be published correctly by hand, and then the next hourly run would overwrite
# it with these stale values. On 2026-09-11 the site was reporting 64.3% in prose and
# 61.8% in the data block feeding the same page.
#
# regen_accuracy.py now writes data/backtest.json, which is committed alongside the code
# and is the single source of these numbers. The dict below is only a fallback for a
# checkout that has not got the file yet, and test_backtest_sync.py fails if the two
# disagree, so the fallback cannot rot unnoticed.
BACKTEST_FALLBACK = {
    "n_games": 1855, "model_su": 0.6442, "market_su": 0.6663, "blend_su": 0.6706,
    "always_home": 0.5288, "margin_mae": 10.24, "market_margin_mae": 9.81,
    "ats": 0.532, "brier_model": 0.2259, "brier_market": 0.2104,
    "note": "opponent-adjusted ratings + QB + injuries + scheme + Elo",
}


def load_prop_audit(datadir="data"):
    """
    What the prop comparison was measured at. Same single-source-of-truth rule as the game
    backtest: written by prop_calibrate.py, read here, never retyped into prose.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for path in [os.path.join(datadir, "prop_verdict.json"),
                 os.path.join(here, "prop_verdict.json")]:
        try:
            with open(path) as f:
                d = json.load(f)
            if isinstance(d, dict) and "hit_rec" in d:
                return d
        except Exception:
            continue
    return None


def load_backtest(datadir="data"):
    """
    The audited figures, preferring the file regen_accuracy.py writes.

    Checked in two places on purpose. Locally it is written to data/, alongside everything
    else the audit touches. In the repo it sits at the top level next to the code, because
    `data/` is scratch that the daily job downloads into and the GitHub upload form has no
    way to place a file inside a directory. Same file either way; first one found wins.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for path in [os.path.join(datadir, "backtest.json"),
                 os.path.join(here, "backtest.json")]:
        try:
            with open(path) as f:
                bt = json.load(f)
            if isinstance(bt, dict) and "model_su" in bt:
                return bt
        except Exception:
            continue
    log("  backtest.json not found; using the built-in audit figures")
    return dict(BACKTEST_FALLBACK)


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
    # Download to a scratch path and only replace the good copy once the new one has been
    # proved readable. The old code deleted the existing file FIRST and then fetched, so a
    # single unreachable host destroyed the pipeline's own input and the run died with a
    # curl error instead of carrying on with yesterday's schedule. Lines and weather move
    # daily, so a fresh copy is still strongly preferred — but stale beats absent.
    tmp = sched_path + ".new"
    r = subprocess.run(["curl", "-sSL", "--max-time", "90", "-o", tmp, SCHEDULES],
                       capture_output=True)
    fresh = False
    if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 100_000:
        try:
            pd.read_csv(tmp, low_memory=False, nrows=5)
            os.replace(tmp, sched_path)
            fresh = True
        except Exception as e:
            log(f"  downloaded schedule was unreadable ({e}); keeping the existing copy")
    if os.path.exists(tmp):
        os.remove(tmp)
    if not fresh:
        if not os.path.exists(sched_path):
            raise SystemExit("no schedule available: the download failed and there is no "
                             "local copy to fall back on")
        age = (datetime.now(timezone.utc).timestamp() - os.path.getmtime(sched_path)) / 3600
        log(f"  schedule download failed — using the local copy, {age:.0f}h old")

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

    # A week is "upcoming" as soon as ONE of its games is unplayed, so on a Friday the
    # Thursday teams already have a real stats row for this week. Giving them a synthetic
    # one too puts the team in here twice, and every per-team lookup downstream
    # (wk_def[oc].get(t), qual.loc[t]) then returns a Series instead of a number, which
    # blows up inside Ridge.predict with a message that names none of this.
    have = set(ratings[(ratings.season == cur) & (ratings.week == target_week)].team)
    teams = [t for t in teams if t not in have]

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

    # Share of the team's targets and carries, from PRIOR games only. Without these the
    # projection is built entirely on what a player did, with no way to know his role
    # changed — and measured against a rolling average of his own recent games the model
    # was WORSE on four of nine stats. Adding usage took the weighted error from 13.04 to
    # 12.71 against a 12.98 baseline. See test_props.py.
    for col, out in [("targets", "tgt_share"), ("carries", "car_share")]:
        if col not in pw.columns:
            pw[out] = 0.0
            continue
        tm = pw.groupby(["team", "season", "week"])[col].transform("sum")
        pw["_sh"] = (pw[col] / tm.replace(0, np.nan)).fillna(0)
        pw[out] = (pw.sort_values(["player_id", "season", "week"])
                   .groupby(["player_id", "season"])["_sh"]
                   .transform(lambda x: x.shift(1).expanding().mean()))
        pw[out] = pw[out].fillna(0)
    pw = pw.drop(columns=["_sh"], errors="ignore")
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
    # The model needs a number, so missing values are median-filled. The SITE must not show
    # a filled value as a forecast, hence the flags: a game with no posted forecast says so.
    g["wx_known"] = (g.wind.notna() & g.temp.notna()).astype(int)
    g["wind_f"] = np.where(g.is_indoor == 1, 0.0, g.wind.fillna(g.wind.median()))
    g["temp_f"] = np.where(g.is_indoor == 1, 70.0, g.temp.fillna(g.temp.median()))

    # ---- injuries: the two biggest absences per side, weighted by position ----
    # This replaced a plain sum of every ruled-out player's snap share, which was measurably
    # WORSE than having no injury features at all (-0.25pp on picks across 8 seeds). Summing
    # made five rotational backups look like one lost left tackle, and it treated a starting
    # corner the same as a starting guard. Weighting by positional value and keeping only the
    # top two absences per side scores +0.49pp instead. Ordering is coherent and that is most
    # of why it is believable: sum < max < weighted sum < weighted top-2.
    #
    # QB carries weight 0 on purpose. qb_change_diff already says "a different quarterback is
    # starting"; letting the QB dominate here too would double-count the one case that matters
    # most. Seed-to-seed noise on any of these numbers is about +/-0.4pp, so single runs mean
    # nothing — everything above is a mean of 8 seeds.
    inj_detail = {}
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

        # The expanding mean above only looks within a season, so in Week 1 nobody has one
        # and every injury would weigh zero — exactly when the site is live for the opener.
        # Last season's average share is fully known before Week 1, so it is a legal
        # point-in-time fallback; shift the season forward to make that explicit.
        prev = (s.groupby(["gsis_id", "season"])[["offense_pct", "defense_pct"]].mean()
                .reset_index().rename(columns={"offense_pct": "pr_off", "defense_pct": "pr_def"}))
        prev["season"] = prev.season.astype("int64") + 1

        i = inj[(inj.game_type == "REG") & (inj.report_status == "Out")][
            ["season", "week", "team", "gsis_id", "position", "full_name"]].dropna(
            subset=["gsis_id"]).copy()
        i[["season", "week"]] = i[["season", "week"]].astype("int64")
        i = i.sort_values("week")
        # a player who is Out has no snap row that week, so reach back to his last appearance
        i = pd.merge_asof(i, usage, on="week", by=["gsis_id", "season"], direction="backward")
        i = i.merge(prev, on=["gsis_id", "season"], how="left")
        i["p_offense_pct"] = i.p_offense_pct.fillna(i.pr_off).fillna(0)
        i["p_defense_pct"] = i.p_defense_pct.fillna(i.pr_def).fillna(0)

        i["pw"] = i.position.map(POS_WEIGHT).fillna(0.7)
        off = i.position.isin(OFF_POS)
        i["share"] = np.where(off, i.p_offense_pct, i.p_defense_pct)
        i["wshare"] = i.share * i.pw
        i["side"] = np.where(off, "off", "def")

        rows = []
        for (se, wk, tm), grp in i.groupby(["season", "week", "team"]):
            rec = {"season": se, "week": wk, "team": tm, "n_out": len(grp)}
            for side in ["off", "def"]:
                sub = grp[grp.side == side].nlargest(2, "wshare")
                rec[f"out_{side}"] = float(sub.wshare.sum())
            rows.append(rec)
            # keep the drivers so the site can name who the number is about
            top = grp.nlargest(3, "wshare")
            inj_detail[(int(se), int(wk), tm)] = [
                {"name": r.full_name, "pos": r.position, "side": r.side,
                 "share": round(float(r.share), 3), "weighted": round(float(r.wshare), 3)}
                for r in top.itertuples() if r.wshare > 0.02]
        ti = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["season", "week", "team", "n_out", "out_off", "out_def"])
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

    # the raw sides of every diff come through too — the explanations quote them, and a
    # factor the reader can't check against a number is just an assertion
    ctx_out = g[["game_id"] + CTX_FEATS + ["wind_f", "temp_f", "is_indoor", "wx_known",
                                         "home_qb_disp", "away_qb_disp",
                                         "home_n_out", "away_n_out",
                                         "home_qb_epa", "away_qb_epa",
                                         "home_qb_rush", "away_qb_rush",
                                         "home_chg", "away_chg",
                                         "home_out_off", "away_out_off",
                                         "home_out_def", "away_out_def"]]
    ctx_out.attrs["inj_detail"] = inj_detail
    return ctx_out


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


XGB_PARAMS = dict(max_depth=3, n_estimators=150, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=2.0)
# XGBoost subsamples rows and columns, so a single fit is one draw from a distribution whose
# spread on this data is about +/-0.4 percentage points of pick accuracy — wider than most of
# the feature effects being tested. Averaging eight seeds removes that lottery: it beats the
# average single seed by +0.51pp and the unluckiest by +1.13pp, lowers Brier, and costs about
# ten seconds. Every published figure is the ensemble, so nothing here depends on a lucky draw.
N_SEEDS = 8

# How much of the published number is the model, and how much is the closing line.
# The old value was 0.4 and had never been checked. Swept over the whole 2019-2025 audit:
# straight-up pick rate peaks at 0.20 (67.12% against 66.90% at 0.40), and Brier and log
# loss both improve as well (0.2124 -> 0.2107, 0.6133 -> 0.6091). Fitting the weight
# walk-forward, so it never saw the season it was scoring, chose 0.15 in each of the last
# three seasons, which is the same region.
#
# Said plainly, because it is the most honest number on the site: log loss alone is
# minimised at w = 0.05, i.e. very nearly "ignore the model". The model buys about half a
# point of pick rate over the bare market and buys nothing at all in probability quality.
BLEND_W = 0.20


def _fit_ensemble(X, y_cls, y_reg):
    import xgboost as xgb
    pairs = []
    for sd in range(N_SEEDS):
        c = xgb.XGBClassifier(**XGB_PARAMS, eval_metric="logloss", random_state=sd)
        c.fit(X, y_cls)
        r = xgb.XGBRegressor(**XGB_PARAMS, random_state=sd)
        r.fit(X, y_reg)
        pairs.append((c, r))
    return pairs


def _ens_predict(pairs, X):
    p = np.mean([c.predict_proba(X)[:, 1] for c, _ in pairs], axis=0)
    m = np.mean([r.predict(X) for _, r in pairs], axis=0)
    return p, m


def fit_predict(df, cur, target_week):
    import xgboost as xgb
    tr = df[(df.season < cur) | ((df.season == cur) & df.home_win.notna())]
    tr = tr.dropna(subset=["home_win", "home_margin"] + FEATS)
    log(f"training on {len(tr)} completed games ({N_SEEDS}-seed ensemble)")

    pairs = _fit_ensemble(tr[FEATS], tr.home_win, tr.home_margin)
    clf, reg = pairs[0]                      # kept only for the SHAP call below

    up = df[(df.season == cur) & (df.week == target_week)].dropna(subset=FEATS).copy()
    up["p_model"], _margin_ens = _ens_predict(pairs, up[FEATS])
    up["p_market"] = up.market_home_wp
    up["p_blend"] = np.where(up.p_market.notna(),
                             BLEND_W * up.p_model + (1 - BLEND_W) * up.p_market, up.p_model)
    up["margin_pred"] = _margin_ens
    tot = up.total_line.fillna(45.0)
    up["predicted_home_score"] = (tot + up.margin_pred) / 2
    up["predicted_away_score"] = (tot - up.margin_pred) / 2
    up["predicted_winner"] = np.where(up.p_blend > 0.5, up.home_team, up.away_team)

    # Exact tree SHAP: each column is one feature's signed contribution to this game's
    # log-odds, last column the bias. This is the model's own arithmetic, so the "why"
    # text cannot drift away from what the model actually did.
    # Averaged across the ensemble: each model's contributions sum to its own raw log-odds,
    # so the mean matrix sums to the mean log-odds. Only relative shares are displayed, and
    # for eight near-identical models those are stable.
    dm = xgb.DMatrix(up[FEATS])
    contribs = np.mean([c.get_booster().predict(dm, pred_contribs=True) for c, _ in pairs], axis=0)

    # The injury counterfactual: the same fitted model asked again with both teams healthy.
    # Because the model is deterministic this is exact, not an estimate — the difference IS
    # what this model attributes to the injury report. Whether that attribution is any good
    # is a separate question the Accuracy tab answers.
    healthy = up[FEATS].copy()
    healthy[["inj_off_diff", "inj_def_diff"]] = 0.0
    up["p_model_healthy"], up["margin_healthy"] = _ens_predict(pairs, healthy)
    # the published number blends with the market, so the shift a reader sees must too
    up["p_blend_healthy"] = np.where(up.p_market.notna(),
                                     BLEND_W * up.p_model_healthy + (1 - BLEND_W) * up.p_market,
                                     up.p_model_healthy)

    imp = pd.Series(np.mean([c.feature_importances_ for c, _ in pairs], axis=0),
                    index=FEATS).sort_values(ascending=False)

    # live out-of-sample scoring: a model that saw only prior seasons, graded on this one
    live = None
    done = df[(df.season == cur) & df.home_win.notna()].dropna(subset=FEATS)
    if len(done) >= 8:
        prior = df[df.season < cur].dropna(subset=["home_win", "home_margin"] + FEATS)
        prior_pairs = _fit_ensemble(prior[FEATS], prior.home_win, prior.home_margin)
        p, mg = _ens_predict(prior_pairs, done[FEATS])
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
    return up, imp, live, contribs


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
# collapsed into one reported "usage" term in the explanation: three near-collinear shares
# are not three readable pieces
USAGE = ["tgt_share", "car_share"]

# The player page plots the projection against what the player has actually done. Box
# scores only, oldest first, capped so a hundred-odd players do not double the payload.
LOG_COLS = {
    "QB": ["attempts", "passing_yards", "passing_tds", "rushing_yards"],
    "RB": ["carries", "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards"],
    "WR": ["targets", "receptions", "receiving_yards", "receiving_tds", "tgt_share"],
    "TE": ["targets", "receptions", "receiving_yards", "receiving_tds", "tgt_share"],
}
LOG_N = 10


def game_logs(pw, ids):
    sub = pw[pw.player_id.isin(set(ids))].sort_values(["player_id", "gameday"])
    out = {}
    for pid, g in sub.groupby("player_id"):
        pos = str(g.position.iloc[-1]) if "position" in g.columns else "WR"
        have = [c for c in LOG_COLS.get(pos, LOG_COLS["WR"]) if c in pw.columns]
        rows = []
        for r in g.tail(LOG_N).itertuples():
            d = {"s": int(r.season), "w": int(r.week), "opp": r.opponent_team,
                 "h": None if pd.isna(r.is_home) else int(r.is_home)}
            for c in have:
                v = getattr(r, c)
                d[c] = None if pd.isna(v) else round(float(v), 3 if "share" in c else 1)
            rows.append(d)
        out[pid] = rows
    return out


def player_projections(pw, ratings, sched, rost, depth, cur, target_week, inj_map=None,
                       opp_scheme=None):
    from sklearn.linear_model import Ridge
    up = sched[(sched.season == cur) & (sched.week == target_week) & (sched.game_type == "REG")]
    opp = {}
    for _, r in up.iterrows():
        opp[r.home_team] = (r.away_team, 1)
        opp[r.away_team] = (r.home_team, 0)
    dcols = ["rating_g_def_pass_epa_pp_allowed", "rating_g_def_rush_epa_pp_allowed"]
    wk_def = ratings[(ratings.season == cur) & (ratings.week == target_week)][["team"] + dcols]
    # keep="last" prefers the real box-score row over any synthetic one; the guard in
    # team_ratings should make this a no-op, and it stays because a duplicate here fails
    # deep inside Ridge.predict with an error that points nowhere near the cause
    wk_def = wk_def.drop_duplicates("team", keep="last")
    wk_def = wk_def.rename(columns={c: f"opp_{c}" for c in dcols}).set_index("team")

    latest = pw.sort_values(["player_id", "gameday"]).groupby("player_id").tail(1)
    active = rost[(rost.season == cur) & (rost.status == "ACT")][
        ["team", "gsis_id", "position", "full_name", "headshot_url"]].rename(
        columns={"gsis_id": "player_id", "full_name": "player_display_name"})
    starters = set()
    if len(depth):
        d = depth[depth.dt == depth.dt.max()]
        starters = set(d[d.pos_rank == 1].gsis_id.dropna())

    rows, why = [], {}
    for out_col, cfg in PTARGETS.items():
        oc = OPPCOL[cfg["opp"]]
        pc = f"proj_{cfg['stat']}"
        sub = pw[pw.position.isin(cfg["pos"])].dropna(subset=[cfg["stat"], pc, oc, "is_home"] + USAGE)
        sub = sub[sub[cfg["vol"]] >= cfg["mn"]]
        if len(sub) < 200:
            continue
        f = [pc, oc, "is_home"] + USAGE
        model = Ridge(alpha=5.0).fit(sub[f], sub[cfg["stat"]])

        cand = active[active.position.isin(cfg["pos"])].merge(
            latest[["player_id", pc, cfg["vol"]] + USAGE], on="player_id", how="inner")
        cand = cand[(cand[cfg["vol"]] >= cfg["mn"]) & (cand.team.isin(opp))]
        if not len(cand):
            continue
        cand["opponent_team"] = cand.team.map(lambda t: opp[t][0])
        cand["is_home"] = cand.team.map(lambda t: opp[t][1])
        cand[oc] = cand.opponent_team.map(lambda t: wk_def[oc].get(t, np.nan))
        cand = cand.dropna(subset=f)
        cand["val"] = model.predict(cand[f])

        # rank 1 = the defense that has allowed the least in this phase, so a low rank
        # is a hard matchup regardless of which stat is being projected
        drank = wk_def[oc].rank(method="min").astype(int).to_dict()
        n_teams = int(wk_def[oc].notna().sum())
        fmean, omean = float(sub[pc].mean()), float(sub[oc].mean())
        umean = [float(sub[u].mean()) for u in USAGE]

        for _, r in cand.iterrows():
            rows.append({"player_id": r.player_id, "player_display_name": r.player_display_name,
                          "position": r.position, "team": r.team, "opponent_team": r.opponent_team,
                          "is_home": int(r.is_home), "stat": out_col, "v": float(r.val),
                          "headshot": r.headshot_url if isinstance(r.headshot_url, str) else None})
            why[(r.player_id, out_col)] = explain.player_reason(
                float(r.val), model.coef_, model.intercept_, float(r[pc]), float(r[oc]), int(r.is_home),
                fmean, omean, drank.get(r.opponent_team, n_teams // 2), n_teams,
                usage=[float(r[u]) for u in USAGE], usage_mean=umean)
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
    recs = pdf.replace({np.nan: None}).to_dict(orient="records")
    inj_map = inj_map or {}
    opp_scheme = opp_scheme or {}
    logs = game_logs(pw, pdf.player_id)
    for rec in recs:
        pid = rec.pop("player_id")
        rec["player_key"] = pid          # the tracker locks projections against this id
        rec["log"] = logs.get(pid, [])
        rec["why"] = {s: w for (p, s), w in why.items() if p == pid}
        st = inj_map.get(pid)
        if st:
            rec["status"] = st
        # the opponent's defensive identity, for the matchup badge. Context only: tested as
        # a prop feature and it moved nothing, which the site says
        osc = opp_scheme.get(rec.get("opponent_team"))
        if osc:
            rec["opp_scheme"] = osc
    return recs


# --------------------------------------------------------------------------- injuries
# The NFL's game-status ladder. Anything below Questionable never appears on the official
# report; a player only practising in a limited way is shown separately and more softly,
# because "limited on Wednesday" is routine maintenance, not doubt about Sunday.
STATUS_RANK = {"Out": 4, "Doubtful": 3, "Questionable": 2}


def injury_status(inj, cur, target_week, espn=None):
    """
    Latest injury report for the upcoming week, per player.

    The report for a given week is published in stages through that week, so early in the
    week the target week's rows may not exist yet. Fall back to the most recent week that
    does have rows, and say which week the information came from rather than implying it
    is current.

    Where ESPN carried the news first, the row also gets the reporter's sentence and the
    minute it was filed. A status with no timestamp is the official report, which is
    published on a schedule; a status with one came off a live feed, and the difference is
    worth showing rather than blending away.
    """
    if not len(inj):
        return {}, {}, None
    i = inj[(inj.season == cur) & inj.gsis_id.notna()].copy()
    if not len(i):
        return {}, {}, None

    weeks = sorted(i.week.unique())
    use = target_week if target_week in weeks else (max([w for w in weeks if w <= target_week],
                                                        default=max(weeks)))
    i = i[i.week == use]

    def clean(v):
        s = str(v).strip() if v is not None and str(v) != "nan" else ""
        return "" if s.lower() in ("", "none", "nan") else s

    # gsis_id -> the sentence the live feed ran, and when, but ONLY for the rows the
    # overlay actually applied. A player whose status still comes from the official report
    # must not be shown wearing a live timestamp; that would be claiming a freshness the
    # number does not have.
    notes = {}
    if espn is not None and len(espn) and "gsis_id" in espn.columns:
        src = espn[espn.get("applied", False) == True] if "applied" in espn.columns else espn
        for _, e in src.dropna(subset=["gsis_id"]).iterrows():
            if bool(e.get("scratch")):
                continue
            ts = e.get("updated")
            notes[e.gsis_id] = {
                "note": (str(e.get("note") or "").strip() or None),
                "updated": None if pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%MZ")}

    out, by_team = {}, {}
    for _, r in i.iterrows():
        rep = clean(r.get("report_status"))
        prac = clean(r.get("practice_status"))
        harm = clean(r.get("report_primary_injury")) or clean(r.get("practice_primary_injury"))
        second = clean(r.get("report_secondary_injury")) or clean(r.get("practice_secondary_injury"))

        # veteran rest days come through the injury feed but are not injuries
        rest = harm.lower().startswith("not injury related")
        if rest:
            harm, second = "", ""

        # practice participation, in the words a reader uses
        pw = ("did not practice" if prac.lower().startswith("did not")
              else "limited in practice" if prac.lower().startswith("limited")
              else "full practice" if prac.lower().startswith("full") else "")

        if rep in STATUS_RANK:
            level, label = rep.lower(), rep
        elif rest:
            level, label = "rest", "Rested"
        elif pw in ("did not practice", "limited in practice"):
            level, label = "limited", ("Did not practice" if pw == "did not practice" else "Limited")
        else:
            continue                       # full practice, no game status — nothing to report

        bits = []
        if harm:
            bits.append(harm.lower() + (f" and {second.lower()}" if second else ""))
        if pw and not (level == "limited" and pw.replace(" in practice", "") in label.lower()):
            bits.append(pw)
        why = ", ".join(bits)

        if level == "out":
            why = f"Ruled out{' — ' + why if why else ''}"
        elif level == "rest":
            why = "Veteran rest day, not an injury"
        elif why:
            why = why[0].upper() + why[1:]

        rec = {"level": level, "label": label, "why": why,
               "injury": harm or None, "week": int(use)}
        e = notes.get(r.gsis_id)
        if e:
            rec["note"] = e["note"] or None
            rec["updated"] = e["updated"]
            rec["src"] = "espn"
        out[r.gsis_id] = rec
        by_team.setdefault(r.team, []).append(
            dict(rec, name=clean(r.get("full_name")), pos=clean(r.get("position")),
                 player_id=r.gsis_id))

    # worst news first, and only the positions a reader is scanning for
    order = {"out": 0, "doubtful": 1, "questionable": 2, "limited": 3, "rest": 4}
    skill = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
    for t, lst in by_team.items():
        lst.sort(key=lambda x: (order.get(x["level"], 9), skill.get(x["pos"], 8), x["name"]))
    return out, by_team, int(use)


def injuries_since_kickoff(sched, espn, cur, week):
    """
    Injuries filed AFTER a game started, for games in this week that are already over.

    This is the gap that made the whole ESPN change necessary. The official injury report
    is a pre-game document: it describes who might not play on Sunday, and it is published
    Wednesday to Friday. An injury suffered DURING Sunday's game cannot appear on it until
    the following Wednesday.

    So for three or four days the site had a finished game sitting on the page, its injury
    report showing the pre-kickoff picture, and no indication whatsoever that the home
    team's quarterback had limped off in the third quarter. That is not a stale number, it
    is a missing event, and no amount of refreshing the official report fixes it.

    The pre-kickoff report on those cards is deliberately left alone. It is what the model
    saw when it locked that prediction and rewriting it would be dishonest about what was
    known at the time. This is a separate list, and the page labels it as one.

    A status is "since kickoff" if its timestamp is later than the scheduled start. Kickoff
    times in the schedule are US Eastern; comparing them to ESPN's UTC stamps without
    converting would shift every game by four or five hours and quietly drop the injuries
    reported in the first few hours after a game -- which is most of them.
    """
    if espn is None or not len(espn) or "updated" not in espn.columns:
        return {}
    d = sched[(sched.season == cur) & (sched.week == week) & sched.home_score.notna()]
    if not len(d):
        return {}
    e = espn[~espn.scratch.fillna(False) & espn.updated.notna()].copy()
    e["status"] = e.espn_status.str.lower().str.strip().map(espn_injuries.STATUS)
    e = e[e.status.notna()]
    if not len(e):
        return {}

    out = {}
    for _, g in d.iterrows():
        when = f"{pd.Timestamp(g.gameday).date()} {g.get('gametime') or '13:00'}"
        try:
            kick = pd.Timestamp(when).tz_localize("America/New_York").tz_convert("UTC")
        except Exception:
            continue
        rows = e[e.team.isin([g.home_team, g.away_team]) & (e.updated > kick)]
        if not len(rows):
            continue
        out[g.game_id] = [
            {"name": r.full_name, "pos": r.position, "team": r.team,
             "label": r.status, "level": r.status.lower(),
             "injury": (r.injury_type or None) if str(r.injury_type) != "Undisclosed" else None,
             "note": (str(r.note).strip() or None) if r.note else None,
             "updated": pd.Timestamp(r.updated).strftime("%Y-%m-%dT%H:%MZ")}
            for r in rows.sort_values("updated", ascending=False).head(8).itertuples()]
    return out


# --------------------------------------------------------------------------- coaches
def coach_tenure(sched, cur):
    """(team, coach) -> seasons in the current unbroken run with that team."""
    h = sched[sched.game_type == "REG"][["season", "home_team", "home_coach"]].rename(
        columns={"home_team": "team", "home_coach": "coach"})
    a = sched[sched.game_type == "REG"][["season", "away_team", "away_coach"]].rename(
        columns={"away_team": "team", "away_coach": "coach"})
    x = pd.concat([h, a]).dropna(subset=["coach"]).drop_duplicates(["season", "team", "coach"])
    out = {}
    for (t, c), grp in x.groupby(["team", "coach"]):
        yrs = sorted(grp.season.unique())
        run, prev = 0, None
        for y in yrs:                       # length of the streak that reaches the current season
            run = run + 1 if prev is not None and y == prev + 1 else 1
            prev = y
        out[(t, c)] = run if prev == cur else 0
    return out


def team_records(sched, cur):
    """W-L-T through completed regular-season games of the current season."""
    d = sched[(sched.season == cur) & (sched.game_type == "REG") & sched.home_score.notna()]
    rec = {}
    for _, r in d.iterrows():
        for t, own, opp in [(r.home_team, r.home_score, r.away_score),
                            (r.away_team, r.away_score, r.home_score)]:
            w, l, ti = rec.get(t, (0, 0, 0))
            rec[t] = (w + (own > opp), l + (own < opp), ti + (own == opp))
    return {t: (f"{w}-{l}" + (f"-{ti}" if ti else "")) for t, (w, l, ti) in rec.items()}


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

    # Live injury status, layered over the official report for the upcoming week only.
    # Scoped to teams that have NOT kicked off yet: a team whose game is already in the
    # books had its prediction locked before kickoff, and rewriting the inputs behind a
    # locked prediction would be rewriting history. See espn_injuries.py.
    tw_un = unplayed[unplayed.week == target_week] if len(unplayed) else unplayed
    live_teams = set(tw_un.home_team) | set(tw_un.away_team) if len(tw_un) else None
    # ESPN is refused by egress policy from both dev machines, so the only way to exercise
    # this path outside CI is a captured response. Set the env var to a .psv fixture.
    inj, espn = espn_injuries.live_overlay(inj, rost, cur, target_week, teams=live_teams,
                                           fixture=os.environ.get("SJ_ESPN_FIXTURE") or None,
                                           log=log)

    ratings, lg = team_ratings(team, sched, cur, target_week)
    pw = player_form(plyr, sched, ratings)
    ctx = context_features(sched, pw, inj, snap, rost, depth, cur)
    df = build_games(sched, ratings, ctx)
    scheme, sch_lg = team_scheme(team, sched, cur, target_week)
    df = add_scheme_cols(df, scheme)
    df = add_adjusted_cols(df, team_adjusted(team, sched, cur, target_week))
    df = add_elo_cols(df)
    up, imp, live, contribs = fit_predict(df, cur, target_week)
    inj_map, inj_teams, inj_week = injury_status(inj, cur, target_week, espn=espn)
    since_kick = injuries_since_kickoff(sched, espn, cur, target_week)
    if since_kick:
        log(f"  {sum(len(v) for v in since_kick.values())} injury update(s) filed after "
            f"kickoff across {len(since_kick)} finished game(s)")
    inj_drivers = ctx.attrs.get("inj_detail", {})
    wk_sc = (scheme[(scheme.season == cur) & (scheme.week == target_week)]
             .drop_duplicates("team", keep="last").set_index("team"))
    opp_sc = {t: {"pk_pressure": round(float(r.get("pk_pressure", 0.5)), 3),
                  "pk_funnel": round(float(r.get("pk_funnel", 0.5)), 3),
                  "pk_havoc": round(float(r.get("pk_havoc", 0.5)), 3)}
              for t, r in wk_sc.iterrows()}
    players = player_projections(pw, ratings, sched, rost, depth, cur, target_week, inj_map, opp_sc)

    # FanDuel comparison. Deliberately AFTER player_projections: the projections above were
    # made from football data alone and cannot see a line. This only reads them.
    prop_model = prop_value.PropModel()
    props_by_game, props_meta = {}, None
    if prop_model.ok:
        props_by_game, props_meta = prop_value.build(
            players, odds_api.fetch(log=log), prop_model, log=log)
    else:
        log("  props: no fitted prop model on disk; Player Prop Value is off this run")

    up = up.copy()
    up["gameday_s"] = pd.to_datetime(up.gameday).dt.strftime("%a %b ") + \
                      pd.to_datetime(up.gameday).dt.day.astype(str)
    reasons = explain.game_reasons(up, contribs, FEATS)
    ten = coach_tenure(sched, cur)
    recs = team_records(sched, cur)

    # Defensive quality as the models actually consume it: EPA allowed per play, split by
    # pass and rush, ranked across the league this week. Rank 1 = stingiest. These are the
    # inputs; the "scheme" measures alongside them are context and were tested as noise.
    wk_r = ratings[(ratings.season == cur) & (ratings.week == target_week)].copy()
    wk_r = wk_r.drop_duplicates("team", keep="last")
    n_rank = int(wk_r.team.nunique())
    for c, nm in [("rating_g_def_pass_epa_pp_allowed", "def_pass"),
                  ("rating_g_def_rush_epa_pp_allowed", "def_rush"),
                  ("rating_g_off_pass_epa_pp", "off_pass"),
                  ("rating_g_off_rush_epa_pp", "off_rush")]:
        wk_r[f"{nm}_rank"] = wk_r[c].rank(method="min").astype(int)
        wk_r[f"{nm}_val"] = wk_r[c]
    # offense ranks read the other way: rank 1 = best offense
    for nm in ["off_pass", "off_rush"]:
        wk_r[f"{nm}_rank"] = (n_rank + 1 - wk_r[f"{nm}_rank"]).astype(int)
    qual = wk_r.set_index("team")[[c for c in wk_r.columns if c.endswith("_rank") or c.endswith("_val")]]

    def side_scheme(r, side):
        """Everything the site needs to draw one team's identity panel."""
        t = r[f"{side}_team"]
        d = {x: float(r.get(f"{side}_sch_{x}", np.nan)) for x in SCHEME}
        d.update({f"pk_{x}": float(r.get(f"{side}_pk_{x}", 0.5)) for x in SCHEME})
        d["label"] = explain.scheme_label(d["pk_pass_rate"], d["pk_adot"], d["pk_pace"])
        d["def_label"] = explain.defense_label(d["pk_pressure"], d["pk_havoc"], d["pk_funnel"])
        if t in qual.index:
            q = qual.loc[t]
            d["quality"] = {k: (int(q[f"{k}_rank"]), round(float(q[f"{k}_val"]), 4))
                            for k in ["def_pass", "def_rush", "off_pass", "off_rush"]}
            d["quality"]["n"] = n_rank
        return d

    kick = sched.drop_duplicates("game_id").set_index("game_id")["gametime"].to_dict() \
        if "gametime" in sched.columns else {}
    games_out = []
    for i, (_, r) in enumerate(up.iterrows()):
        hs, as_ = side_scheme(r, "home"), side_scheme(r, "away")
        hc, ac = r.get("home_coach"), r.get("away_coach")
        indoor = int(r.get("is_indoor", 0))
        known = indoor or int(r.get("wx_known", 0) or 0)
        wind = float(r.get("wind_f")) if known and not pd.isna(r.get("wind_f")) else None
        temp = float(r.get("temp_f")) if known and not pd.isna(r.get("temp_f")) else None
        games_out.append({
            "game_id": r.game_id, "gameday": r.gameday_s,
            "gameday_iso": str(pd.Timestamp(r.gameday).date()),
            "stadium_id": None if pd.isna(r.get("stadium_id")) else str(r.get("stadium_id")),
            "kickoff": (lambda k: None if k is None or (isinstance(k, float) and pd.isna(k)) else str(k))(kick.get(r.game_id)),
            "home_team": r.home_team, "away_team": r.away_team,
            "home_record": recs.get(r.home_team, "0-0"), "away_record": recs.get(r.away_team, "0-0"),
            "spread_line": None if pd.isna(r.spread_line) else float(r.spread_line),
            "total_line": None if pd.isna(r.total_line) else float(r.total_line),
            "p_model": float(r.p_model), "p_market": None if pd.isna(r.p_market) else float(r.p_market),
            "p_blend": float(r.p_blend), "margin_pred": float(r.margin_pred),
            "predicted_winner": r.predicted_winner,
            "predicted_home_score": float(r.predicted_home_score),
            "predicted_away_score": float(r.predicted_away_score),
            "home_qb": r.get("home_qb_disp"), "away_qb": r.get("away_qb_disp"),
            "wind": wind, "temp": temp, "indoor": indoor,
            "roof": None if pd.isna(r.get("roof")) else str(r.get("roof")),
            "surface": None if pd.isna(r.get("surface")) else str(r.get("surface")),
            "stadium": None if pd.isna(r.get("stadium")) else str(r.get("stadium")),
            "home_out": int(r.get("home_n_out", 0)), "away_out": int(r.get("away_n_out", 0)),
            "props": props_by_game.get(f"{r.away_team}@{r.home_team}", []),
            "injuries": {
                "week": inj_week,
                "since": since_kick.get(r.game_id, []),
                "home": [p for p in inj_teams.get(r.home_team, []) if p["level"] != "rest"][:8],
                "away": [p for p in inj_teams.get(r.away_team, []) if p["level"] != "rest"][:8],
                # exact counterfactual: this model, same fit, both teams healthy
                "impact": {
                    "p_now": round(float(r.p_blend), 4),
                    "p_healthy": round(float(r.p_blend_healthy), 4),
                    "shift": round(float(r.p_blend - r.p_blend_healthy), 4),
                    "margin_shift": round(float(r.margin_pred - r.margin_healthy), 2),
                    "drivers": {
                        "home": inj_drivers.get((cur, target_week, r.home_team), []),
                        "away": inj_drivers.get((cur, target_week, r.away_team), []),
                    },
                },
            },
            "why": reasons[i],
            "weather_note": explain.weather_note(indoor, temp, wind, r.get("roof"),
                                                 r.get("surface"), hs["pk_pass_rate"],
                                                 as_["pk_pass_rate"]),
            "coaching": {
                "home": {"name": None if pd.isna(hc) else str(hc),
                         "years": ten.get((r.home_team, hc), 0), "scheme": hs},
                "away": {"name": None if pd.isna(ac) else str(ac),
                         "years": ten.get((r.away_team, ac), 0), "scheme": as_},
                "clash": explain.style_clash(r.home_team, r.away_team, hs, as_),
            },
        })

    # ---- season tracker: lock this week's calls, grade anything that has now been played ----
    hist_path = os.path.join(a.outdir, "history.json")
    hist = tracker.load(hist_path)
    gid_by_team = {}
    for _, r in up.iterrows():
        gid_by_team[r.home_team] = str(r.game_id)
        gid_by_team[r.away_team] = str(r.game_id)
    scheme_rows = [
        {"team": t, "game_id": gid_by_team.get(t),
         **{m: round(float(v), 4) for m, v in
            scheme[(scheme.season == cur) & (scheme.week == target_week) &
                   (scheme.team == t)][[f"sch_{x}" for x in tracker.SCHEME_TRACK]]
            .iloc[0].items() if not pd.isna(v)}}
        for t in gid_by_team
        if len(scheme[(scheme.season == cur) & (scheme.week == target_week) & (scheme.team == t)])
    ]
    for row in scheme_rows:                       # strip the sch_ prefix to match the actuals
        for m in tracker.SCHEME_TRACK:
            if f"sch_{m}" in row:
                row[m] = row.pop(f"sch_{m}")

    locked = tracker.lock_week(hist, up, players, scheme_rows, cur, target_week)
    graded = tracker.grade(hist, sched, plyr, sf_per_game(team, sched))
    tracker.prune_players(hist, cur)
    tracker.save(hist, hist_path)
    log(f"tracker: {locked} new predictions locked, {graded} newly graded, "
        f"{len(hist['games'])} games on file")


    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "season": cur, "week": target_week,
        "games": games_out,
        "players": players,
        "feature_importance": [{"feature": k, "importance": float(v),
                                "label": explain.FEATURE_INFO.get(k, k)} for k, v in imp.items()],
        "backtest": load_backtest(a.datadir),
        "props_meta": props_meta,
        "prop_audit": load_prop_audit(a.datadir),
        # the fitted spread of each projection, so the page can draw an expected range
        # around every number from the same distribution the prop probabilities use
        "prop_range": prop_value.range_table(prop_model) if prop_model.ok else None,
        "live": live,
        "scheme_league": {x: float(sch_lg[x]) for x in SCHEME},
        "tracker": tracker.summarize(hist, cur),
    }
    # ---- live layer: attach the live context to every game and record a prediction version.
    # The engine above never read anything from the live store; this only annotates and
    # records what it produced. See live/README.md.
    try:
        lstate = live_store.hydrate(live_store.load_state(), cur, target_week, [g["game_id"] for g in games_out])
        mv = live_versions.model_version(XGB_PARAMS, FEATS, BLEND_W, N_SEEDS, f"{cur}-w{target_week}")
        req_path = os.path.join(live_store.ROOT, "refresh_request.json")
        reason = "Scheduled refresh: upstream data changed"
        if os.path.exists(req_path):
            try:
                reason = "Live refresh: " + "; ".join(json.load(open(req_path)).get("reasons", [])[:3])
            except Exception:
                pass
        live_versions.record(payload, mv, data_version=payload["generated"], injury_snapshot_id=lstate.get("snapshot_id"),
                             weather_snapshot_id=(lstate.get("last_sync") or {}).get("weather"), default_reason=reason, log=log)
        if os.path.exists(req_path):
            os.remove(req_path)
        for g in games_out:                      # after recording, so the page sees the new version
            g["live"] = live_context.game_context(g, lstate)
        payload["live_meta"] = live_context.live_meta(lstate, {"run": "full", "at": payload["generated"]})
    except Exception as e:                       # the live layer must never stop a publish
        log(f"  live layer skipped: {e}")

    # Full float repr costs ~40% of the payload for digits nothing renders. Four places is
    # more than any display uses and still exact enough for the charts.
    def trim(o):
        if isinstance(o, float):
            return None if (np.isnan(o) or np.isinf(o)) else round(o, 4)
        if isinstance(o, dict):
            return {k: trim(v) for k, v in o.items()}
        if isinstance(o, list):
            return [trim(v) for v in o]
        return o

    out = os.path.join(a.outdir, "payload.json")
    json.dump(trim(payload), open(out, "w"), separators=(",", ":"), default=str)
    log(f"wrote {out}: {len(games_out)} games, {len(players)} players"
        + (f", live {live['n']} scored" if live else ""))


if __name__ == "__main__":
    main()
