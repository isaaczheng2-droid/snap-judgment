"""
NBA pipeline: data -> models -> projections for the upcoming slate -> nba_payload.json.

Run on the Actions runner after nba/collect.py (which adds runner-only sources: rosters,
injury report, lines). Runs fine without any of those: every block the page shows says where
its inputs came from and when, and says "unavailable" rather than guessing.

    python3 nba/run.py --out repo/nba_payload.json [--days 3] [--now 2026-10-20T12:00Z]

Forecast archive: every projection written to the page is also appended to
nba/data/forecasts.ndjson with forecast_at and the available-at time of each input, so a
finished game is graded on the last forecast made before tipoff, never on a re-run.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from nba import data, team_model, player_model, props, fantasy, grades, status  # noqa: E402

DATA = os.path.join(HERE, "data")
ARCHIVE = os.path.join(DATA, "forecasts.ndjson")
LEDGER = os.path.join(DATA, "ledger.json")
SCHEMA = "nba.v1"


def log(m):
    print(f"[nba] {m}", flush=True)


def now_iso(t=None):
    return (t or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _f(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


# ----------------------------------------------------------------------------- availability inputs
def load_injuries():
    """Latest official injury-report snapshot written by collect.py, or None."""
    p = os.path.join(DATA, "injury_snapshots.ndjson")
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    if not rows:
        return None
    latest = max(r["fetched_at"] for r in rows)
    # one snapshot may hold both ESPN's feed and the official report; the official report is
    # ordered last so it wins when both name a player
    cur = sorted([r for r in rows if r["fetched_at"] == latest], key=lambda r: 0 if r.get("source") == "espn" else 1)
    return {"fetched_at": latest, "report_time": cur[0].get("report_time"), "rows": cur, "sources": sorted({r.get("source") or "official" for r in cur})}


def load_rosters():
    """Current rosters from stats.nba.com (collect.py) keyed by team abbreviation, or None."""
    p = os.path.join(DATA, "rosters_current.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


STATUS_P = {"out": 0.0, "doubtful": 0.25, "questionable": 0.5, "day-to-day": 0.5, "probable": 0.85, "available": 1.0}


# ----------------------------------------------------------------------------- slate
def upcoming(G, now, days):
    g = G[(G.tipoff_utc >= now - timedelta(hours=6)) & (G.tipoff_utc <= now + timedelta(days=days)) & ~G.final]
    if g.empty:                                                     # off-season: show the first slate of the schedule
        g = G[(G.tipoff_utc > now) & ~G.final].sort_values("tipoff_utc")
        if not g.empty:
            first = g.tipoff_utc.iloc[0].normalize()
            g = g[g.tipoff_utc < first + timedelta(days=days)]
    return g.sort_values("tipoff_utc")


def future_rows(P, T, slate, injuries, rosters):
    """Synthetic player-game rows for the slate: each team's last-known roster (or the collected
    current roster), with availability from the injury report when present."""
    rows = []
    last_by_team = {}
    for tid, grp in P.sort_values("tipoff_utc").groupby("team_id"):
        # union of the team's last 5 game rosters (a single playoff box score lists only 12-13)
        gids = grp.game_id.drop_duplicates().tail(5)
        lb = grp[grp.game_id.isin(gids)].sort_values("tipoff_utc").drop_duplicates("player_id", keep="last")
        last_by_team[tid] = lb
    inj_map = {}
    if injuries:
        for r in injuries["rows"]:
            if r.get("player"):
                inj_map[(props.norm_name(r["player"]), r.get("team"))] = r
    for g in slate.itertuples():
        for side, tid, abbr, opp_id, opp in (("home", g.home_id, g.home, g.away_id, g.away), ("away", g.away_id, g.away, g.home_id, g.home)):
            base = None
            src = None
            if rosters and abbr in rosters:
                base = pd.DataFrame(rosters[abbr]["players"])
                src = {"source": f"{rosters[abbr].get('source', 'collected')} roster", "asof": rosters[abbr]["fetched_at"]}
            elif tid in last_by_team:
                lb = last_by_team[tid]
                base = lb[["player_id", "player", "position", "jersey", "headshot"]].copy()
                src = {"source": f"last five box scores (through {lb.game_date.max()})", "asof": lb.tipoff_utc.max().strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "note": "offseason moves not reflected until current rosters are collected"}
            if base is None or base.empty:
                continue
            for r in base.itertuples():
                st = inj_map.get((props.norm_name(r.player), abbr)) or inj_map.get((props.norm_name(r.player), None))
                status_txt = ((st.get("status") or "unknown").lower() if st else ("available" if injuries else "unknown"))
                if status_txt not in STATUS_P and status_txt != "unknown":
                    status_txt = "questionable"
                rows.append({"game_id": g.game_id, "season": g.season, "player_id": str(r.player_id), "player": r.player, "team_id": tid, "team": abbr,
                             "opp_id": opp_id, "opp": opp, "home": side == "home", "position": r.position, "jersey": getattr(r, "jersey", None),
                             "starter": False, "played": status_txt != "out", "dnp": False, "active": True, "reason": (st or {}).get("reason"),
                             "ejected": False, "min": np.nan, "headshot": getattr(r, "headshot", None), "game_date": g.tipoff_utc.strftime("%Y-%m-%d"),
                             "team_pts": np.nan, "opp_pts": np.nan, "plus_minus": np.nan, "tipoff_utc": g.tipoff_utc, "phase": g.phase, "periods": np.nan,
                             "avail_status": status_txt, "avail_source": (st or {}).get("source") or "no injury report", "roster_source": src["source"], "roster_asof": src["asof"],
                             "roster_note": src.get("note"), **{k: np.nan for k in ("fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "pts")}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main build
def build(out_path, days=3, now=None, sims=3000, refresh=False):
    now = now or datetime.now(timezone.utc)
    t0 = now_iso(now)
    log(f"building at {t0}")
    d = data.build(2022, 2027, refresh=refresh) if refresh or not os.path.exists(os.path.join(DATA, "games.parquet")) else data.load()
    manifest = json.load(open(os.path.join(DATA, "manifest.json")))
    G, T, P = d["games"], d["team_games"], d["player_games"]
    cur_season = int(G[G.tipoff_utc > now].season.min()) if (G.tipoff_utc > now).any() else int(G.season.max())
    last_done = int(G[G.final].season.max())
    injuries, rosters = load_injuries(), load_rosters()
    slate = upcoming(G, now, days)
    log(f"season {cur_season}, last finished season {last_done}, slate {len(slate)} games, injuries {'yes' if injuries else 'none'}, rosters {'collected' if rosters else 'last box score'}")

    # ---- team model: features for history + slate (slate rows carry NaN results)
    Tf = T.copy()
    slate_T = []
    for g in slate.itertuples():
        for side, tid, abbr, opp in (("home", g.home_id, g.home, g.away_id), ("away", g.away_id, g.away, g.home_id)):
            slate_T.append({"game_id": g.game_id, "season": g.season, "team_id": tid, "team": abbr, "home": side == "home", "opp_id": opp,
                            "pts": np.nan, "opp_pts": np.nan, "won": False, "tipoff_utc": g.tipoff_utc, "phase": g.phase, "periods": np.nan,
                            "neutral": bool(g.neutral), "poss": np.nan, "poss_game": np.nan, "pace": np.nan, "ortg": np.nan, "drtg": np.nan,
                            **{k: np.nan for k in ("fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf")}})
    if slate_T:
        Tf = pd.concat([Tf, pd.DataFrame(slate_T)], ignore_index=True)
        Tf = Tf.sort_values(["team_id", "tipoff_utc"]).reset_index(drop=True)
        prev = Tf.groupby(["team_id", "season"])["tipoff_utc"].shift(1)
        Tf["rest_days"] = (Tf["tipoff_utc"] - prev).dt.total_seconds() / 86400
        Tf["b2b"] = Tf["rest_days"].between(0.5, 1.5)
        Tf["games_last7"] = Tf.groupby(["team_id", "season"])["tipoff_utc"].transform(
            lambda s: pd.Series([((s > t - pd.Timedelta(days=7)) & (s < t)).sum() for t in s], index=s.index))
    Th = team_model.team_history(Tf)
    gt = team_model.game_table(Th)
    hist = gt[gt.margin.notna()]
    tm = team_model.train_final(hist, last_done)
    blind = tm["lineup_blind"]
    gs = gt[gt.game_id.isin(slate.game_id)].copy()
    gp = team_model.predict(blind, gs)
    gs["p_home"], gs["margin_pred"], gs["total_pred"] = gp.p_home, gp.margin_pred, gp.total_pred
    backtest_team = json.load(open(os.path.join(DATA, "team_backtest.json"))) if os.path.exists(os.path.join(DATA, "team_backtest.json")) else None

    # ---- player projections for the slate
    fut = future_rows(P, T, slate, injuries, rosters)
    players_out, lineups = [], {}
    if not fut.empty:
        # two finished seasons of history are enough to fit the minutes model and give every
        # returning player a trailing record; keeps the daily run to about a minute
        Pall = pd.concat([P[P.season >= last_done - 1], fut], ignore_index=True)
        gfe = gt[["game_id", "sum_pace"]].copy()
        gfe["exp_pace"] = gfe.sum_pace / 2
        gfe["margin_pred_pre"] = np.nan
        gfe = gfe.set_index("game_id")
        gfe.loc[gs.game_id, "margin_pred_pre"] = gs.set_index("game_id").margin_pred
        # historical expected margins for the minutes model's training rows: the team model fitted
        # on the seasons before each one (walk-forward, no look-ahead), computed here so the runner
        # needs no cached features
        for s_ in sorted(gt.season.unique()):
            if s_ < last_done - 1 or s_ > last_done:
                continue
            tr_ = hist[hist.season < s_]
            if len(tr_) < 500:
                continue
            m_ = team_model.fit(tr_, team_model.BLIND)
            idx_ = gt.season == s_
            pred_ = team_model.predict(m_, gt[idx_])
            gfe.loc[gt.loc[idx_, "game_id"].values, "margin_pred_pre"] = pred_.margin_pred.values
        Pf = player_model.player_features({"player_games": Pall, "team_games": Tf}, gfe.reset_index())
        train = Pf[(Pf.season <= last_done) & Pf.min_ewm5.notna()]
        mm = player_model.fit_minutes(train[train.abs_margin.notna()])
        F = Pf[Pf.game_id.isin(slate.game_id)].copy()
        F["playing"] = F.played
        F = player_model.predict_minutes(mm, F, weight_by_p_play=True)
        F = player_model.stat_means(F)
        # availability: injury status caps P(play); no report -> the model's own P(play)
        F["p_play_model"] = F.p_play
        F["p_play"] = [min(pp if pd.notna(pp) else 0.5, STATUS_P.get(s, 1.0)) if s != "unknown" else (pp if pd.notna(pp) else 0.5)
                       for pp, s in zip(F.p_play, F.avail_status)]
        F.loc[F.avail_status == "out", ["p_play"]] = 0.0
        rng = np.random.default_rng(11)
        lines_latest = props.latest_by_key(props.read_quotes()) if os.path.exists(props.QUOTES) else {}
        by_player_lines = {}
        for k, q in lines_latest.items():
            by_player_lines.setdefault(props.norm_name(q["player"]), []).append(q)
        prop_rows = []
        for _, r in F.iterrows():
            if pd.isna(r.min_mu):
                # no usable game history (rookie, long absence, or unmatched id): listed, not projected
                players_out.append({"game_id": int(r.game_id), "player_id": r.player_id, "player": r.player, "team": r.team, "team_id": int(r.team_id), "opp": r.opp,
                                    "home": bool(r.home), "position": r.position, "jersey": r.jersey, "headshot": r.headshot,
                                    "availability": {"status": r.avail_status, "source": r.avail_source, "reason": r.reason, "p_play": None, "p_play_model": None},
                                    "p_start": None, "minutes": {"mean": None, "sd": None, "p10": None, "p90": None, "trailing5": None, "trailing15": None, "reconcile_scale": None},
                                    "conditional": {}, "availability_adjusted": {}, "fantasy": {"points": {}, "categories": {}}, "inputs": {},
                                    "unavailable": "no NBA game history in the loaded seasons: not projected", "model_version": player_model.MODEL_VERSION})
                continue
            sim = player_model.simulate(r, n=sims, rng=rng)
            # market lines for this player, if any: P(over) at each quoted line
            my_lines = by_player_lines.get(props.norm_name(r.player), [])
            line_map = {}
            for q in my_lines:
                st = props.MARKETS.get(q["market"])
                if st and st in sim:
                    line_map.setdefault(st, set()).add(float(q["line"]))
            summ = player_model.summarize(sim, p_play=r.p_play)
            fpts = {k: _f(fantasy.score_points(sim, f).mean(), 1) for k, f in fantasy.POINTS_FORMATS.items()}
            cats = fantasy.category_line(sim)
            row = {"game_id": int(r.game_id), "player_id": r.player_id, "player": r.player, "team": r.team, "team_id": int(r.team_id), "opp": r.opp,
                   "home": bool(r.home), "position": r.position, "jersey": r.jersey, "headshot": r.headshot,
                   "availability": {"status": r.avail_status, "source": r.avail_source, "reason": r.reason, "p_play": _f(r.p_play), "p_play_model": _f(r.p_play_model)},
                   "p_start": _f(r.p_start), "minutes": {"mean": _f(r.min_mu, 1), "sd": _f(r.min_sd, 1), "p10": _f(np.percentile(sim["min"], 10), 1), "p90": _f(np.percentile(sim["min"], 90), 1),
                                                          "trailing5": _f(r.min_ewm5, 1), "trailing15": _f(r.min_ewm15, 1), "reconcile_scale": _f(r.get("min_scale"), 3)},
                   "conditional": {k: {kk: _f(vv, 1) for kk, vv in v.items()} for k, v in summ.items() if isinstance(v, dict) and k != "min"},
                   "availability_adjusted": {k: _f(v["mean"] * r.p_play, 1) for k, v in summ.items() if isinstance(v, dict) and k != "min" and "mean" in v},
                   "fantasy": {"points": fpts, "categories": {k: _f(v, 3) for k, v in cats.items()}},
                   "inputs": {"opponent_factor": {k: _f(r[f"of_{k}"], 2) for k in ("fga", "fta", "oreb", "dreb", "ast")},
                              "abs_margin_pred": _f(r.abs_margin, 1), "absent_share": _f(r.absent_share, 3), "b2b": bool(r.b2b), "rest_days": _f(r.rest, 1)},
                   "model_version": player_model.MODEL_VERSION}
            # P(over) at quoted lines and prop evaluations
            if line_map:
                row["p_over"] = {}
                for st, ls in line_map.items():
                    row["p_over"][st] = {str(L): {"p_over": _f((sim[st] > L).mean()), "p_push": _f((sim[st] == L).mean())} for L in sorted(ls)}
                for q in my_lines:
                    st = props.MARKETS.get(q["market"])
                    if not st or st not in sim or q["side"] != "over":
                        continue
                    other = lines_latest.get((q["book"], q["market"], props.norm_name(q["player"]), float(q["line"]), "under"))
                    L = float(q["line"])
                    ss = dict(summ[st]); ss["p_over"] = float((sim[st] > L).mean()); ss["p_push"] = float((sim[st] == L).mean())
                    flags = []
                    if not (P.season == cur_season).any():
                        flags.append("preseason: no current-season games")   # forced to insufficient evidence below
                    if st in ("stl", "blk", "sb"):
                        flags.append("no baseline win")          # last-10 median beats the model's mean on these (see Results)
                    if r.n_prior < 10:
                        flags.append("small sample")
                    if r.b2b:
                        flags.append("back-to-back")
                    if r.avail_status in ("questionable", "doubtful"):
                        flags.append(f"listed {r.avail_status}")
                    if r.absent_share > 0.15:
                        flags.append("key teammates absent: usage estimate less certain")
                    ev_ = props.evaluate(q, other, ss, r.p_play, flags=flags)
                    if "preseason: no current-season games" in flags:
                        ev_["verdict"], ev_["why"] = "insufficient evidence", "preseason: rates and minutes come from last season and new rosters; graded but not recommended until games are played"
                    ev_.update({"game_id": int(r.game_id), "team": r.team, "player_id": r.player_id})
                    prop_rows.append(ev_)
            players_out.append(row)
        # lineups per team: projected starters = top-5 P(start) among available; labelled estimate
        for (gid, tid), grp in F.groupby(["game_id", "team_id"]):
            avail = grp[grp.p_play > 0].sort_values("p_start", ascending=False)
            starters = avail.head(5)
            lineups[f"{int(gid)}:{int(tid)}"] = {
                "starters": [{"player_id": x.player_id, "player": x.player, "position": x.position, "p_start": _f(x.p_start), "minutes": _f(x.min_mu, 1)} for x in starters.itertuples()],
                "bench": [{"player_id": x.player_id, "player": x.player, "position": x.position, "minutes": _f(x.min_mu, 1), "p_play": _f(x.p_play)} for x in avail.iloc[5:].itertuples()],
                "out": [{"player_id": x.player_id, "player": x.player, "reason": x.reason} for x in grp[grp.p_play == 0].itertuples()],
                "label": "projected closing lineup (estimate)", "basis": "P(start) from trailing start share and minutes; confirmed starters arrive ~30 min before tipoff and are not collected yet",
                "source": grp.roster_source.iloc[0], "asof": grp.roster_asof.iloc[0], "note": grp.roster_note.iloc[0]}
    else:
        prop_rows = []

    # ---- games block
    games = []
    fut_by_game = {}
    if players_out:
        for p in players_out:
            fut_by_game.setdefault(p["game_id"], []).append(p)
    for g in gs.itertuples():
        row = {"game_id": int(g.game_id), "season": int(g.season), "phase": g.phase, "tipoff_utc": g.tipoff_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "home": {"id": int(g.home_id), "abbr": g.home_team}, "away": {"id": int(g.away_id), "abbr": g.away_team}, "neutral": bool(g.neutral),
               "p_home": _f(g.p_home), "margin_pred": _f(g.margin_pred, 1), "total_pred": _f(g.total_pred, 1),
               "home_pts_pred": _f((g.total_pred + g.margin_pred) / 2, 1), "away_pts_pred": _f((g.total_pred - g.margin_pred) / 2, 1),
               "basis": "lineup-blind team model (team strength, rest, schedule, home court); availability enters the player projections, not the game number",
               "model_version": tm["model_version"], "trained_through": tm["trained_through"], "forecast_at": t0,
               "factors": {"net_rating_diff": _f(g.d_net, 1), "off_rating_diff": _f(g.d_off, 1), "def_rating_diff": _f(g.d_def, 1), "elo_diff": _f(g.d_elo * 100, 0),
                           "rest_diff": _f(g.d_rest, 1), "b2b_diff": _f(g.d_b2b, 0), "expected_pace": _f(g.sum_pace / 2, 1), "sos_diff": _f(g.d_sos, 1)},
               "lineups": {"home": lineups.get(f"{int(g.game_id)}:{int(g.home_id)}"), "away": lineups.get(f"{int(g.game_id)}:{int(g.away_id)}")},
               "market": None, "scenarios": {"overtime": "not folded into the mean; OT adds ~25 team minutes per period",
                                              "blowout": f"|expected margin| {abs(float(g.margin_pred)):.1f}: {'starters likely rest late' if abs(float(g.margin_pred)) > 12 else 'competitive expectation'}"}}
        games.append(row)
    # market lines per game (latest fetch), if collected
    gl_path = os.path.join(DATA, "game_lines.ndjson")
    if os.path.exists(gl_path):
        gl = [json.loads(l) for l in open(gl_path) if l.strip()]
        latest = max((r["fetched_at"] for r in gl), default=None)
        cur = [r for r in gl if r["fetched_at"] == latest]
        for row in games:
            hits = [r for r in cur if r.get("home") and r["home"].split()[-1].upper()[:3] == row["home"]["abbr"][:3]]
            if hits:
                row["market"] = {"fetched_at": latest, "quotes": [{k: r.get(k) for k in ("book", "market", "name", "point", "price", "quoted_at")} for r in hits]}

    # ---- grades (current season if it has games, else last finished season, labelled)
    gseason = cur_season if (P.season == cur_season).any() else last_done
    Pg = P.merge(G[["game_id"]], on="game_id")
    grade_rows = grades.compute(Pg, T, gseason)
    gval = json.load(open(os.path.join(DATA, "grade_validation.json"))) if os.path.exists(os.path.join(DATA, "grade_validation.json")) else None

    # ---- fantasy rankings (season-long view from the projections when a slate exists, else last-season per-game)
    fant = {"formats": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in fantasy.POINTS_FORMATS.items()},
            "categories": {"9cat": fantasy.CATEGORIES_9, "8cat": fantasy.CATEGORIES_8}, "note": "basketball scoring only; NFL PPR settings are never applied here"}
    if players_out and any(not p.get("unavailable") for p in players_out):
        pool = [{"player_id": p["player_id"], "player": p["player"], "team": p["team"], "position": p["position"], "game_id": p["game_id"],
                 "points": p["fantasy"]["points"], "cats": p["fantasy"]["categories"], "p_play": p["availability"]["p_play"], "minutes": p["minutes"]["mean"]}
                for p in players_out if not p.get("unavailable")]
        z = fantasy.z_scores([{**c["cats"]} for c in pool])
        for c, zz in zip(pool, z):
            c["z9"] = _f(zz["z_total"], 2)
            c["z"] = {k: _f(zz[f"z_{k}"], 2) for k in fantasy.CATEGORIES_9}
        fant["slate"] = sorted(pool, key=lambda c: -(c["points"].get("espn_points") or 0))
    # streaming: games per team over the next 7 days from the schedule
    wk = G[(G.tipoff_utc > now) & (G.tipoff_utc <= now + timedelta(days=7))]
    fant["games_next7"] = {**wk.groupby("home").size().to_dict()}
    for k, v in wk.groupby("away").size().to_dict().items():
        fant["games_next7"][k] = fant["games_next7"].get(k, 0) + v

    # ---- rosters block (all 30 teams): last-known or collected
    rost = {}
    Pl = P.sort_values("tipoff_utc")
    grade_by = {r["player_id"]: r for r in grade_rows}
    for tid, grp in Pl.groupby("team_id"):
        abbr = grp.team.iloc[-1]
        last_gid = grp.game_id.iloc[-1]
        lb = grp[grp.game_id == last_gid]
        if rosters and abbr in rosters:
            plist = rosters[abbr]["players"]; src = {"source": f"{rosters[abbr].get('source', 'collected')} roster", "asof": rosters[abbr]["fetched_at"]}
        else:
            plist = [{"player_id": x.player_id, "player": x.player, "position": x.position, "jersey": x.jersey, "headshot": x.headshot, "starter": bool(x.starter)} for x in lb.itertuples()]
            src = {"source": f"last box score {lb.game_date.iloc[0]}", "asof": lb.tipoff_utc.iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ"), "note": "offseason moves not reflected until current rosters are collected"}
        for p in plist:
            gr = grade_by.get(str(p["player_id"]))
            p["grade"] = None if not gr else gr["grade"]
            p["grade_label"] = None if not gr else gr["label"]
            p["player_id"] = str(p["player_id"])
        rost[abbr] = {"team_id": int(tid), "players": plist, **src}

    # ---- results: backtests + forward ledger
    ledger = grade_ledger(G, now)
    results = {"team_backtest": backtest_team, "player_backtest": json.load(open(os.path.join(DATA, "player_backtest.json"))) if os.path.exists(os.path.join(DATA, "player_backtest.json")) else None,
               "grade_validation": gval, "forward": ledger,
               "market_note": "No historical NBA lines are held, so no real-market backtest is claimed. Lines are collected forward from the first fetch; the market comparison fills in as games settle."}

    # ---- status
    st = status.table()
    payload = {"schema": SCHEMA, "generated_at": t0, "season": cur_season, "phase": "preseason" if not (P.season == cur_season).any() else "in season",
               "season_label": f"{cur_season - 1}-{str(cur_season)[2:]}", "slate_days": days,
               "notes": preseason_notes(cur_season, injuries, rosters, slate),
               "games": games, "players": players_out, "props": prop_rows,
               "players_meta": {"rates_from": "EWMA of prior games (half-life 10 games), shrunk to the position prior for small samples, adjusted by the opponent's trailing allowed rates and expected pace",
                                "minutes_from": "gradient-boosted minutes model (trailing minutes, start share, rest, schedule, expected margin, absent teammates), reconciled to 240 team minutes",
                                "simulation": "attempts ~ Poisson, makes ~ Binomial, shared minutes and pace draws; combos and fantasy computed per simulation",
                                "conditional": "given the player plays", "availability_adjusted": "conditional mean x P(play); a DNP voids a prop but scores zero in fantasy",
                                "lineups_note": "court placement is illustrative; positions are ESPN's listed positions"},
               "grades": {"rows": grade_rows, "season": gseason, "season_label": f"{gseason - 1}-{str(gseason)[2:]}",
                          "timeframe": ("current season to date" if gseason == cur_season else f"{gseason - 1}-{str(gseason)[2:]} full season (last finished season; the new season has no games yet)"),
                          "updated_at": t0, "meta": grades.META, "methodology_version": grades.METHOD_VERSION, "validation": gval,
                          "definition": "Snap Grade is a percentile rank among the comparison group of a weighted, shrunk box-production summary; descriptive, not a probability"},
               "fantasy": fant, "rosters": rost, "results": results,
               "data_status": {"sources": st, "manifest": manifest, "injury_report": None if not injuries else {"fetched_at": injuries["fetched_at"], "rows": len(injuries["rows"]), "sources": injuries.get("sources")},
                               "rosters": "collected" if rosters else "last box score per team", "lines": prop_line_status(),
                               "models": {"team": {"version": tm["model_version"], "trained_through": tm["trained_through"], "features": team_model.BLIND},
                                          "player": {"version": player_model.MODEL_VERSION}}}}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = _clean(payload)
    # allow_nan=False: a NaN would be written as a bare token that browsers refuse to parse,
    # taking the whole section down; a missing value is null, never NaN and never zero
    json.dump(payload, open(out_path, "w"), separators=(",", ":"), default=_json_default, allow_nan=False)
    archive(payload, manifest, injuries)
    log(f"wrote {out_path}: {len(games)} games, {len(players_out)} player projections, {len(prop_rows)} prop evaluations, {len(grade_rows)} grades, {os.path.getsize(out_path) // 1024} KB")
    return payload


def _clean(o):
    """Recursively turn NaN/NaT/numpy scalars into JSON-safe values (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return None if pd.isna(o) else o.strftime("%Y-%m-%dT%H:%M:%SZ")
    if o is pd.NaT:
        return None
    return o


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def preseason_notes(cur_season, injuries, rosters, slate):
    n = []
    if not injuries:
        n.append("No injury feed has been collected yet. Availability is the model's own P(play) from last season's pattern, labelled 'no injury report'.")
    if not rosters:
        n.append("Rosters are the union of each team's last five box scores of 2025-26; offseason trades and signings are not reflected until the runner collects current rosters (ESPN site API).")
    if not os.path.exists(props.QUOTES):
        n.append("No lines collected yet. Prop evaluations appear once the runner's Odds API fetch returns NBA player props; there is no historical NBA line archive here.")
    if not slate.empty:
        n.append(f"Slate shown: {slate.tipoff_utc.min().strftime('%Y-%m-%d')} to {slate.tipoff_utc.max().strftime('%Y-%m-%d')} UTC, {len(slate)} games.")
    return n


def prop_line_status():
    if not os.path.exists(props.QUOTES):
        return {"quotes": 0, "latest": None}
    q = props.read_quotes()
    return {"quotes": len(q), "latest": max((r.get("fetched_at") or "" for r in q), default=None), "books": sorted({r["book"] for r in q})}


# ----------------------------------------------------------------------------- archive + ledger
def archive(payload, manifest, injuries):
    os.makedirs(DATA, exist_ok=True)
    avail = {"boxscores": manifest.get("ingested_at"), "injury_report": injuries["fetched_at"] if injuries else None, "lines": prop_line_status().get("latest")}
    with open(ARCHIVE, "a") as f:
        for g in payload["games"]:
            f.write(json.dumps({"kind": "game", "game_id": g["game_id"], "tipoff_utc": g["tipoff_utc"], "forecast_at": payload["generated_at"], "inputs_available_at": avail,
                                "model_version": g["model_version"], "p_home": g["p_home"], "margin_pred": g["margin_pred"], "total_pred": g["total_pred"], "basis": g["basis"]}) + "\n")
        for p in payload["players"]:
            if p.get("unavailable"):
                continue
            c = p["conditional"]
            f.write(json.dumps({"kind": "player", "game_id": p["game_id"], "player_id": p["player_id"], "forecast_at": payload["generated_at"], "inputs_available_at": avail,
                                "model_version": p["model_version"], "p_play": p["availability"]["p_play"], "min": p["minutes"]["mean"],
                                "pts": c["pts"]["mean"], "reb": c["reb"]["mean"], "ast": c["ast"]["mean"], "fg3m": c["fg3m"]["mean"], "pra": c["pra"]["mean"],
                                "pts_med": c["pts"]["median"], "pts_p10": c["pts"]["p10"], "pts_p90": c["pts"]["p90"]}) + "\n")
        for q in payload["props"]:
            f.write(json.dumps({"kind": "prop", "game_id": q.get("game_id"), "player_id": q.get("player_id"), "forecast_at": payload["generated_at"], "book": q["book"], "market": q["market"],
                                "line": q["line"], "p_over": q.get("p_over"), "sides": q.get("sides"), "verdict": q.get("verdict"), "quoted_at": q.get("quoted_at")}) + "\n")


def grade_ledger(G, now):
    """Grade every finished game/player-game on the LAST archived forecast before tipoff."""
    if not os.path.exists(ARCHIVE):
        return {"games": {"n": 0}, "players": {"n": 0}, "props": {"n": 0}, "note": "forward tracking starts with the first published slate"}
    rows = [json.loads(l) for l in open(ARCHIVE) if l.strip()]
    fin = G[G.final].set_index("game_id")
    best = {}
    for r in rows:
        gid = r["game_id"]
        if gid not in fin.index:
            continue
        tip = fin.loc[gid, "tipoff_utc"].strftime("%Y-%m-%dT%H:%M:%SZ")
        if r["forecast_at"] > tip:
            continue                                  # made after tipoff: never graded
        key = (r["kind"], gid, r.get("player_id"), r.get("book"), r.get("market"), r.get("line"))
        if key not in best or r["forecast_at"] > best[key]["forecast_at"]:
            best[key] = r
    gm = [r for r in best.values() if r["kind"] == "game"]
    out = {"games": {"n": len(gm)}, "players": {"n": 0}, "props": {"n": 0}}
    if gm:
        y = np.array([float(fin.loc[r["game_id"], "home_score"] > fin.loc[r["game_id"], "away_score"]) for r in gm])
        p = np.array([r["p_home"] for r in gm], dtype=float)
        mg = np.array([fin.loc[r["game_id"], "home_score"] - fin.loc[r["game_id"], "away_score"] for r in gm], dtype=float)
        out["games"].update({"su": _f(((p > 0.5) == (y > 0.5)).mean()), "brier": _f(((p - y) ** 2).mean()),
                             "margin_mae": _f(np.abs(mg - np.array([r["margin_pred"] for r in gm], dtype=float)).mean(), 2),
                             "rows": [{"game_id": r["game_id"], "p_home": r["p_home"], "margin_pred": r["margin_pred"], "forecast_at": r["forecast_at"],
                                       "home_score": int(fin.loc[r["game_id"], "home_score"]), "away_score": int(fin.loc[r["game_id"], "away_score"]),
                                       "correct": bool((r["p_home"] > 0.5) == (fin.loc[r["game_id"], "home_score"] > fin.loc[r["game_id"], "away_score"]))} for r in gm][-200:]})
    pl = [r for r in best.values() if r["kind"] == "player"]
    if pl:
        pg = data.load()["player_games"].set_index(["game_id", "player_id"])
        errs = []
        for r in pl:
            k = (r["game_id"], r["player_id"])
            if k in pg.index and bool(pg.loc[k, "played"]):
                a = pg.loc[k]
                errs.append({"pts": abs(a.pts - r["pts"]), "reb": abs(a.reb - r["reb"]), "ast": abs(a.ast - r["ast"]), "min": abs(a["min"] - r["min"]),
                             "in80": bool(r["pts_p10"] <= a.pts <= r["pts_p90"]), "played": True})
            elif k in pg.index:
                errs.append({"played": False, "p_play": r["p_play"]})
        played = [e for e in errs if e["played"]]
        out["players"] = {"n": len(errs), "n_played": len(played),
                          "mae": {k: _f(np.mean([e[k] for e in played]), 2) for k in ("pts", "reb", "ast", "min")} if played else None,
                          "pts_coverage_80": _f(np.mean([e["in80"] for e in played])) if played else None}
    pr = [r for r in best.values() if r["kind"] == "prop"]
    out["props"] = {"n": len(pr), "note": "graded against the actual stat once box scores land; EV realised only on real quotes"}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "repo", "nba_payload.json"))
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--now", default=None)
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    now = datetime.strptime(a.now, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc) if a.now else None
    build(a.out, days=a.days, now=now, sims=a.sims, refresh=a.refresh)
