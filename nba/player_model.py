"""
NBA player projections: minutes first, production second, simulation for the distributions.

Minutes  P(play), P(start) and the minutes distribution are modelled separately from production.
         Minutes are predicted per player from trailing usage, role, rest, schedule, the team's
         expected game competitiveness (|predicted margin| from the team model), and the share of
         the team's usual minutes that is unavailable. Team minutes are reconciled to 240
         (regulation); overtime is a scenario on top, never folded into the mean.
Rates    Per-minute rates for attempts and non-shooting stats (fga, fg3a share, fta, reb, ast,
         stl, blk, tov) and conversion rates (fg2%, fg3%, ft%) from an exponentially weighted
         history, shrunk toward the position prior for small samples, then adjusted for the
         opponent's trailing allowed rates and the game's expected pace.
Sim      Points come from simulated shooting (attempts ~ Poisson, makes ~ Binomial), so a
         player's points, 3PM and FGA are internally consistent, and combos (PRA, P+R, P+A, R+A,
         fantasy) are computed per simulation so their correlation is preserved. Minutes and
         pace are shared draws within a simulation.
Output   Both a conditional-on-playing projection and an availability-adjusted one
         (P(play) x conditional), each with mean, median, intervals and P(over x) for any line.

Walk-forward evaluation is in evaluate(): validation 2024-25, untouched test 2025-26, against
last-10 mean, last-10 median and trailing-minutes x trailing-rate baselines. No historical
prop lines are held, so no P(over) grading against the market is claimed here; see props.py.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_VERSION = "nba_player_v1"
RATES = ["fga", "fg3a", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov"]
HL = 10                # half-life in games for per-minute rate EWMAs
SHRINK_N = 8           # games of prior weight toward the position mean
MIN_FEATS = ["min_ewm5", "min_ewm15", "min_last", "min_sd10", "start_rate10", "play_rate10", "gap_days",
             "absent_share", "abs_margin", "b2b", "rest", "last7", "season_frac", "team_min_ewm_sum", "is_home"]
PLAY_FEATS = ["play_rate10", "played_last", "gap_days", "min_ewm15", "b2b", "rest", "season_frac", "n_prior"]


# ----------------------------------------------------------------------------- features
def player_features(d, game_feats):
    """Per player-game rows with pre-game trailing features. `game_feats` is the team-model game
    table (for expected margin/pace and rest). Historical availability comes from the box score."""
    P = d["player_games"].sort_values(["player_id", "tipoff_utc"]).copy()
    P["season_frac"] = P.groupby(["team_id", "season"])["tipoff_utc"].rank(method="dense") / 82
    a5, a15, ar = 1 - 0.5 ** 0.2, 1 - 0.5 ** (1 / 15), 1 - 0.5 ** (1 / HL)
    gp = P.groupby("player_id")
    played_min = P["min"].where(P.played)
    P["min_ewm5"] = played_min.groupby(P.player_id).transform(lambda s: s.ewm(alpha=a5, ignore_na=True).mean().shift(1))
    P["min_ewm15"] = played_min.groupby(P.player_id).transform(lambda s: s.ewm(alpha=a15, ignore_na=True).mean().shift(1))
    P["min_last"] = played_min.groupby(P.player_id).transform(lambda s: s.ffill().shift(1))
    P["min_sd10"] = played_min.groupby(P.player_id).transform(lambda s: s.rolling(10, min_periods=3).std().shift(1))
    P["start_rate10"] = gp["starter"].transform(lambda s: s.astype(float).rolling(10, min_periods=1).mean().shift(1))
    P["play_rate10"] = gp["played"].transform(lambda s: s.astype(float).rolling(10, min_periods=1).mean().shift(1))
    P["played_last"] = gp["played"].transform(lambda s: s.astype(float).shift(1))
    P["n_prior"] = gp["played"].transform(lambda s: s.astype(float).cumsum().shift(1)).fillna(0)
    last_t = gp["tipoff_utc"].shift(1)
    P["gap_days"] = ((P.tipoff_utc - last_t).dt.total_seconds() / 86400).clip(upper=30)
    # per-minute rates (EWMA over played games, prior games only) and conversion
    for k in RATES:
        r = (P[k] / P["min"]).where(P.played)
        P[f"r_{k}"] = r.groupby(P.player_id).transform(lambda s: s.ewm(alpha=ar, ignore_na=True).mean().shift(1))
    for num, den, name in (("fgm", "fga", "fg_pct"), ("fg3m", "fg3a", "fg3_pct"), ("ftm", "fta", "ft_pct")):
        m = P[num].where(P.played).groupby(P.player_id).transform(lambda s: s.ewm(alpha=ar, ignore_na=True).mean().shift(1))
        a = P[den].where(P.played).groupby(P.player_id).transform(lambda s: s.ewm(alpha=ar, ignore_na=True).mean().shift(1))
        P[f"{name}_num"], P[f"{name}_den"] = m, a
    # position prior for shrinkage: league per-minute rates by listed position over prior seasons is
    # approximated by the running league mean per position (shifted by season to avoid look-ahead)
    P["pos"] = P.position.fillna("G").str[0]
    pri = P[P.played].groupby(["season", "pos"]).apply(
        lambda g: pd.Series({**{f"pr_{k}": g[k].sum() / g["min"].sum() for k in RATES},
                             "pr_fg_pct": g.fgm.sum() / g.fga.sum(), "pr_fg3_pct": g.fg3m.sum() / max(g.fg3a.sum(), 1),
                             "pr_ft_pct": g.ftm.sum() / max(g.fta.sum(), 1)}), include_groups=False).reset_index()
    pri["season"] += 1                                   # prior season's league figure applies
    P = P.merge(pri, on=["season", "pos"], how="left")
    for k in [f"pr_{k}" for k in RATES] + ["pr_fg_pct", "pr_fg3_pct", "pr_ft_pct"]:
        P[k] = P[k].fillna(P[k].mean())
    w = P.n_prior / (P.n_prior + SHRINK_N)
    for k in RATES:
        P[f"r_{k}"] = w * P[f"r_{k}"].fillna(P[f"pr_{k}"]) + (1 - w) * P[f"pr_{k}"]
    for name in ("fg_pct", "fg3_pct", "ft_pct"):
        den = P[f"{name}_den"].fillna(0)
        obs = (P[f"{name}_num"].fillna(0) + 20 * P[f"pr_{name}"]) / (den + 20)   # 20-attempt prior
        P[name] = obs
    # team context: game features (home perspective) mapped to each side
    gf = game_feats.set_index("game_id")
    P["abs_margin"] = P.game_id.map(gf["margin_pred_pre"].abs()) if "margin_pred_pre" in gf else np.nan
    P["exp_pace"] = P.game_id.map(gf["exp_pace"]) if "exp_pace" in gf else np.nan
    tg = d["team_games"].set_index(["game_id", "team_id"])
    P["rest"] = [tg.rest_days.get((g, t), np.nan) for g, t in zip(P.game_id, P.team_id)]
    P["b2b"] = [float(tg.b2b.get((g, t), False)) for g, t in zip(P.game_id, P.team_id)]
    P["last7"] = [tg.games_last7.get((g, t), np.nan) for g, t in zip(P.game_id, P.team_id)]
    P["rest"] = P["rest"].clip(upper=5)
    P["is_home"] = P.home.astype(float)
    # absent share: usual minutes (min_ewm15) of the team's roster for this game that did not play
    P["usual"] = P.min_ewm15.fillna(0)
    team_usual = P.groupby(["game_id", "team_id"])["usual"].transform("sum")
    absent = P["usual"].where(~P.played, 0).groupby([P.game_id, P.team_id]).transform("sum")
    P["absent_share"] = (absent / team_usual.replace(0, np.nan)).fillna(0)
    P["team_min_ewm_sum"] = P["usual"].where(P.played, 0).groupby([P.game_id, P.team_id]).transform("sum")
    # opponent allowed factor per stat: opponent's trailing allowed per-minute rate / league
    P = _opponent_factors(P)
    return P


def _opponent_factors(P):
    """For each team-game, the opponent's EWMA of (stat allowed per 48 team minutes) relative to
    league average, from prior games only."""
    played = P[P.played]
    tot = played.groupby(["game_id", "team_id", "opp_id", "tipoff_utc", "season"])[RATES + ["pts"]].sum().reset_index()
    tot = tot.sort_values("tipoff_utc")
    league = tot.groupby("season")[RATES + ["pts"]].transform("mean")
    a = 1 - 0.5 ** (1 / 12)
    for k in RATES + ["pts"]:
        rel = tot[k] / league[k]
        # allowed BY the opponent = what teams scored against opp_id; index by the defending team
        tot[f"allowed_{k}"] = rel
    allowed = tot.rename(columns={"opp_id": "def_id"}).sort_values("tipoff_utc")
    for k in RATES + ["pts"]:
        allowed[f"of_{k}"] = allowed.groupby(["def_id", "season"])[f"allowed_{k}"].transform(
            lambda s: s.ewm(alpha=a).mean().shift(1))
    key = allowed.set_index(["game_id", "def_id"])[[f"of_{k}" for k in RATES + ["pts"]]]
    idx = pd.MultiIndex.from_arrays([P.game_id, P.opp_id])
    of = key.reindex(idx)
    for k in RATES + ["pts"]:
        P[f"of_{k}"] = of[f"of_{k}"].values
        P[f"of_{k}"] = P[f"of_{k}"].fillna(1.0)
    return P


# ----------------------------------------------------------------------------- minutes
def fit_minutes(P):
    tr = P[P.played & P.min_ewm5.notna()].dropna(subset=["abs_margin"])
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=5, min_samples_leaf=40, random_state=1)
    reg.fit(tr[MIN_FEATS], tr["min"])
    res = tr["min"] - reg.predict(tr[MIN_FEATS])
    bins = pd.cut(reg.predict(tr[MIN_FEATS]), [0, 8, 14, 20, 26, 32, 60], labels=False)
    sd = pd.Series(res).groupby(bins).std().to_dict()
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, min_samples_leaf=60, random_state=1)
    tp = P[P.play_rate10.notna()]
    clf.fit(tp[PLAY_FEATS], tp.played.astype(int))
    starter = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=60, random_state=1)
    ts = P[P.played & P.start_rate10.notna()]
    starter.fit(ts[["start_rate10", "min_ewm5", "min_ewm15", "absent_share"]], ts.starter.astype(int))
    return {"reg": reg, "sd_by_bin": sd, "p_play": clf, "p_start": starter}


def minutes_sd(model, mu):
    b = pd.cut(mu, [0, 8, 14, 20, 26, 32, 60], labels=False)
    return pd.Series(b).map(model["sd_by_bin"]).fillna(6.0).values


def predict_minutes(model, P, reconcile=True):
    """Conditional-on-playing minutes with team reconciliation to 240 regulation minutes across
    the players expected to play (P.playing flag, or played for historical rows)."""
    out = P.copy()
    # gradient boosting handles missing values natively; the only hard requirements are a
    # minutes history and the game-context inputs the model was trained with
    ok = out.min_ewm5.notna() & out.abs_margin.notna()
    out["min_mu"] = np.nan
    if ok.any():
        out.loc[ok, "min_mu"] = model["reg"].predict(out.loc[ok, MIN_FEATS]).clip(0, 48)
    okp = out.play_rate10.notna()
    out["p_play"] = np.nan
    if okp.any():
        out.loc[okp, "p_play"] = model["p_play"].predict_proba(out.loc[okp, PLAY_FEATS])[:, 1]
    oks = out.start_rate10.notna() & out.min_ewm5.notna()
    out["p_start"] = np.nan
    if oks.any():
        out.loc[oks, "p_start"] = model["p_start"].predict_proba(out.loc[oks, ["start_rate10", "min_ewm5", "min_ewm15", "absent_share"]])[:, 1]
    if reconcile:
        playing = out["playing"] if "playing" in out else out["played"]
        s = out["min_mu"].where(playing, 0).groupby([out.game_id, out.team_id]).transform("sum")
        scale = (240 / s.replace(0, np.nan)).clip(0.7, 1.3)
        out["min_mu"] = np.where(playing & out.min_mu.notna(), (out.min_mu * scale).clip(0, 48), out.min_mu)
        out["min_scale"] = scale
    out["min_sd"] = minutes_sd(model, out["min_mu"].fillna(0))
    return out


# ----------------------------------------------------------------------------- rates & sim
def stat_means(P):
    """Analytic per-game expectations given min_mu: rate x minutes x opponent factor x pace factor."""
    out = P.copy()
    pace = (out.exp_pace / 100.0).fillna(1.0).clip(0.9, 1.1)
    for k in RATES:
        out[f"e_{k}"] = out.min_mu * out[f"r_{k}"] * (out[f"of_{k}"] ** 0.5) * pace
    out["e_fg2a"] = (out.e_fga - out.e_fg3a).clip(lower=0)
    fg2_pct = ((out.fg_pct * out.r_fga - out.fg3_pct * out.r_fg3a) / (out.r_fga - out.r_fg3a).replace(0, np.nan)).clip(0.3, 0.75).fillna(0.5)
    out["fg2_pct"] = fg2_pct
    out["e_fgm"] = out.e_fg2a * fg2_pct + out.e_fg3a * out.fg3_pct
    out["e_fg3m"] = out.e_fg3a * out.fg3_pct
    out["e_ftm"] = out.e_fta * out.ft_pct
    out["e_pts"] = 2 * out.e_fg2a * fg2_pct + 3 * out.e_fg3m + out.e_ftm
    out["e_reb"] = out.e_oreb + out.e_dreb
    out["e_pra"] = out.e_pts + out.e_reb + out.e_ast
    return out


def simulate(row, n=4000, rng=None, ot_prob=0.0):
    """Monte Carlo for one player-game row (from stat_means). Returns dict of arrays."""
    rng = rng or np.random.default_rng(7)
    mu, sd = float(row.min_mu), float(row.min_sd)
    mins = np.clip(rng.normal(mu, sd, n), 0, 48)
    if ot_prob:
        mins = mins + (rng.random(n) < ot_prob) * rng.normal(3.5, 1.5, n).clip(0, 5)
    pace = np.clip(rng.normal(1.0, 0.045, n), 0.85, 1.15) * float((row.exp_pace or 100.0) / 100.0 if pd.notna(row.exp_pace) else 1.0)
    def lam(k):
        return np.maximum(mins * float(row[f"r_{k}"]) * float(row[f"of_{k}"]) ** 0.5 * pace, 1e-9)
    fga = rng.poisson(lam("fga"))
    share3 = float(row.r_fg3a / row.r_fga) if row.r_fga > 0 else 0.0
    fg3a = rng.binomial(fga, min(max(share3, 0), 1))
    fg2a = fga - fg3a
    fg2m = rng.binomial(fg2a, float(row.fg2_pct))
    fg3m = rng.binomial(fg3a, float(row.fg3_pct))
    fta = rng.poisson(lam("fta"))
    ftm = rng.binomial(fta, float(row.ft_pct))
    reb = rng.poisson(lam("oreb")) + rng.poisson(lam("dreb"))
    ast, stl, blk, tov = (rng.poisson(lam(k)) for k in ("ast", "stl", "blk", "tov"))
    pts = 2 * fg2m + 3 * fg3m + ftm
    return {"min": mins, "pts": pts, "reb": reb, "ast": ast, "fg3m": fg3m, "stl": stl, "blk": blk, "tov": tov,
            "fgm": fg2m + fg3m, "fga": fga, "ftm": ftm, "fta": fta, "pra": pts + reb + ast, "pr": pts + reb,
            "pa": pts + ast, "ra": reb + ast, "sb": stl + blk,
            "dd": ((pts >= 10).astype(int) + (reb >= 10) + (ast >= 10) + (stl >= 10) + (blk >= 10)) >= 2}


def summarize(sim, lines=None, p_play=None):
    """Mean/median/intervals per stat, P(over) for supplied lines, both conditional and
    availability-adjusted (P(play) x conditional; the 'no-play' outcome is a void, not a zero,
    for props, but counts as zero for fantasy)."""
    out = {}
    for k, v in sim.items():
        if k == "dd":
            out[k] = {"p": float(v.mean())}
            continue
        q = np.percentile(v, [10, 25, 50, 75, 90])
        out[k] = {"mean": round(float(v.mean()), 2), "median": float(q[2]), "p10": float(q[0]), "p25": float(q[1]),
                  "p75": float(q[3]), "p90": float(q[4])}
        if lines and k in lines:
            L = lines[k]
            out[k]["line"] = L
            out[k]["p_over"] = float((v > L).mean())
            out[k]["p_push"] = float((v == L).mean())
    if p_play is not None:
        out["p_play"] = float(p_play)
    return out


# ----------------------------------------------------------------------------- evaluation
def evaluate(P, first_test=2024, last_test=2026, sample_sims=2500, seed=3):
    """Walk-forward: fit minutes on seasons < s, project season s. Conditional-on-playing metrics
    on rows that played; availability metrics (P(play)) on all roster rows."""
    reports = []
    P = P[P.min_ewm5.notna() | ~P.played].copy()
    for s in range(first_test, last_test + 1):
        tr, te = P[P.season < s], P[(P.season == s) & P.abs_margin.notna()].copy()
        if te.empty:
            continue
        m = fit_minutes(tr)
        te = predict_minutes(m, te)
        te = stat_means(te)
        pl = te[te.played & te.min_mu.notna()]
        rep = {"season": s, "role": "test" if s == last_test else "validation", "n_player_games": int(len(pl)),
               "n_roster_rows": int(te.p_play.notna().sum())}
        rep["minutes"] = {"mae": float((pl["min"] - pl.min_mu).abs().mean()),
                          "mae_last5_ewm": float((pl["min"] - pl.min_ewm5).abs().mean()),
                          "mae_last15_ewm": float((pl["min"] - pl.min_ewm15).abs().mean()),
                          "bias": float((pl.min_mu - pl["min"]).mean())}
        # reconciliation check: team sums of projected minutes on regulation games
        reg = pl[pl.periods <= 4].groupby(["game_id", "team_id"]).min_mu.sum()
        rep["minutes"]["team_sum_regulation"] = {"mean": float(reg.mean()), "sd": float(reg.std())}
        pp = te[te.p_play.notna()]
        y, p = pp.played.astype(float), pp.p_play.clip(1e-6, 1 - 1e-6)
        rep["p_play"] = {"brier": float(((p - y) ** 2).mean()), "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                         "base_rate": float(y.mean()), "brier_play_rate10": float(((pp.play_rate10.clip(1e-6, 1 - 1e-6) - y) ** 2).mean())}
        ps = pl[pl.p_start.notna()]
        rep["p_start"] = {"brier": float(((ps.p_start - ps.starter.astype(float)) ** 2).mean()),
                          "brier_start_rate10": float(((ps.start_rate10 - ps.starter.astype(float)) ** 2).mean())}
        # production, conditional on playing
        base = _baselines(P, s)
        pl = pl.merge(base, on=["player_id", "game_id"], how="left")
        stats = {}
        for k, e in (("pts", "e_pts"), ("reb", "e_reb"), ("ast", "e_ast"), ("fg3m", "e_fg3m"), ("stl", "e_stl"), ("blk", "e_blk"), ("tov", "e_tov")):
            row = {"model_mae": float((pl[k] - pl[e]).abs().mean()), "model_bias": float((pl[e] - pl[k]).mean())}
            for b in ("l10mean", "l10median", "minxrate"):
                c = f"{b}_{k}"
                if c in pl:
                    mm = pl[c].notna()
                    row[f"{b}_mae"] = float((pl.loc[mm, k] - pl.loc[mm, c]).abs().mean())
            stats[k] = row
        pra = pl.pts + pl.reb + pl.ast
        stats["pra"] = {"model_mae": float((pra - pl.e_pra).abs().mean()),
                        "l10mean_mae": float((pra - (pl.l10mean_pts + pl.l10mean_reb + pl.l10mean_ast)).abs().mean())}
        rep["production"] = stats
        # simulation coverage on a sample (interval calibration + P(over) calibration against
        # the player's own median as a stand-in line; NOT a market line)
        rng = np.random.default_rng(seed)
        samp = pl.sample(min(sample_sims, len(pl)), random_state=seed)
        cov = {k: [] for k in ("pts", "reb", "ast", "pra")}
        pov = []
        for _, r in samp.iterrows():
            sim = simulate(r, n=1500, rng=rng)
            for k in cov:
                lo, hi = np.percentile(sim[k], [10, 90])
                actual = r.pts + r.reb + r.ast if k == "pra" else r[k]
                cov[k].append(lo <= actual <= hi)
            L = np.median(sim["pts"]) - 0.5
            pov.append(((sim["pts"] > L).mean(), float(r.pts > L)))
        rep["simulation"] = {"n": int(len(samp)), "coverage_80": {k: float(np.mean(v)) for k, v in cov.items()},
                             "p_over_median_line": {"mean_pred": float(np.mean([a for a, _ in pov])), "hit_rate": float(np.mean([b for _, b in pov]))}}
        reports.append(rep)
    return reports


def _baselines(P, season):
    Q = P[P.played].sort_values(["player_id", "tipoff_utc"]).copy()
    out = Q[["player_id", "game_id"]].copy()
    for k in ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov"):
        g = Q.groupby("player_id")[k]
        out[f"l10mean_{k}"] = g.transform(lambda s: s.rolling(10, min_periods=3).mean().shift(1))
        out[f"l10median_{k}"] = g.transform(lambda s: s.rolling(10, min_periods=3).median().shift(1))
        rate = (Q[k] / Q["min"]).groupby(Q.player_id).transform(lambda s: s.rolling(10, min_periods=3).mean().shift(1))
        out[f"minxrate_{k}"] = rate * Q.min_ewm5
    return out[out.game_id.isin(P[P.season == season].game_id)]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    from nba import data, team_model
    d = data.load()
    g = team_model.features(d)
    # pre-game expected margin/pace from the lineup-blind team model, walk-forward per season
    g["margin_pred_pre"], g["exp_pace"] = np.nan, g.sum_pace / 2
    for s in sorted(g.season.unique()):
        tr = g[(g.season < s) & g.margin.notna()]
        if len(tr) < 500:
            continue
        m = team_model.fit(tr, team_model.BLIND)
        idx = g.season == s
        g.loc[idx, "margin_pred_pre"] = team_model.predict(m, g[idx]).margin_pred.values
    cache = os.path.join(HERE, "data", "player_features.parquet")
    if os.path.exists(cache) and "--refresh" not in sys.argv:
        P = pd.read_parquet(cache)
    else:
        P = player_features(d, g)
        P.to_parquet(cache, index=False)
    reps = evaluate(P)
    json.dump({"model_version": MODEL_VERSION, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "seasons": reps, "market_comparison": "none: no historical NBA prop lines are held; forward collection only"},
              open(os.path.join(HERE, "data", "player_backtest.json"), "w"), indent=1)
    for r in reps:
        print(r["season"], r["role"], "minutes", {k: round(v, 2) for k, v in r["minutes"].items() if isinstance(v, float)},
              "\n   p_play", {k: round(v, 3) for k, v in r["p_play"].items()}, "p_start", {k: round(v, 3) for k, v in r["p_start"].items()})
        for k, v in r["production"].items():
            print("   ", k, {a: round(b, 2) for a, b in v.items()})
        print("   sim", r["simulation"])
