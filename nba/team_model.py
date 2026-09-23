"""
NBA team game model: win probability, margin and total from the model alone.

No market input anywhere in the features. Every feature is computed from games that tipped
off before the game being predicted (walk-forward), with a previous-season prior that fades
as the current season accumulates.

Two feature sets are evaluated and reported separately because they answer different questions:
  * "lineup-blind"  - team strength, rest, schedule and home court only. This is what the model
                      knows days ahead, and what a scheduled forecast uses.
  * "availability"  - adds the share of the team's usual production that is on the floor,
                      computed from the players who actually played. Historically that means
                      the box score told us who played, which is information a bettor only has
                      close to tipoff (from the injury report). The forward pipeline fills the
                      same feature from the official injury report + probable starters, so the
                      backtest number for this set is an upper bound on what the live model sees.

Baselines reported alongside: home team always, season-to-date net rating (regressed), Elo.
There are no historical NBA closing lines in this repository, so no market comparison is
claimed for the backtest; the market comparison starts with forward collection (see nba/props.py
and the runner collector) and is graded as real lines arrive.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_VERSION = "nba_team_v1"
HALF_LIFE = 12          # games; EWMA decay for team ratings
PRIOR_GAMES = 15        # weight of last season's rating expressed in games
LEAGUE_ORTG = 113.0

BLIND = ["home", "d_net", "d_off", "d_def", "sum_pace", "d_rest", "d_b2b", "d_last7", "d_elo", "d_sos", "neutral"]
AVAIL = BLIND + ["d_avail"]


def _ewm(values, half_life):
    a = 1 - 0.5 ** (1 / half_life)
    out, cur = [], None
    for v in values:
        cur = v if cur is None else cur + a * (v - cur)
        out.append(cur)
    return out


def team_history(T):
    """Per team-game pre-game ratings: EWMA of ORtg/DRtg/pace from prior games, with a
    previous-season prior, an Elo, and the availability index. Returns T with feature cols."""
    T = T.sort_values("tipoff_utc").copy()
    T["net"] = T.ortg - T.drtg
    feats = {k: np.full(len(T), np.nan) for k in ("r_off", "r_def", "r_pace", "elo_pre", "sos")}
    idx = {g: i for i, g in enumerate(T.index)}
    state = {}                          # team_id -> dict(off, def, pace, n, season, prev_off, prev_def)
    elo = {}
    last_season = {}
    K = 20.0
    for i, r in T.iterrows():
        t, s = r.team_id, r.season
        st = state.get(t)
        if st is None or st["season"] != s:
            prev = st
            st = {"season": s, "n": 0, "off": None, "def": None, "pace": None,
                  "p_off": prev["off"] if prev else LEAGUE_ORTG, "p_def": prev["def"] if prev else LEAGUE_ORTG,
                  "p_pace": prev["pace"] if prev else 100.0, "opp_net": []}
            # regress last season's ratings 40% to the mean as the prior
            for k, m in (("p_off", LEAGUE_ORTG), ("p_def", LEAGUE_ORTG), ("p_pace", 100.0)):
                st[k] = m + 0.6 * (st[k] - m)
            state[t] = st
            if t in elo and last_season.get(t) != s:
                elo[t] = 1500 + 0.7 * (elo[t] - 1500)
            last_season[t] = s
        elo.setdefault(t, 1500.0)
        w = st["n"] / (st["n"] + PRIOR_GAMES)
        j = idx[i]
        feats["r_off"][j] = w * (st["off"] if st["off"] is not None else st["p_off"]) + (1 - w) * st["p_off"]
        feats["r_def"][j] = w * (st["def"] if st["def"] is not None else st["p_def"]) + (1 - w) * st["p_def"]
        feats["r_pace"][j] = w * (st["pace"] if st["pace"] is not None else st["p_pace"]) + (1 - w) * st["p_pace"]
        feats["elo_pre"][j] = elo[t]
        feats["sos"][j] = float(np.mean(st["opp_net"][-15:])) if st["opp_net"] else 0.0
        # update after the game (only if it was played and boxed)
        if pd.notna(r.ortg) and pd.notna(r.drtg):
            a = 1 - 0.5 ** (1 / HALF_LIFE)
            for k, v in (("off", r.ortg), ("def", r.drtg), ("pace", r.pace)):
                st[k] = v if st[k] is None else st[k] + a * (v - st[k])
            st["n"] += 1
    # Elo needs both sides; second pass per game in time order
    T2 = T.reset_index()
    by_game = T2.groupby("game_id", sort=False)
    elo = {}
    elo_pre = np.full(len(T2), np.nan)
    seen_season = {}
    for gid, grp in sorted(by_game, key=lambda kv: kv[1].tipoff_utc.iloc[0]):
        if len(grp) != 2:
            continue
        a, b = grp.iloc[0], grp.iloc[1]
        for r in (a, b):
            if seen_season.get(r.team_id) != r.season:
                elo[r.team_id] = 1500 + 0.7 * (elo.get(r.team_id, 1500) - 1500)
                seen_season[r.team_id] = r.season
        ea, eb = elo[a.team_id], elo[b.team_id]
        elo_pre[a.name], elo_pre[b.name] = ea, eb
        if pd.notna(a.pts) and pd.notna(b.pts):
            ha = 60 if (a.home and not a.neutral) else (-60 if (b.home and not b.neutral) else 0)
            exp_a = 1 / (1 + 10 ** ((eb - ea - ha) / 400))
            mov = abs(a.pts - b.pts)
            mult = ((mov + 3) ** 0.8) / (7.5 + 0.006 * abs(ea - eb))
            d = K * mult * ((1 if a.pts > b.pts else 0) - exp_a)
            elo[a.team_id] = ea + d
            elo[b.team_id] = eb - d
    T2["elo_pre"] = elo_pre
    for k in ("r_off", "r_def", "r_pace", "sos"):
        T2[k] = feats[k]
    # opponent net for SOS: filled from the opponent's pre-game net via merge
    opp = T2[["game_id", "team_id", "r_off", "r_def"]].rename(columns={"team_id": "opp_id", "r_off": "opp_off", "r_def": "opp_def"})
    T2 = T2.merge(opp, on=["game_id", "opp_id"], how="left")
    return T2.set_index("index")


def availability_index(P, T):
    """Share of a team's usual production that played. Each player's value = trailing mean of
    box production per game (prior games only, EWMA); team usual = sum over the players who
    appeared in the team's last 5 games; available = sum over players with minutes today."""
    P = P.sort_values("tipoff_utc").copy()
    prod = (P.pts + 1.2 * P.reb + 1.5 * P.ast + 3 * (P.stl + P.blk) - P.tov - 0.7 * (P.fga - P.fgm)).where(P.played)
    P["prod"] = prod
    a = 1 - 0.5 ** (1 / 10)
    P["val"] = P.groupby("player_id")["prod"].transform(lambda s: s.ewm(alpha=a, ignore_na=True).mean().shift(1))
    P["val"] = P["val"].fillna(0.0)
    played = P[P.played]
    got = played.groupby(["game_id", "team_id"])["val"].sum().rename("avail_val")
    # usual: sum of val over the union of players who played in the team's previous 5 games (per team, by time)
    usual = {}
    per_team = {}
    for (gid, tid), grp in played.groupby(["game_id", "team_id"], sort=False):
        pass
    games = T[["game_id", "team_id", "tipoff_utc"]].sort_values("tipoff_utc")
    last = {}
    out = {}
    val_by = played.groupby(["game_id", "team_id"]).apply(lambda g: dict(zip(g.player_id, g.val)), include_groups=False).to_dict()
    for r in games.itertuples():
        hist = last.setdefault(r.team_id, [])
        pool = {}
        for d in hist[-5:]:
            pool.update(d)
        cur = val_by.get((r.game_id, r.team_id), {})
        # value the pool at today's estimates where a player played today, else at last known
        usual_val = sum(max(cur.get(p, v), 0) for p, v in pool.items()) if pool else np.nan
        out[(r.game_id, r.team_id)] = usual_val
        if cur:
            hist.append(cur)
    T = T.copy()
    T["usual_val"] = [out.get((g, t), np.nan) for g, t in zip(T.game_id, T.team_id)]
    T = T.merge(got.reset_index(), on=["game_id", "team_id"], how="left")
    T["avail"] = (T.avail_val / T.usual_val).clip(0.5, 1.3)
    return T


def game_table(T):
    """One row per game (home perspective) with difference features and targets."""
    h = T[T.home].set_index("game_id")
    a = T[~T.home].set_index("game_id")
    common = h.index.intersection(a.index)
    h, a = h.loc[common], a.loc[common]
    g = pd.DataFrame(index=common)
    g["season"], g["tipoff_utc"], g["phase"] = h.season, h.tipoff_utc, h.phase
    g["home_team"], g["away_team"] = h.team, a.team
    g["home_id"], g["away_id"] = h.team_id, a.team_id
    g["home"] = (~h.neutral.astype(bool)).astype(float)
    g["neutral"] = h.neutral.astype(bool).astype(float)
    g["d_off"] = h.r_off - a.r_off
    g["d_def"] = a.r_def - h.r_def              # positive = home defence better (lower DRtg)
    g["d_net"] = (h.r_off - h.r_def) - (a.r_off - a.r_def)
    g["sum_pace"] = h.r_pace + a.r_pace
    g["exp_total_rate"] = (h.r_off + a.r_off + h.r_def + a.r_def) / 4
    g["d_rest"] = h.rest_days.clip(upper=5).fillna(3) - a.rest_days.clip(upper=5).fillna(3)
    g["d_b2b"] = h.b2b.astype(float) - a.b2b.astype(float)
    g["d_last7"] = h.games_last7 - a.games_last7
    g["d_elo"] = (h.elo_pre - a.elo_pre) / 100
    g["d_sos"] = h.sos - a.sos
    g["d_avail"] = h.get("avail", np.nan) - a.get("avail", np.nan) if "avail" in h else np.nan
    g["margin"] = h.pts - a.pts
    g["total"] = h.pts + a.pts
    g["home_win"] = (g.margin > 0).astype(float).where(g.margin.notna())
    g["ot"] = (h.periods > 4)
    g["home_pts"], g["away_pts"] = h.pts, a.pts
    return g.reset_index().rename(columns={"index": "game_id"})


def features(d, P_avail=True):
    T = team_history(d["team_games"])
    if P_avail:
        T = availability_index(d["player_games"], T)
    return game_table(T)


# ----------------------------------------------------------------------------- fitting
def fit(train, cols):
    tr = train.dropna(subset=cols + ["margin"])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500)).fit(tr[cols], tr.home_win)
    reg_m = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit(tr[cols], tr.margin)
    tcols = cols + ["exp_total_rate"]
    reg_t = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit(tr[tcols], tr.total)
    resid_m = float(np.std(tr.margin - reg_m.predict(tr[cols])))
    resid_t = float(np.std(tr.total - reg_t.predict(tr[tcols])))
    return {"clf": clf, "reg_m": reg_m, "reg_t": reg_t, "cols": cols, "tcols": tcols, "sd_margin": resid_m, "sd_total": resid_t,
            "n_train": int(len(tr))}


def predict(model, g):
    cols, tcols = model["cols"], model["tcols"]
    ok = g[cols + ["exp_total_rate"]].notna().all(axis=1)
    out = pd.DataFrame(index=g.index)
    out["p_home"] = np.nan
    out["margin_pred"] = np.nan
    out["total_pred"] = np.nan
    if ok.any():
        out.loc[ok, "p_home"] = model["clf"].predict_proba(g.loc[ok, cols])[:, 1]
        out.loc[ok, "margin_pred"] = model["reg_m"].predict(g.loc[ok, cols])
        out.loc[ok, "total_pred"] = model["reg_t"].predict(g.loc[ok, tcols])
    return out


def metrics(y, p, margin=None, mpred=None, total=None, tpred=None):
    m = y.notna() & p.notna()
    y, p = y[m].astype(float), p[m].astype(float).clip(1e-6, 1 - 1e-6)
    r = {"n": int(m.sum()), "su": float(((p > 0.5) == (y > 0.5)).mean()) if m.any() else None,
         "brier": float(((p - y) ** 2).mean()) if m.any() else None,
         "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()) if m.any() else None}
    if margin is not None and mpred is not None:
        mm = margin.notna() & mpred.notna()
        r["margin_mae"] = float((margin[mm] - mpred[mm]).abs().mean()) if mm.any() else None
    if total is not None and tpred is not None:
        mt = total.notna() & tpred.notna()
        r["total_mae"] = float((total[mt] - tpred[mt]).abs().mean()) if mt.any() else None
    return r


def baselines(g):
    """Home-always, regressed net-rating, Elo. Probabilities via fixed logistic maps that are
    NOT fitted on the evaluated season (constants chosen from 2022-23 only)."""
    b = {}
    b["home_always"] = pd.Series(0.58, index=g.index)
    b["net_rating"] = 1 / (1 + np.exp(-(0.11 * g.d_net + 0.32 * g.home)))
    b["elo"] = 1 / (1 + np.exp(-(np.log(10) / 4) * (g.d_elo + 0.6 * g.home)))
    return b


def walk_forward(g, first_test=2024, last_test=2026, phases=("regular", "play-in", "postseason")):
    """Train on all seasons before s, evaluate season s. Returns per-season report and the
    out-of-sample predictions."""
    g = g[g.phase.isin(phases) & g.margin.notna()].copy()
    rows, preds = [], []
    for s in range(first_test, last_test + 1):
        tr, te = g[g.season < s], g[g.season == s]
        if te.empty or len(tr) < 500:
            continue
        rep = {"season": s, "n_train": int(len(tr)), "n_test": int(len(te)), "role": "test" if s == last_test else "validation"}
        for name, cols in (("lineup_blind", BLIND), ("availability", AVAIL)):
            if name == "availability" and ("d_avail" not in tr or tr.d_avail.notna().sum() < 500):
                rep[name] = {"n": 0, "su": None, "brier": None, "logloss": None, "margin_mae": None, "total_mae": None}
                continue
            m = fit(tr, cols)
            p = predict(m, te)
            rep[name] = metrics(te.home_win, p.p_home, te.margin, p.margin_pred, te.total, p.total_pred)
            if name == "lineup_blind":
                pr = te[["game_id", "season", "tipoff_utc", "home_team", "away_team", "margin", "total"]].copy()
                pr["p_home"], pr["margin_pred"], pr["total_pred"] = p.p_home, p.margin_pred, p.total_pred
            else:
                pr["p_home_avail"], pr["margin_pred_avail"], pr["total_pred_avail"] = p.p_home, p.margin_pred, p.total_pred
        for name, p in baselines(te).items():
            rep[f"baseline_{name}"] = metrics(te.home_win, p)
        # recent-average total baseline: sum of both teams' EWMA rates
        rep["baseline_rate_total"] = {"total_mae": float((te.total - (te.exp_total_rate * 2 * te.sum_pace / 2 / 100)).abs().mean())}
        rows.append(rep)
        preds.append(pr)
    return rows, (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame())


def train_final(g, through_season):
    """Fit the deployable models on every finished game up to and including through_season."""
    tr = g[(g.season <= through_season) & g.margin.notna() & g.phase.isin(("regular", "play-in", "postseason"))]
    out = {"lineup_blind": fit(tr, BLIND), "availability": None, "trained_through": int(through_season),
            "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "model_version": MODEL_VERSION}
    if "d_avail" in tr and tr.d_avail.notna().sum() > 500:
        out["availability"] = fit(tr, AVAIL)
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    from nba import data
    d = data.load()
    g = features(d)
    rows, preds = walk_forward(g)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump({"model_version": MODEL_VERSION, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "features": {"lineup_blind": BLIND, "availability": AVAIL}, "seasons": rows,
               "market_comparison": "none: no historical NBA closing lines are held; forward collection only"},
              open(os.path.join(HERE, "data", "team_backtest.json"), "w"), indent=1)
    preds.to_parquet(os.path.join(HERE, "data", "team_oos.parquet"), index=False)
    for r in rows:
        print(r["season"], r["role"], "blind", {k: round(v, 3) for k, v in r["lineup_blind"].items() if v is not None},
              "\n      avail", {k: round(v, 3) for k, v in r["availability"].items() if v is not None},
              "\n      elo", {k: round(v, 3) for k, v in r["baseline_elo"].items() if v is not None},
              "net", round(r["baseline_net_rating"]["su"], 3), "home", round(r["baseline_home_always"]["su"], 3))
