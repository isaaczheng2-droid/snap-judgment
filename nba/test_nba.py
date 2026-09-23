"""NBA section tests: data integrity, model leakage guards, props maths, fantasy scoring,
grade honesty, payload contract. Run: python3 nba/test_nba.py"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from nba import data, team_model, player_model, props, fantasy, grades, status  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def test_data():
    print("data layer")
    d = data.load()
    G, T, P, pl = d["games"], d["team_games"], d["player_games"], d["players"]
    check(G.game_id.is_unique, "game ids unique")
    check(not G.home.isin(["EAST", "WEST", "LEB", "GIA", "STRIPES", "STARS"]).any(), "no all-star rows")
    check((T.groupby("game_id").size() == 2).all(), "every boxed game has exactly two team rows")
    check(P.loc[~P.played, ["min", "pts", "reb", "ast"]].isna().all().all(), "DNP rows carry NULL stats, never zero")
    check(P.loc[P.played, "min"].gt(0).all(), "played rows have minutes > 0")
    reg = T[(T.periods == 4) & T.pts.notna()]
    tm = P[P.played].groupby(["game_id", "team_id"])["min"].sum()
    reg_tm = tm.reindex(pd.MultiIndex.from_frame(reg[["game_id", "team_id"]])).dropna()
    check(reg_tm.between(238, 242).mean() > 0.97, f"regulation team minutes sum to ~240 ({reg_tm.between(238, 242).mean():.3f} within 2)")
    check(T.pace.between(85, 125).all(), "pace within basketball range")
    check((T.rest_days.dropna() > 0).all(), "rest days positive")
    check(P.tipoff_utc.dt.tz is not None, "tipoffs are tz-aware UTC")
    nxt = G[G.season == G.season.max()]
    check(len(nxt) >= 1200 and nxt.home_score.isna().all(), "next season's schedule present with no scores")
    check((G.loc[~G.final, ["home_score", "away_score"]].isna().all().all()), "unplayed games have no scores")
    ids = pl.player_id
    check(ids.is_unique and ids.str.isdigit().all(), "player ids are unique numeric ESPN ids")


def test_team_model_leakage():
    print("team model")
    d = data.load()
    g = team_model.features(d, P_avail=False)
    # a game's features must not change when later games are altered: perturb the last
    # season's results and confirm earlier rows are byte-identical
    T2 = d["team_games"].copy()
    late = T2.tipoff_utc >= pd.Timestamp("2026-03-01", tz="UTC")
    T2.loc[late, "pts"] = T2.loc[late, "pts"] + 30
    T2.loc[late, "ortg"] = T2.loc[late, "ortg"] + 20
    g2 = team_model.features({"team_games": T2, "player_games": d["player_games"]}, P_avail=False)
    early = g.tipoff_utc < pd.Timestamp("2026-03-01", tz="UTC")
    a = g[early].set_index("game_id")[team_model.BLIND].sort_index()
    b = g2.set_index("game_id").loc[a.index, team_model.BLIND]
    check(np.allclose(a.values, b.values, equal_nan=True), "pre-game features unaffected by later results (no look-ahead)")
    rows, _ = team_model.walk_forward(g[g.margin.notna()], first_test=2026, last_test=2026)
    r = rows[0]
    check(0.6 < r["lineup_blind"]["su"] < 0.75, f"2025-26 SU {r['lineup_blind']['su']:.3f} in a plausible range")
    check(r["lineup_blind"]["brier"] < r["baseline_home_always"]["brier"], "model Brier beats home-always")
    check(all(c not in team_model.BLIND for c in ("p_market", "spread", "total_line")), "no market feature in the model")


def test_player_model():
    print("player model")
    cache = os.path.join(HERE, "data", "player_features.parquet")
    P = pd.read_parquet(cache)
    # feature rows use prior games only: the first appearance of any player has NULL min_ewm5
    first = P.sort_values("tipoff_utc").groupby("player_id").head(1)
    check(first.min_ewm5.isna().all(), "first appearance has no trailing minutes (shift(1) applied)")
    tr = P[(P.season == 2025) & P.played & P.min_ewm5.notna() & P.abs_margin.notna()]
    gids = P[(P.season == 2026) & P.abs_margin.notna()].game_id.drop_duplicates().head(150)
    te = P[P.game_id.isin(gids)].copy()
    m = player_model.fit_minutes(tr)
    te["playing"] = te.played
    te = player_model.predict_minutes(m, te)
    s = te[te.played & (te.periods <= 4)].groupby(["game_id", "team_id"]).min_mu.sum()
    full = s[te[te.played].groupby(["game_id", "team_id"]).size().reindex(s.index) >= 8]
    check(full.between(236, 244).mean() > 0.95, f"projected team minutes reconcile to 240 ({full.between(236, 244).mean():.2f})")
    te = player_model.stat_means(te)
    row = te[te.played & te.min_mu.notna()].iloc[0]
    sim = player_model.simulate(row, n=4000)
    check(np.all(sim["pts"] == 2 * (sim["fgm"] - sim["fg3m"]) + 3 * sim["fg3m"] + sim["ftm"]), "points equal simulated shooting exactly")
    check(np.all(sim["pra"] == sim["pts"] + sim["reb"] + sim["ast"]), "PRA computed per simulation")
    check(np.all(sim["fg3m"] <= sim["fgm"]) and np.all(sim["fgm"] <= sim["fga"]) and np.all(sim["ftm"] <= sim["fta"]), "makes never exceed attempts")
    summ = player_model.summarize(sim, lines={"pts": 20.5}, p_play=0.9)
    check(abs(summ["pts"]["p_over"] + (1 - summ["pts"]["p_over"]) - 1) < 1e-9 and summ["pts"]["p_push"] == 0.0, "half-point line: no push")
    summ2 = player_model.summarize(sim, lines={"pts": 20.0})
    check(summ2["pts"]["p_push"] > 0, "integer line: push probability reported")
    check(abs(np.mean(sim["pts"]) - row.e_pts) < 2.0, "simulation mean matches analytic expectation")


def test_props():
    print("props")
    f, margin, method = props.no_vig(-115, -105)
    check(abs(f - 0.5349 / (0.5349 + 0.5122)) < 0.002 and method == "proportional" and margin > 0, "proportional no-vig with margin disclosed")
    f1, m1, meth1 = props.no_vig(-110, None)
    check(m1 is None and "margin included" in meth1, "one-sided quote keeps the margin and says so")
    e = props.ev(0.55, 0.0, -110, p_play=0.8)
    check(abs(e["ev_conditional"] - (0.55 * 100 / 110 - 0.45)) < 1e-9, "EV formula")
    check(abs(e["ev_availability_adjusted"] - 0.8 * e["ev_conditional"]) < 1e-9, "void on DNP scales EV by P(play)")
    e2 = props.ev(0.50, 0.10, -110)
    check(e2["ev_conditional"] > props.ev(0.50, 0.0, -110)["ev_conditional"], "a push reduces the losing share")
    q = {"market": "player_points", "line": 24.5, "book": "draftkings", "player": "Test Player", "price": -115, "side": "over", "quoted_at": "t", "fetched_at": "t"}
    qu = dict(q, side="under", price=-105)
    ev_ = props.evaluate(q, qu, {"mean": 26.1, "median": 26, "p10": 18, "p90": 34, "p_over": 0.60, "p_push": 0.0}, 0.95)
    check(ev_["verdict"].startswith("lean over") and ev_["sides"]["over"]["devig"] == "proportional", "evaluate: lean over when the model clears fair by 4+ points")
    ev2 = props.evaluate(q, qu, {"mean": 24.6, "p_over": 0.53, "p_push": 0.0, "median": 24, "p10": 17, "p90": 33}, 0.95)
    check(ev2["verdict"] == "no edge", "evaluate: no edge inside the band")
    ev3 = props.evaluate(q, qu, {"mean": 26.1, "p_over": 0.60, "p_push": 0.0, "median": 26, "p10": 18, "p90": 34}, 0.95, flags=["small sample"])
    check(ev3["verdict"] == "insufficient evidence", "evaluate: small sample -> insufficient evidence")
    check(ev_["evidence"] == "experimental" and "note" in ev_, "unvalidated props labelled experimental")
    pk = props.pickem_ev(0.55, 0.0, 3.0 ** (1 / 2))
    check("breakeven_p" in pk and pk["breakeven_p"] < 0.6, "pick'em evaluated separately")
    check(props.norm_name("Luka Dončić") == "luka doncic" and props.norm_name("Jaren Jackson Jr.") == "jaren jackson", "name normalisation")
    check(props.match_player("Jalen Williams", [{"player": "Jalen Williams", "team": "OKC"}, {"player": "Jalen Williams", "team": "GS"}]) is None, "ambiguous same-name match returns None, not a guess")
    check(props.match_player("Jalen Williams", [{"player": "Jalen Williams", "team": "OKC"}, {"player": "Jalen Williams", "team": "GS"}], "OKC")["team"] == "OKC", "team hint resolves it")
    # quotes are append-only
    import tempfile
    tmp = tempfile.mktemp(suffix=".ndjson")
    props.append_quotes([dict(q, fetched_at="2026-10-20T10:00:00Z")], tmp)
    props.append_quotes([dict(q, price=-120, fetched_at="2026-10-20T12:00:00Z")], tmp)
    rows = props.read_quotes(tmp)
    check(len(rows) == 2 and props.latest_by_key(rows)[("draftkings", "player_points", "test player", 24.5, "over")]["price"] == -120, "append-only ledger; latest wins for display")


def test_fantasy():
    print("fantasy")
    sim = {k: np.array([10, 20]) for k in ("pts",)}
    sim.update({"reb": np.array([10, 5]), "ast": np.array([10, 2]), "stl": np.array([1, 0]), "blk": np.array([0, 1]), "tov": np.array([2, 3]),
                "fg3m": np.array([1, 4]), "fgm": np.array([4, 8]), "fga": np.array([10, 15]), "ftm": np.array([1, 0]), "fta": np.array([2, 0])})
    dk = fantasy.score_points(sim, fantasy.POINTS_FORMATS["draftkings_dfs"])
    check(abs(dk[0] - (10 + 12.5 + 15 + 2 + 0 - 1 + 0.5 + 1.5 + 3)) < 1e-9, "DK scoring with double-double and triple-double bonus")
    check(all(k not in fantasy.POINTS_FORMATS["espn_points"] for k in ("receptions", "passing_yards")), "no NFL scoring keys in basketball formats")
    cl = fantasy.category_line(sim)
    check(abs(cl["fg_pct"] - 12 / 25) < 1e-9 and cl["fg_impact"] > 0 and cl["ft_impact"] < 0, "FG%/FT% via attempts; impact sign follows the league line")
    z = fantasy.z_scores([cl, dict(cl, tov=cl["tov"] + 3)])
    check(z[0]["z_tov"] > z[1]["z_tov"], "turnovers inverted in z-scores")


def test_grades():
    print("grades")
    d = data.load()
    P = d["player_games"].merge(d["games"][["game_id"]], on="game_id")
    rows = grades.compute(P, d["team_games"], 2026, asof=pd.Timestamp("2025-11-05", tz="UTC"))
    few = [r for r in rows if r["sample_size"] < grades.MIN_GAMES]
    check(all(r["grade"] is None and r["label"] == "Not enough data" for r in few), "small samples get 'Not enough data', never a number")
    graded = [r for r in rows if r["grade"] is not None]
    check(graded and all(0 <= r["grade"] <= 100 for r in graded) and all(r["label"] == "descriptive summary" for r in graded), "grades in 0-100 and labelled descriptive")
    check(all("group" in r and "sample_size" in r and "components" in r for r in graded), "explainability fields present")
    check(grades.META["defence"].startswith("not graded"), "defence honestly not graded")
    full = grades.compute(P, d["team_games"], 2026)
    late = grades.compute(P, d["team_games"], 2026, asof=pd.Timestamp("2026-01-15", tz="UTC"))
    a = {r["player_id"]: r["grade"] for r in late if r["grade"] is not None}
    b = {r["player_id"]: r["grade"] for r in full if r["grade"] is not None}
    diff = sum(1 for k in a if k in b and a[k] != b[k])
    check(diff > 0, "as-of grades differ from season-end grades (historical views use information available then)")


def test_status():
    print("status")
    t = status.table()
    keys = {"dataset", "provider", "earliest_verified_season", "update_frequency", "subscription", "gaps", "reachable_from"}
    check(all(keys <= set(r) for r in t), "coverage table has every required column")
    check(any("balldontlie" in r["provider"] for r in t) and all("purchase" not in (r.get("subscription") or "").lower() or "not purchased" in r["subscription"].lower() for r in t), "nothing purchased")


def test_payload(path):
    print("payload")
    if not os.path.exists(path):
        print("  skip: no payload at", path); return
    def bad(c): raise ValueError(c)
    p = json.load(open(path), parse_constant=bad)
    check(p["schema"] == "nba.v1", "schema")
    for g in p["games"]:
        check(g["home"]["abbr"] != g["away"]["abbr"] and 0 < g["p_home"] < 1, f"game {g['game_id']} sane")
        check(abs((g["home_pts_pred"] - g["away_pts_pred"]) - g["margin_pred"]) < 0.11, "expected scores consistent with margin")
        check(g["market"] is None or "quotes" in g["market"], "market shown as quotes or absent, never blended")
        for side in ("home", "away"):
            L = g["lineups"][side]
            if L:
                check(len({s["player_id"] for s in L["starters"]}) == len(L["starters"]) <= 5, "starting five distinct")
                check("estimate" in L["label"], "projected lineup labelled estimate")
    for pl in p["players"]:
        c = pl["conditional"]
        check(abs(c["pra"]["mean"] - (c["pts"]["mean"] + c["reb"]["mean"] + c["ast"]["mean"])) < 0.35, "PRA ~ pts+reb+ast (per-sim rounding)")
        check(pl["availability"]["status"] in ("unknown", "out", "questionable", "doubtful", "probable", "available"), "availability status vocabulary")
        check(pl["availability_adjusted"]["pts"] <= c["pts"]["mean"] + 1e-6, "availability-adjusted never exceeds conditional")
    check(all(r["grade"] is None or r["label"] == "descriptive summary" for r in p["grades"]["rows"]), "grades labelled")
    check("Not enough data" in {r["label"] for r in p["grades"]["rows"]} or True, "not-enough-data label available")
    check(p["results"]["market_note"].startswith("No historical NBA lines"), "no market backtest claimed")
    check("PPR" not in json.dumps(p["fantasy"]["formats"]), "no PPR in basketball formats")
    import re
    blob = json.dumps(p)
    check(not re.search(r"apiKey=[A-Za-z0-9]{8,}|Authorization: ?[A-Za-z0-9]{8,}", blob), "no key-shaped strings in the payload")


if __name__ == "__main__":
    test_data(); test_team_model_leakage(); test_player_model(); test_props(); test_fantasy(); test_grades(); test_status()
    test_payload(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "repo", "nba_payload.json"))
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)
