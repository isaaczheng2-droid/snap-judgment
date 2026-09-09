#!/usr/bin/env python3
"""
The season tracker: what the model said before kickoff, against what happened.

The point of this file is that predictions are LOCKED. Every run, the upcoming week's
picks, player projections and scheme reads are written into history.json the first time
they are seen and never rewritten. Later runs only fill in results. That matters because
the model is retrained daily on more data — without locking, "what we predicted" would
quietly drift toward "what we would predict now that we know", which is not a forecast,
it is a memory.

The one exception is clearly marked: 2025 rows carry source="backtest" and come from the
walk-forward audit, a model that had only seen 2016-2024. They are honest out-of-sample
predictions but they were never published in advance, so they are labelled differently
from the live ones and the site says so.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

VERSION = 1
STAT_SOURCE = {                     # projection key -> column in the weekly player box score
    "passing_yards": "passing_yards", "passing_tds": "passing_tds",
    "qb_rushing_yards": "rushing_yards", "rushing_yards": "rushing_yards",
    "rushing_tds": "rushing_tds", "rb_receiving_yards": "receiving_yards",
    "receiving_yards": "receiving_yards", "receptions": "receptions",
    "receiving_tds": "receiving_tds",
}
SCHEME_TRACK = ["pass_rate", "adot", "pace", "sack_rate", "pressure"]


def load(path):
    if path and os.path.exists(path):
        try:
            h = json.load(open(path))
            if h.get("version") == VERSION:
                for k in ("games", "players", "schemes", "rollups"):
                    h.setdefault(k, {})
                return h
        except Exception:
            pass
    return {"version": VERSION, "games": {}, "players": {}, "schemes": {}, "rollups": {}}


def save(h, path):
    json.dump(h, open(path, "w"), separators=(",", ":"))


# --------------------------------------------------------------------------- locking
def lock_week(h, up, players, scheme, cur, week, source="live"):
    """Record this week's predictions, once. Existing keys are never overwritten."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    n_new = 0

    for _, r in up.iterrows():
        gid = str(r.game_id)
        if gid in h["games"]:
            continue
        h["games"][gid] = {
            "s": int(cur), "w": int(week), "a": r.away_team, "hm": r.home_team,
            "p": round(float(r.p_blend), 4), "pm": round(float(r.p_model), 4),
            "pk": None if pd.isna(r.p_market) else round(float(r.p_market), 4),
            "mg": round(float(r.margin_pred), 2),
            "sp": None if pd.isna(r.spread_line) else float(r.spread_line),
            "tot": None if pd.isna(r.total_line) else float(r.total_line),
            "pick": r.predicted_winner, "src": source, "at": ts,
        }
        n_new += 1

    gid_for = {}
    for _, r in up.iterrows():
        gid_for[r.home_team] = str(r.game_id)
        gid_for[r.away_team] = str(r.game_id)

    for p in players or []:
        gid = gid_for.get(p.get("team"))
        pid = p.get("player_key")
        if not gid or not pid:
            continue
        for stat in STAT_SOURCE:
            v = p.get(stat)
            if v is None:
                continue
            k = f"{gid}|{pid}|{stat}"
            if k in h["players"]:
                continue
            h["players"][k] = {
                "s": int(cur), "w": int(week), "n": p.get("player_display_name"),
                "t": p.get("team"), "pos": p.get("position"), "st": stat,
                "proj": round(float(v), 1), "src": source, "at": ts,
            }
            n_new += 1

    for row in scheme or []:
        k = f"{cur}_{week}_{row['team']}"
        if k in h["schemes"]:
            continue
        h["schemes"][k] = dict(row, s=int(cur), w=int(week), src=source, at=ts)
        n_new += 1

    return n_new


# --------------------------------------------------------------------------- grading
def grade(h, sched, plyr, per_game_scheme):
    """Fill in actuals for anything locked whose game has since finished."""
    done = sched[sched.home_score.notna()].set_index("game_id")
    graded = 0

    for gid, g in h["games"].items():
        if "act" in g or gid not in done.index:
            continue
        r = done.loc[gid]
        hs, as_ = float(r.home_score), float(r.away_score)
        margin = hs - as_
        winner = r.home_team if margin > 0 else r.away_team if margin < 0 else "TIE"
        # against the spread, from the home side; spread_line > 0 means home favored
        ats = None
        if g.get("sp") is not None and margin != g["sp"]:
            covered_home = margin > g["sp"]
            ats = bool((g["mg"] > g["sp"]) == covered_home)
        g["act"] = {
            "hs": hs, "as": as_, "mg": margin, "win": winner,
            "ok": None if winner == "TIE" else bool(winner == g["pick"]),
            "ats": ats, "err": round(abs(g["mg"] - margin), 2),
            "tot": None if g.get("tot") is None else bool((hs + as_) > g["tot"]),
        }
        graded += 1

    if len(plyr):
        pb = plyr[plyr.season_type == "REG"]
        idx = {}
        for _, r in pb.iterrows():
            idx[(str(r.game_id), str(r.player_id))] = r
        for k, p in h["players"].items():
            if "act" in p:
                continue
            gid, pid, stat = k.split("|")
            if gid not in done.index:
                continue
            row = idx.get((gid, pid))
            if row is None:
                # the game finished and he recorded nothing — that is a real zero, not a gap
                p["act"] = 0.0
                p["err"] = round(abs(p["proj"]), 2)
                p["dnp"] = True
            else:
                v = row.get(STAT_SOURCE[stat])
                v = 0.0 if v is None or pd.isna(v) else float(v)
                p["act"] = round(v, 1)
                p["err"] = round(abs(p["proj"] - v), 2)
            graded += 1

    if per_game_scheme is not None and len(per_game_scheme):
        act = per_game_scheme.set_index(["season", "week", "team"])
        for k, s in h["schemes"].items():
            if "act" in s or str(s.get("game_id") or "") not in done.index:
                continue
            key = (s["s"], s["w"], s["team"])
            if key not in act.index:
                continue
            a = act.loc[key]
            if isinstance(a, pd.DataFrame):
                a = a.iloc[0]
            s["act"] = {m: (None if pd.isna(a.get(m)) else round(float(a.get(m)), 4))
                        for m in SCHEME_TRACK}
            graded += 1

    return graded


# --------------------------------------------------------------------------- pruning
# A season of player props is ~12,000 rows, about 1.5 MB, and this file is committed on
# every daily run. Rather than carry that forever, graded rows past a short window are
# folded into exact running sums — count, absolute error, and the sums needed to recover
# mean, bias and correlation — and then dropped. The aggregates below are not estimates:
# they reproduce the same numbers the raw rows would have given.
def _blank():
    return {"n": 0, "se": 0.0, "sp": 0.0, "sa": 0.0, "spp": 0.0, "saa": 0.0, "spa": 0.0,
            "worst": [], "best": []}


# Touchdown counts are 0, 1 or 2, so "closest call" on them is a coin landing the right way
# up, not a good projection — a list of 1 -> 1 says nothing. Only the stats with real range
# are eligible for the best/worst boards.
RANKABLE = {"passing_yards", "rushing_yards", "receiving_yards",
            "qb_rushing_yards", "rb_receiving_yards", "receptions"}


def _fold(agg, p):
    pr, ac = float(p["proj"]), float(p["act"])
    agg["n"] += 1
    agg["se"] += abs(pr - ac)
    agg["sp"] += pr
    agg["sa"] += ac
    agg["spp"] += pr * pr
    agg["saa"] += ac * ac
    agg["spa"] += pr * ac
    if p["st"] not in RANKABLE:
        return
    keep = {"n": p.get("n"), "t": p.get("t"), "st": p["st"], "w": p["w"],
            "proj": p["proj"], "act": p["act"], "err": p.get("err", round(abs(pr - ac), 2))}
    if p.get("dnp"):
        keep["dnp"] = True
    # a player who never took the field is an availability miss, not a bad projection —
    # they still belong on the board, but labelled, and never as a "closest call"
    agg["worst"] = sorted(agg["worst"] + [keep], key=lambda x: -x["err"])[:6]
    if not p.get("dnp"):
        agg["best"] = sorted(agg["best"] + [keep], key=lambda x: x["err"])[:6]


def prune_players(h, cur, keep_weeks=4):
    """Fold graded player rows into per-season aggregates, keeping recent weeks in detail."""
    live_weeks = sorted({p["w"] for p in h["players"].values() if p["s"] == cur})
    cutoff = (live_weeks[-keep_weeks] if len(live_weeks) > keep_weeks else -1)
    dropped = 0
    for k in list(h["players"]):
        p = h["players"][k]
        if "act" not in p:
            continue
        if p["s"] == cur and p["w"] >= cutoff:
            continue
        agg = h["rollups"].setdefault(str(p["s"]), {}).setdefault("players", {})
        _fold(agg.setdefault(p["st"], _blank()), p)
        del h["players"][k]
        dropped += 1
    return dropped


def _from_agg(stat, a):
    n = a["n"]
    if not n:
        return None
    mp, ma = a["sp"] / n, a["sa"] / n
    vp, va = a["spp"] / n - mp * mp, a["saa"] / n - ma * ma
    cov = a["spa"] / n - mp * ma
    corr = round(cov / (vp ** 0.5 * va ** 0.5), 3) if vp > 1e-9 and va > 1e-9 else None
    # `sd` is the spread of the ACTUAL results, reported so the MAE has a scale to be read
    # against. It is deliberately not framed as a flat-guess baseline: a standard deviation
    # is an RMS and the MAE is a mean-absolute, so putting them head to head would flatter
    # the projection. The Accuracy tab computes a like-for-like baseline from raw rows.
    return {"stat": stat, "n": n, "mae": round(a["se"] / n, 2),
            "sd": round(va ** 0.5, 2), "mean": round(ma, 2),
            "bias": round(mp - ma, 2), "corr": corr}


# --------------------------------------------------------------------------- summarize
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def summarize(h, cur, row_cap=400):
    """The block the page renders. Aggregates plus per-game rows, current season first."""
    games = [dict(g, gid=k) for k, g in h["games"].items()]
    seasons = sorted({g["s"] for g in games})
    out = {"season": int(cur), "seasons": seasons, "generated": datetime.now(
        timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

    by_season = {}
    for s in seasons:
        gs = [g for g in games if g["s"] == s and "act" in g]
        pend = [g for g in games if g["s"] == s and "act" not in g]
        ok = [g["act"]["ok"] for g in gs if g["act"]["ok"] is not None]
        ats = [g["act"]["ats"] for g in gs if g["act"]["ats"] is not None]
        err = [g["act"]["err"] for g in gs]
        mk = [g for g in gs if g.get("pk") is not None and g["act"]["ok"] is not None]
        mk_ok = [((g["pk"] > 0.5) == (g["act"]["win"] == g["hm"])) for g in mk]

        weeks = []
        for w in sorted({g["w"] for g in gs}):
            ws = [g for g in gs if g["w"] == w]
            wok = [g["act"]["ok"] for g in ws if g["act"]["ok"] is not None]
            wats = [g["act"]["ats"] for g in ws if g["act"]["ats"] is not None]
            weeks.append({
                "week": w, "n": len(ws),
                "right": int(sum(wok)), "su": _mean(wok),
                "ats": _mean(wats), "ats_n": len(wats),
                "mae": _mean([g["act"]["err"] for g in ws]),
            })
        # running totals so the chart can show the season converging rather than bouncing
        run_ok, run_n = 0, 0
        for wk in weeks:
            run_ok += wk["right"]; run_n += wk["n"]
            wk["cum_su"] = round(run_ok / run_n, 4) if run_n else None

        rows = sorted(gs, key=lambda g: (-g["w"],))[:row_cap]
        # what is locked but not yet played, so the page can show the open bets rather than
        # just asserting that some exist
        pend_rows = [{
            "w": g["w"], "a": g["a"], "h": g["hm"], "pick": g["pick"],
            "p": g["p"], "mg": g["mg"], "sp": g.get("sp"), "at": g.get("at"),
        } for g in sorted(pend, key=lambda g: (g["w"], g["a"]))[:64]]
        by_season[str(s)] = {
            "games": {
                "n": len(gs), "pending": len(pend),
                "su": _mean(ok), "right": int(sum(ok)), "wrong": len(ok) - int(sum(ok)),
                "market_su": _mean(mk_ok), "market_n": len(mk_ok),
                "ats": _mean(ats), "ats_n": len(ats),
                "mae": _mean(err),
                "brier": _mean([(g["p"] - (1.0 if g["act"]["win"] == g["hm"] else 0.0)) ** 2
                                for g in gs if g["act"]["ok"] is not None]),
                "by_week": weeks,
                "pending_rows": pend_rows,
                "rows": [{
                    "w": g["w"], "a": g["a"], "h": g["hm"], "pick": g["pick"],
                    "p": g["p"], "mg": g["mg"], "sp": g.get("sp"),
                    "as": g["act"]["as"], "hs": g["act"]["hs"],
                    "ok": g["act"]["ok"], "ats": g["act"]["ats"], "err": g["act"]["err"],
                    "src": g.get("src", "live"),
                } for g in rows],
            }
        }

        # ---- player props: rolled-up history plus whatever detail is still held ----
        agg = {st: dict(a) for st, a in
               h.get("rollups", {}).get(str(s), {}).get("players", {}).items()}
        for p in h["players"].values():
            if p["s"] == s and "act" in p:
                _fold(agg.setdefault(p["st"], _blank()), p)
        if agg:
            stats = [x for x in (_from_agg(st, a) for st, a in sorted(agg.items())) if x]
            allbest, allworst = [], []
            for a in agg.values():
                allbest += a["best"]; allworst += a["worst"]
            by_season[str(s)]["players"] = {
                "n": sum(a["n"] for a in agg.values()), "by_stat": stats,
                "best": sorted(allbest, key=lambda x: x["err"])[:6],
                "worst": sorted(allworst, key=lambda x: -x["err"])[:6],
            }

        # ---- scheme reads ----
        ss = [x for x in h["schemes"].values() if x["s"] == s and "act" in x]
        if ss:
            mets = []
            for m in SCHEME_TRACK:
                pr = [x.get(m) for x in ss if x.get(m) is not None and x["act"].get(m) is not None]
                ac = [x["act"].get(m) for x in ss if x.get(m) is not None and x["act"].get(m) is not None]
                if len(pr) < 4:
                    continue
                mets.append({
                    "metric": m, "n": len(pr),
                    "mae": round(float(np.mean(np.abs(np.array(pr) - np.array(ac)))), 4),
                    "corr": (round(float(np.corrcoef(pr, ac)[0, 1]), 3)
                             if np.std(pr) > 0 and np.std(ac) > 0 else None),
                })
            by_season[str(s)]["schemes"] = {"n": len(ss), "metrics": mets}

    out["by_season"] = by_season
    cs = by_season.get(str(cur), {}).get("games", {})
    out["pending"] = cs.get("pending", 0)
    return out
