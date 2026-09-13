"""
Forward paper test for the player-prop feature.

Every prop the site prices is written down ONCE, at the decision time (the first poll at or
after kickoff - 60 min, never later than kickoff), with the quote it saw, the quote's own
fetch time, the projection, the probability, the price-derived numbers, the model version,
the data version and the decision the shipped filters made. Rows are appended and never
edited. After the game, each row is graded from the box score, at the odds it recorded,
with pushes voided and non-appearances voided, and compared with the closing quote.

This does not reconstruct any past offer. It starts the day it is switched on.

Tables (live/data):
  paper_trail.ndjson   one row per prop at decision time (see FIELDS)
  paper_grades.ndjson  one row per graded prop: result, return, closing line/price, CLV
"""
import json
import os
from datetime import datetime, timedelta, timezone

from . import store
from .weather import kickoff_utc, _iso

DECISION_MIN = 60          # decide this many minutes before kickoff
STALE_MIN = 180            # a quote fetched more than this long before the decision is not "the quote at the decision"
ACTUAL_COL = {"passing_yards": "passing_yards", "qb_rushing_yards": "rushing_yards", "rushing_yards": "rushing_yards",
              "rb_receiving_yards": "receiving_yards", "receiving_yards": "receiving_yards", "receptions": "receptions",
              "passing_tds": "passing_tds", "rushing_tds": "rushing_tds", "receiving_tds": "receiving_tds"}
MARKET = {"passing_yards": "player_pass_yds", "qb_rushing_yards": "player_rush_yds", "rushing_yards": "player_rush_yds",
          "rb_receiving_yards": "player_reception_yds", "receiving_yards": "player_reception_yds", "receptions": "player_receptions"}


def _parse(iso):
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _payout(price):
    return price / 100.0 if price > 0 else 100.0 / abs(price)


def _breakeven(price):
    o = float(price)
    return abs(o) / (abs(o) + 100) if o < 0 else 100 / (o + 100)


def record(payload, state, now=None, log=print):
    """Write the decision-time snapshot for every game whose decision time has arrived."""
    now = now or datetime.now(timezone.utc)
    meta = payload.get("props_meta") or {}
    fetched = _parse(meta.get("fetched"))
    done = state.setdefault("paper", {})
    n_rows, n_games = 0, 0
    for g in payload.get("games") or []:
        gid = g.get("game_id")
        if gid in done:
            continue
        kick = kickoff_utc(g.get("gameday_iso"), g.get("kickoff"))
        if kick is None:
            continue
        cutoff = kick - timedelta(minutes=DECISION_MIN)
        if now < cutoff:
            continue
        if now >= kick:
            done[gid] = {"at": _iso(now), "status": "missed", "note": "first poll after the decision window was past kickoff"}
            log(f"  paper: {gid} missed the decision window (now {_iso(now)}, kickoff {_iso(kick)})")
            continue
        props = g.get("props") or []
        if not props or fetched is None:
            done[gid] = {"at": _iso(now), "status": "no_quotes"}
            continue
        age_min = (now - fetched).total_seconds() / 60
        stale = age_min > STALE_MIN
        rows = []
        for p in props:
            price = p.get("odds")
            rows.append({
                "season": payload.get("season"), "week": payload.get("week"), "game_id": gid,
                "kickoff_utc": _iso(kick), "cutoff_utc": _iso(cutoff), "decided_at": _iso(now),
                "quote_fetched_at": meta.get("fetched"), "quote_age_min": round(age_min, 1), "stale": stale,
                "bookmaker": meta.get("book", "FanDuel"), "quote_source": meta.get("source"),
                "player": p.get("player"), "player_key": p.get("player_key"), "team": p.get("team"),
                "position": p.get("position"), "stat": p.get("stat"), "market": MARKET.get(p.get("stat")),
                "line": p.get("line"), "side": p.get("side"), "odds": price,
                "odds_over": p.get("odds_over"), "odds_under": p.get("odds_under"),
                "line_open": p.get("line_open"), "line_prev": p.get("line_prev"),
                "projection": p.get("projection"), "model_prob": p.get("confidence"),
                "market_prob_novig": p.get("fair"), "market_prob_raw": p.get("implied"),
                "breakeven_prob": None if price is None else round(_breakeven(price), 4),
                "ev_per_unit": p.get("ev"), "score": p.get("score"), "tier": p.get("tier"),
                "recommended": bool(p.get("recommended")), "blocked_by": p.get("blocked_by"),
                "robust": p.get("robust"), "evidence": p.get("evidence"),
                "model_version": (payload.get("live_meta") or {}).get("modelVersion") or state.get("model_version"),
                "data_version": payload.get("generated"),
                "status": "pending",
            })
        store.append("paper_trail", rows)
        done[gid] = {"at": _iso(now), "status": "recorded", "rows": len(rows), "stale": stale}
        n_rows += len(rows); n_games += 1
        log(f"  paper: {gid} recorded {len(rows)} props at {_iso(now)} (quote {round(age_min)} min old{', STALE' if stale else ''})")
    return n_games, n_rows


def _closing(player, market, kick):
    """Last recorded FanDuel line/prices for this player+market before kickoff, from prop_lines."""
    path = os.path.join(store.ROOT, "prop_lines.ndjson")
    best = None
    try:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("player") != player or r.get("market") != market:
                    continue
                at = _parse(r.get("at"))
                if at is None or at > kick:
                    continue
                if best is None or at >= _parse(best["at"]):
                    best = r
    except OSError:
        pass
    return best


def grade(stats_df, season, log=print):
    """Grade pending rows whose game has a box score. stats_df: nflverse weekly player stats."""
    import pandas as pd
    trail = store.read("paper_trail")
    graded = {(r.get("game_id"), r.get("player_key"), r.get("stat"), r.get("line"), r.get("side")) for r in store.read("paper_grades")}
    s = stats_df[(stats_df.season == season)]
    have_games = set(s.game_id.unique()) if "game_id" in s.columns else set()
    out = []
    for r in trail:
        key = (r.get("game_id"), r.get("player_key"), r.get("stat"), r.get("line"), r.get("side"))
        if key in graded or r.get("game_id") not in have_games:
            continue
        rows = s[(s.game_id == r["game_id"]) & (s.player_id == r.get("player_key"))]
        col = ACTUAL_COL.get(r.get("stat"))
        g = {"game_id": r["game_id"], "player_key": r.get("player_key"), "player": r.get("player"), "stat": r.get("stat"),
             "line": r.get("line"), "side": r.get("side"), "odds": r.get("odds"), "recommended": r.get("recommended"),
             "stale": r.get("stale"), "model_prob": r.get("model_prob"), "market_prob_novig": r.get("market_prob_novig"),
             "ev_per_unit": r.get("ev_per_unit"), "decided_at": r.get("decided_at"), "graded_at": store.now_iso()}
        if not len(rows) or col is None or col not in rows.columns:
            g.update({"result": "void", "reason": "did not appear in the box score", "actual": None, "return": 0.0})
        else:
            actual = float(rows[col].iloc[0])
            g["actual"] = actual
            line = float(r.get("line"))
            if actual == line:
                g.update({"result": "push", "return": 0.0})
            else:
                won = actual > line if str(r.get("side")).lower() == "over" else actual < line
                g.update({"result": "win" if won else "loss",
                          "return": round(_payout(int(r["odds"])), 4) if won else -1.0})
        # closing line value
        kick = _parse(r.get("kickoff_utc"))
        close = _closing(r.get("player"), r.get("market"), kick) if kick else None
        if close:
            g["close_line"] = close.get("line")
            side = str(r.get("side")).lower()
            g["close_odds"] = close.get("over") if side == "over" else close.get("under")
            if close.get("line") is not None and r.get("line") is not None:
                mv = float(close["line"]) - float(r["line"])
                g["clv_line"] = round(mv if side == "over" else -mv, 2)   # + means the market moved toward our side
            if g.get("close_odds") is not None and r.get("odds") is not None:
                g["clv_prob"] = round(_breakeven(g["close_odds"]) - _breakeven(r["odds"]), 4)  # + means our price was better than the close
        out.append(g)
    if out:
        store.append("paper_grades", out)
        log(f"  paper: graded {len(out)} props")
    return len(out)


def summary():
    """What the Model Performance page shows: counts, hit rate, return, calibration, CLV, by market and side."""
    gr = store.read("paper_grades")
    tr = store.read("paper_trail")
    settled = [g for g in gr if g.get("result") in ("win", "loss")]
    def block(rows):
        w = [g for g in rows if g["result"] in ("win", "loss")]
        n = len(w)
        if not n:
            return {"n": 0}
        wins = sum(1 for g in w if g["result"] == "win")
        ret = sum(g["return"] for g in w)
        clv = [g["clv_prob"] for g in w if g.get("clv_prob") is not None]
        p = wins / n
        se = (p * (1 - p) / n) ** 0.5
        return {"n": n, "wins": wins, "hit": round(p, 4), "ci95": [round(max(0, p - 1.96 * se), 4), round(min(1, p + 1.96 * se), 4)],
                "roi": round(ret / n, 4), "claimed": round(sum(g["model_prob"] or 0 for g in w) / n, 4),
                "clv_prob_mean": round(sum(clv) / len(clv), 4) if clv else None, "clv_n": len(clv),
                "pushes": sum(1 for g in rows if g["result"] == "push"), "voids": sum(1 for g in rows if g["result"] == "void")}
    rec = [g for g in gr if g.get("recommended")]
    out = {"started": min((r.get("decided_at") for r in tr), default=None), "recorded": len(tr), "graded": len(gr),
           "pending": len(tr) - len(gr), "games_recorded": len({r["game_id"] for r in tr}),
           "all": block(gr), "recommended": block(rec),
           "by_side": {sd: block([g for g in rec if str(g.get("side")).lower() == sd]) for sd in ("over", "under")},
           "by_stat": {st: block([g for g in rec if g.get("stat") == st]) for st in sorted({g.get("stat") for g in rec if g.get("stat")})},
           "calibration": []}
    for lo, hi in [(0.5, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        b = [g for g in settled if g.get("model_prob") is not None and lo <= g["model_prob"] < hi]
        if len(b) >= 10:
            out["calibration"].append({"band": f"{lo:.2f}-{min(hi, 1):.2f}", "n": len(b),
                                       "claimed": round(sum(g["model_prob"] for g in b) / len(b), 4),
                                       "won": round(sum(1 for g in b if g["result"] == "win") / len(b), 4)})
    return out
