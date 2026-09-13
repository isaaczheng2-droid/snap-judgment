#!/usr/bin/env python3
"""
Real-market prop backtest: FanDuel quotes at a fixed decision time, from The Odds API's
historical snapshots, scored against projections that only knew what was knowable before
each game.

Three subcommands, because the three steps run in three places:

  prepare   (dev machine, needs data/) -> live/data/history/
              prop_model_2024.json   residual scale + per-side calibration fitted on 2019-2024
                                     walk-forward rows ONLY (2025 untouched)
              projections_2025.csv   walk-forward projections for every 2025 player-game the
                                     site would have projected (models trained on <2025)
              actuals_2025.csv       box-score outcomes for every player who appeared

  collect   (GitHub Actions, needs ODDS_API_KEY) -> live/data/history/quotes_<season>_w<week>.ndjson
              one row per (event, market, player, side) at the latest snapshot at or before
              kickoff - 60 min, with the snapshot's own timestamp kept; a snapshot older
              than STALE_H hours before the cutoff is recorded and flagged, never used.
              Cost: 10 credits x markets returned x regions per event (4 markets -> 40).

  evaluate  (anywhere the three files above are) -> live/data/history/backtest_<season>.json
              joins quotes to projections and outcomes, scores the model, always-under,
              always-over and the rolling-mean rule on the IDENTICAL eligible quotes, at the
              actual prices, with pushes voided and non-appearances voided.

Nothing here invents a quote. A player with no FanDuel offer at the cutoff is not in the
sample; a quote whose snapshot is stale is excluded and counted as excluded.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np

HIST = os.path.join(os.environ.get("SJ_LIVE_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "live", "data"), "history")
BASE = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl"
MARKETS = ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_receptions"]
BOOK = "fanduel"
REGION = "us"
CUTOFF_MIN = 60
STALE_H = 3.0            # a snapshot older than this before the cutoff is not "the quote at the cutoff"
COST_PER_EVENT = 10 * len(MARKETS)

ET = timezone(timedelta(hours=-4))   # NFL schedule times are Eastern; Sep-early Nov is EDT


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- prepare
def prepare(season=2025, fit_through=2024):
    import pandas as pd
    from prop_calibrate import fit_platt, fit_scale, scale_of, logit, sigmoid, OU
    from prop_audit import build
    from test_prop_edge import half_point
    os.makedirs(HIST, exist_ok=True)
    d = build()
    d = d[d.stat.isin(OU)].copy()
    hist = d[(d.season <= fit_through)].dropna(subset=["median"])
    # residual scale + quantiles per stat, on the fit window only
    stats_ = {}
    for st in OU:
        h = hist[hist.stat == st]
        res = h.actual.values - h.proj.values
        a, b = fit_scale(h.proj.values, res)
        z = np.sort(res / scale_of(a, b, h.proj.values))
        stats_[st] = {"a": float(a), "b": float(b), "z": [float(v) for v in np.quantile(z, np.linspace(0, 1, 201))],
                      "n": int(len(z)), "fit_seasons": [int(hist.season.min()), int(fit_through)]}
    # per-side calibration: walk-forward raw probabilities inside the fit window, against the
    # synthetic line (the only line those seasons have), then Platt per side. This is the
    # same recipe as the shipped file, restricted to the fit window.
    hist = hist.assign(line=half_point(hist["median"]))
    hist["over"] = hist.proj > hist.line
    hist["won"] = np.where(hist.over, hist.actual > hist.line, hist.actual < hist.line)
    rows = []
    for s in sorted(hist.season.unique()):
        h0, c = hist[hist.season < s], hist[hist.season == s]
        if len(h0) < 2000:
            continue
        for st in OU:
            hh, cc = h0[h0.stat == st], c[c.stat == st]
            if len(hh) < 300 or not len(cc):
                continue
            res = hh.actual.values - hh.proj.values
            a, b = fit_scale(hh.proj.values, res)
            zs = np.sort(res / scale_of(a, b, hh.proj.values))
            thr = (cc.line.values - cc.proj.values) / scale_of(a, b, cc.proj.values)
            p_over = 1.0 - np.searchsorted(zs, thr) / len(zs)
            rows.append(pd.DataFrame({"p_raw": np.where(cc.over.values, p_over, 1 - p_over),
                                      "won": cc.won.values, "over": cc.over.values}))
    prev = pd.concat(rows, ignore_index=True)
    cal = {}
    for side, nm in [(True, "over"), (False, "under")]:
        h = prev[prev.over == side]
        a, b = fit_platt(h.p_raw.values, h.won.values.astype(float))
        cal[nm] = {"a": a, "b": b}
    json.dump({"stats": stats_, "calibration": cal, "fit_through": fit_through,
               "note": "fitted on walk-forward rows through the fit season only; the evaluated season never touched"},
              open(os.path.join(HIST, f"prop_model_{fit_through}.json"), "w"))
    # projections for the evaluated season: one row per player-game-stat
    t = d[d.season == season][["season", "week", "game_id", "player_id", "player_display_name", "position", "team",
                                "opponent_team", "stat", "proj", "mean", "median", "actual", "vol_actual", "vol_proj", "n_prior"]]
    t.to_csv(os.path.join(HIST, f"projections_{season}.csv"), index=False)
    # actuals for everyone who appeared (settlement + DNP detection)
    wk_path = f"data/stats_player_week_{season}.parquet"
    s = pd.read_parquet(wk_path) if os.path.exists(wk_path) else pd.read_parquet("data/player_stats_all.parquet")
    s = s[(s.season == season) & (s.season_type == "REG")]
    tc = "recent_team" if "recent_team" in s.columns else "team"
    keep = ["player_id", "player_display_name", "position", tc, "week", "attempts", "passing_yards", "carries",
            "rushing_yards", "targets", "receptions", "receiving_yards"]
    a = s[[c for c in keep if c in s.columns]].rename(columns={tc: "team"})
    a.to_csv(os.path.join(HIST, f"actuals_{season}.csv"), index=False)
    log(f"prepared {HIST}: model fitted through {fit_through} ({ {k: v['n'] for k, v in stats_.items()} }), "
        f"{len(t)} projection rows, {len(a)} actual rows for {season}")


# ----------------------------------------------------------------------------- collect
def _get(url, timeout=30):
    """(parsed json or None, headers). Never raises; never logs the URL (the key is in it)."""
    try:
        r = subprocess.run(["curl", "-sSL", "-D", "-", "--max-time", str(timeout), url], capture_output=True, timeout=timeout + 10)
    except Exception as e:
        return None, {"_status": "0", "_error": f"curl: {e}"[:200]}
    if r.returncode != 0:
        return None, {"_status": "0", "_error": f"curl exit {r.returncode}: {r.stderr.decode('utf-8', 'replace').strip()[:160]}"}
    raw = r.stdout.decode("utf-8", "replace")
    head, _, body = raw.partition("\r\n\r\n")
    if not body:
        head, _, body = raw.partition("\n\n")
    # a redirect or a 100-continue leaves a second header block in front of the body
    while body.lstrip().upper().startswith("HTTP/"):
        head, _, body = body.partition("\r\n\r\n") if "\r\n\r\n" in body else body.partition("\n\n")
    hdrs, status = {}, ""
    for ln in head.splitlines():
        if ln.upper().startswith("HTTP/"):
            parts = ln.split(); status = parts[1] if len(parts) > 1 else ""
        elif ":" in ln:
            k, _, v = ln.partition(":"); hdrs[k.strip().lower()] = v.strip()
    hdrs["_status"] = status
    try:
        return json.loads(body), hdrs
    except Exception:
        hdrs["_error"] = body.strip()[:200] or "unparseable response"
        return None, hdrs


def _kickoff_utc(gameday, gametime):
    h, m = [int(x) for x in str(gametime).split(":")[:2]]
    return datetime.fromisoformat(str(gameday)).replace(hour=h, minute=m, tzinfo=ET).astimezone(timezone.utc)


def _schedule(season, weeks, datadir="data"):
    import pandas as pd
    p = os.path.join(datadir, "games.csv")
    if not os.path.exists(p):
        os.makedirs(datadir, exist_ok=True)
        subprocess.run(["curl", "-sSL", "--max-time", "60", "-o", p,
                        "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"], check=True)
    g = pd.read_csv(p)
    g = g[(g.season == season) & (g.game_type == "REG") & (g.week.isin(weeks))]
    return g[["game_id", "season", "week", "gameday", "gametime", "home_team", "away_team"]]


def collect(season, weeks, cutoff_min=CUTOFF_MIN, closing=False, max_events=None, max_credits=None, datadir="data"):
    from live import teams
    key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not key:
        log("ODDS_API_KEY not set; nothing collected"); return 2
    os.makedirs(HIST, exist_ok=True)
    sched = _schedule(season, weeks, datadir)
    names = {teams.NAMES[c]: c for c in teams.NAMES}
    ping, h = _get(f"https://api.the-odds-api.com/v4/sports?apiKey={key}")
    log(f"pre-flight: HTTP {h.get('_status')}, {h.get('x-requests-remaining', '?')} credits remaining"
        f"{' (' + h['_error'] + ')' if h.get('_error') else ''}")
    if not isinstance(ping, list):
        log("the key was not accepted; nothing collected"); return 2
    used_total, remaining, n_events, n_rows = 0, None, 0, 0
    log_rows = []
    for wk in sorted(set(weeks)):
        out_path = os.path.join(HIST, f"quotes_{season}_w{wk:02d}.ndjson")
        done = set()
        if os.path.exists(out_path):
            for line in open(out_path):
                try:
                    done.add(json.loads(line)["game_id"])
                except Exception:
                    pass
        f = open(out_path, "a")
        for g in sched[sched.week == wk].itertuples():
            if g.game_id in done:
                continue
            if max_events is not None and n_events >= max_events:
                break
            if max_credits is not None and used_total + COST_PER_EVENT > max_credits:
                log(f"credit budget {max_credits} reached; stopping"); break
            kick = _kickoff_utc(g.gameday, g.gametime)
            snaps = [("t-60", kick - timedelta(minutes=cutoff_min))] + ([("close", kick)] if closing else [])
            # find the event id: the historical events list at the cutoff, narrowed to this kickoff
            d0 = snaps[0][1]
            ev, h = _get(f"{BASE}/events?apiKey={key}&date={d0.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                         f"&commenceTimeFrom={(kick - timedelta(hours=6)).strftime('%Y-%m-%dT%H:%M:%SZ')}"
                         f"&commenceTimeTo={(kick + timedelta(hours=6)).strftime('%Y-%m-%dT%H:%M:%SZ')}")
            remaining = h.get("x-requests-remaining", remaining)
            used_total += int(float(h.get("x-requests-last", 0) or 0))
            if not isinstance(ev, dict) or not isinstance(ev.get("data"), list):
                log(f"  {g.game_id}: events list failed HTTP {h.get('_status')} {h.get('_error', '')}")
                log_rows.append({"game_id": g.game_id, "status": "events_failed", "http": h.get("_status"), "error": h.get("_error")})
                continue
            eid = None
            for e in ev["data"]:
                if names.get(e.get("home_team")) == g.home_team and names.get(e.get("away_team")) == g.away_team:
                    eid = e.get("id"); commence = e.get("commence_time"); break
            if not eid:
                log(f"  {g.game_id}: no event in the snapshot at {d0.isoformat()}")
                log_rows.append({"game_id": g.game_id, "status": "no_event"})
                continue
            n_events += 1
            for tag, when in snaps:
                url = (f"{BASE}/events/{eid}/odds?apiKey={key}&regions={REGION}&markets={','.join(MARKETS)}"
                       f"&oddsFormat=american&bookmakers={BOOK}&date={when.strftime('%Y-%m-%dT%H:%M:%SZ')}")
                data, h = _get(url)
                remaining = h.get("x-requests-remaining", remaining)
                cost = int(float(h.get("x-requests-last", 0) or 0)); used_total += cost
                if not isinstance(data, dict) or not isinstance(data.get("data"), dict):
                    log(f"  {g.game_id} {tag}: odds failed HTTP {h.get('_status')} {h.get('_error', '')}")
                    log_rows.append({"game_id": g.game_id, "snapshot": tag, "status": "odds_failed", "http": h.get("_status"), "error": h.get("_error")})
                    continue
                ts = data.get("timestamp")
                try:
                    age_h = (when - datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds() / 3600
                except Exception:
                    age_h = None
                stale = age_h is None or age_h > STALE_H
                rows_here = 0
                for bm in data["data"].get("bookmakers", []):
                    if bm.get("key") != BOOK:
                        continue
                    for mk in bm.get("markets", []):
                        for oc in mk.get("outcomes", []):
                            if oc.get("description") is None or oc.get("point") is None:
                                continue
                            f.write(json.dumps({
                                "season": season, "week": wk, "game_id": g.game_id, "event_id": eid,
                                "home": g.home_team, "away": g.away_team, "commence": commence,
                                "kickoff_utc": kick.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "snapshot": tag, "requested_at": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "snapshot_ts": ts, "snapshot_age_h": None if age_h is None else round(age_h, 3),
                                "stale": stale, "bookmaker": BOOK, "market_last_update": mk.get("last_update"),
                                "market": mk.get("key"), "player": oc.get("description"),
                                "side": str(oc.get("name")).lower(), "line": float(oc["point"]),
                                "price": int(oc.get("price")) if oc.get("price") is not None else None,
                            }, separators=(",", ":")) + "\n")
                            rows_here += 1
                n_rows += rows_here
                log(f"  {g.game_id} {tag}: {rows_here} outcomes, snapshot {ts} (age {age_h and round(age_h, 2)}h{', STALE' if stale else ''}), cost {cost}, {remaining} left")
                log_rows.append({"game_id": g.game_id, "snapshot": tag, "status": "ok", "rows": rows_here,
                                 "snapshot_ts": ts, "stale": stale, "cost": cost})
        f.close()
    summary = {"season": season, "weeks": sorted(set(weeks)), "events": n_events, "rows": n_rows,
               "credits_used": used_total, "credits_remaining": remaining, "cutoff_min": cutoff_min,
               "closing": closing, "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "log": log_rows}
    json.dump(summary, open(os.path.join(HIST, f"collect_{season}_{int(time.time())}.json"), "w"), indent=1)
    log(f"collected {n_rows} outcome rows over {n_events} events for {used_total} credits ({remaining} remaining)")
    return 0


# ---------------------------------------------------------------------------- evaluate
STAT_FOR = {  # market + position -> our stat key
    "player_pass_yds": {"QB": "passing_yards"},
    "player_rush_yds": {"QB": "qb_rushing_yards", "RB": "rushing_yards"},
    "player_reception_yds": {"WR": "receiving_yards", "TE": "receiving_yards", "RB": "rb_receiving_yards"},
    "player_receptions": {"WR": "receptions", "TE": "receptions", "RB": "receptions"},
}
ACTUAL_COL = {"passing_yards": "passing_yards", "qb_rushing_yards": "rushing_yards", "rushing_yards": "rushing_yards",
              "rb_receiving_yards": "receiving_yards", "receiving_yards": "receiving_yards", "receptions": "receptions"}


def _norm(name):
    import re, unicodedata
    n = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n)
    return re.sub(r"[^a-z]", "", n)


def _payout(price, stake=1.0):
    return stake * (price / 100.0) if price > 0 else stake * (100.0 / abs(price))


def evaluate(season, model_path=None, min_p=0.65):
    import pandas as pd
    import prop_value
    model_path = model_path or os.path.join(HIST, "prop_model_2024.json")
    model = prop_value.PropModel(model_path)
    if not model.ok:
        log(f"no model at {model_path}"); return 2
    proj = pd.read_csv(os.path.join(HIST, f"projections_{season}.csv"))
    act = pd.read_csv(os.path.join(HIST, f"actuals_{season}.csv"))
    proj["nk"] = proj.player_display_name.map(_norm)
    act["nk"] = act.player_display_name.map(_norm)
    quotes = []
    for fn in sorted(os.listdir(HIST)):
        if fn.startswith(f"quotes_{season}_w") and fn.endswith(".ndjson"):
            for line in open(os.path.join(HIST, fn)):
                try:
                    quotes.append(json.loads(line))
                except Exception:
                    pass
    if not quotes:
        log("no quotes collected yet"); return 2
    q = pd.DataFrame(quotes)
    q = q[q.snapshot == "t-60"]
    n_stale = int(q.stale.sum())
    q = q[~q.stale]
    # pair over/under prices per (game, market, player, line)
    key = ["game_id", "week", "market", "player", "line"]
    over = q[q.side == "over"][key + ["price", "snapshot_ts", "market_last_update"]].rename(columns={"price": "over"})
    under = q[q.side == "under"][key + ["price"]].rename(columns={"price": "under"})
    m = over.merge(under, on=key, how="outer")
    m["nk"] = m.player.map(_norm)
    rows = []
    for r in m.itertuples():
        a = act[(act.week == r.week) & (act.nk == r.nk)]
        pr = proj[(proj.week == r.week) & (proj.nk == r.nk)]
        pos = (pr.position.iloc[0] if len(pr) else (a.position.iloc[0] if len(a) else None))
        stat = STAT_FOR.get(r.market, {}).get(pos)
        row = {"game_id": r.game_id, "week": r.week, "market": r.market, "player": r.player, "line": r.line,
               "over": r.over, "under": r.under, "snapshot_ts": r.snapshot_ts, "position": pos, "stat": stat}
        if stat is None:
            row["status"] = "no_position_match"; rows.append(row); continue
        # settlement
        if not len(a):
            row["status"] = "void_dnp"; rows.append(row); continue
        actual = float(a[ACTUAL_COL[stat]].iloc[0]) if ACTUAL_COL[stat] in a.columns else np.nan
        row["actual"] = actual
        if np.isnan(actual):
            row["status"] = "no_actual"; rows.append(row); continue
        row["push"] = actual == r.line
        pr = pr[pr.stat == stat]
        if not len(pr):
            row["status"] = "no_projection"; rows.append(row); continue
        p = pr.iloc[0]
        row["proj"] = float(p.proj); row["hist_mean"] = float(p["mean"]); row["n_prior"] = int(p.n_prior)
        ev = prop_value.evaluate(stat, float(p.proj), r.line,
                                 None if pd.isna(r.over) else int(r.over), None if pd.isna(r.under) else int(r.under), model)
        if ev is None:
            row["status"] = "no_price"; rows.append(row); continue
        row.update({"status": "scored", "side": ev["side"], "odds": ev["odds"], "confidence": ev["confidence"],
                    "fair": ev["fair"], "implied": ev["implied"], "score": ev["score"], "recommended": ev["recommended"],
                    "edge_pct": ev["edge_pct"]})
        rows.append(row)
    R = pd.DataFrame(rows)
    S = R[R.status == "scored"].copy()
    S["model_over"] = S.side == "Over"
    S["won"] = np.where(S.model_over, S.actual > S.line, S.actual < S.line)
    S["hist_over"] = S.hist_mean > S.line
    S["hist_won"] = np.where(S.hist_over, S.actual > S.line, S.actual < S.line)
    S["under_won"] = S.actual < S.line
    S["over_won"] = S.actual > S.line
    def ret(df, take_over, won):
        # unit stake at the actual price of the side taken; pushes return 0
        out = []
        for push, o, u, t, w in zip(df.push.values, df.over.values, df.under.values, take_over, won):
            if push:
                out.append(0.0); continue
            price = o if t else u
            if price is None or (isinstance(price, float) and np.isnan(price)):
                out.append(np.nan); continue
            out.append(_payout(price) if w else -1.0)
        return np.array(out, dtype=float)
    S["ret_model"] = ret(S, S.model_over.values, S.won.values)
    S["ret_hist"] = ret(S, S.hist_over.values, S.hist_won.values)
    S["ret_under"] = ret(S, np.zeros(len(S), bool), S.under_won.values)
    S["ret_over"] = ret(S, np.ones(len(S), bool), S.over_won.values)
    def block(df, label):
        d = df[~df.push]
        n = len(d)
        if n == 0:
            return {"label": label, "n": 0}
        from scipy import stats as st
        lo, hi = st.binomtest(int(d.won.sum()), n, 0.5).proportion_ci(0.95)
        # cluster by game for the CI on the model's hit rate
        g = d.groupby("game_id").won.agg(["sum", "size"])
        p_hat = g["sum"].sum() / g["size"].sum(); mg = len(g)
        se = np.sqrt(((g["sum"] - p_hat * g["size"]) ** 2).sum() / max(mg - 1, 1) * mg) / g["size"].sum() if mg > 1 else np.nan
        return {"label": label, "n": int(n), "pushes": int(df.push.sum()), "games": int(mg),
                "model": {"hit": float(d.won.mean()), "ci95": [float(lo), float(hi)],
                          "ci95_cluster": [float(p_hat - 1.96 * se), float(p_hat + 1.96 * se)] if mg > 1 else None,
                          "roi": float(np.nanmean(d.ret_model)), "claimed": float(d.confidence.mean()),
                          "under_share": float(1 - d.model_over.mean())},
                "always_under": {"hit": float(d.under_won.mean()), "roi": float(np.nanmean(d.ret_under))},
                "always_over": {"hit": float(d.over_won.mean()), "roi": float(np.nanmean(d.ret_over))},
                "rolling_mean_rule": {"hit": float(d.hist_won.mean()), "roi": float(np.nanmean(d.ret_hist)),
                                      "agrees_with_model": float((d.hist_over == d.model_over).mean())},
                "breakeven_at_actual_odds": float(np.mean([prop_value.implied_probability(o) for o in d.odds]))}
    out = {"season": season, "decision_time": "kickoff - 60 min", "bookmaker": BOOK, "model": os.path.basename(model_path),
           "quotes_total": int(len(m)), "stale_excluded": n_stale,
           "status_counts": {k: int(v) for k, v in R.status.value_counts().items()},
           "all_scored": block(S, "every eligible quote"),
           "recommended": block(S[S.recommended], "quotes the shipped filters would have recommended"),
           "by_market": {mk: block(S[S.market == mk], mk) for mk in sorted(S.market.unique())},
           "by_side": {sd: block(S[S.model_over == (sd == "over")], sd) for sd in ("over", "under")},
           "recommended_by_market": {mk: block(S[(S.recommended) & (S.market == mk)], mk) for mk in sorted(S.market.unique())},
           "calibration": []}
    for lo_b, hi_b in [(0.5, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        b = S[(S.confidence >= lo_b) & (S.confidence < hi_b) & (~S.push)]
        if len(b) >= 20:
            out["calibration"].append({"band": f"{lo_b:.2f}-{min(hi_b, 1):.2f}", "n": int(len(b)),
                                       "claimed": float(b.confidence.mean()), "won": float(b.won.mean())})
    json.dump(out, open(os.path.join(HIST, f"backtest_{season}.json"), "w"), indent=1, default=str)
    S.to_csv(os.path.join(HIST, f"backtest_{season}_rows.csv"), index=False)
    log(json.dumps({k: out[k] for k in ("quotes_total", "stale_excluded", "status_counts")}))
    log(json.dumps(out["all_scored"], indent=1)); log(json.dumps(out["recommended"], indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "collect", "evaluate"])
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--weeks", default="1")
    ap.add_argument("--fit-through", type=int, default=2024)
    ap.add_argument("--cutoff-min", type=int, default=CUTOFF_MIN)
    ap.add_argument("--closing", action="store_true")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--max-credits", type=int, default=None)
    ap.add_argument("--datadir", default="data")
    a = ap.parse_args()
    weeks = []
    for part in str(a.weeks).split(","):
        if "-" in part:
            lo, hi = part.split("-"); weeks += list(range(int(lo), int(hi) + 1))
        elif part.strip():
            weeks.append(int(part))
    if a.cmd == "prepare":
        return prepare(a.season, a.fit_through)
    if a.cmd == "collect":
        return collect(a.season, weeks, a.cutoff_min, a.closing, a.max_events, a.max_credits, a.datadir)
    return evaluate(a.season)


if __name__ == "__main__":
    sys.exit(main() or 0)
