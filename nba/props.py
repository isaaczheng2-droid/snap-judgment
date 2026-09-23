"""
NBA player props: quotes in, evaluations out, and nothing in between touches a projection.

Quote schema (one row per side of one market at one book at one time):
  {"sport":"nba","game_id":<espn id or null>,"event_ref":<provider event id>,"commence":<iso>,
   "book":"draftkings","market":"player_points","player":"<name as quoted>","player_id":<espn id or null>,
   "line":24.5,"side":"over","price":-115,"price_format":"american",
   "quoted_at":<iso, when the provider says it was last updated>,"fetched_at":<iso, when we pulled it>,
   "source":"odds_api"|"import:<file>","raw":{...provider fields kept verbatim...}}
Quotes are append-only (nba/data/prop_quotes.ndjson on the runner); a re-fetch never rewrites an
earlier row, and a corrected line is a new row with a later fetched_at.

No-vig:   both sides present -> multiplicative (proportional) devig: p_side / (p_over + p_under).
          Disclosed on every evaluation as devig="proportional"; with one side only the raw implied
          probability is kept and devig="none (margin included)".
EV:       per unit staked, with pushes (line is an integer the player can land on) returning the
          stake, and voids (player does not play) returning the stake: EV = P(play) x [p_win x payout
          - p_lose] + (1 - P(play)) x 0 for books that void on DNP. The conditional-on-playing EV is
          reported alongside because it is what the price is about.
Pick'em:  (PrizePicks / Underdog style, no price, fixed multipliers) are evaluated separately with
          their own break-even and are NOT mixed into the sportsbook EV table.
Flags:    small sample, questionable/absent teammate scenario, back-to-back, line moved since first
          seen, stat the model has not beaten its baseline on. "insufficient evidence" is a valid verdict.
"""
import json
import math
import os
import re
import time
import unicodedata
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
QUOTES = os.path.join(DATA, "prop_quotes.ndjson")
BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
BOOKS = ["fanduel", "draftkings"]                # verified individually per fetch; Sleeper is not offered
MARKETS = {                                       # Odds API market key -> simulated stat key
    "player_points": "pts", "player_rebounds": "reb", "player_assists": "ast", "player_threes": "fg3m",
    "player_blocks": "blk", "player_steals": "stl", "player_turnovers": "tov", "player_blocks_steals": "sb",
    "player_points_rebounds_assists": "pra", "player_points_rebounds": "pr", "player_points_assists": "pa",
    "player_rebounds_assists": "ra", "player_double_double": "dd",
}
STAT_LABEL = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "3-pointers", "blk": "Blocks",
              "stl": "Steals", "tov": "Turnovers", "sb": "Steals + blocks", "pra": "Pts + reb + ast",
              "pr": "Pts + reb", "pa": "Pts + ast", "ra": "Reb + ast", "dd": "Double-double"}
QUOTA_FLOOR = 500


# ----------------------------------------------------------------------------- odds maths
def implied(american):
    o = float(american)
    return abs(o) / (abs(o) + 100) if o < 0 else 100 / (o + 100)


def payout(american):
    o = float(american)
    return o / 100.0 if o > 0 else 100.0 / abs(o)


def no_vig(price_side, price_other):
    """Returns (fair_p_side, margin, method)."""
    if price_other is None:
        return implied(price_side), None, "none (margin included)"
    a, b = implied(price_side), implied(price_other)
    return a / (a + b), a + b - 1.0, "proportional"


def ev(p_win, p_push, price, p_play=1.0):
    """Expected net return per unit, conditional on playing and availability-adjusted (void on DNP)."""
    p_lose = max(0.0, 1.0 - p_win - p_push)
    cond = p_win * payout(price) - p_lose
    return {"ev_conditional": round(cond, 4), "ev_availability_adjusted": round(p_play * cond, 4),
            "breakeven_p": round(1 / (1 + payout(price)), 4)}


def pickem_ev(p_win, p_push, multiplier):
    """Pick'em style: fixed multiplier on a win (e.g. 3x on a 2-leg power play per leg-equivalent is
    NOT what we do; we report per-leg break-even and the leg's fair probability). The caller supplies
    the per-leg effective multiplier it wants to test."""
    return {"p_win": round(p_win, 4), "p_push": round(p_push, 4), "breakeven_p": round(1 / multiplier, 4),
            "edge_pp": round((p_win / max(1 - p_push, 1e-9) - 1 / multiplier) * 100, 2)}


# ----------------------------------------------------------------------------- identity
def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[.'\-]", "", s.lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def match_player(name, players, team_hint=None):
    """Exact normalised-name match; a same-name collision is resolved by team when given, else
    returned as ambiguous (None) rather than guessed."""
    n = norm_name(name)
    hits = [p for p in players if norm_name(p["player"]) == n]
    if len(hits) > 1 and team_hint:
        hits = [p for p in hits if p.get("team") == team_hint] or hits
    return hits[0] if len(hits) == 1 else None


# ----------------------------------------------------------------------------- evaluation
def evaluate(quote_over, quote_under, sim_summary, p_play, flags=None, validated=False):
    """quote_over/under: quote rows (either may be None). sim_summary: summarize() output for the
    stat with p_over/p_push at this line. Returns the evaluation record for the page."""
    q = quote_over or quote_under
    stat = MARKETS.get(q["market"]) or q.get("stat")
    s = sim_summary
    line = float(q["line"])
    p_over, p_push = s.get("p_over"), s.get("p_push", 0.0)
    if p_over is None:
        return {"verdict": "insufficient evidence", "why": "no simulation at this line", "market": q["market"], "line": line}
    p_under = max(0.0, 1 - p_over - p_push)
    out = {"market": q["market"], "stat": stat, "label": STAT_LABEL.get(stat, stat), "line": line, "book": q["book"],
           "player": q["player"], "player_id": q.get("player_id"), "quoted_at": q.get("quoted_at"), "fetched_at": q.get("fetched_at"),
           "projection": {"mean": s.get("mean"), "median": s.get("median"), "p10": s.get("p10"), "p90": s.get("p90")},
           "p_over": round(p_over, 4), "p_under": round(p_under, 4), "p_push": round(p_push, 4), "p_play": round(float(p_play), 4),
           "sides": {}, "flags": list(flags or []), "evidence": "validated" if validated else "experimental"}
    po = quote_over["price"] if quote_over else None
    pu = quote_under["price"] if quote_under else None
    for side, price, other, p in (("over", po, pu, p_over), ("under", pu, po, p_under)):
        if price is None:
            continue
        fair, margin, method = no_vig(price, other)
        rec = {"price": price, "implied": round(implied(price), 4), "fair": round(fair, 4), "margin": None if margin is None else round(margin, 4),
               "devig": method, "edge_pp": round((p - fair) * 100, 2)}
        rec.update(ev(p, p_push, price, p_play))
        out["sides"][side] = rec
    best = max(out["sides"].items(), key=lambda kv: kv[1]["ev_conditional"], default=None)
    if best is None:
        out["verdict"] = "insufficient evidence"
        out["why"] = "no price on either side"
    else:
        side, rec = best
        if "no baseline win" in out["flags"] or "small sample" in out["flags"]:
            out["verdict"] = "insufficient evidence"
            out["why"] = "; ".join(f for f in out["flags"] if f in ("no baseline win", "small sample"))
        elif rec["edge_pp"] >= 4 and rec["ev_conditional"] > 0:
            out["verdict"] = f"lean {side}"
            out["why"] = f"model {round((p_over if side == 'over' else p_under) * 100)}% vs fair {round(rec['fair'] * 100)}%"
        else:
            out["verdict"] = "no edge"
            out["why"] = f"model within {abs(rec['edge_pp'])} points of the fair price"
    if not validated:
        out["note"] = "experimental: this stat's P(over) has not been graded against real closing lines yet"
    return out


# ----------------------------------------------------------------------------- storage
def append_quotes(rows, path=QUOTES):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return len(rows)


def read_quotes(path=QUOTES):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def latest_by_key(rows):
    """Newest row per (book, market, player, line, side)."""
    out = {}
    for r in sorted(rows, key=lambda r: r.get("fetched_at") or ""):
        out[(r["book"], r["market"], norm_name(r["player"]), float(r["line"]), r["side"])] = r
    return out


# ----------------------------------------------------------------------------- import
def import_file(path, book="import", fetched_at=None):
    """CSV or JSON of quotes. CSV columns: player, market (odds-api key or stat), line, side,
    price[, book, commence, quoted_at, game]. JSON: a list of quote dicts in the schema above."""
    import csv
    fetched_at = fetched_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    if path.lower().endswith(".json"):
        data = json.load(open(path))
        for r in data:
            r = dict(r); r.setdefault("sport", "nba"); r.setdefault("fetched_at", fetched_at); r.setdefault("source", f"import:{os.path.basename(path)}")
            rows.append(r)
    else:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                mk = r.get("market", "").strip()
                if mk in STAT_LABEL:                       # stat key given; map back to a market key
                    mk = next(k for k, v in MARKETS.items() if v == mk)
                rows.append({"sport": "nba", "game_id": None, "event_ref": r.get("game"), "commence": r.get("commence"),
                             "book": (r.get("book") or book).lower(), "market": mk, "player": r["player"].strip(), "player_id": None,
                             "line": float(r["line"]), "side": r["side"].strip().lower(), "price": int(float(r["price"])),
                             "price_format": "american", "quoted_at": r.get("quoted_at"), "fetched_at": fetched_at,
                             "source": f"import:{os.path.basename(path)}", "raw": dict(r)})
    return rows


# ----------------------------------------------------------------------------- Odds API (runner only)
def fetch_odds_api(key=None, books=BOOKS, markets=tuple(MARKETS), log=print, get=None, path=QUOTES):
    """Pull NBA player props for upcoming events from the listed books, one event request per
    game (cost = markets x regions, US region). The key comes from the environment only; the
    URL is never logged. Returns a status dict for the data-status page; quotes are appended."""
    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    if get is None:
        from odds_api import _get as get
    key = (key or os.environ.get("ODDS_API_KEY") or "").strip()
    status = {"provider": "the-odds-api", "sport_key": "basketball_nba", "books_requested": list(books), "books_seen": [],
              "events": 0, "quotes": 0, "credits_remaining": None, "fetched_at": None, "ok": False, "error": None}
    if not key:
        status["error"] = "ODDS_API_KEY not set"
        log("  nba odds: no key; adapter idle")
        return status
    events, h = get(f"{BASE}/events?apiKey={key}")
    if not isinstance(events, list):
        status["error"] = f"events HTTP {h.get('_status')}: {h.get('_error')}"
        log(f"  nba odds: {status['error']}")
        return status
    status["credits_remaining"] = h.get("x-requests-remaining")
    fetched = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status["fetched_at"] = fetched
    rows, seen = [], set()
    for ev_ in events:
        rem = status["credits_remaining"]
        try:
            if rem is not None and float(rem) < QUOTA_FLOOR:
                status["error"] = f"stopped at credit floor {QUOTA_FLOOR}"
                break
        except ValueError:
            pass
        url = (f"{BASE}/events/{ev_['id']}/odds?apiKey={key}&regions=us&markets={','.join(markets)}"
               f"&oddsFormat=american&bookmakers={','.join(books)}")
        data, h = get(url)
        status["credits_remaining"] = h.get("x-requests-remaining", status["credits_remaining"])
        if not isinstance(data, dict):
            continue
        status["events"] += 1
        for bm in data.get("bookmakers", []):
            seen.add(bm.get("key"))
            for mk in bm.get("markets", []):
                for oc in mk.get("outcomes", []):
                    if oc.get("point") is None or not oc.get("description"):
                        continue
                    rows.append({"sport": "nba", "game_id": None, "event_ref": data.get("id"), "commence": data.get("commence_time"),
                                 "home": data.get("home_team"), "away": data.get("away_team"),
                                 "book": bm.get("key"), "market": mk.get("key"), "player": oc["description"], "player_id": None,
                                 "line": float(oc["point"]), "side": str(oc.get("name", "")).lower(), "price": int(oc["price"]),
                                 "price_format": "american", "quoted_at": mk.get("last_update") or bm.get("last_update"),
                                 "fetched_at": fetched, "source": "odds_api", "raw": {"outcome": oc}})
    status["books_seen"] = sorted(b for b in seen if b)
    status["quotes"] = append_quotes(rows, path) if rows else 0
    status["ok"] = status["events"] > 0
    for b in books:
        if b not in status["books_seen"]:
            log(f"  nba odds: {b} returned no NBA player-prop markets this fetch")
    log(f"  nba odds: {status['events']} events, {status['quotes']} quotes, books {status['books_seen']}, {status['credits_remaining']} credits left")
    return status


def fetch_game_lines(key=None, books=BOOKS, log=print, get=None, path=os.path.join(DATA, "game_lines.ndjson")):
    """Featured h2h/spreads/totals for every upcoming NBA event in one request (cost 3)."""
    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    if get is None:
        from odds_api import _get as get
    key = (key or os.environ.get("ODDS_API_KEY") or "").strip()
    if not key:
        return {"ok": False, "error": "ODDS_API_KEY not set"}
    data, h = get(f"{BASE}/odds?apiKey={key}&regions=us&markets=h2h,spreads,totals&oddsFormat=american&bookmakers={','.join(books)}")
    if not isinstance(data, list):
        return {"ok": False, "error": f"HTTP {h.get('_status')}: {h.get('_error')}"}
    fetched = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for ev_ in data:
        for bm in ev_.get("bookmakers", []):
            for mk in bm.get("markets", []):
                for oc in mk.get("outcomes", []):
                    rows.append({"sport": "nba", "event_ref": ev_.get("id"), "commence": ev_.get("commence_time"), "home": ev_.get("home_team"),
                                 "away": ev_.get("away_team"), "book": bm.get("key"), "market": mk.get("key"), "name": oc.get("name"),
                                 "point": oc.get("point"), "price": oc.get("price"), "quoted_at": mk.get("last_update"), "fetched_at": fetched})
    append_quotes(rows, path)
    return {"ok": True, "events": len(data), "rows": len(rows), "credits_remaining": h.get("x-requests-remaining"), "fetched_at": fetched}
