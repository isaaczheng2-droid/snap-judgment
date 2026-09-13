#!/usr/bin/env python3
"""
FanDuel player prop lines, via The Odds API.

WHY NOT FANDUEL DIRECTLY: their own endpoints are not a public API, and pulling pricing off
a sportsbook you have not agreed terms with is not something to build into a job that runs
every hour forever. The Odds API is a licensed aggregator that carries FanDuel's lines, so
the data is the same and the arrangement is not a problem waiting to happen.

CREDITS, WHICH ARE THE REAL CONSTRAINT: player props are only available one event at a time
(`/events/{id}/odds`), and each request costs markets x regions credits. One NFL week is
~16 events x 4 markets = 64 credits per refresh. The plan is 20,000 credits a month (it was
500 when this module was written, which is why everything below is so careful). Budget:
a refresh every 3 hours through the week plus hourly inside 6 hours of a kickoff is about
3,000 a week, 13,000 a month, with the rest as headroom. This module therefore:

  - fetches props at most once every `refresh_hours`, caching to disk in between; the cache
    lives under live/data so it is committed and survives between Actions runs (before that
    the cache sat in the ignored data/ directory, so every rebuild on the runner paid again)
  - reads the remaining quota out of the response headers and logs it every time
  - stops immediately if the quota drops below a floor, rather than silently failing later
  - records every line or price change to live/data/prop_lines.ndjson, so line movement
    (open -> now) and closing-line value can be computed from the site's own history

Set ODDS_API_KEY in the environment (a GitHub Actions secret). With no key the module does
nothing at all and says so, and the rest of the site publishes exactly as before -- there is
no state in which a missing or broken odds feed can stop the predictions going out.

NOTHING HERE FEEDS THE PREDICTION MODEL. This runs after the projections are final. It
cannot reach them, and the projections cannot see it.
"""
import json
import os
import subprocess
import time

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
BOOK = "fanduel"
REGION = "us"
REFRESH_HOURS = 3
REFRESH_HOURS_NEAR = 1     # inside NEAR_KICKOFF_H of a kickoff
NEAR_KICKOFF_H = 6
QUOTA_FLOOR = 500          # stop before the quota is gone, not after (~8 refreshes of slack)
_LIVE = os.environ.get("SJ_LIVE_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "live", "data")
CACHE = os.path.join(_LIVE, "odds_cache.json")
LINES = os.path.join(_LIVE, "prop_lines.ndjson")

# The Odds API market key -> the stat name in our payload. Two of them are position
# dependent: the API has one "rush yards" market, we project QB rushing separately from
# running back rushing, and one "reception yards" market against separate WR/TE and RB
# projections. Resolved per player by position.
MARKETS = {
    "player_pass_yds": {"QB": "passing_yards"},
    "player_rush_yds": {"QB": "qb_rushing_yards", "RB": "rushing_yards"},
    "player_reception_yds": {"WR": "receiving_yards", "TE": "receiving_yards",
                             "RB": "rb_receiving_yards"},
    "player_receptions": {"WR": "receptions", "TE": "receptions", "RB": "receptions"},
}
MARKET_LIST = ",".join(MARKETS)


def _get(url, timeout=25):
    """
    Returns (parsed_json, headers_dict). Never raises.

    `headers` always carries `_status` (the HTTP code) and, on a failure, `_error` (whatever
    the API said went wrong). The first version of this swallowed both and logged only
    "could not list events", which is true, useless, and cost a debugging round trip: the
    API had actually returned a perfectly clear message and nothing was reading it.

    The URL is never logged or returned, because the API key is in it.
    """
    try:
        r = subprocess.run(["curl", "-sSL", "-D", "-", "--max-time", str(timeout), url],
                           capture_output=True, timeout=timeout + 10)
        if r.returncode != 0:
            return None, {"_status": "0", "_error": f"curl exit {r.returncode}"}
        if not r.stdout:
            return None, {"_status": "0", "_error": "empty response"}
        raw = r.stdout.decode("utf-8", "replace")
        head, _, body = raw.partition("\r\n\r\n")
        if not body:
            head, _, body = raw.partition("\n\n")
        hdrs = {}
        status = ""
        for ln in head.splitlines():
            if ln.upper().startswith("HTTP/"):
                parts = ln.split()
                status = parts[1] if len(parts) > 1 else ""
            elif ":" in ln:
                k, _, v = ln.partition(":")
                hdrs[k.strip().lower()] = v.strip()
        hdrs["_status"] = status
        try:
            data = json.loads(body)
        except Exception:
            hdrs["_error"] = body.strip()[:200] or "unparseable response"
            return None, hdrs
        # The Odds API reports problems as an object with a message, not an HTTP-only code
        if isinstance(data, dict) and ("message" in data or "error_code" in data):
            hdrs["_error"] = str(data.get("message") or data.get("error_code"))[:200]
            return None, hdrs
        return data, hdrs
    except Exception as e:
        return None, {"_status": "0", "_error": f"{type(e).__name__}: {e}"}


def _fresh(path, hours):
    try:
        return (time.time() - os.path.getmtime(path)) < hours * 3600
    except OSError:
        return False


def refresh_hours_for(kickoffs_iso, now=None):
    """3h normally, 1h when any listed kickoff is within NEAR_KICKOFF_H hours (and not yet past)."""
    now = now or time.time()
    for k in kickoffs_iso or []:
        try:
            t = time.mktime(time.strptime(str(k)[:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
        except Exception:
            continue
        if -1 * 3600 <= t - now <= NEAR_KICKOFF_H * 3600:
            return REFRESH_HOURS_NEAR
    return REFRESH_HOURS


def _read_lines(path=LINES):
    """History per player|market|commence: the recorded rows, oldest first."""
    hist = {}
    try:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                hist.setdefault(f"{r.get('player')}|{r.get('market')}|{r.get('commence')}", []).append(r)
    except OSError:
        pass
    return hist


def record_lines(props, fetched, path=LINES):
    """
    Append one row per prop whose line or price changed since the last recorded row (or that
    is new). Then annotate each prop from the history: line_open (first line seen this week),
    line_prev (the last different line) and line_moved_at (when the current line appeared),
    so the page can show movement without a second file.
    """
    hist = _read_lines(path)
    rows = []
    for slot in props.values():
        k = f"{slot.get('player')}|{slot.get('market')}|{slot.get('commence')}"
        seq = hist.setdefault(k, [])
        prev = seq[-1] if seq else None
        if (prev is None or prev.get("line") != slot.get("line")
                or prev.get("over") != slot.get("over") or prev.get("under") != slot.get("under")):
            r = {"at": fetched, "player": slot.get("player"), "market": slot.get("market"),
                 "commence": slot.get("commence"), "game": slot.get("game"),
                 "line": slot.get("line"), "over": slot.get("over"), "under": slot.get("under")}
            rows.append(r)
            seq.append(r)
        slot["line_open"] = seq[0]["line"]
        slot["line_open_at"] = seq[0]["at"]
        cur = slot.get("line")
        different = [r for r in seq if r.get("line") != cur]
        slot["line_prev"] = different[-1]["line"] if different else None
        # when did the current line first appear (after the last different one)?
        i = seq.index(different[-1]) + 1 if different else 0
        slot["line_moved_at"] = seq[i]["at"] if different and i < len(seq) else None
    if rows:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
        except Exception:
            pass
    return len(rows)


def fetch(key=None, cache=CACHE, refresh_hours=REFRESH_HOURS, log=print):
    """
    FanDuel player props for the upcoming slate.

    Returns a dict keyed by "<normalised player name>|<stat>" holding line and both prices,
    or None if there is nothing to work with. Falls back to the cached copy on any failure,
    which is the right behaviour: a six-hour-old line is far more useful than no line, and
    the page timestamps it so the reader can judge.
    """
    key = (key or os.environ.get("ODDS_API_KEY") or "").strip()
    # Development only. The Odds API is refused by egress policy from both dev machines, so
    # the UI has to be built against a captured shape. A fixture is never published: it is
    # opt-in via an env var that CI does not set, and the payload records where lines came
    # from either way.
    fix = os.environ.get("SJ_ODDS_FIXTURE")
    if fix and os.path.exists(fix):
        try:
            d = json.load(open(fix))
            d["source"] = "fixture"
            log(f"  odds: FIXTURE {fix} ({len(d.get('props', {}))} props) - not live data")
            return d
        except Exception as e:
            log(f"  odds: fixture unreadable ({e})")
            return None
    if _fresh(cache, refresh_hours):
        try:
            d = json.load(open(cache))
            log(f"  odds: using cached FanDuel lines from {d.get('fetched', 'unknown')}")
            return d
        except Exception:
            pass
    if not key:
        log("  odds: ODDS_API_KEY not set; no FanDuel lines this run")
        return _stale(cache, log)

    if len(key) < 20:
        log(f"  odds: ODDS_API_KEY looks too short ({len(key)} chars) - check it was pasted whole")
    events, hdrs = _get(f"{BASE}/events?apiKey={key}")
    if not isinstance(events, list):
        log(f"  odds: could not list events - HTTP {hdrs.get('_status', '?')}"
            f"{': ' + hdrs['_error'] if hdrs.get('_error') else ''}")
        return _stale(cache, log)
    remaining = hdrs.get("x-requests-remaining")
    log(f"  odds: {len(events)} events listed, {remaining or '?'} credits remaining")
    if remaining is not None:
        try:
            if float(remaining) < QUOTA_FLOOR:
                log(f"  odds: below the {QUOTA_FLOOR}-credit floor; not fetching props")
                return _stale(cache, log)
        except ValueError:
            pass

    props, used, errors = {}, 0, 0
    for ev in events:
        eid = ev.get("id")
        if not eid:
            continue
        url = (f"{BASE}/events/{eid}/odds?apiKey={key}&regions={REGION}"
               f"&markets={MARKET_LIST}&oddsFormat=american&bookmakers={BOOK}")
        data, h = _get(url)
        used += 1
        remaining = h.get("x-requests-remaining", remaining)
        if not isinstance(data, dict):
            if not errors:
                log(f"  odds: event odds failed - HTTP {h.get('_status', '?')}"
                    f"{': ' + h['_error'] if h.get('_error') else ''}")
            errors += 1
            continue
        for bm in data.get("bookmakers", []):
            if bm.get("key") != BOOK:
                continue
            for mk in bm.get("markets", []):
                mkey = mk.get("key")
                if mkey not in MARKETS:
                    continue
                for oc in mk.get("outcomes", []):
                    who, name, pt = oc.get("description"), oc.get("name"), oc.get("point")
                    if not who or pt is None:
                        continue
                    slot = props.setdefault(f"{who}|{mkey}", {
                        "player": who, "market": mkey, "line": float(pt),
                        "over": None, "under": None,
                        "game": f"{data.get('away_team')} @ {data.get('home_team')}",
                        "commence": data.get("commence_time"),
                    })
                    if str(name).lower() == "over":
                        slot["over"] = int(oc.get("price"))
                    elif str(name).lower() == "under":
                        slot["under"] = int(oc.get("price"))
        if remaining is not None:
            try:
                if float(remaining) < QUOTA_FLOOR:
                    log(f"  odds: hit the credit floor after {used} events; stopping early")
                    break
            except ValueError:
                pass

    if not props:
        log(f"  odds: no player props returned across {used} event(s), {errors} errored. "
            f"If the events listed but no props came back, FanDuel props may not be covered "
            f"for this sport on this plan.")
        return _stale(cache, log)

    fetched = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
    n_changed = record_lines(props, fetched)
    out = {"fetched": fetched,
           "book": "FanDuel", "props": props,
           "events_fetched": used, "credits_remaining": remaining, "lines_changed": n_changed}
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(out, open(cache, "w"))
    except Exception:
        pass
    log(f"  odds: {len(props)} FanDuel props across {used} events, {n_changed} line/price changes recorded, "
        f"{remaining or '?'} credits left")
    return out


def _stale(cache, log):
    """Any cached copy at all beats nothing, as long as its age is carried with it."""
    try:
        d = json.load(open(cache))
        log(f"  odds: falling back to cached lines from {d.get('fetched', 'unknown')}")
        return d
    except Exception:
        return None


def stat_for(market, position):
    """The API's market plus our position gives our stat name. None if we do not project it."""
    return MARKETS.get(market, {}).get(str(position or "").upper())
