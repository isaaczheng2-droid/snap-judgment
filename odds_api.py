#!/usr/bin/env python3
"""
FanDuel player prop lines, via The Odds API.

WHY NOT FANDUEL DIRECTLY: their own endpoints are not a public API, and pulling pricing off
a sportsbook you have not agreed terms with is not something to build into a job that runs
every hour forever. The Odds API is a licensed aggregator that carries FanDuel's lines, so
the data is the same and the arrangement is not a problem waiting to happen.

CREDITS, WHICH ARE THE REAL CONSTRAINT: player props are only available one event at a time
(`/events/{id}/odds`), and each request costs markets x regions credits. One NFL week is
~16 events x 5 markets = 80 credits per refresh. The free tier is 500 credits A MONTH, so an
hourly refresh would burn roughly 13,000 a week. This module therefore:

  - fetches props at most once every REFRESH_HOURS, caching to disk in between
  - reads the remaining quota out of the response headers and logs it every time
  - stops immediately if the quota drops below a floor, rather than silently failing later

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
REFRESH_HOURS = 6
QUOTA_FLOOR = 40           # stop before the quota is gone, not after
CACHE = "data/fanduel_props.json"

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
    """Returns (parsed_json, headers_dict) or (None, {}). Never raises."""
    try:
        r = subprocess.run(["curl", "-sSL", "-D", "-", "--max-time", str(timeout), url],
                           capture_output=True, timeout=timeout + 10)
        if r.returncode != 0 or not r.stdout:
            return None, {}
        raw = r.stdout.decode("utf-8", "replace")
        head, _, body = raw.partition("\r\n\r\n")
        if not body:
            head, _, body = raw.partition("\n\n")
        hdrs = {}
        for ln in head.splitlines():
            if ":" in ln:
                k, _, v = ln.partition(":")
                hdrs[k.strip().lower()] = v.strip()
        return json.loads(body), hdrs
    except Exception:
        return None, {}


def _fresh(path, hours):
    try:
        return (time.time() - os.path.getmtime(path)) < hours * 3600
    except OSError:
        return False


def fetch(key=None, cache=CACHE, refresh_hours=REFRESH_HOURS, log=print):
    """
    FanDuel player props for the upcoming slate.

    Returns a dict keyed by "<normalised player name>|<stat>" holding line and both prices,
    or None if there is nothing to work with. Falls back to the cached copy on any failure,
    which is the right behaviour: a six-hour-old line is far more useful than no line, and
    the page timestamps it so the reader can judge.
    """
    key = key or os.environ.get("ODDS_API_KEY") or ""
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

    events, hdrs = _get(f"{BASE}/events?apiKey={key}")
    if not isinstance(events, list):
        log("  odds: could not list events")
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

    props, used = {}, 0
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
        log("  odds: no player props returned")
        return _stale(cache, log)

    out = {"fetched": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
           "book": "FanDuel", "props": props,
           "events_fetched": used, "credits_remaining": remaining}
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(out, open(cache, "w"))
    except Exception:
        pass
    log(f"  odds: {len(props)} FanDuel props across {used} events, "
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
