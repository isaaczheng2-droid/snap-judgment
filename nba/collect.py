"""
Runner-side collectors for sources the build container cannot reach. Each one is independent,
records its outcome in nba/data/source_state.json (shown on the Data status tab), never raises,
and appends timestamped snapshots rather than overwriting.

    python3 nba/collect.py [--only rosters,injuries,lines,props,pbpstats,validate,boxscores]

Secrets: ODDS_API_KEY from the environment only. Nothing here writes a key anywhere.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from nba import status, props, data  # noqa: E402

DATA = os.path.join(HERE, "data")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
NBA_HEADERS = ["-H", f"User-Agent: {UA}", "-H", "Referer: https://www.nba.com/", "-H", "Origin: https://www.nba.com",
               "-H", "Accept: application/json, text/plain, */*", "-H", "x-nba-stats-origin: stats", "-H", "x-nba-stats-token: true"]


def log(m):
    print(f"[nba collect] {m}", flush=True)


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def curl(url, headers=(), timeout=40, binary=False):
    r = subprocess.run(["curl", "-sSL", "--max-time", str(timeout), "-w", "\n%{http_code}", *headers, url], capture_output=True, timeout=timeout + 10)
    out = r.stdout
    body, _, code = out.rpartition(b"\n")
    code = code.decode().strip()
    return code, (body if binary else body.decode("utf-8", "replace"))


def season_str(now=None):
    now = now or datetime.now(timezone.utc)
    y = now.year if now.month >= 9 else now.year - 1
    return f"{y}-{str(y + 1)[2:]}"


# ----------------------------------------------------------------------------- box scores (always)
def collect_boxscores():
    try:
        d = data.build(2022, 2027, refresh=True)
        status.record("sdv_boxscores", True, f"{len(d['games'])} games, {len(d['player_games'])} player-games")
        status.record("sdv_schedule_next", True, f"{int((d['games'].season == d['games'].season.max()).sum())} games in the latest schedule")
    except Exception as e:
        status.record("sdv_boxscores", False, repr(e))
        log(f"boxscores failed: {e!r}")


# ----------------------------------------------------------------------------- rosters
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"


def _espn_teams():
    """ESPN team id -> abbreviation, from the games table (same id namespace as the box scores)."""
    import pandas as pd
    G = pd.read_parquet(os.path.join(DATA, "games.parquet"))
    m = {}
    for a, b in (("home_id", "home"), ("away_id", "away")):
        for tid, ab in G[[a, b]].dropna().drop_duplicates().itertuples(index=False):
            if ab not in ("TBD",):
                m[int(tid)] = ab
    return m


def espn_get(url, timeout=40):
    """ESPN's site API answers curl's own user agent from the runner (the NFL injury feed has
    used exactly that for weeks); a browser UA without browser headers is what bot filters
    reject. Try plain first, browser-like second, and report the code either way."""
    code, body = curl(url, ["-H", "Accept: application/json"], timeout=timeout)
    if code != "200":
        code2, body2 = curl(url, ["-H", f"User-Agent: {UA}", "-H", "Accept: application/json"], timeout=timeout)
        if code2 == "200":
            return code2, body2
        log(f"espn {url.split('/nba/')[-1][:60]}: HTTP {code} / {code2}")
    return code, body


def collect_rosters():
    """Current rosters. ESPN's site API first (same athlete/team ids as the box scores, so the
    identity match is exact, and it is reachable from the runner); stats.nba.com via nba_api as
    the fallback, which the 2026-09-23 runner probe showed timing out (datacenter block)."""
    teams = _espn_teams()
    out, ok = {}, 0
    for tid, abbr in sorted(teams.items()):
        code, body = espn_get(f"{ESPN}/teams/{tid}/roster")
        if code != "200":
            continue
        try:
            d = json.loads(body)
        except Exception:
            continue
        players = []
        for a in d.get("athletes", []):
            st = (a.get("status") or {}).get("name")
            inj = a.get("injuries") or []
            players.append({"player_id": str(a["id"]), "player": a.get("displayName") or a.get("fullName"), "position": ((a.get("position") or {}).get("abbreviation") or "")[:1] or None,
                            "jersey": a.get("jersey"), "headshot": (a.get("headshot") or {}).get("href"), "roster_status": st,
                            "injury": ({"status": inj[0].get("status"), "detail": (inj[0].get("details") or {}).get("type")} if inj else None), "id_match": "espn"})
        out[abbr] = {"espn_team_id": tid, "season": (d.get("season") or {}).get("displayName"), "fetched_at": now_iso(), "source": "espn site api", "players": players}
        ok += 1
        time.sleep(0.3)
    if ok >= 25:
        json.dump(out, open(os.path.join(DATA, "rosters_current.json"), "w"), indent=0)
        status.record("espn_site", True, f"{ok}/30 rosters, {sum(len(v['players']) for v in out.values())} players")
        log(f"rosters (espn): {ok} teams")
        return
    status.record("espn_site", False, f"only {ok}/30 rosters from ESPN")
    log(f"rosters (espn): only {ok}/30; trying stats.nba.com (expected to time out from a hosted runner)")
    if os.environ.get("NBA_TRY_STATS_NBA") == "1":
        collect_rosters_nba()


def collect_injuries_espn():
    """ESPN's league injury feed (timestamped per entry). Appended as a snapshot with source=espn;
    the official NBA report, when it is published, is appended by collect_injuries and wins ties."""
    code, body = espn_get(f"{ESPN}/injuries")
    if code != "200":
        status.record("espn_site", False, f"injuries HTTP {code}")
        return
    try:
        d = json.loads(body)
    except Exception as e:
        status.record("espn_site", False, f"injuries unparseable: {e!r}")
        return
    teams = _espn_teams()
    fetched, rows = now_iso(), []
    for t in d.get("injuries", []):
        abbr = teams.get(int(t.get("id", 0) or 0))
        for i in t.get("injuries", []):
            ath = i.get("athlete") or {}
            rows.append({"team": abbr, "team_name": t.get("displayName"), "player": ath.get("displayName"), "player_id": str(ath.get("id")) if ath.get("id") else None,
                         "status": i.get("status"), "reason": i.get("shortComment") or (i.get("details") or {}).get("type"), "detail": (i.get("details") or {}).get("type"),
                         "report_time": i.get("date"), "fetched_at": fetched, "source": "espn"})
    with open(os.path.join(DATA, "injury_snapshots.ndjson"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    status.record("espn_site", True, f"{len(rows)} injury entries at {d.get('timestamp')}")
    log(f"injuries (espn): {len(rows)} rows")


def collect_rosters_nba():
    """CommonTeamRoster for all 30 teams via nba_api; writes rosters_current.json keyed by ESPN
    abbreviation with ESPN player ids attached by exact name match (unmatched keep nba id only)."""
    try:
        from nba_api.stats.endpoints import commonteamroster
        from nba_api.stats.static import teams as nba_teams
    except Exception as e:
        status.record("stats_nba", False, f"nba_api import failed: {e!r}")
        return
    season = season_str()
    espn = {}
    try:
        import pandas as pd
        pl = pd.read_parquet(os.path.join(DATA, "players.parquet"))
        for r in pl.itertuples():
            espn.setdefault(props.norm_name(r.player), []).append({"player_id": r.player_id, "headshot": r.headshot, "position": r.position, "team": r.last_team})
    except Exception:
        pass
    abbr_map = {"GSW": "GS", "NOP": "NO", "NYK": "NY", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH", "PHX": "PHX"}
    out, ok, fail = {}, 0, 0
    for t in nba_teams.get_teams():
        try:
            df = commonteamroster.CommonTeamRoster(team_id=t["id"], season=season, timeout=45).get_data_frames()[0]
            time.sleep(0.7)
        except Exception as e:
            fail += 1
            log(f"roster {t['abbreviation']} failed: {e!r}"[:200])
            continue
        abbr = abbr_map.get(t["abbreviation"], t["abbreviation"])
        players = []
        for r in df.itertuples():
            cands = espn.get(props.norm_name(r.PLAYER), [])
            hit = cands[0] if len(cands) == 1 else next((c for c in cands if c["team"] == abbr), None)
            players.append({"player_id": hit["player_id"] if hit else f"nba:{r.PLAYER_ID}", "nba_id": int(r.PLAYER_ID), "player": r.PLAYER,
                            "position": (hit["position"] if hit else (r.POSITION or "")[:1] or None), "jersey": str(r.NUM) if r.NUM else None,
                            "headshot": hit["headshot"] if hit else f"https://cdn.nba.com/headshots/nba/latest/260x190/{r.PLAYER_ID}.png",
                            "id_match": "espn" if hit else "unmatched (nba id only)"})
        out[abbr] = {"nba_team_id": t["id"], "season": season, "fetched_at": now_iso(), "players": players}
        ok += 1
    if ok:
        json.dump(out, open(os.path.join(DATA, "rosters_current.json"), "w"), indent=0)
        n_un = sum(1 for v in out.values() for p in v["players"] if p["id_match"] != "espn")
        status.record("stats_nba", True, f"{ok}/30 rosters for {season}; {n_un} players without an ESPN id match")
        log(f"rosters: {ok} ok, {fail} failed, {n_un} unmatched")
    else:
        status.record("stats_nba", False, f"0/30 rosters for {season}: {fail} failures")


# ----------------------------------------------------------------------------- official injury report
def collect_injuries():
    """Find the newest injury-report PDF linked from official.nba.com and parse it. Rows are
    appended with fetched_at (and the report's own timestamp) to injury_snapshots.ndjson."""
    code, html = curl("https://official.nba.com/nba-injury-report-2026-27-season/", ["-H", f"User-Agent: {UA}"])
    if code != "200":
        code, html = curl("https://official.nba.com/nba-injury-report-2025-26-season/", ["-H", f"User-Agent: {UA}"])
    if code != "200":
        status.record("official_injury", False, f"index page HTTP {code}")
        return
    links = re.findall(r'https://ak-static\.cms\.nba\.com/referee/injury/Injury-Report_[\w\-]+\.pdf', html)
    if not links:
        status.record("official_injury", False, "no report PDF linked (expected out of season)")
        log("injuries: no PDF linked")
        return
    url = links[0]
    code, pdf = curl(url, ["-H", f"User-Agent: {UA}"], binary=True)
    if code != "200":
        status.record("official_injury", False, f"PDF HTTP {code}")
        return
    path = os.path.join(DATA, "injury_latest.pdf")
    open(path, "wb").write(pdf)
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pdfplumber"], capture_output=True, timeout=180)
        import pdfplumber
    except Exception as e:
        status.record("official_injury", False, f"pdfplumber unavailable: {e!r}")
        return
    rows, fetched = [], now_iso()
    m = re.search(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})(AM|PM)", url)
    report_time = f"{m.group(1)} {m.group(2)}{m.group(3)} ET" if m else None
    try:
        with pdfplumber.open(path) as doc:
            game, team = None, None
            for page in doc.pages:
                for line in (page.extract_text() or "").splitlines():
                    mg = re.search(r"(\w{3})@(\w{3})", line)
                    if mg:
                        game = mg.group(0)
                    mt = re.search(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+(\w[\w\-']+, [\w\-'\. ]+?)\s+(Out|Available|Questionable|Doubtful|Probable)\s+(.*)$", line)
                    if mt:
                        team = mt.group(1)
                        last, first = [x.strip() for x in mt.group(2).split(",", 1)]
                        rows.append({"game": game, "team_name": team, "player": f"{first} {last}", "status": mt.group(3), "reason": mt.group(4).strip(),
                                     "report_time": report_time, "fetched_at": fetched, "source": url})
    except Exception as e:
        status.record("official_injury", False, f"parse failed: {e!r}")
        return
    with open(os.path.join(DATA, "injury_snapshots.ndjson"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    status.record("official_injury", True, f"{len(rows)} rows from {os.path.basename(url)}")
    log(f"injuries: {len(rows)} rows from {url}")


# ----------------------------------------------------------------------------- pbpstats lineups
def collect_pbpstats():
    season = season_str()
    code, body = curl(f"https://api.pbpstats.com/get-totals/nba?Season={season}&SeasonType=Regular%20Season&Type=Lineup", ["-H", f"User-Agent: {UA}"], timeout=60)
    if code != "200":
        status.record("pbpstats", False, f"HTTP {code}")
        return
    try:
        d = json.loads(body)
        rows = d.get("multi_row_table_data", [])
    except Exception as e:
        status.record("pbpstats", False, f"unparseable: {e!r}")
        return
    json.dump({"season": season, "fetched_at": now_iso(), "rows": rows}, open(os.path.join(DATA, "lineups_pbpstats.json"), "w"))
    status.record("pbpstats", True, f"{len(rows)} lineup rows for {season}")
    log(f"pbpstats: {len(rows)} lineup rows")


# ----------------------------------------------------------------------------- lines
def collect_lines():
    st = props.fetch_game_lines(log=log)
    status.record("odds_api", bool(st.get("ok")), st.get("error") or f"game lines: {st.get('events')} events, {st.get('credits_remaining')} credits left")
    st2 = props.fetch_odds_api(log=log)
    if st2.get("ok"):
        status.record("odds_api", True, f"props: {st2['quotes']} quotes from {st2['books_seen']}, {st2['credits_remaining']} credits left")
    elif st2.get("error"):
        status.record("odds_api", False, st2["error"])


# ----------------------------------------------------------------------------- validation vs stats.nba.com
def validate_boxscores(n_games=40):
    """Compare ESPN team totals for the most recent finished games with stats.nba.com league
    game log. Any mismatch is recorded, not corrected silently. stats.nba.com times out from
    hosted runners (2026-09-23 probe), so this only runs when NBA_TRY_STATS_NBA=1."""
    if os.environ.get("NBA_TRY_STATS_NBA") != "1":
        status.record("stats_nba", False, "skipped: stats.nba.com times out from hosted runners (set NBA_TRY_STATS_NBA=1 to retry)")
        return
    try:
        import pandas as pd
        from nba_api.stats.endpoints import leaguegamelog
    except Exception as e:
        status.record("stats_nba", False, f"nba_api import failed: {e!r}")
        return
    T = pd.read_parquet(os.path.join(DATA, "team_games.parquet"))
    last = T[T.pts.notna()].sort_values("tipoff_utc").tail(n_games * 2)
    season = int(last.season.max())
    s = f"{season - 1}-{str(season)[2:]}"
    try:
        lg = leaguegamelog.LeagueGameLog(season=s, player_or_team_abbreviation="T", timeout=45).get_data_frames()[0]
    except Exception as e:
        status.record("stats_nba", False, f"leaguegamelog {s} failed: {e!r}"[:200])
        return
    lg["date"] = pd.to_datetime(lg.GAME_DATE).dt.strftime("%Y-%m-%d")
    abbr_map = {"GSW": "GS", "NOP": "NO", "NYK": "NY", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH"}
    lg["abbr"] = lg.TEAM_ABBREVIATION.map(lambda a: abbr_map.get(a, a))
    key = lg.set_index(["abbr", "date"])
    matched, mism = 0, []
    for r in last.itertuples():
        d = pd.Timestamp(r.tipoff_utc).tz_convert("US/Eastern").strftime("%Y-%m-%d")
        if (r.team, d) in key.index:
            off = key.loc[(r.team, d)]
            off = off.iloc[0] if hasattr(off, "iloc") and getattr(off, "ndim", 1) == 2 else off
            matched += 1
            if int(off.PTS) != int(r.pts) or int(off.FGA) != int(r.fga):
                mism.append({"game_id": int(r.game_id), "team": r.team, "espn": [int(r.pts), int(r.fga)], "nba": [int(off.PTS), int(off.FGA)]})
    json.dump({"checked_at": now_iso(), "season": s, "matched": matched, "mismatches": mism}, open(os.path.join(DATA, "boxscore_validation.json"), "w"), indent=1)
    status.record("stats_nba", True, f"box validation: {matched} team-games matched, {len(mism)} mismatches")
    log(f"validation: {matched} matched, {len(mism)} mismatches")


def collect_injuries_all():
    collect_injuries_espn()
    collect_injuries()


STEPS = {"boxscores": collect_boxscores, "rosters": collect_rosters, "injuries": collect_injuries_all, "pbpstats": collect_pbpstats,
         "lines": collect_lines, "validate": validate_boxscores}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(STEPS))
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    for name in a.only.split(","):
        name = name.strip()
        if name in STEPS:
            log(f"== {name}")
            try:
                STEPS[name]()
            except Exception as e:
                log(f"{name} crashed: {e!r}")
                status.record({"boxscores": "sdv_boxscores", "rosters": "espn_site", "injuries": "official_injury", "pbpstats": "pbpstats",
                               "lines": "odds_api", "validate": "stats_nba"}[name], False, f"crash: {e!r}")
