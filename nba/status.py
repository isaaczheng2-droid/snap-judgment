"""
Source coverage table for the NBA Data status tab. Facts here were verified during the build
(dates in `verified_on`); the runner collector updates `last_success`, `last_attempt` and
`last_error` per source on every run so the page shows live health, never an assumed one.
"""
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "data", "source_state.json")

SOURCES = [
    {"id": "sdv_boxscores", "dataset": "Box scores (player + team), schedules, game rosters", "provider": "SportsDataverse data releases (ESPN-sourced, hoopR)",
     "earliest_verified_season": "2021-22 loaded (releases go back further; 2002+ published)", "update_frequency": "daily during the season (release assets rebuilt)",
     "subscription": "none", "gaps": "no play-by-play possessions, no lineups, no tracking; minutes are whole numbers; DNP reasons are ESPN's text",
     "reachable_from": "everywhere (GitHub release assets)", "role": "historical backbone + daily results", "verified_on": "2026-09-23"},
    {"id": "sdv_schedule_next", "dataset": "2026-27 schedule", "provider": "SportsDataverse data releases (ESPN)", "earliest_verified_season": "2026-27 (1,205 regular-season games, Oct 20 2026 - Apr 12 2027)",
     "update_frequency": "daily", "subscription": "none", "gaps": "no preseason games; NBA Cup final placeholder TBD vs TBD", "reachable_from": "everywhere", "role": "slate", "verified_on": "2026-09-23"},
    {"id": "stats_nba", "dataset": "Official box scores, rosters, player index, advanced/tracking tables", "provider": "stats.nba.com via nba_api (community client, MIT; not an officially supported API)",
     "earliest_verified_season": "1996-97 (league game logs); tracking 2013-14+", "update_frequency": "live", "subscription": "none",
     "gaps": "blocks datacenter traffic intermittently; rate-limit and header sensitive; unavailable from the build container (probe pending from the Actions runner)",
     "reachable_from": "browser: yes; container: no; runner: probe pending", "role": "validation of ESPN box scores, current rosters, player-id crosswalk", "verified_on": "2026-09-23"},
    {"id": "cdn_nba_live", "dataset": "Live scoreboard / boxscore JSON, headshots", "provider": "cdn.nba.com", "earliest_verified_season": "current season only",
     "update_frequency": "live", "subscription": "none", "gaps": "static JSON returned 403 (Akamai) even from a browser; headshots served fine",
     "reachable_from": "headshots: browser yes, container no; runner: probe pending", "role": "headshots (hotlinked, with fallback)", "verified_on": "2026-09-23"},
    {"id": "pbpstats", "dataset": "Possession totals, lineups, on/off", "provider": "api.pbpstats.com", "earliest_verified_season": "2000-01 (site); 2024-25 team totals verified",
     "update_frequency": "nightly", "subscription": "none for the endpoints tested (no key)", "gaps": "unofficial; no published SLA; blocked from the container",
     "reachable_from": "browser: yes; container: no; runner: probe pending", "role": "lineup units, on/off, possession-level pace (needed before defence is graded)", "verified_on": "2026-09-23"},
    {"id": "nba_data_archive", "dataset": "Play-by-play archives (nbastats, nbastatsv3, pbpstats, datanba, matchups)", "provider": "github.com/shufinskiy/nba_data",
     "earliest_verified_season": "1996-97 through 2024-25 (2024-25 pbpstats + nbastatsv3 downloaded)", "update_frequency": "per season (not current)", "subscription": "none",
     "gaps": "no 2025-26 or later; no licence stated; not used in the models yet", "reachable_from": "everywhere (raw.githubusercontent.com)", "role": "future possession-level features", "verified_on": "2026-09-23"},
    {"id": "official_injury", "dataset": "Official NBA injury report (PDF, several times daily in season)", "provider": "official.nba.com / ak-static.cms.nba.com",
     "earliest_verified_season": "2019-20 (published PDFs); no historical archive held here", "update_frequency": "1pm/5pm ET and game-day updates in season", "subscription": "none",
     "gaps": "PDF parsing; no report until the season starts (Oct 2026); historical snapshots begin with forward collection", "reachable_from": "container: no; runner: probe pending", "role": "availability, P(play) inputs", "verified_on": "2026-09-23"},
    {"id": "odds_api", "dataset": "Game lines (h2h/spreads/totals) and player props, basketball_nba", "provider": "The Odds API",
     "earliest_verified_season": "featured markets from mid-2020, props from May 2023 (historical endpoint is paid and NOT purchased)", "update_frequency": "per request; quota-limited",
     "subscription": "existing key (GitHub secret ODDS_API_KEY); props cost markets x regions per event", "gaps": "no PrizePicks/Underdog/Sleeper; props mainly US books; FanDuel and DraftKings verified per fetch, not assumed",
     "reachable_from": "runner only (key never leaves GitHub secrets)", "role": "timestamped lines for evaluation and forward tracking", "verified_on": "2026-09-23"},
    {"id": "balldontlie", "dataset": "Teams, players, games (free); stats/injuries (ALL-STAR $9.99/mo); odds/props/lineups/PBP (GOAT $39.99/mo)", "provider": "balldontlie.io",
     "earliest_verified_season": "games 1946-; season averages 1996+; advanced 2015+; lineups/PBP 2025+", "update_frequency": "live", "subscription": "key required for every call (401 without); free tier 5 req/min",
     "gaps": "adapter present and disabled: no key configured, nothing purchased", "reachable_from": "runner (with key)", "role": "optional cross-check", "verified_on": "2026-09-23"},
    {"id": "sleeper", "dataset": "Sleeper NBA pick'em / fantasy lines", "provider": "Sleeper", "earliest_verified_season": "n/a", "update_frequency": "n/a", "subscription": "n/a",
     "gaps": "no public API for lines; not offered by The Odds API; import-only (CSV/JSON)", "reachable_from": "import only", "role": "pick'em comparison via import", "verified_on": "2026-09-23"},
    {"id": "basketball_reference", "dataset": "Historical reference tables", "provider": "basketball-reference.com", "earliest_verified_season": "n/a",
     "update_frequency": "n/a", "subscription": "n/a", "gaps": "terms prohibit automated collection; not used", "reachable_from": "not used", "role": "none", "verified_on": "2026-09-23"},
]


def load_state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {}


def record(source_id, ok, detail=None, path=STATE):
    st = load_state()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = st.setdefault(source_id, {})
    row["last_attempt"] = now
    if ok:
        row["last_success"] = now
        row["last_error"] = None
    else:
        row["last_error"] = (detail or "failed")[:300]
    if detail and ok:
        row["last_detail"] = str(detail)[:300]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(st, open(path, "w"), indent=1)
    return row


def table():
    st = load_state()
    out = []
    for s in SOURCES:
        r = dict(s)
        r.update({k: st.get(s["id"], {}).get(k) for k in ("last_attempt", "last_success", "last_error", "last_detail")})
        out.append(r)
    return out
