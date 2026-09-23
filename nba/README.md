# Snap Judgment · NBA section

Everything here is separate from the NFL pipeline. It shares the page shell, the HTTP helper,
the append-only store pattern and the Odds API quota discipline, and nothing else.

## Layout

| File | Role |
| --- | --- |
| `data.py` | Fetch SportsDataverse (ESPN) release assets, normalise to `games`, `team_games`, `player_games`, `players` (parquet under `nba/data/`, rebuilt every run; never committed) |
| `team_model.py` | Pre-game team ratings (EWMA ORtg/DRtg/pace with a fading prior-season prior, Elo, SOS), logistic win model, ridge margin and total; walk-forward evaluation with baselines |
| `player_model.py` | Minutes model (P(play), P(start), minutes, 240-minute reconciliation), per-minute rates with position shrinkage and opponent/pace factors, Monte Carlo simulation, walk-forward evaluation |
| `props.py` | Quote schema, append-only quote ledger, proportional no-vig, EV with push/void, pick'em separately, Odds API `basketball_nba` adapter (runner only), CSV/JSON import |
| `fantasy.py` | Points formats (ESPN, Yahoo, DK, FD, Sleeper defaults) and 8/9-cat z-scores with attempt-weighted FG%/FT% |
| `grades.py` | Snap Grade (NBA): descriptive percentile with components, comparison group, sample size; chronological validation |
| `status.py` | Source coverage table + per-source health written by the collectors |
| `collect.py` | Runner-only collectors: box-score refresh, ESPN rosters + injury feed, official injury PDF, pbpstats lineups, Odds API lines/props, box-score validation |
| `run.py` | Builds `nba_payload.json` and appends every forecast to `nba/data/forecasts.ndjson` |
| `test_nba.py` | Data integrity, leakage, simulation consistency, props maths, fantasy, grades, payload contract |

The NFL side gained `player_grades.py` (NFL Snap Grades + depth-chart block), the
`rosters`/`grades` payload keys, and the page pieces `build/v2/js_lineup.js`, `js_nba.js`, `lineup.css`.

## Running

```
pip install -r requirements.txt -r nba/requirements.txt
python3 nba/data.py                       # fetch + normalise (about 30s, 5 seasons + next schedule)
python3 nba/team_model.py                 # walk-forward backtest -> nba/data/team_backtest.json
python3 nba/player_model.py               # walk-forward backtest -> nba/data/player_backtest.json (3 min first time)
python3 nba/grades.py                     # chronological grade check -> nba/data/grade_validation.json
python3 nba/collect.py                    # runner only; each step records its own outcome
python3 nba/run.py --out nba_payload.json # the page's data
python3 nba/test_nba.py
```

The GitHub Actions job `.github/workflows/nba.yml` runs collect + run daily at 12:40 UTC and
every three hours in season, commits `nba_payload.json` and `nba/data/*.ndjson|json`, and
shares the `publish` concurrency group with the NFL jobs. `ODDS_API_KEY` is read from the
repository secret by `props.py` only; nothing writes a key to the page or the repository.

## Data dictionary

`games` (one row per game): `game_id` ESPN id (int, the key everywhere); `season` (end year:
2026 = 2025-26); `phase` preseason/regular/play-in/postseason; `phase_label` ESPN type (STD,
RD16, QTR, SEMI, FINAL, CC = NBA Cup final); `tipoff_utc`; `home_id`/`away_id` ESPN team ids;
`home`/`away` abbreviations; `home_score`/`away_score` NULL until final; `final`; `periods`
(5+ = overtime); `venue`, `city`, `neutral`, `attendance`; `cup_final`; `ingested_at`.

`team_games` (one row per team per game): box totals (`fgm fga fg3m fg3a ftm fta oreb dreb reb
ast stl blk tov pf`), `pts`, `opp_pts`, `won`, `poss` (FGA − OREB + TOV + 0.44·FTA for this
side), `poss_game` (mean of both sides), `pace` (per 48), `ortg`, `drtg`, `rest_days` (NULL for
a season's first game), `b2b`, `games_last7`.

`player_games` (one row per rostered player per game): `player_id` ESPN athlete id (string);
`played` (minutes > 0); `min` and every stat NULL when the player did not play; `dnp`,
`active`, `reason` (ESPN text); `starter`; `ejected`; `plus_minus`; `headshot` (ESPN URL);
`tipoff_utc`, `phase`, `periods`.

`players`: latest team, position (G/F/C), jersey, headshot, first season, games played.

`nba/data/forecasts.ndjson` (append-only): `kind` game/player/prop, `forecast_at`,
`inputs_available_at` {boxscores, injury_report, lines}, `model_version`, the numbers shown.
A finished game is graded on the last row whose `forecast_at` precedes tipoff.

`nba/data/prop_quotes.ndjson` (append-only): the quote schema in `props.py` (book, market,
player, line, side, price, `quoted_at` from the book, `fetched_at` from us, source, raw).

`nba/data/injury_snapshots.ndjson` (append-only): per fetch, every entry from the ESPN feed
(`source=espn`) and, when published, the official report (`source=<pdf url>`); the official
report wins ties.

## What is unavailable, and why

* stats.nba.com (nba_api): times out from GitHub-hosted runners and from the build container
  (2026-09-23 probe). Adapter kept; ESPN's site API supplies rosters and identity instead.
* cdn.nba.com static/live JSON: 403 from every location tested; headshots load.
* Historical NBA odds/props: The Odds API's historical endpoint is paid and was not purchased,
  so there is no real-market backtest. Forward collection starts with the first in-season run.
* Historical injury-report snapshots: none exist here; forward collection only.
* balldontlie: no key configured, nothing purchased; adapter disabled.
* Sleeper / PrizePicks / Underdog lines: no API; import-only via CSV/JSON.
* Lineup on/off, tracking, play-by-play features: pbpstats is reachable from the runner and
  the collector stores lineup totals, but none of it feeds a model yet; defence is not graded.
* Confirmed starters (~30 min before tipoff): not collected; the lineup shown is a labelled
  estimate from P(start).
