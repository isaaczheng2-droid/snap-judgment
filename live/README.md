# Snap Judgment live data layer

Player availability and stadium weather, monitored continuously, fed into the prediction engine only when it matters, and shown on the site with its age and source attached. Nothing here retrains a model; the layer's only lever on the predictions is a refresh request that the full pipeline honours, and every prediction that results is kept as a new version next to the old one.

Ground rules the code enforces (each has a test in `test_live.py`):

- Missing injury data is UNKNOWN, never healthy. Only a positive statement ("Active", "Full participation") becomes an availability claim.
- Questionable is Questionable. It is never collapsed into inactive or active.
- A retractable roof is `pending` until a source says open or closed; the page says so instead of guessing.
- Weather is fetched for the stadium's coordinates, never the city.
- No API key is ever read by the frontend, logged, or placed in a URL that is logged. Query strings are redacted before they hit the sync log.
- The model reruns only for CRITICAL/HIGH player or depth-chart events, never for weather (weather as a model input tested as noise in the audit; it is shown, not modelled).
- Old predictions are never overwritten. A change produces a new row in `prediction_versions` and a reason.
- Sportsbook prices never enter the football projection. The odds module and the live layer share no state.
- Stale is labelled stale. Every live object carries `fetchedAt`, `sourceUpdatedAt`, and a freshness verdict computed from hours-to-kickoff.

## 1. Architecture

```
  SOURCES              INGESTION            VALIDATION          NORMALIZED STORE
  ESPN injuries  ──┐   run_live.py          normalize.validate  live/data/*.ndjson
  nflverse         ├─► ingest_players ────► identity.Crosswalk ─► player_status
    injuries       │   ingest_weather       teams.to_canonical    player_practice_reports
    rosters        │                        stadiums.roof_status  depth_charts
    depth charts ──┘                                              weather_forecasts
  Open-Meteo     ──┐                                              weather_alerts
  NWS hourly       ├─────────────────────────────────────────────► api_sync_log
  NWS alerts     ──┘                                              data_quality
                                                                  source_conflicts
                                                                        │
                                            CHANGE DETECTION            ▼
                                            events.player_changes   live_events
                                            events.depth_changes        │
                                            events.weather_changes      ▼
                                                                  IMPACT (impact.classify)
                                                                  LOW / MODERATE / HIGH / CRITICAL
                                                                        │
                              ┌─────────────────────────────────────────┤
                              ▼ CRITICAL/HIGH player or depth event     ▼ always
                    live/data/refresh_request.json           payload.json  games[i].live
                              │                                          live_meta
                              ▼                                          │
                    run_pipeline.py (full)                               ▼
                    versions.record ─► prediction_versions        merge_payload.py
                                       prediction_change_log      splices into index.html
                                                                         │
                                                                         ▼
                                                              FRONTEND js_live.js
                                                              prediction change box, live feed,
                                                              weather card, injury impact,
                                                              stale banners, change highlighting
```

Backend = GitHub Actions. Database = append-only NDJSON files committed to the repo. API = `payload.json` (the page polls it every 4 minutes) and `admin.html` (diagnostics). There is no server to run and nothing to pay for.

## 2. Selected sources and why

| Need | Primary | Fallback / validation | Why |
|---|---|---|---|
| Injury status, gameday inactives | ESPN injuries endpoint (`espn_injuries.py`) | nflverse `injuries_YYYY.parquet` (official Wed/Thu/Fri report) | ESPN is the only free source with a per-item timestamp and in-game updates; the official report is authoritative for practice participation and designations |
| IR / PUP / NFI / suspension | nflverse `rosters_YYYY.parquet` (status RES/PUP/NON/SUS) | | Roster status is the ground truth for long-term unavailability, and it carries every vendor id used for identity |
| Depth charts | nflverse `depth_charts_YYYY.parquet` | | Only free machine-readable depth chart; drives STARTING_QB_CHANGE and DEPTH_CHART_CHANGE |
| Hourly forecast | Open-Meteo (`forecast` API) | NWS `api.weather.gov` gridpoint hourly | Open-Meteo: one call per stadium, 16-day horizon, no key, 10k calls/day free tier, CC-BY 4.0 attribution shown on the site. NWS: public domain, US stadiums only, needs a User-Agent, sometimes slow |
| Severe weather alerts | NWS `alerts/active?point=` | | The only authoritative source for watches/warnings |
| Stadium coordinates, roof, surface, tz, elevation | `live/stadiums.json` (38 venues incl. London, Munich, Madrid, São Paulo, Dublin, Melbourne, Frankfurt, Mexico City) | | Compiled from nflverse stadium ids cross-checked against public venue data; roof types fixed per venue, retractable state per game |

Paid upgrades if the site ever needs official inactives within seconds of the 90-minute release: SportsDataIO (indicative $99–149/mo entry tier), Sportradar (quote), Fantasy Nerds ($499/yr). The ingestion module is written so a new source is one adapter function that returns `normalize.record(...)` rows.

## 3. API cost

| Provider | Calls per poll | Polls per week (peak) | Cost |
|---|---|---|---|
| ESPN injuries | 1 | ~400 | $0 (undocumented public endpoint; failure falls back to nflverse) |
| nflverse parquet | 3 (cached to `data/` when unchanged) | ~400 | $0 |
| Open-Meteo | ≤ 16 (one per outdoor/retractable game due) | ~2,400/week worst case | $0, well under 10,000/day free tier |
| NWS points | cached per stadium in `cache.json` after first hit | ~40 total | $0 |
| NWS hourly + alerts | ≤ 2 per US outdoor game due | ~4,800/week worst case | $0 |
| The Odds API (FanDuel props) | 4 credits per event (4 markets × 1 region), ~64 per refresh | every 3 h, hourly inside 6 h of a kickoff: ~45 refreshes | ~3,000 credits/week, ~13,000/month on the 20,000/month plan; `QUOTA_FLOOR` 500 stops fetching before it runs out |
| GitHub Actions | ~1 min per live poll, ~6 min per full rebuild | ~400 + ~170 | $0 on a public repo |

The adaptive cadence (`run_live.CADENCE`) keeps most polls far below the peak: a game more than 3 days out is checked every 6 hours, one within 90 minutes every 15 minutes, and games already played are skipped.

## 4. Database schema

All tables are NDJSON under `live/data/` (override with `SJ_LIVE_DIR`), one JSON object per line, append-only, `system_received_at` stamped on every row by `store.append`. Current state is always "latest row per key" (`store.latest`), rebuilt by `store.hydrate` at the start of every run. `cache.json` holds only the small operational cache (`nws_grid`, `last_sync`, `last_poll`, `sources`, `kickoffs`, `depth`, `alerts`, `snapshot_id`, `conflicts_seen`).

**player_status** (one row per player × source, appended only when something changed)
`internal_player_id, player_name, team, position, game_id, season, week, original_status, normalized_status, injury_body_part, injury_description, original_practice, practice_status, practice_date, roster_status, game_status, estimated_return_date, depth_chart_position, depth_order, active, inactive, source, source_updated_at, system_received_at, last_verified_at, snapshot_id`

**player_status_history** — `player_id, game_id, previous_status, new_status, source, changed_at, snapshot_id`

**player_practice_reports** — `player_id, team, practice_date, practice_status, original_practice, source`

**depth_charts** — `team, season, week, position, depth_order, player_id, player_name, source, snapshot_id`

**weather_forecasts** — `game_id, stadium_id, provider, forecast_created_at, fetched_at, kickoff_utc, window_start, window_end, kickoff_temp_f, kickoff_wind_mph, kickoff_gust_mph, kickoff_precip_prob, max_window_wind, max_window_gust, precip_prob_max, precipitation_in, snowfall_in, visibility_min_mi, condition, impact_level, impact_reasons, hourly[]`

**weather_alerts** — `game_id, stadium_id, alert_id, event, severity, headline, onset, ends, source`

**weather_observed** — post-game observed conditions (schema mirrors the forecast summary; the fetch is stubbed in `weather.observed_url` and not yet scheduled)

**live_events** — `event_id, event_type, game_id, player_id, team_id, severity, previous_value, new_value, detail, source, occurred_at, received_at, processed, prediction_recalculated`
Event types: `PLAYER_STATUS_CHANGE, GAMEDAY_INACTIVE, GAMEDAY_ACTIVE, STARTING_QB_CHANGE, PRACTICE_CHANGE, RETURN_DATE_CHANGE, DEPTH_CHART_CHANGE, WEATHER_CHANGE_EVENT, SEVERE_WEATHER_ALERT`. Marker rows `{event_id, processed: true, prediction_recalculated}` fold into the original on read.

**prediction_versions** — `version_id, game_id, season, week, created_at, model_version, data_version, injury_snapshot_id, weather_snapshot_id, p_home, p_model, p_market, home_score, away_score, margin, players{key: proj}, reason, event_ids[]`

**prediction_change_log** — `from_version, to_version, game_id, field, before, after, change, n_fields, reason, timestamp`

**api_sync_log** — `provider, endpoint, url (query redacted), status, latency_ms, retry_count, fixture, error, at`

**source_conflicts** — `player_id, game_id, sources{espn, nflverse:injuries}, resolved_to, rule, timestamp`

**data_quality** — `kind (unknown_player | bad_team | team_mismatch | unknown_game | bad_timestamp | implausible_timestamp | duplicate | impossible_transition), detail, player_id, source, system_received_at`

Sidecars: `source_mapping.json` (identity crosswalk, rewritten only when its digest changes), `refresh_request.json` (present only between a qualifying event and the next full run), `odds_cache.json` (last FanDuel fetch, so a run within the refresh window spends no credits).

**prop_lines** (`prop_lines.ndjson`) — `at, player, market, commence, game, line, over, under`; one row per line or price change, first sighting included. Feeds `line_open`, `line_prev`, `line_moved_at` on every prop card and, later, closing-line value.

## 5. Environment variables and secrets

| Name | Where | Required | Purpose |
|---|---|---|---|
| `ODDS_API_KEY` | GitHub secret | for props only | Passed to both jobs. The live poll refreshes FanDuel lines on the cadence above; the projections never see them |
| `NWS_USER_AGENT` | GitHub secret (optional) | no | NWS asks for a contact string. Default is `snap-judgment (github.com/isaaczheng2-droid/snap-judgment)`; set the secret if you want an email in it. Nothing in the code ever hard-codes a person |
| `SJ_LIVE_DIR` | env | no | Relocate the tables (tests use a temp dir) |
| `SJ_ESPN_FIXTURE` | env | no | Path to a saved ESPN response; skips the network |
| `SJ_WX_FIXTURES` | env | no | Directory of `openmeteo*.json`, `nws_points*.json`, `nws_hourly*.json`, `nws_alerts*.json`; skips the network |

Keys are read only by Python inside the Actions runner. The published page and `payload.json` contain no credentials; `http.get_json` strips query strings before logging and `odds_api.py` never logs URLs.

## 6. Polling schedule

GitHub cron (UTC), in `.github/workflows/refresh.yml`:

| Cron | Job | When |
|---|---|---|
| `17 * * * *` | `refresh` (full rebuild, skips if upstream unchanged and no refresh request) | hourly |
| `*/15 13-23 * * 0` and `*/15 0-5 * * 1` | `live` | Sunday 9am ET to Monday 1am ET, every 15 min |
| `*/15 22-23 * * 1,4` and `*/15 0-5 * * 2,5` | `live` | Monday and Thursday nights, every 15 min |
| `47 * * * *` | `live` | every other hour |

`workflow_dispatch` takes `mode` = `full` or `live` and `force`. GitHub's floor is 5 minutes and 10–30 minutes of drift is normal; within a run, `run_live.py` decides per game whether a poll is actually due from hours-to-kickoff:

| Hours to kickoff | Poll every |
|---|---|
| ≤ 1.5 | 15 min |
| ≤ 6 | 1 h |
| ≤ 24 | 2 h |
| ≤ 72 | 6 h |
| ≤ 168 | 12 h |
| beyond | 24 h |

Played games are never polled. Domes and closed roofs skip weather entirely.

## 7. "Backend routes"

There is no server. The equivalents are:

| Route | Implementation | Consumer |
|---|---|---|
| `GET /payload.json` | committed by both jobs; `games[i].live` and `live_meta` are the live API | page (polls every 4 min, `LIVE.fresh` keys) |
| `GET /admin.html` | `run_live.write_admin` (noindex) | you |
| `GET /live/data/*.ndjson` | raw files on Pages | research, DuckDB/pandas |
| `POST /refresh` | `live/data/refresh_request.json` → `gh workflow run refresh.yml -f mode=full` | workflow |
| `POST /poll` | `gh workflow run refresh.yml -f mode=live` | manual |

## 8. Normalization logic (`live/normalize.py`, `live/teams.py`, `live/identity.py`)

Status vocabulary: `HEALTHY, FULL_PRACTICE, LIMITED_PRACTICE, DID_NOT_PRACTICE, QUESTIONABLE, DOUBTFUL, OUT, IR, PUP, NFI, SUSPENDED, INACTIVE, ACTIVE, UNKNOWN`. Both the original vendor text and the normalized value are stored. ESPN's "Out / Coach's Decision" becomes `INACTIVE` (a scratch, not an injury); "Suspension" becomes `SUSPENDED`; unrecognised text becomes `UNKNOWN` and is logged. `active`/`inactive` booleans are set only from a positive ACTIVE/INACTIVE statement.

Team codes: every vendor code maps to nflverse canonical through `teams.to_canonical` (ESPN `LAR→LA`, `WSH→WAS`; legacy `OAK→LV`, `SD→LAC`, `STL→LA`).

Identity: `internal_player_id` is the GSIS id. `Crosswalk.resolve` tries the vendor id first (ESPN, PFR, Sportradar, ESB…), then a unique normalized name within the team (suffixes, hyphens, apostrophes, diacritics stripped), else `unresolved` and a `data_quality` row. Ambiguous names never resolve.

Merge: ESPN is the base record; the official report supplies practice participation. When both give a game status and disagree: rule 1, fail-safe, the unavailable status stands; rule 2, ESPN wins if timestamped within 48 hours, else the official report. Every disagreement is written to `source_conflicts` once.

## 9. Change-detection logic (`live/events.py`, `live/weather.py`)

Player: compares the merged current record with the previous merged record per player. Bootstrap (no previous state) records baselines silently. A player seen for the first time raises an event only if newsworthy (status in OUT/IR/PUP/NFI/SUSPENDED/INACTIVE/DOUBTFUL/QUESTIONABLE and not a roster-only row). A status change raises `PLAYER_STATUS_CHANGE`, or `GAMEDAY_INACTIVE`/`GAMEDAY_ACTIVE` on game day, or `STARTING_QB_CHANGE` (CRITICAL) when the player is the depth-1 QB. Practice changes need a previous value. Return-date changes are LOW.

Depth: starter at a key slot changed → `DEPTH_CHART_CHANGE` (CRITICAL for QB, HIGH for RB/WR/TE/T, MEDIUM otherwise).

Weather: `weather.diff` compares the new summary with the last stored one; a move beyond tolerance (wind 6 mph, gust 10, precip probability 25 pts, temp 12 °F, snow 0.1 in) or a change of impact level appends a new forecast row. `WEATHER_CHANGE_EVENT` (HIGH) fires for a ≥10 mph wind/gust move or an impact jump to HIGH; `SEVERE_WEATHER_ALERT` for a new NWS alert: CRITICAL when severity is Extreme or Severe and the product is a Warning, HIGH for a severe Watch, MEDIUM for advisories and statements. Provider disagreement beyond tolerance (wind 6, precip 30, temp 10) sets `uncertainty` on the game's weather object and the page shows "Forecast uncertainty".

Impact thresholds (from the audit's evidence, roughly per 10 mph of wind ≈ −7% passing yards): wind ≥10 LOW, ≥15 MODERATE, ≥20 HIGH; gusts ≥30 bump, ≥40 HIGH; rain ≥60% and ≥0.1 in MODERATE, ≥0.25 in HIGH; snow MODERATE, ≥1 in HIGH; temp ≤25 °F MODERATE, ≤10 HIGH; severe alert HIGH.

## 10. Prediction-refresh logic (`live/impact.py`, `live/versions.py`)

`impact.classify(player, usage)`: position base tier (QB CRITICAL only as the starter or depth 1, MODERATE if depth unknown, else LOW; RB/WR/T HIGH; TE/G/C MODERATE), adjusted by depth order (≥3 LOW, 2 down one) and by usage from the payload's game logs (snap share ≥0.75 up, <0.30 down; target/carry share ≥0.25 up, <0.08 down). Skill positions cap at HIGH.

`impact.wants_refresh(event)` is true for a `PLAYER_STATUS_CHANGE`, `GAMEDAY_INACTIVE`, `DEPTH_CHART_CHANGE` or `STARTING_QB_CHANGE` whose tier is CRITICAL or HIGH. When any event wants a refresh, `run_live.py` writes `refresh_request.json` with the reasons and the workflow dispatches a full run.

In the full run, `versions.record` compares each game's new prediction with its latest stored version (tolerances: win probability 0.005, margin 0.25, player projection 0.5). A change appends a `prediction_versions` row with `model_version` (hash of hyperparameters, features, blend weights, seeds, training cutoff), `data_version`, both snapshot ids, and a reason built from the pending HIGH/CRITICAL events for that game ("Live refresh: Patrick Mahomes OUT (STARTING_QB_CHANGE)"); otherwise "Scheduled refresh: upstream data changed". Field-level diffs go to `prediction_change_log`; the events are marked processed with `prediction_recalculated` true or false. Nothing is ever deleted.

## 11. Frontend components (`build/v2/js_live.js`, wired in `js_games.js`)

- `predictionChange(g)` — old → new win probability and score, reason, "What changed?" with the version history.
- `liveFeed(g)` — newest-first events with severity chips and source, shown on the game overview.
- `weatherCardLive(g)` — indoor / pending-roof / open, temperature, wind, gusts, rain grid, impact badge, uncertainty note, active alerts, provider attribution and forecast time. Stale warning when freshness fails.
- `injuryImpactBox(g)` — team-level tier with the players driving it, first thing on the Injuries tab.
- Card chips on the Games list: "Weather high/moderate", "Prediction updated", "Injury data may be stale".
- `#liveBanner` — site-wide notice when the live layer itself is stale.
- `snapshotLive` / `highlightChanges` — values that changed since the last poll pulse once (`.pulse`).
- `dataWarning` / `sourceLine` — the shared "as of … from …" line under every live object.

Every live object the page consumes is the clean shape produced by `context.game_context`:
`{injuryImpact, weather, alerts, recentEvents, versions, projectionChanged, lastUpdated, freshness}`; no raw provider payloads reach the browser.

## 12. Error handling, fallback, freshness

- `http.get_json`: 3 retries with exponential backoff on 429/5xx/network, per-host circuit breaker (4 consecutive failures → host skipped 15 min), every call logged with latency and retry count.
- ESPN down → official report + rosters; both down → the previous merged state is kept and marked stale (never blanked, never assumed healthy).
- Open-Meteo down → NWS hourly; both down → last forecast row kept and marked stale. Non-US stadiums use Open-Meteo only.
- nflverse parquet unreachable → the copy cached under `data/` from the last run.
- Validation never drops silently: every rejected row is a `data_quality` row, and impossible transitions are kept but flagged.
- The live layer is wrapped in try/except inside `run_pipeline.py`, so a live failure can never block a publish.
- Freshness (`context.STALE_HOURS`): injuries are stale after 45 min within 6 h of kickoff, 3 h within a day, 8 h within 3 days, else 24 h; weather 1.5 h / 3 h / 8 h / 24 h. Stale sets the warning on the object, the chip on the card, and (when the whole layer is stale) the banner.

## 13. Testing

`python3 test_live.py` — 70+ checks: team mapping, status normalization (scratch vs injury, unknown text), identity (suffixes, diacritics, ambiguity), validation (bad team, unknown game, implausible timestamp, duplicate, impossible transition), stadium roof logic, weather parse/summarize/classify/compare/diff against real captured fixtures (Arrowhead, Sep 2026), impact tiers, event rules (bootstrap silence, first-sight, QB change), versioning (initial, unchanged, changed with reason, processed markers), context freshness.
Also: `test_fresh_keys.py` (page keys match `merge_payload.FRESH_KEYS`, now including `live_meta`), `test_embed.py` (page renders, injuries route), `test_backtest_sync.py`, `test_prop_value.py`.
Offline runs: `SJ_ESPN_FIXTURE=tests/fixtures/espn_injuries_2026-09-11.psv SJ_WX_FIXTURES=tests/fixtures SJ_LIVE_DIR=/tmp/live python3 run_live.py --force`.
End-to-end verified: bootstrap is silent; steady-state polls append nothing; a simulated Mahomes Out produces `STARTING_QB_CHANGE` CRITICAL, a refresh request, a new prediction version with that reason, and the page shows the change box, feed and weather card on desktop and mobile.

## 14. Files

New: `live/{__init__,store,teams,identity,normalize,stadiums,http,weather,events,impact,versions,context}.py`, `live/stadiums.json`, `live/README.md`, `run_live.py`, `test_live.py`, `tests/fixtures/{openmeteo_KAN00_2026-09-12,nws_points_KAN00,nws_hourly_EAX_47_48,nws_alerts_sample,nws_alerts_none}.json`, `build/v2/js_live.js`.
Modified: `run_pipeline.py` (live hooks after the payload is built; `kickoff`, `gameday_iso`, `stadium_id` on games), `merge_payload.py` (`live_meta` fresh key), `build/v2/{js_core,js_games,components.css,assemble}.*`, `test_embed.py`, `.github/workflows/refresh.yml` (live job, schedules, dispatch mode), `.gitignore` (`!live/data/`).

## 15. Manual configuration still required

1. Push the files above and dispatch `refresh.yml` with `mode=live` once; `live/data/` is created by that first run.
2. Optional: add the `NWS_USER_AGENT` secret if you want NWS to have a contact address.
3. Optional: schedule the observed-weather job (post-game `weather.observed_url`) if you want forecast-vs-actual history.
4. `admin.html` is public on Pages (noindex). Move it behind something if that ever matters.
5. Notifications: the event log is ready for it; nothing sends anything yet.
6. Open-Meteo's free tier is non-commercial. If the site ever charges, switch to their paid plan or make NWS primary for US venues.
