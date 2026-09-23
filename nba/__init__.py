"""
Snap Judgment — NBA section.

Everything here is separate from the NFL pipeline: its own data layer, models, ledger and
payload key (`nba`). It shares only the page shell, the HTTP helper with retries, the append-
only store pattern and the Odds API quota discipline.

Data sources, in the order they are trusted (see nba/status.py for the coverage table):
  1. SportsDataverse "sportsdataverse-data" releases (ESPN box scores, schedules, game
     rosters) — reachable from everywhere this project runs; the historical backbone.
  2. api.pbpstats.com — possession-level team totals and lineups, no key; runner only.
  3. stats.nba.com via the community nba_api client — official box scores for validation,
     rosters and the player index; runner only; not an officially supported API.
  4. official.nba.com injury report (PDF) — runner only, in season.
  5. The Odds API, basketball_nba — game lines and per-event player props; key lives in the
     GitHub secret ODDS_API_KEY and is never written to the page or the repository.
  6. balldontlie — adapter present, disabled without a key; free tier is teams/players/games.
"""
