# Fantasy workspace

What it is: weekly Full-PPR (configurable) projections, rankings, start/sit, watchlists and a
forecast-vs-result record for QB/RB/WR/TE, built on the same per-stat ridge models the prop
pages use, plus four fantasy-only targets (RB receptions and receiving TDs, QB rushing TDs and
interceptions). Points are recomputed in the page from the per-stat projections, so any
scoring setting applies at once.

Files: `scoring.py` (settings, presets, points), `engine.py` (targets, ranges, availability,
flags, competition), `build.py` (the payload block, called from run_pipeline), `backtest.py`
(walk-forward validation → data/fantasy_backtest.json, fantasy_model.json, data/fantasy_oof.parquet),
`startsit.py` (verdict strength from the record), `script.py` (game-script scenarios from
play-by-play; context only), `script_test.py` (its harness test).

Validated (walk-forward, 2019-2024 selection, 2025 untouched holdout, 27,469 player-weeks):
points MAE beats the rolling-average baseline by 2.5-4.2% per position on 2025 (and last-season
average by ~11%); weekly Spearman 0.43-0.60; top-N precision 46-59%; 10-90 range covers 73%
(QB) to 82% (WR) of results, 25-75 covers 45-52%; start/sit pairs: 5+ pt gap 82% (rolling avg
80%), 3-5 pt 70%, 1-3 pt 60% (a toss-up, and the page says so). Questionable players have
appeared 50% of the time since 2019 (by position on the card); Doubtful 0.5%.

Not validated / not built: custom-scoring ranges (scaled from the Full PPR range and labelled),
fumbles / 2-pt / return yards (not projected; actuals include them), roster-specific advice,
Sleeper import, waivers, trades (architecture notes in the Settings page). Game-script volume
features tested at +0.26% (p<0.001) and kept out of the number; shown as context.
