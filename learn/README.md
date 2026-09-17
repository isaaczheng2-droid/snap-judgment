# Controlled learning cycle

`python3 learn/cycle.py` runs: collect (data manifest with digests → data_version), forecasts
(fantasy projections into learn/data/forecasts.ndjson, before kickoff only, revisions appended,
never edited), actuals (finished games with posted box scores; changed results appended as
corrections; did-not-play recorded explicitly), measure (learn/data/prospective.json), train +
compare + promote (candidate menu in config.json, walk-forward on the selection seasons,
row-paired against the active model, cluster bootstrap by season-week, Holm-adjusted p-values,
segment checks, gate in gate.py). `--dry-run` skips registry writes; `--rollback "reason"`;
`--status`; `paused: true` in config.json stops training while the ledger keeps recording.

The cycle never edits config.json or code. The registry (learn/registry.json) holds every
version with params, metrics, reason and status; run_pipeline applies the active version's
parameters at startup, so a promotion changes the site's model without a code change.

Selection vs final evaluation: candidates are chosen on 2019-2024 only; the 2025 holdout is off
by default (report_holdout=false) so it is not tuned against repeatedly; the prospective
ledger is the final test and is never used to choose.

Scheduling: a Python job. The site is static and does not need to be open. A local scheduler
needs the computer on at run time; running it from the GitHub Actions workflow does not.
First run (2026-09-17): 5 candidates, all rejected below the 1% bar (best: script_volume +0.26%).
