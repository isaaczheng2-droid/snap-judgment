"""
Snap Judgment live data layer.

    sources -> ingestion -> validation -> normalized store -> change detection
            -> impact evaluation -> prediction refresh -> frontend

Everything here sits BESIDE the prediction engine (run_pipeline.py), never inside it. The
engine keeps producing its independent number from football data; this layer watches the
world change, records every change with its provenance, decides whether the engine needs to
run again, and explains the result to the reader.

Persistence is append-only NDJSON under live/data/, committed by the GitHub Actions job. In
this deployment GitHub Actions is the backend, the repo is the database, and payload.json is
the API the page reads. No provider is ever called from the browser.
"""
from . import store, teams, identity, normalize, stadiums, http, weather, events, impact, versions, context  # noqa: F401

__all__ = ["store", "teams", "identity", "normalize", "stadiums", "http", "weather", "events",
           "impact", "versions", "context"]
