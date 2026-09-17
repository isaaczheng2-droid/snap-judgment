#!/usr/bin/env python3
"""
The controlled learning cycle.

    python3 learn/cycle.py                 # every step, promotion only through the gate
    python3 learn/cycle.py --dry-run       # everything except writing to the registry
    python3 learn/cycle.py --steps collect,forecasts,actuals,measure
    python3 learn/cycle.py --rollback "reason"
    python3 learn/cycle.py --status

Steps, in order:
  1 collect     data manifest: every source with its digest and timestamp -> data_version
  2 forecasts   the current payload's fantasy projections into the ledger, before kickoff only
  3 actuals     results for finished games into the ledger, corrections kept
  4 measure     errors by position, week, status and model version on the prospective record
  5 train       candidate models from the fixed menu, walk-forward on the selection seasons
  6 compare     row-paired against the current model and the naive baseline
  7 promote     only if the gate in learn/config.json passes; otherwise recorded as rejected

The cycle never edits learn/config.json or any code. `paused: true` in the config stops steps
5-7 (the ledger keeps recording). Everything it does is appended to learn/data/cycle_log.ndjson
and experiments to learn/data/experiments.ndjson.

Scheduling: this is a Python job, not part of the website. The site is static; nothing here
needs it to be open. Run it wherever Python runs: a local scheduler (Task Scheduler / cron)
means that computer has to be on at the scheduled time; the GitHub Actions workflow already
used for the hourly rebuild can run it too, in which case no computer of yours needs to be on.
"""
import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import run_pipeline as rp                                                  # noqa: E402
from learn import registry, manifest, ledger, evaluate, candidates, gate     # noqa: E402
from fantasy import backtest                                               # noqa: E402

STORE = os.path.join(HERE, "data")
LOG = os.path.join(STORE, "cycle_log.ndjson")
EXPERIMENTS = os.path.join(STORE, "experiments.ndjson")
CONFIG = os.path.join(HERE, "config.json")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_event(kind, **kw):
    os.makedirs(STORE, exist_ok=True)
    rec = {"at": now(), "event": kind, **kw}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{rec['at'][11:19]}] {kind}: " + ", ".join(f"{k}={v}" for k, v in kw.items() if k not in ("detail",)))


def load_config():
    return json.load(open(CONFIG))


def step_collect(datadir, cur):
    m = manifest.snapshot(datadir, cur)
    log_event("collect", data_version=m["data_version"], sources=sum(1 for s in m["sources"] if s.get("present")))
    return m


def step_forecasts(payload_path, sched, data_version):
    try:
        payload = json.load(open(payload_path))
    except Exception as e:
        log_event("forecasts_skipped", reason=f"no payload at {payload_path}: {e}")
        return 0
    fantasy = payload.get("fantasy")
    if not fantasy:
        log_event("forecasts_skipped", reason="payload has no fantasy block")
        return 0
    v = registry.active()
    n, late = ledger.record_forecasts(fantasy, sched, v["id"] if v else "unregistered", data_version, log=lambda *a: None)
    log_event("forecasts", recorded=n, skipped_after_kickoff=late, week=fantasy.get("week"))
    return n


def step_actuals(plyr, sched, cur, digest):
    n, corr = ledger.record_actuals(plyr, sched, cur, source_digest=digest, log=lambda *a: None)
    log_event("actuals", graded=n, corrections=corr, pending=len(ledger.pending()))
    return n


def step_measure():
    g = ledger.graded()
    res = evaluate.prospective(g)
    os.makedirs(STORE, exist_ok=True)
    json.dump(res, open(os.path.join(STORE, "prospective.json"), "w"), indent=1)
    log_event("measure", graded_rows=res.get("n", 0), mae=(res.get("all") or {}).get("mae"), vs_naive=(res.get("all") or {}).get("vs_naive"))
    return res


def step_train_compare_promote(config, datadir, dry_run):
    if config.get("paused"):
        log_event("train_skipped", reason="paused in learn/config.json")
        return None
    reg = registry.init(rp)
    active = reg["versions"][reg["active"]]
    pw, inj, plyr, cur = backtest.build_rows(datadir, log=lambda *a: None)
    sched = pd.read_csv(os.path.join(datadir, "games.csv"), low_memory=False)
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    seasons = config["selection_seasons"]
    frames, used = candidates.run(rp, pw, sched, config, active["params"], config["candidates"], seasons, log=lambda *a: None)
    exp_id = f"exp-{now()[:10]}-{uuid.uuid4().hex[:6]}"
    assessments = []
    for spec in config["candidates"]:
        j = evaluate.paired(frames["current"], frames[spec["id"]])
        a = gate.assess(j, config)
        a["candidate"] = spec["id"]
        a["params"] = used[spec["id"]]
        a["note"] = spec.get("note")
        assessments.append(a)
    assessments = gate.decide(assessments, config)
    cur_mae = float((frames["current"].act_pts - frames["current"].proj_pts).abs().mean())
    naive_mae = float((frames["current"].act_pts - frames["current"].naive_pts).abs().mean())
    record = {"id": exp_id, "at": now(), "active_version": active["id"], "selection_seasons": seasons,
              "n_rows": int(len(frames["current"])), "current_mae": round(cur_mae, 4), "naive_mae": round(naive_mae, 4),
              "config": {k: config[k] for k in ("min_samples", "min_rel_gain", "alpha", "max_segment_loss", "bootstrap")},
              "candidates": [{k: v for k, v in a.items() if k != "segments"} | {"segments": a["segments"]} for a in assessments],
              "dry_run": dry_run}
    accepted = [a for a in assessments if a["accepted"]]
    promoted = None
    if accepted:
        best = max(accepted, key=lambda a: a["rel_gain"])
        record["winner"] = best["candidate"]
        if not dry_run:
            v = registry.add_candidate(reg, best["params"], active["id"], {k: best[k] for k in ("n", "rel_gain", "gain_ci95", "p_adj", "mae_current", "mae_candidate")},
                                       "candidate", "; ".join(best["reasons"]), exp_id)
            registry.promote(reg, v["id"], f"experiment {exp_id}: {best['candidate']} {'; '.join(best['reasons'])}")
            promoted = v["id"]
            record["promoted"] = promoted
    else:
        record["winner"] = None
    if not dry_run:
        for a in assessments:
            if not a["accepted"]:
                registry.add_candidate(reg, a["params"], active["id"], {k: a[k] for k in ("n", "rel_gain", "gain_ci95", "p_adj", "mae_current", "mae_candidate")},
                                       "rejected", "; ".join(a["reasons"]), exp_id)
    os.makedirs(STORE, exist_ok=True)
    with open(EXPERIMENTS, "a") as f:
        f.write(json.dumps(record) + "\n")
    log_event("experiment", id=exp_id, candidates=len(assessments), accepted=len(accepted), promoted=promoted or "none",
              current_mae=round(cur_mae, 3), naive_mae=round(naive_mae, 3))
    for a in assessments:
        print(f"    {a['candidate']:<16} gain {a['rel_gain']:+.2%}  CI {a['gain_ci95'][0]:+.2%}..{a['gain_ci95'][1]:+.2%}  p_adj {a['p_adj']:.3f}  "
              f"worst {a['worst_segment']['segment'] if a['worst_segment'] else '-'} {a['worst_segment']['gain'] if a['worst_segment'] else 0:+.2%}  -> {'ACCEPTED' if a['accepted'] else 'rejected: ' + a['reasons'][0]}")
    return record


def status():
    reg = registry.load()
    cfg = load_config()
    exps = ledger._read(EXPERIMENTS)
    pros = None
    try:
        pros = json.load(open(os.path.join(STORE, "prospective.json")))
    except Exception:
        pass
    out = {"paused": cfg.get("paused", False), "active": reg["versions"][reg["active"]]["id"] if reg else None,
           "previous": reg.get("previous") if reg else None, "versions": len(reg["versions"]) if reg else 0,
           "experiments": len(exps), "last_experiment": exps[-1] if exps else None,
           "ledger": {"forecasts": len(ledger._read(ledger.FORECASTS)), "actuals": len(ledger._read(ledger.ACTUALS)), "pending": len(ledger.pending())},
           "prospective": (pros or {}).get("all"), "manifest": (manifest.latest() or {}).get("data_version")}
    return out


def summary_for_payload(limit=6):
    """The block the site shows on the Model health page."""
    reg = registry.load()
    cfg = load_config()
    exps = ledger._read(EXPERIMENTS)[-limit:]
    def slim(e):
        return {"id": e["id"], "at": e["at"], "active_version": e["active_version"], "n_rows": e["n_rows"],
                "current_mae": e["current_mae"], "naive_mae": e["naive_mae"], "winner": e.get("winner"), "promoted": e.get("promoted"),
                "dry_run": e.get("dry_run", False),
                "candidates": [{"candidate": c["candidate"], "note": c.get("note"), "rel_gain": c["rel_gain"], "gain_ci95": c["gain_ci95"],
                                "p_adj": c["p_adj"], "accepted": c["accepted"], "reasons": c["reasons"], "worst_segment": c.get("worst_segment")}
                               for c in e["candidates"]]}
    pros = None
    try:
        pros = json.load(open(os.path.join(STORE, "prospective.json")))
    except Exception:
        pass
    events = ledger._read(LOG)[-12:]
    return {"paused": cfg.get("paused", False), "gate": {k: cfg[k] for k in ("min_samples", "min_rel_gain", "alpha", "max_segment_loss")},
            "selection_seasons": cfg.get("selection_seasons"), "registry": registry.summary(), "experiments": [slim(e) for e in exps],
            "prospective": pros, "ledger": {"forecasts": len(ledger._read(ledger.FORECASTS)), "actuals": len(ledger._read(ledger.ACTUALS)),
                                            "pending": len(ledger.pending())},
            "recent_events": events, "candidate_menu": [{"id": c["id"], "note": c.get("note")} for c in cfg.get("candidates", [])],
            "scheduling": "learn/cycle.py is a Python job; the site is static and does not need to be open. On a local scheduler the computer must be on at run time; on GitHub Actions it need not be."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--payload", default=os.path.join(ROOT, "payload.json"))
    ap.add_argument("--steps", default="collect,forecasts,actuals,measure,train")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rollback", metavar="REASON")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.status:
        print(json.dumps(status(), indent=1)); return 0
    if a.rollback:
        reg = registry.load()
        registry.rollback(reg, a.rollback)
        log_event("rollback", to=reg["active"], reason=a.rollback); return 0
    config = load_config()
    registry.init(rp)
    steps = [s.strip() for s in a.steps.split(",")]
    # On a fresh checkout (the Actions runner) data/ is not committed; fetch it the same way
    # the full rebuild does before reading anything.
    os.makedirs(a.datadir, exist_ok=True)
    if not os.path.exists(os.path.join(a.datadir, "games.csv")):
        log_event("fetch", reason="data directory empty; downloading sources")
        rp.load_all(a.datadir, None)
    sched = pd.read_csv(os.path.join(a.datadir, "games.csv"), low_memory=False)
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    cur = int(sched[sched.game_type == "REG"].season.max())
    m = None
    if "collect" in steps:
        m = step_collect(a.datadir, cur)
    dv = (m or manifest.latest() or {}).get("data_version")
    if "forecasts" in steps:
        step_forecasts(a.payload, sched, dv)
    if "actuals" in steps:
        try:
            plyr = pd.read_parquet(os.path.join(a.datadir, "stats_player", f"stats_player_week_{cur}.parquet"))
            digest = next((s.get("digest") for s in (m or manifest.latest() or {}).get("sources", []) if s["source"] == "player_stats"), None)
            step_actuals(plyr, sched, cur, digest)
        except Exception as e:
            log_event("actuals_skipped", reason=str(e)[:120])
    if "measure" in steps:
        step_measure()
    if "train" in steps:
        step_train_compare_promote(config, a.datadir, a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
