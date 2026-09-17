"""
Model registry: which parameter set the site runs, every version that was ever a candidate,
why each was promoted or rejected, and rollback.

A "model" here is a parameter set for the player stat ridges: the ridge penalty and the
per-stat feature tables (EXTRA_FEATS, ADJ_SHARES) in run_pipeline. The pipeline applies the
active version at startup (`apply`), so a promotion changes what the site runs without any
code being rewritten, and a rollback is one registry write.
"""
import copy
import hashlib
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "registry.json")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fingerprint(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def shipped_params(rp):
    """The parameters the code ships with, as the registry's baseline version."""
    return {"alpha": float(rp.RIDGE_ALPHA), "extra_feats": copy.deepcopy(rp.EXTRA_FEATS),
            "adj_shares": copy.deepcopy(rp.ADJ_SHARES), "script_feats": False}


def load(path=PATH):
    try:
        return json.load(open(path))
    except Exception:
        return None


def save(reg, path=PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(reg, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def init(rp, path=PATH, reason="shipped defaults"):
    """Create the registry with the code's defaults as version 1, if it does not exist."""
    reg = load(path)
    if reg:
        return reg
    params = shipped_params(rp)
    vid = f"v1-{fingerprint(params)}"
    reg = {"active": vid, "previous": None,
           "versions": {vid: {"id": vid, "params": params, "fingerprint": fingerprint(params), "created_at": now(),
                              "status": "active", "reason": reason, "parent": None, "metrics": None}},
           "history": [{"at": now(), "event": "init", "version": vid, "reason": reason}]}
    save(reg, path)
    return reg


def active(path=PATH):
    reg = load(path)
    if not reg:
        return None
    return reg["versions"].get(reg["active"])


def apply(rp, path=PATH, log=print):
    """Point run_pipeline at the active version's parameters. No registry: shipped defaults."""
    v = active(path)
    if not v:
        return None
    p = v["params"]
    rp.RIDGE_ALPHA = float(p.get("alpha", rp.RIDGE_ALPHA))
    rp.EXTRA_FEATS = copy.deepcopy(p.get("extra_feats", rp.EXTRA_FEATS))
    rp.ADJ_SHARES = copy.deepcopy(p.get("adj_shares", rp.ADJ_SHARES))
    log(f"  model registry: active {v['id']} (alpha {rp.RIDGE_ALPHA}, {len(rp.EXTRA_FEATS)} stats with extras)")
    return v


def add_candidate(reg, params, parent, metrics, status, reason, experiment_id=None, path=PATH):
    fp = fingerprint(params)
    n = 1 + max([int(k.split("-")[0][1:]) for k in reg["versions"]] or [0])
    vid = f"v{n}-{fp}"
    if vid in reg["versions"]:
        return reg["versions"][vid]
    reg["versions"][vid] = {"id": vid, "params": params, "fingerprint": fp, "created_at": now(), "status": status,
                            "reason": reason, "parent": parent, "metrics": metrics, "experiment": experiment_id}
    reg["history"].append({"at": now(), "event": status, "version": vid, "reason": reason, "experiment": experiment_id})
    save(reg, path)
    return reg["versions"][vid]


def promote(reg, vid, reason, path=PATH):
    if vid not in reg["versions"]:
        raise KeyError(vid)
    prev = reg["active"]
    if prev and prev in reg["versions"]:
        reg["versions"][prev]["status"] = "superseded"
    reg["previous"] = prev
    reg["active"] = vid
    reg["versions"][vid]["status"] = "active"
    reg["versions"][vid]["promoted_at"] = now()
    reg["versions"][vid]["promotion_reason"] = reason
    reg["history"].append({"at": now(), "event": "promoted", "version": vid, "from": prev, "reason": reason})
    save(reg, path)
    return reg


def rollback(reg, reason="manual rollback", path=PATH):
    prev = reg.get("previous")
    if not prev or prev not in reg["versions"]:
        raise RuntimeError("nothing to roll back to")
    cur = reg["active"]
    reg["versions"][cur]["status"] = "rolled_back"
    reg["versions"][prev]["status"] = "active"
    reg["active"], reg["previous"] = prev, cur
    reg["history"].append({"at": now(), "event": "rollback", "version": prev, "from": cur, "reason": reason})
    save(reg, path)
    return reg


def summary(path=PATH, experiments=None, limit=8):
    reg = load(path)
    if not reg:
        return None
    v = reg["versions"][reg["active"]]
    hist = reg["history"][-limit:]
    return {"active": {k: v.get(k) for k in ("id", "fingerprint", "created_at", "promoted_at", "reason", "promotion_reason", "status")},
            "active_params": v["params"], "previous": reg.get("previous"),
            "n_versions": len(reg["versions"]), "history": hist}
