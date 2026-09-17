"""
The promotion gate. A candidate is promoted only when every predefined requirement in
learn/config.json holds on the selection seasons:

  samples     at least min_samples paired rows
  gain        relative MAE improvement over the current model of at least min_rel_gain
  evidence    the improvement survives a cluster bootstrap (resampling whole season-weeks,
              because players in the same week share game environments) with the p-value
              Holm-adjusted for the number of candidates tried in the cycle
  segments    no position and no season gets worse than max_segment_loss relative to the
              current model

The gate reads thresholds; it never changes them. Improvement has to be demonstrated.
"""
import numpy as np


def cluster_bootstrap(j, n_boot=400, seed=0):
    """Distribution of the relative gain when whole (season, week) clusters are resampled."""
    rng = np.random.default_rng(seed)
    clusters = j.groupby(["season", "week"]).agg(cur=("e_cur", "sum"), cand=("e_cand", "sum"), n=("e_cur", "size")).reset_index()
    cur, cand, n = clusters.cur.values, clusters.cand.values, clusters.n.values
    gains = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(clusters), len(clusters))
        c, k = cur[idx].sum(), cand[idx].sum()
        gains.append(1 - k / c if c else 0.0)
    return np.array(gains)


def holm(pvals):
    """Holm step-down adjusted p-values, in the original order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def assess(j, config, n_boot=None):
    """Everything the decision needs for one candidate, before multiplicity adjustment."""
    n = int(len(j))
    mae_cur, mae_cand = float(j.e_cur.mean()), float(j.e_cand.mean())
    gain = 1 - mae_cand / mae_cur if mae_cur else 0.0
    bs = cluster_bootstrap(j, n_boot or config.get("bootstrap", 400))
    p = float((bs <= 0).mean())
    segs = {}
    worst = None
    for col in config.get("segments", ["position", "season"]):
        for val, g in j.groupby(col):
            c, k = float(g.e_cur.mean()), float(g.e_cand.mean())
            rg = 1 - k / c if c else 0.0
            segs[f"{col}={val}"] = {"n": int(len(g)), "gain": round(rg, 4)}
            if worst is None or rg < worst[1]:
                worst = (f"{col}={val}", rg)
    return {"n": n, "mae_current": round(mae_cur, 4), "mae_candidate": round(mae_cand, 4), "rel_gain": round(gain, 4),
            "gain_ci95": [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)],
            "p_raw": round(p, 4), "segments": segs, "worst_segment": {"segment": worst[0], "gain": round(worst[1], 4)} if worst else None}


def decide(assessments, config):
    """Apply the gate to a list of assessments (one per candidate); adds p_adj, accepted, reasons."""
    ps = [a["p_raw"] for a in assessments]
    adj = holm(ps) if ps else []
    for a, pa in zip(assessments, adj):
        a["p_adj"] = round(float(pa), 4)
        reasons = []
        if a["n"] < config["min_samples"]:
            reasons.append(f"only {a['n']} paired rows; {config['min_samples']} required")
        if a["rel_gain"] < config["min_rel_gain"]:
            reasons.append(f"gain {a['rel_gain']:+.2%} is below the {config['min_rel_gain']:.1%} bar")
        if a["p_adj"] > config["alpha"]:
            reasons.append(f"not significant after adjusting for {len(assessments)} candidates (p={a['p_adj']:.3f}, alpha={config['alpha']})")
        ws = a.get("worst_segment")
        if ws and ws["gain"] < -config["max_segment_loss"]:
            reasons.append(f"{ws['segment']} gets worse by {-ws['gain']:.1%}, beyond the {config['max_segment_loss']:.0%} allowance")
        a["accepted"] = not reasons
        a["reasons"] = reasons or [f"gain {a['rel_gain']:+.2%} (95% {a['gain_ci95'][0]:+.2%} to {a['gain_ci95'][1]:+.2%}), p={a['p_adj']:.3f}, no segment worse than {config['max_segment_loss']:.0%}"]
    return assessments
