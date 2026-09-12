"""
Model impact: how much a player's availability matters to the prediction.

Classifies LOW / MODERATE / HIGH / CRITICAL from things the pipeline already measures:
position, depth-chart order, recent snap share and usage shares. It does NOT assign a
numerical adjustment. The game model's own injury features (position-weighted snap share of
ruled-out players, QB change) and the pipeline's exact healthy-vs-published counterfactual
are the only quantities that move a number, and they only move it when the model re-runs.

The tier also drives refresh triggers: CRITICAL and HIGH events ask for a model recalculation;
MEDIUM and LOW are recorded and shown but do not rerun anything.
"""
TIERS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

# starting-lineup weights, roughly the pipeline's POS_WEIGHT plus offensive line and kickers
POS_BASE = {"QB": "CRITICAL", "RB": "HIGH", "WR": "HIGH", "TE": "MODERATE", "T": "HIGH", "G": "MODERATE", "C": "MODERATE",
            "OL": "MODERATE", "DE": "MODERATE", "DT": "MODERATE", "EDGE": "MODERATE", "LB": "MODERATE", "CB": "MODERATE",
            "S": "MODERATE", "DB": "MODERATE", "K": "MODERATE", "P": "LOW", "LS": "LOW", "FB": "LOW"}


def _down(t, n=1):
    return TIERS[max(0, TIERS.index(t) - n)]


def _up(t, n=1):
    return TIERS[min(len(TIERS) - 1, TIERS.index(t) + n)]


def classify(player, usage=None):
    """
    player: dict with position, depth_order (1 = starter), and optionally snap_share,
            tgt_share, car_share, is_starting_qb.
    usage:  optional dict of the same usage numbers when they live elsewhere.
    -> (tier, reasons)
    """
    u = {**(usage or {}), **{k: player.get(k) for k in ("snap_share", "tgt_share", "car_share") if player.get(k) is not None}}
    pos = str(player.get("position") or "").upper()
    depth = player.get("depth_order")
    tier = POS_BASE.get(pos, "LOW")
    why = [f"{pos or 'unknown position'} baseline {tier.lower()}"]
    if pos == "QB":
        if player.get("is_starting_qb") or depth == 1:
            why.append("starting quarterback")
        elif depth is None:
            tier = "MODERATE"; why.append("quarterback, depth chart position unknown")
        else:
            tier = "LOW"; why.append("backup quarterback")
        return tier, why
    if depth is None:
        tier = _down(tier); why.append("depth chart position unknown")
    elif depth >= 3:
        tier = "LOW"; why.append(f"depth {depth}")
    elif depth == 2:
        tier = _down(tier); why.append("second on the depth chart")
    ss = u.get("snap_share")
    if ss is not None:
        if ss >= 0.75: tier = _up(tier); why.append(f"{ss:.0%} of snaps")
        elif ss < 0.30: tier = _down(tier); why.append(f"{ss:.0%} of snaps")
    share = max([v for v in (u.get("tgt_share"), u.get("car_share")) if v is not None] or [None]) if any(v is not None for v in (u.get("tgt_share"), u.get("car_share"))) else None
    if share is not None:
        if share >= 0.25: tier = _up(tier); why.append(f"{share:.0%} usage share")
        elif share < 0.08: tier = _down(tier); why.append(f"{share:.0%} usage share")
    if pos in ("RB", "WR", "TE") and tier == "CRITICAL":
        tier = "HIGH"                       # only a quarterback is critical on his own
    return tier, why


# events that ask the engine to run again. Everything else is recorded and displayed only.
REFRESH_TIERS = {"CRITICAL", "HIGH"}


def wants_refresh(event):
    return event.get("severity") in REFRESH_TIERS and event.get("event_type") in (
        "PLAYER_STATUS_CHANGE", "GAMEDAY_INACTIVE", "DEPTH_CHART_CHANGE", "STARTING_QB_CHANGE")
