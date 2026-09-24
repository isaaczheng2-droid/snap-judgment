"""
Model impact v2: how much a player's availability matters to the prediction, as a NUMBER.

v1 classified LOW/MODERATE/HIGH/CRITICAL from position and depth order alone, and it had
a failure mode the site actually shipped: when a team rules its starting quarterback out,
the published depth chart often promotes the backup BEFORE kickoff — so the injured
starter drops to rank 2, v1 read "backup quarterback", and a card could show a starting
QB OUT next to MODEL IMPACT: LOW. That is corrected here three ways:

  1. Starter detection no longer trusts the current depth rank alone. A player is treated
     as a starter if ANY of: depth rank 1, recent snap share >= 55%, he is the projected
     starting QB the game model used (payload home_qb/away_qb), or a usage share that only
     starters have. A demoted-because-injured starter stays a starter.
  2. The classification is a 0-100 MODEL IMPACT SCORE:
         importance (0-100) x availability severity (0-1) x replacement factor (0.6-1.2)
     mapped to tiers at 25/50/75 — so the tier is explainable, sortable and auditable.
  3. Sanity floors that arithmetic can never talk its way out of:
         starting QB OUT/INACTIVE/IR  -> CRITICAL, always
         starting QB DOUBTFUL         -> at least HIGH
         starting QB QUESTIONABLE     -> at least MODERATE
         starter LT / edge / CB1 OUT  -> at least MODERATE
         >=25% usage-share starter OUT-> at least HIGH
     A floor that fires is named in the reasons, so the UI can say why.

The score still does not adjust a prediction number. The game model's own injury features
and the pipeline's healthy-vs-published counterfactual are the only quantities that move a
probability, and they only move when the model re-runs. The tier drives refresh triggers
(CRITICAL and HIGH ask for a recalculation) and the ordering of everything shown.
"""
TIERS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

# position importance 0-100 for a STARTER at the position (backups are scaled down).
# Ordering follows the game model's own POS_WEIGHT plus football judgment for QB;
# these are display/triage weights, not model coefficients.
POS_IMPORTANCE = {"QB": 100, "T": 62, "LT": 66, "RT": 58, "DE": 62, "EDGE": 62, "CB": 60, "WR": 60,
                  "RB": 55, "DT": 50, "NT": 48, "DL": 55, "S": 48, "FS": 48, "SS": 48, "DB": 50,
                  "G": 45, "LG": 45, "RG": 45, "C": 46, "OL": 48, "TE": 45, "LB": 45, "ILB": 45,
                  "MLB": 46, "OLB": 50, "K": 35, "FB": 15, "P": 15, "LS": 8}

# availability severity 0-1 (how much of the player is lost / at risk)
SEVERITY = {"OUT": 1.0, "INACTIVE": 1.0, "IR": 1.0, "PUP": 1.0, "NFI": 1.0, "SUSPENDED": 1.0,
            "DOUBTFUL": 0.85, "QUESTIONABLE": 0.5, "DID_NOT_PRACTICE": 0.4, "LIMITED_PRACTICE": 0.2,
            "FULL_PRACTICE": 0.05, "ACTIVE": 0.0, "HEALTHY": 0.0, "UNKNOWN": 0.3}
GONE = {"OUT", "INACTIVE", "IR", "PUP", "NFI", "SUSPENDED"}

SNAP_STARTER = 0.55         # recent snap share that marks a starter whatever the chart says
USAGE_KEY = 0.25            # target/carry share that marks a primary playmaker


def tier_of(score):
    return "CRITICAL" if score >= 75 else "HIGH" if score >= 50 else "MODERATE" if score >= 25 else "LOW"


def _tier_floor(tier, floor):
    return floor if TIERS.index(floor) > TIERS.index(tier) else tier


def is_starter(player, usage=None):
    """(bool, how). Depth rank 1 OR snap share OR projected-starter flag OR starter usage.
    A missing depth chart makes a player 'unknown', never 'backup'."""
    u = {**(usage or {}), **{k: player.get(k) for k in ("snap_share", "tgt_share", "car_share", "is_starting_qb", "is_projected_starter") if player.get(k) is not None}}
    depth = player.get("depth_order")
    if u.get("is_starting_qb") or u.get("is_projected_starter"):
        return True, "projected starter in the game model"
    if depth == 1:
        return True, "depth chart rank 1"
    ss = u.get("snap_share")
    if ss is not None and ss >= SNAP_STARTER:
        return True, f"{ss:.0%} of recent snaps"
    share = max([v for v in (u.get("tgt_share"), u.get("car_share")) if v is not None] or [0])
    if share >= USAGE_KEY:
        return True, f"{share:.0%} usage share"
    if depth is None:
        return None, "depth chart position unknown"
    return False, f"depth rank {depth}"


def score(player, usage=None):
    """
    player: dict with position, normalized_status (or status), depth_order, and optionally
            snap_share / tgt_share / car_share / is_starting_qb / is_projected_starter /
            replacement_gap (0-1, starter grade minus replacement grade / 100).
    -> {"score": 0-100, "tier": ..., "role": "STARTER"|"BACKUP"|"DEPTH"|"ROLE UNKNOWN",
        "starter": bool|None, "reasons": [...], "parts": {...}}
    """
    u = usage or {}
    pos = str(player.get("position") or "").upper()
    has_status = bool(player.get("normalized_status") or player.get("status"))
    st = str(player.get("normalized_status") or player.get("status") or "UNKNOWN").upper()
    # No status at all means the caller is asking "how much does this PLAYER matter"
    # (event classification does its own status handling), so severity is full — a
    # starting QB must classify CRITICAL there, exactly as v1 did.
    sev = SEVERITY.get(st, 0.3) if has_status else 1.0
    imp = POS_IMPORTANCE.get(pos, 40)
    starter, how = is_starter(player, u)
    reasons = []
    if starter:
        role_mult, role = 1.0, "STARTER"
        reasons.append(f"starter ({how})")
    elif starter is None:
        role_mult, role = (0.85 if pos == "QB" else 0.6), "ROLE UNKNOWN"
        reasons.append(how)
    else:
        depth = player.get("depth_order") or 3
        role_mult = 0.45 if depth == 2 else 0.2
        role = "BACKUP" if depth == 2 else "DEPTH"
        reasons.append(how)
    # usage sharpens importance for skill players
    uu = {**u, **{k: player.get(k) for k in ("snap_share", "tgt_share", "car_share") if player.get(k) is not None}}
    share = max([v for v in (uu.get("tgt_share"), uu.get("car_share")) if v is not None] or [0])
    if share >= USAGE_KEY:
        imp = min(100, imp * 1.15)
        reasons.append(f"{share:.0%} of team targets/carries")
    ss = uu.get("snap_share")
    if ss is not None and ss < 0.30 and not starter:
        imp *= 0.7
        reasons.append(f"only {ss:.0%} of snaps")
    # replacement quality: a big starter-vs-backup grade gap raises the impact, a tiny one lowers it
    gap = player.get("replacement_gap")
    repl = 1.0 if gap is None else max(0.6, min(1.2, 0.75 + float(gap) * 1.5))
    if gap is not None:
        reasons.append(f"replacement grade gap {round(float(gap) * 100)}")
    s = imp * role_mult * sev * repl
    s = max(0.0, min(100.0, s))
    tier = tier_of(s)
    # ---- sanity floors: arithmetic never overrides football
    if pos == "QB" and starter:
        if st in GONE:
            tier, s = "CRITICAL", max(s, 80.0)
            reasons.append("floor: a starting quarterback ruled out is never below CRITICAL")
        elif st == "DOUBTFUL":
            tier = _tier_floor(tier, "HIGH"); s = max(s, 60.0)
            reasons.append("floor: starting QB doubtful is at least HIGH")
        elif st in ("QUESTIONABLE", "DID_NOT_PRACTICE"):
            tier = _tier_floor(tier, "MODERATE"); s = max(s, 35.0)
            reasons.append("floor: starting QB in doubt is at least MODERATE")
    elif starter and st in GONE:
        if share >= USAGE_KEY:
            tier = _tier_floor(tier, "HIGH"); s = max(s, 55.0)
            reasons.append("floor: a primary playmaker out is at least HIGH")
        elif pos in ("LT", "T", "DE", "EDGE", "CB"):
            tier = _tier_floor(tier, "MODERATE"); s = max(s, 30.0)
            reasons.append(f"floor: a starting {pos} out is at least MODERATE")
    if pos == "QB" and starter is None and st in GONE:
        # unknown-role QB who is out: fail up, never down
        tier = _tier_floor(tier, "HIGH"); s = max(s, 55.0)
        reasons.append("floor: a quarterback out with unknown depth is treated as HIGH until the chart says otherwise")
    return {"score": round(s, 1), "tier": tier, "role": role, "starter": starter,
            "reasons": reasons, "parts": {"importance": round(imp, 1), "role_mult": role_mult,
                                          "severity": sev, "replacement": repl, "status": st}}


def classify(player, usage=None):
    """Back-compatible wrapper: -> (tier, reasons)."""
    r = score(player, usage)
    return r["tier"], r["reasons"]


# events that ask the engine to run again. Everything else is recorded and displayed only.
REFRESH_TIERS = {"CRITICAL", "HIGH"}


def wants_refresh(event):
    return event.get("severity") in REFRESH_TIERS and event.get("event_type") in (
        "PLAYER_STATUS_CHANGE", "GAMEDAY_INACTIVE", "DEPTH_CHART_CHANGE", "STARTING_QB_CHANGE")
