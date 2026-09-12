"""
Normalization of player availability, and the validation that gates it.

The rule that matters most: absence of information is UNKNOWN, never HEALTHY. A player who
is not on any report has normalized_status UNKNOWN with a note that no report exists; only a
positive source statement ("Active", "Full Participation") turns into an availability claim.
Both the vendor's original text and the normalized value are stored on every row.
"""
from datetime import datetime, timezone, timedelta

from . import teams, store

# --------------------------------------------------------------- the vocabulary
STATUSES = ["HEALTHY", "FULL_PRACTICE", "LIMITED_PRACTICE", "DID_NOT_PRACTICE", "QUESTIONABLE",
            "DOUBTFUL", "OUT", "IR", "PUP", "NFI", "SUSPENDED", "INACTIVE", "ACTIVE", "UNKNOWN"]
# ordering used for "worse / better" comparisons in change detection (higher = less available)
SEVERITY_RANK = {"ACTIVE": 0, "HEALTHY": 0, "FULL_PRACTICE": 1, "LIMITED_PRACTICE": 2,
                 "QUESTIONABLE": 3, "DID_NOT_PRACTICE": 3, "DOUBTFUL": 4, "OUT": 5, "INACTIVE": 5,
                 "SUSPENDED": 5, "PUP": 6, "NFI": 6, "IR": 6, "UNKNOWN": -1}
UNAVAILABLE = {"OUT", "INACTIVE", "IR", "PUP", "NFI", "SUSPENDED"}

# game status (official report vocabulary and ESPN's), lower-cased
GAME_STATUS = {
    "out": "OUT", "doubtful": "DOUBTFUL", "questionable": "QUESTIONABLE", "probable": "QUESTIONABLE",
    "injured reserve": "IR", "ir": "IR", "reserve/injured": "IR", "physically unable to perform": "PUP",
    "pup": "PUP", "reserve/pup": "PUP", "nfi": "NFI", "non-football injury": "NFI", "reserve/nfi": "NFI",
    "suspension": "SUSPENDED", "suspended": "SUSPENDED", "reserve/suspended": "SUSPENDED",
    "active": "ACTIVE", "inactive": "INACTIVE", "day-to-day": "QUESTIONABLE", "day to day": "QUESTIONABLE",
    "healthy": "HEALTHY", "": None, "none": None,
}
PRACTICE = {
    "full participation in practice": "FULL_PRACTICE", "full": "FULL_PRACTICE", "fp": "FULL_PRACTICE",
    "limited participation in practice": "LIMITED_PRACTICE", "limited": "LIMITED_PRACTICE", "lp": "LIMITED_PRACTICE",
    "did not participate in practice": "DID_NOT_PRACTICE", "dnp": "DID_NOT_PRACTICE", "did not practice": "DID_NOT_PRACTICE",
    "": None, "none": None,
}
ROSTER = {"ACT": "ACTIVE", "RES": "IR", "PUP": "PUP", "NON": "NFI", "SUS": "SUSPENDED", "EXE": "EXEMPT",
          "DEV": "PRACTICE_SQUAD", "CUT": "CUT", "RET": "RETIRED", "TRC": "TRADED", "TRD": "TRADED",
          "TRT": "TRADED", "UFA": "FREE_AGENT"}
# ESPN files gameday scratches under status Out with a details.type of "Coach's Decision";
# that is an INACTIVE, not an injury, and must never reach injury features as OUT
SCRATCH_TYPES = {"coach's decision", "coaches decision", "not injury related", "healthy scratch"}


def norm_status(text, injury_type=None):
    """Vendor status text -> normalized status, or None if it is not a status."""
    t = str(text or "").strip().lower()
    if t == "out" and str(injury_type or "").strip().lower() in SCRATCH_TYPES:
        return "INACTIVE"
    if str(injury_type or "").strip().lower() == "suspension":
        return "SUSPENDED"
    return GAME_STATUS.get(t, "UNKNOWN" if t else None)


def norm_practice(text):
    t = str(text or "").strip().lower()
    return PRACTICE.get(t, "UNKNOWN" if t else None)


def norm_roster(code):
    c = str(code or "").strip().upper()
    return ROSTER.get(c, "UNKNOWN" if c else None)


# --------------------------------------------------------------- the record
FIELDS = ["internal_player_id", "player_name", "team", "position", "game_id", "season", "week",
          "original_status", "normalized_status", "injury_body_part", "injury_description",
          "original_practice", "practice_status", "practice_date", "roster_status", "game_status",
          "estimated_return_date", "depth_chart_position", "depth_order", "active", "inactive",
          "source", "source_updated_at", "system_received_at", "last_verified_at", "snapshot_id"]


def record(**kw):
    """Build a player_status row with every field present; missing values are None, never a guess."""
    r = {k: None for k in FIELDS}
    r.update({k: v for k, v in kw.items() if k in FIELDS})
    if r["normalized_status"] is None:
        r["normalized_status"] = "UNKNOWN"
    # active / inactive are only ever set from a positive statement
    st = r["normalized_status"]
    if st == "ACTIVE":
        r["active"], r["inactive"] = True, False
    elif st == "INACTIVE":
        r["active"], r["inactive"] = False, True
    r["team"] = teams.to_canonical(r["team"]) or r["team"]
    return r


# --------------------------------------------------------------- validation
def _parse_ts(s):
    if not s:
        return None
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except Exception:
        return None


# transitions a real player cannot make inside one report cycle; anything listed is logged,
# kept, and flagged rather than silently accepted or dropped
IMPOSSIBLE = {("IR", "ACTIVE"), ("IR", "FULL_PRACTICE"), ("PUP", "ACTIVE"), ("SUSPENDED", "QUESTIONABLE")}


def validate(rows, crosswalk=None, games=None, now=None, previous=None):
    """
    Returns (accepted_rows, problems). Problems are also written to the data_quality table.
      - team must be canonical and, when the crosswalk knows the player, match his roster team
      - game_id must be on the slate when given
      - timestamps must be parseable, not in the future, not older than 30 days
      - duplicate (player, source, source_updated_at) rows are collapsed
      - impossible transitions from the previous known status are flagged
    """
    now = now or datetime.now(timezone.utc)
    ok, problems, seen = [], [], set()
    slate = set(games or [])
    for r in rows:
        pid = r.get("internal_player_id")
        prob = None
        if not pid:
            prob = ("unknown_player", f"{r.get('player_name')} ({r.get('team')}) could not be resolved to an id")
        elif not teams.is_canonical(r.get("team")):
            prob = ("bad_team", f"{pid}: team {r.get('team')!r} is not canonical")
        elif crosswalk and pid in crosswalk.meta and crosswalk.meta[pid].get("team") not in (None, r.get("team")):
            prob = ("team_mismatch", f"{pid}: report says {r.get('team')}, roster says {crosswalk.meta[pid].get('team')}")
        elif r.get("game_id") and slate and r["game_id"] not in slate:
            prob = ("unknown_game", f"{pid}: game {r.get('game_id')} not on the slate")
        else:
            ts = _parse_ts(r.get("source_updated_at"))
            if r.get("source_updated_at") and ts is None:
                prob = ("bad_timestamp", f"{pid}: {r.get('source_updated_at')!r}")
            elif ts and (ts > now + timedelta(hours=1) or ts < now - timedelta(days=30)):
                prob = ("implausible_timestamp", f"{pid}: {ts.isoformat()}")
        key = (pid, r.get("source"), r.get("source_updated_at"), r.get("normalized_status"), r.get("practice_status"))
        if not prob and key in seen:
            prob = ("duplicate", f"{pid}: duplicate report from {r.get('source')}")
        if not prob and previous and pid in previous:
            was = previous[pid].get("normalized_status")
            if (was, r.get("normalized_status")) in IMPOSSIBLE:
                prob = ("impossible_transition", f"{pid}: {was} -> {r.get('normalized_status')}")
                r["flag"] = "impossible_transition"      # kept, but marked
                seen.add(key); ok.append(r); problems.append(prob)
                store.quality(prob[0], prob[1], player_id=pid, source=r.get("source"))
                continue
        if prob:
            problems.append(prob)
            store.quality(prob[0], prob[1], player_id=pid, source=r.get("source"))
            continue
        seen.add(key)
        ok.append(r)
    return ok, problems


def _age_h(iso):
    t = _parse_ts(iso)
    return None if t is None else (datetime.now(timezone.utc) - t).total_seconds() / 3600


def merge_sources(rows):
    """
    One current record per player. ESPN wins for game status when it is newer (it is the only
    timestamped source and the only one that can see an in-game injury); the official report
    supplies practice participation, which ESPN does not carry. Disagreements are kept.
    """
    by = {}
    for r in rows:
        by.setdefault(r["internal_player_id"], []).append(r)
    merged, conflicts = {}, []
    for pid, rs in by.items():
        espn = [r for r in rs if r["source"] == "espn"]
        offi = [r for r in rs if r["source"] == "nflverse:injuries"]
        other = [r for r in rs if r["source"] not in ("espn", "nflverse:injuries")]
        base = (espn or offi or other)[0]
        cur = dict(base)
        if espn and offi:
            e, o = espn[0], offi[0]
            cur["practice_status"] = o.get("practice_status"); cur["original_practice"] = o.get("original_practice")
            cur["injury_body_part"] = cur.get("injury_body_part") or o.get("injury_body_part")
            es, os_ = e["normalized_status"], o["normalized_status"]
            if es != os_ and os_ not in ("UNKNOWN", None) and es not in ("UNKNOWN", None):
                # Both speak and disagree. Rules, in order:
                #   1. fail-safe: if either says the player is unavailable, that stands. A stale
                #      "Questionable" must never overrule a fresh "Out", and vice versa the
                #      cost of wrongly showing Out is far smaller than wrongly showing available.
                #   2. otherwise the timestamped source (ESPN) wins if filed within 48 hours;
                #      the official report has no timestamp finer than the week.
                e_un, o_un = es in UNAVAILABLE, os_ in UNAVAILABLE
                if e_un != o_un:
                    winner, rule = ("espn" if e_un else "nflverse:injuries"), "fail-safe: the unavailable status stands"
                else:
                    age = _age_h(e["source_updated_at"]) if e.get("source_updated_at") else None
                    winner, rule = ("espn" if age is not None and age < 48 else "nflverse:injuries"), "timestamped source within 48h, else the official report"
                if winner != "espn":
                    cur["normalized_status"], cur["original_status"], cur["source"] = os_, o["original_status"], o["source"]
                    cur["source_updated_at"] = o.get("source_updated_at")
                conflicts.append({"player_id": pid, "game_id": cur.get("game_id"), "sources": {"espn": es, "nflverse:injuries": os_},
                                  "resolved_to": winner, "rule": rule, "timestamp": store.now_iso()})
        elif offi and not espn:
            cur["practice_status"] = offi[0].get("practice_status")
        merged[pid] = cur
    return merged, conflicts


