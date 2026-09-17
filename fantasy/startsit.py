"""
Start/sit: two or more players side by side, and how much the evidence supports a choice.

The verdict strength comes from the backtest, not from the size of the number: for a
projected gap in a band, the share of pairs where the higher-projected player actually scored
more (fantasy_model.json, startsit_holdout). Under 60% is a toss-up and the page says so.
"""
from . import engine

BANDS = [(1, 3, "1-3 pts"), (3, 5, "3-5 pts"), (5, 99, "5+ pts")]
TOSS_UP = 0.60          # historical hit rate below which no lean is offered
LEAN = 0.70             # between TOSS_UP and LEAN: "slight lean"


def band_for(gap):
    for lo, hi, lab in BANDS:
        if lo <= gap < hi:
            return lab
    return None


def compare(players, model, settings_name="Full PPR"):
    """
    players: list of fantasy projection records (engine.build output). Returns the ordered
    comparison, the gap between the top two, the historical hit rate for that gap band, a
    verdict, and the factors that drive the difference between the top two.
    """
    ps = [p for p in players if p and p.get("exp_pts") is not None]
    if len(ps) < 2:
        return {"players": ps, "verdict": "Need two players with a projection.", "strength": "none"}
    ps = sorted(ps, key=lambda p: -p["exp_pts"])
    a, b = ps[0], ps[1]
    gap = round(a["exp_pts"] - b["exp_pts"], 1)
    band = band_for(gap)
    ss = (model or {}).get("startsit_holdout", {})
    rec = ss.get(band) if band else None
    hit = rec["higher_scored_more"] if rec else None
    weak = [p for p in (a, b) if p.get("availability", {}).get("status") in ("questionable", "doubtful", "out", "bye")]
    if a.get("availability", {}).get("status") in ("out", "bye"):
        verdict, strength = f"{a['name']} is not playing; {b['name']} by default.", "availability"
    elif gap < 1 or hit is None:
        verdict, strength = "Too close to call: the projections are within a point, which the backtest cannot separate.", "toss-up"
    elif hit <= TOSS_UP:
        verdict, strength = f"Toss-up leaning {a['name']}: gaps of {band} have gone the projection's way only {hit:.0%} of the time since 2019.", "toss-up"
    elif hit < LEAN:
        verdict, strength = f"Slight lean {a['name']}: gaps of {band} have gone the projection's way {hit:.0%} of the time.", "lean"
    else:
        verdict, strength = f"Start {a['name']}: gaps of {band} have gone the projection's way {hit:.0%} of the time (n={rec['n']:,}).", "start"
    if weak and strength in ("lean", "start"):
        verdict += " Availability is the bigger risk here; check the designation before kickoff."
    # what drives the gap: sum of the breakdown terms across the position's stats, in points
    drivers = []
    for term in ("form", "usage", "matchup", "venue", "absence"):
        da, db = (a.get("drivers") or {}).get(term, 0.0), (b.get("drivers") or {}).get(term, 0.0)
        d = round(da - db, 1)
        if abs(d) >= 0.3:
            drivers.append({"term": term, "favors": a["name"] if d > 0 else b["name"], "points": abs(d)})
    drivers.sort(key=lambda x: -x["points"])
    return {"players": ps, "top": a["player_key"], "gap": gap, "band": band,
            "history": rec, "verdict": verdict, "strength": strength, "drivers": drivers[:4],
            "settings": settings_name}


def flex_eligible(rec, settings):
    return rec.get("position") in tuple(settings.get("flex_positions", ("RB", "WR", "TE")))
