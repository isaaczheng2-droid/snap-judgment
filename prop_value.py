#!/usr/bin/env python3
"""
Compare the player projections this site already produces against FanDuel's posted lines.

THE ONE RULE THIS FILE EXISTS TO OBEY: nothing here ever touches a projection. The
projection is made first, by the existing model, from historical football data, with no
knowledge that a sportsbook exists. This module reads that finished number and a posted
line and reports where they disagree. If it were ever allowed to nudge a projection toward a
line, the disagreement -- the only thing worth measuring -- would quietly disappear.

WHAT THE CONFIDENCE NUMBER IS, AND IS NOT

The site had no per-player confidence score. There was no "84% confident" anywhere in the
payload, only projections and a factor breakdown. So rather than invent one, this computes
a real probability: for each stat, the distribution of (actual - projection) is measured
across 41,853 walk-forward props, with a spread that grows with the size of the projection.
A projection of 58.4 against a line of 44.5 becomes P(he clears 44.5), read off that
distribution. That is a quantity with units, and it is the only kind of number it is valid
to compare against a sportsbook's implied probability.

Measured raw, it came out overconfident by 2 to 5 points, worse the more confident it got.
A single shrink toward 0.5, fitted walk-forward, took the expected calibration error from
2.69% to 0.80% -- and that number, while true, was hiding the actual problem. Broken out by
how far the projection sits from the line, the bias FLIPS SIGN: too optimistic on extreme
overs (claimed 72.0%, won 65.5%), too pessimistic on extreme unders (claimed 81.6%, won
90.2%). One shrink moves both tails the same way, so it could never fix that; the average
looked good because the two errors cancelled. The calibration is now fitted per side, in
log-odds space, walk-forward. Worst-band error 8.5% -> 3.7%. It matters because the
recommendation filter ONLY ever fires in those tails -- nothing near the line clears 65%.

WHAT THE BACKTEST SUPPORTS, STATED PLAINLY

Against a line set at each player's own rolling median, over 2019-2025:
    props at 65%+ confidence won 70.7%      (n=11,606, CI 69.8-71.5%)
    everything else won 53.7%
    break-even at -110 is 52.4%
Both sides work -- overs 66.2%, unders 72.5% -- so it is not purely the skew.

AND THE CAVEAT THAT TRAVELS WITH IT: betting the under blindly on every one of those same
lines returns 54.9%, already past break-even. A line a constant strategy beats is a soft
line, and a real book does not post one. So none of the above demonstrates an edge over
FanDuel. It demonstrates that the projections carry information relative to a naive
baseline. Whether that survives contact with a real market is untested, and the page says
so.
"""
import json
import os

import numpy as np

MODEL_PATH = "prop_model.json"

# Per the spec. Volume-based props are steadier than event-based ones, so the same
# percentage disagreement means much less on a touchdown line than on a receptions line.
VOLATILITY = {
    "receptions": "low",
    "passing_yards": "medium", "rushing_yards": "medium",
    "receiving_yards": "medium", "rb_receiving_yards": "medium",
    "qb_rushing_yards": "medium",
    "passing_tds": "high", "rushing_tds": "high", "receiving_tds": "high",
}
# Edge (as a fraction of the line) that earns full marks on the edge component. A volatile
# prop has to disagree by much more before the disagreement means anything.
EDGE_FULL = {"low": 0.16, "medium": 0.30, "high": 0.60}
VOL_POINTS = {"low": 1.0, "medium": 0.65, "high": 0.30}

# Stats whose model loses to a naive rolling average of the player's own recent games.
# The site's own audit publishes this. A projection that cannot beat "what he usually does"
# has no business being ranked against a sportsbook, so these are shown and never
# recommended -- the reason is printed on the page rather than hidden by omission.
NOT_RECOMMENDABLE = {"passing_tds", "rushing_tds", "receiving_tds"}

PRETTY = {
    "passing_yards": "Passing Yards", "passing_tds": "Passing TDs",
    "qb_rushing_yards": "Rushing Yards", "rushing_yards": "Rushing Yards",
    "rushing_tds": "Rushing TDs", "rb_receiving_yards": "Receiving Yards",
    "receiving_yards": "Receiving Yards", "receptions": "Receptions",
    "receiving_tds": "Receiving TDs",
}
CATEGORY = {
    "passing_yards": "passing", "passing_tds": "passing",
    "qb_rushing_yards": "rushing", "rushing_yards": "rushing", "rushing_tds": "touchdowns",
    "rb_receiving_yards": "receiving", "receiving_yards": "receiving",
    "receptions": "receiving", "receiving_tds": "touchdowns",
}

# The spec's filters.
ODDS_MIN, ODDS_MAX = -200, 100
MIN_CONFIDENCE = 0.65
MIN_SCORE = 65
MIN_EDGE_PCT = 0.05          # "meaningfully differs from the line"

TIERS = [(85, "Elite Model Edge", "elite"), (75, "Strong Model Edge", "strong"),
         (65, "Moderate Model Edge", "moderate")]


# ------------------------------------------------------------------------------- odds
def implied_probability(american):
    """The spec's formula. Note this INCLUDES the bookmaker's margin."""
    o = float(american)
    return abs(o) / (abs(o) + 100) if o < 0 else 100 / (o + 100)


def fair_probability(price_side, price_other):
    """
    The same thing with the margin divided out, using both sides of the market.

    Raw implied probabilities sum to more than 1 -- that surplus is the bookmaker's margin,
    and it is how the book gets paid. Comparing the model against the RAW number credits the
    model with an edge that is really just vig, on every single prop, which would inflate
    every score on the page. Where both prices are posted the margin can be removed exactly.
    Where only one is, the raw number is returned and the page says the margin is still in it.
    """
    if price_other is None:
        return implied_probability(price_side), None
    a, b = implied_probability(price_side), implied_probability(price_other)
    total = a + b
    if total <= 0:
        return a, None
    return a / total, total - 1.0


# ------------------------------------------------------------------- probability model
class PropModel:
    """Turns (stat, projection, line) into a calibrated P(the side we take wins)."""

    # Calibration is PER SIDE, and that is not a detail. Measured against the backtest, the
    # raw probability is too optimistic on overs and too pessimistic on unders -- the bias
    # flips sign between the tails. A single shrink toward 0.5 moves both tails the same
    # way by construction, so it cannot fix one without worsening the other; it only looked
    # fine because the two errors cancelled in the average. Broken out, the shipped shrink
    # claimed 72.0% and delivered 65.5% on the extreme overs, which is exactly the region
    # the recommendation filter fires in. Two parameters per side in log-odds space cut the
    # worst band error from 8.5% to 3.7%. See test_prop_tail.py.
    DEFAULT_CAL = {"over": {"a": 0.60, "b": -0.05}, "under": {"a": 1.00, "b": -0.10}}

    def __init__(self, path=None):
        self.ok = False
        self.cal = dict(self.DEFAULT_CAL)
        self.stats = {}
        for p in [path, os.path.join("data", MODEL_PATH),
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_PATH)]:
            if not p:
                continue
            try:
                with open(p) as f:
                    m = json.load(f)
                if isinstance(m, dict) and m.get("stats"):
                    self.stats = m["stats"]
                    c = m.get("calibration")
                    if isinstance(c, dict) and "over" in c and "under" in c:
                        self.cal = c
                    elif "shrink" in m:
                        # an older model file: honour its shrink rather than silently
                        # applying a calibration it was never fitted with
                        k = float(m["shrink"])
                        self.cal = {"over": {"a": k, "b": 0.0}, "under": {"a": k, "b": 0.0}}
                    self.ok = True
                    return
            except Exception:
                continue

    def _raw_over(self, stat, projection, line):
        m = self.stats.get(stat)
        if not m or projection is None or line is None:
            return None
        scale = max(m["a"] + m["b"] * float(projection), 1e-6) * np.sqrt(np.pi / 2)
        z = np.asarray(m["z"])
        thr = (float(line) - float(projection)) / scale
        return 1.0 - float(np.searchsorted(z, thr)) / len(z)

    def p_side(self, stat, projection, line):
        """
        (take_over, calibrated P(that side wins)). None if the stat has no fitted spread.

        The side is chosen by the projection alone, before any price is seen, and the
        calibration applied is the one fitted for that side.
        """
        raw = self._raw_over(stat, projection, line)
        if raw is None:
            return None, None
        take_over = float(projection) > float(line)
        p = raw if take_over else 1.0 - raw
        c = self.cal["over" if take_over else "under"]
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        z = float(c["a"]) * np.log(p / (1 - p)) + float(c["b"])
        return take_over, float(np.clip(1 / (1 + np.exp(-z)), 0.01, 0.99))

    def p_over(self, stat, projection, line):
        """Calibrated P(actual > line), for callers that want the over side specifically."""
        take_over, p = self.p_side(stat, projection, line)
        if p is None:
            return None
        return p if take_over else 1.0 - p


RANGE_Q = {"q10": 0.10, "q25": 0.25, "q50": 0.50, "q75": 0.75, "q90": 0.90}


def range_table(model):
    """
    Per stat: the scale coefficients and a few quantiles of the standardised residual, so a
    page can draw "expected range" around a projection as
        projection + z_q * (a + b * projection) * sqrt(pi / 2).
    This is the distribution the prop probabilities already come from; publishing its
    quantiles adds no new model, it shows the one that exists.
    """
    out = {}
    for stat, m in (model.stats or {}).items():
        z = np.asarray(m["z"], dtype=float)
        if not len(z):
            continue
        n = len(z)
        out[stat] = {"a": round(float(m["a"]), 4), "b": round(float(m["b"]), 4),
                     "k": round(float(np.sqrt(np.pi / 2)), 4),
                     **{key: round(float(z[min(n - 1, int(round(q * (n - 1))))]), 3)
                        for key, q in RANGE_Q.items()}}
    return out


# --------------------------------------------------------------------- the comparison
def evaluate(stat, projection, line, price_over, price_under, model):
    """
    One prop, fully scored. Returns None only if it cannot be evaluated at all.

    The side is chosen by the projection alone -- over if the model projects above the line,
    under if below -- before any price is looked at.
    """
    if projection is None or line is None:
        return None
    take_over = float(projection) > float(line)
    price = price_over if take_over else price_under
    if price is None:
        return None
    other = price_under if take_over else price_over
    _, confidence_cal = model.p_side(stat, projection, line)

    if confidence_cal is None:
        # No fitted residual distribution for this stat, so there is no honest probability
        # to put next to the sportsbook's. The row is still shown -- the projection and the
        # line are real and worth seeing -- but with the confidence column blank and the
        # reason stated, rather than a made-up number or a silent disappearance.
        edge_ = (float(projection) - float(line)) if take_over else (float(line) - float(projection))
        return {
            "stat": stat, "prop": PRETTY.get(stat, stat),
            "category": CATEGORY.get(stat, "other"),
            "side": "Over" if take_over else "Under",
            "line": round(float(line), 1), "odds": int(price),
            "projection": round(float(projection), 1),
            "confidence": None, "edge": round(edge_, 1),
            "edge_pct": round(edge_ / max(abs(float(line)), 0.5), 4),
            "implied": round(implied_probability(price), 4),
            "fair": round(fair_probability(price, other)[0], 4),
            "margin": None, "prob_edge": None, "prob_edge_raw": None,
            "volatility": VOLATILITY.get(stat, "high"),
            "score": None, "tier": "No Strong Betting Edge", "tier_class": "none",
            "recommended": False,
            "blocked_by": ["this stat has no measured error distribution, so no honest "
                           "probability can be put against the price"],
        }

    confidence = confidence_cal
    edge = (float(projection) - float(line)) if take_over else (float(line) - float(projection))
    edge_pct = edge / max(abs(float(line)), 0.5)

    implied = implied_probability(price)
    fair, margin = fair_probability(price, other)
    vol = VOLATILITY.get(stat, "medium")

    # --- Bet Value Score, at the weights the spec sets ---
    conf_pts = np.clip((confidence - 0.50) / 0.35, 0, 1)             # 50% -> 85%
    edge_pts = np.clip(edge_pct / (2 * EDGE_FULL[vol]), 0, 1)        # volatility-scaled
    prob_pts = np.clip((confidence - fair) / 0.20, 0, 1)             # vs the de-vigged price
    vol_pts = VOL_POINTS[vol]
    score = 100 * (0.40 * conf_pts + 0.30 * edge_pts + 0.20 * prob_pts + 0.10 * vol_pts)
    score = float(round(score))

    label, klass = "No Strong Betting Edge", "none"
    for cut, nm, cl in TIERS:
        if score >= cut:
            label, klass = nm, cl
            break

    # --- the filters, each recorded so the page can say WHICH one blocked it ---
    blocks = []
    if not (ODDS_MIN <= price <= ODDS_MAX):
        blocks.append(f"odds {price:+.0f} outside {ODDS_MIN:+.0f} to {ODDS_MAX:+.0f}")
    if confidence < MIN_CONFIDENCE:
        blocks.append(f"confidence {confidence:.0%} below {MIN_CONFIDENCE:.0%}")
    if edge_pct < MIN_EDGE_PCT:
        blocks.append(f"projection within {MIN_EDGE_PCT:.0%} of the line")
    if score < MIN_SCORE:
        blocks.append(f"bet value {score:.0f} below {MIN_SCORE}")
    if stat in NOT_RECOMMENDABLE:
        blocks.append("this stat's model loses to a rolling average of the player's own games")

    return {
        "stat": stat, "prop": PRETTY.get(stat, stat), "category": CATEGORY.get(stat, "other"),
        "side": "Over" if take_over else "Under",
        "line": round(float(line), 1),
        "odds": int(price),
        "projection": round(float(projection), 1),
        "confidence": round(confidence, 4),
        "edge": round(edge, 1),
        "edge_pct": round(edge_pct, 4),
        "implied": round(implied, 4),
        "fair": round(fair, 4),
        "margin": None if margin is None else round(margin, 4),
        "prob_edge": round(confidence - fair, 4),
        "prob_edge_raw": round(confidence - implied, 4),
        "volatility": vol,
        "score": score,
        "tier": label, "tier_class": klass,
        "recommended": not blocks,
        "blocked_by": blocks,
    }


def reason(p, player, opponent):
    """One honest sentence. No promises, and no adjectives the numbers do not earn."""
    st = p["prop"].lower()
    side = "above" if p["side"] == "Over" else "below"
    # match how the page prints a line: 4.5 stays 4.5, 308.0 shows as 308
    num = lambda v: f"{v:g}"
    bits = [f"The model projects {player} at {num(p['projection'])} {st}, "
            f"{num(abs(p['edge']))} {side} FanDuel's {num(p['line'])}."]
    if p["confidence"] >= 0.75:
        bits.append(f" On the historical spread of its {st} misses that clears the line "
                    f"{p['confidence']:.0%} of the time")
    else:
        bits.append(f" That works out to {p['confidence']:.0%} against the line")
    if p["prob_edge"] > 0:
        bits.append(f", against {p['fair']:.0%} priced in once the margin is removed.")
    else:
        bits.append(f", which is no better than the {p['fair']:.0%} already priced in.")
    if p["volatility"] == "high":
        bits.append(" This is a volatile stat, so the gap has to be larger to mean as much.")
    return "".join(bits)


# ------------------------------------------------------------------------ assembly
def _norm(name):
    """Match on names the way two different feeds spell them."""
    import re
    n = str(name or "").lower().replace(".", "").replace("'", "").replace("-", " ")
    n = re.sub(r"[^a-z ]", "", n).strip()
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)$", "", n).strip()
    return re.sub(r"\s+", " ", n)


def build(players, odds, model, log=print):
    """
    Match this week's projections to FanDuel's posted lines and score every overlap.

    Reads `players` and never writes to it. The projection that goes into the comparison is
    the identical float already published on the player card -- if those two numbers ever
    disagreed, the feature would be lying about its own inputs.
    """
    import odds_api

    if not odds or not odds.get("props"):
        return {}, None
    by_name = {}
    for p in players:
        by_name.setdefault(_norm(p.get("player_display_name")), p)

    rows, unmatched = [], 0
    for slot in odds["props"].values():
        pl = by_name.get(_norm(slot.get("player")))
        if pl is None:
            unmatched += 1
            continue
        stat = odds_api.stat_for(slot.get("market"), pl.get("position"))
        if not stat:
            continue
        proj = pl.get(stat)
        if proj is None:
            continue
        ev = evaluate(stat, proj, slot["line"], slot.get("over"), slot.get("under"), model)
        if ev is None:
            continue
        ev.update({
            "player": pl.get("player_display_name"),
            "player_key": pl.get("player_key"),
            "team": pl.get("team"), "opponent": pl.get("opponent_team"),
            "position": pl.get("position"), "headshot": pl.get("headshot"),
            "is_home": pl.get("is_home"),
        })
        ev["reason"] = reason(ev, ev["player"], ev["opponent"])
        rows.append(ev)

    rows.sort(key=lambda r: (-(r["score"] or -1), -(r["confidence"] or 0)))
    by_game = {}
    for r in rows:
        home = r["team"] if r["is_home"] else r["opponent"]
        away = r["opponent"] if r["is_home"] else r["team"]
        by_game.setdefault(f"{away}@{home}", []).append(r)

    n_rec = sum(1 for r in rows if r["recommended"])
    log(f"  props: {len(rows)} FanDuel lines matched to a projection, "
        f"{n_rec} clear every filter ({unmatched} players not in our projections)")
    meta = {"book": odds.get("book", "FanDuel"), "fetched": odds.get("fetched"),
            "matched": len(rows), "recommended": n_rec,
            "source": odds.get("source", "live"),
            "credits_remaining": odds.get("credits_remaining")}
    return by_game, meta
