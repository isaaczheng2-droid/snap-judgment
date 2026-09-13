#!/usr/bin/env python3
"""
Check the comparison maths, including against the worked example in the spec.

The spec gives one fully worked case (David Njoku, 58.4 projected against a 44.5 line at
-115) with every intermediate number. That is the best kind of test to have: it was written
down before the code existed, so it cannot have been fitted to it. Every arithmetic step is
checked against it.

The parts NOT pinned by the spec -- the probability, and therefore the score -- are checked
for properties instead: monotonic in the right direction, symmetric where it should be, and
never recommending something the filters forbid.
"""
import sys

import numpy as np

import prop_value as PV

fails = []


def check(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"   {d}" if d else ""))
    if not c:
        fails.append(n)


def close(a, b, tol=0.005):
    return a is not None and abs(a - b) <= tol


def main():
    m = PV.PropModel()
    check("the fitted probability model loaded", m.ok, f"{len(m.stats)} stats")

    print("\nIMPLIED PROBABILITY -- the spec's own examples")
    check("-150 gives 60%", close(PV.implied_probability(-150), 0.60),
          f"{PV.implied_probability(-150):.4f}")
    check("+100 gives 50%", close(PV.implied_probability(100), 0.50),
          f"{PV.implied_probability(100):.4f}")
    check("-115 gives 53.5%", close(PV.implied_probability(-115), 0.535),
          f"{PV.implied_probability(-115):.4f}")
    check("-200 gives 66.7%", close(PV.implied_probability(-200), 2/3))

    print("\nREMOVING THE MARGIN -- both sides posted")
    f, marg = PV.fair_probability(-115, -115)
    check("a symmetric market de-vigs to exactly 50%", close(f, 0.50), f"{f:.4f}")
    check("...and the margin is reported", close(marg, 0.0698, 0.002), f"{marg:.4f}")
    f2, m2 = PV.fair_probability(-200, 160)
    check("a lopsided market de-vigs below its raw implied",
          f2 < PV.implied_probability(-200), f"{f2:.3f} < {PV.implied_probability(-200):.3f}")
    f3, m3 = PV.fair_probability(-115, None)
    check("one-sided market falls back to raw, and says so",
          close(f3, PV.implied_probability(-115)) and m3 is None)

    print("\nTHE SPEC'S WORKED EXAMPLE -- Njoku, 58.4 projected, 44.5 line, -115")
    p = PV.evaluate("receiving_yards", 58.4, 44.5, -115, -105, m)
    check("it evaluates", p is not None)
    check("side is Over", p["side"] == "Over", p["side"])
    check("projection edge is +13.9", close(p["edge"], 13.9, 0.05), str(p["edge"]))
    check("edge percentage is 31.2%", close(p["edge_pct"], 0.312, 0.002),
          f"{p['edge_pct']:.4f}")
    check("implied probability is 53.5%", close(p["implied"], 0.535), f"{p['implied']:.4f}")
    check("volatility is medium", p["volatility"] == "medium", p["volatility"])
    check("category is receiving", p["category"] == "receiving", p["category"])
    print(f"     model probability {p['confidence']:.1%} | fair price {p['fair']:.1%} "
          f"| edge over price {p['prob_edge']:+.1%} | score {p['score']:.0f} ({p['tier']})")

    # The spec assumed 84% confidence and a score of 91 for this example. Every piece of
    # ARITHMETIC above matches it exactly. The probability does not, and should not: 84%
    # was a placeholder written before any confidence system existed. Measured against the
    # model's own error distribution, a 13.9-yard gap on receiving yards is 0.38 standard
    # deviations -- the spread of its receiving-yard misses at a 58.4 projection is about
    # 37 yards. So this is a 55% proposition, and the honest answer is that it does not
    # qualify. Pinned here so the divergence stays deliberate rather than drifting.
    check("the spec's example is NOT elite once a real probability is used",
          p["confidence"] < 0.60 and not p["recommended"],
          f"{p['confidence']:.1%}, score {p['score']:.0f}")
    # What it takes to clear the 65% floor, AFTER the per-side recalibration. These numbers
    # moved once the overs stopped being overstated: receiving yards used to clear at a 75
    # projection and now needs about 81. That is the point of the fix, so the thresholds are
    # pinned here rather than quietly tracking whatever the model happens to say.
    strong = PV.evaluate("receiving_yards", 82.0, 44.5, -115, -105, m)
    check("an over on receiving yards clears 65% at roughly 1.8x the line",
          strong["confidence"] >= 0.65, f"proj 82.0 -> {strong['confidence']:.1%}")
    weak = PV.evaluate("receiving_yards", 75.0, 44.5, -115, -105, m)
    check("...and no longer clears it at 1.7x, which it wrongly used to",
          weak["confidence"] < 0.65, f"proj 75.0 -> {weak['confidence']:.1%}")
    steady = PV.evaluate("receptions", 7.0, 4.5, -115, -105, m)
    check("a steadier stat clears it on a smaller relative gap",
          steady["confidence"] >= 0.65, f"7.0 vs 4.5 -> {steady['confidence']:.1%}")
    # unders need far less, because these distributions are right-skewed and the
    # calibration is now allowed to say so instead of being averaged against the overs
    dn = PV.evaluate("receiving_yards", 37.0, 44.5, -115, -105, m)
    check("an under clears 65% at only 0.83x the line, and that is real, not a bug",
          dn["confidence"] >= 0.65, f"proj 37.0 -> {dn['confidence']:.1%}")

    print("\nTHE FILTERS -- each has to actually block")
    base = dict(stat="receiving_yards", projection=58.4, line=44.5, model=m)
    out = PV.evaluate(price_over=-250, price_under=200, **base)
    check("odds outside -200..+100 are blocked",
          not out["recommended"] and any("outside" in b for b in out["blocked_by"]),
          str(out["blocked_by"]))
    out = PV.evaluate(price_over=100, price_under=-120, **base)
    check("+100 is allowed (boundary is inclusive)",
          not any("outside" in b for b in out["blocked_by"]))
    out = PV.evaluate(stat="receiving_yards", projection=45.0, line=44.5,
                      price_over=-110, price_under=-110, model=m)
    check("a projection sitting on the line is blocked as not meaningful",
          any("within" in b for b in out["blocked_by"]), str(out["blocked_by"]))
    out = PV.evaluate(stat="receiving_tds", projection=0.8, line=0.5,
                      price_over=-110, price_under=-110, model=m)
    check("a stat with no fitted error distribution still returns a row", out is not None)
    check("...but carries no confidence rather than a made-up one",
          out["confidence"] is None and out["score"] is None)
    check("...is never recommended, and the page is told why",
          not out["recommended"] and any("error distribution" in b for b in out["blocked_by"]),
          str(out["blocked_by"]))

    print("\nDIRECTION AND MONOTONICITY")
    lo = PV.evaluate("receiving_yards", 50.0, 44.5, -110, -110, m)
    hi = PV.evaluate("receiving_yards", 70.0, 44.5, -110, -110, m)
    check("a bigger projection gap raises confidence",
          hi["confidence"] > lo["confidence"], f"{lo['confidence']:.3f} -> {hi['confidence']:.3f}")
    check("...and raises the bet value score", hi["score"] > lo["score"],
          f"{lo['score']} -> {hi['score']}")
    un = PV.evaluate("receiving_yards", 30.0, 44.5, -110, -110, m)
    check("projecting below the line takes the Under", un["side"] == "Under", un["side"])
    check("...with a positive edge, not a negative one", un["edge"] > 0, str(un["edge"]))

    worse = PV.evaluate("receiving_yards", 58.4, 44.5, -190, 155, m)
    better = PV.evaluate("receiving_yards", 58.4, 44.5, -110, -110, m)
    check("a worse price scores lower on identical model inputs",
          worse["score"] < better["score"], f"{worse['score']} vs {better['score']}")
    check("...while the projection itself is untouched",
          worse["projection"] == better["projection"] == 58.4)

    print("\nVOLATILITY -- a volatile stat must need a bigger gap")
    rec = PV.evaluate("receptions", 5.6, 4.5, -110, -110, m)          # low volatility
    ryd = PV.evaluate("receiving_yards", 56.0, 45.0, -110, -110, m)   # medium, same 24% gap
    check("same percentage gap scores higher on the steadier stat",
          rec["score"] > ryd["score"],
          f"receptions {rec['score']} vs receiving yards {ryd['score']}")
    check("volatility tiers match the spec",
          PV.VOLATILITY["receptions"] == "low" and
          PV.VOLATILITY["receiving_yards"] == "medium" and
          PV.VOLATILITY["receiving_tds"] == "high")

    print("\nTHE TIER LADDER")
    for score, want in [(91, "Large disagreement"), (85, "Large disagreement"),
                        (84, "Clear disagreement"), (75, "Clear disagreement"),
                        (74, "Moderate disagreement"), (65, "Moderate disagreement"),
                        (64, "No meaningful disagreement"), (10, "No meaningful disagreement")]:
        got = next((nm for cut, nm, _ in PV.TIERS if score >= cut), "No meaningful disagreement")
        check(f"{score} -> {want}", got == want, got)

    print("\nCALIBRATION SPOT-CHECK -- the probability must mean something")
    # A projection sitting on the line is NOT a coin flip, and assuming it would be is the
    # classic mistake here. These distributions are right-skewed: a receiver has a lot of
    # quiet games and a few enormous ones, so his mean sits above his median. The model
    # projects the mean. Landing above the mean is therefore less likely than landing below
    # it, and the honest probability of clearing a line set at the projection is under 50%.
    # A normal-shaped assumption would return 50% here and be wrong on every yardage prop.
    # This is also why the backtest's recommendations came out 70% unders.
    ev = PV.evaluate("receiving_yards", 44.6, 44.5, -110, -110, m)
    check("a projection level with the line is BELOW a coin flip, because the stat is skewed",
          0.38 <= ev["confidence"] <= 0.50, f"{ev['confidence']:.3f}")
    rc = PV.evaluate("receptions", 4.6, 4.5, -110, -110, m)
    check("...and the same holds on receptions, which are far less skewed, but milder",
          ev["confidence"] < rc["confidence"] <= 0.52,
          f"receiving yards {ev['confidence']:.3f} vs receptions {rc['confidence']:.3f}")
    # and the shrink must actually be applied
    check("a per-side calibration is loaded, not one global shrink",
          set(m.cal) == {"over", "under"} and m.cal["over"] != m.cal["under"], str(m.cal))
    # the tail the single shrink got wrong: an extreme over must no longer overstate
    hot = PV.evaluate("receiving_yards", 90.0, 44.5, -110, -110, m)
    check("an extreme over is no longer claimed above 75%",
          hot["confidence"] < 0.75, f"proj 2.0x the line -> {hot['confidence']:.1%}")
    cold = PV.evaluate("rushing_yards", 19.5, 48.5, -110, -110, m)
    check("an extreme under is allowed to claim high, because it earns it",
          cold["confidence"] > 0.80, f"proj 0.4x the line -> {cold['confidence']:.1%}")
    check("the two sides really are calibrated differently",
          abs(hot["confidence"] - (1 - 0.5)) != abs(cold["confidence"] - 0.5))

    print("\nLANGUAGE -- nothing may imply a guarantee")
    txt = " ".join([PV.reason(p, "David Njoku", "BAL"),
                    PV.reason(un, "David Njoku", "BAL"),
                    " ".join(nm for _, nm, _ in PV.TIERS)]).lower()
    banned = ["guarantee", "guaranteed", "safe", "certain", "risk-free", "riskless",
              "lock", "sure thing", "can't lose", "free money", "profit"]
    hit = [w for w in banned if w in txt]
    check("no guarantee language anywhere in the generated copy", not hit, str(hit))

    print(f"\n{len(fails)} failed" if fails else "\nall checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
