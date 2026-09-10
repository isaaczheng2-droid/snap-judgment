#!/usr/bin/env python3
"""
Why the model says what it says.

Game reasons come from XGBoost's exact tree SHAP contributions, not from a hand-written
rule that guesses at the model's logic. Every factor shown is a real, signed, additive
piece of the log-odds the model actually produced; the shares add to 100% of the moved
log-odds. If the text says the passing matchup is doing the work, the model agrees.

Player reasons decompose a three-term ridge exactly: form, opponent, venue.

The prose deliberately stays hedged. A 62% pick is a 62% pick, and the wording should
not read like a lock — see the accuracy audit for why that matters.
"""
import numpy as np
import pandas as pd

# label, and how to describe the underlying numbers for one game
FEATURE_INFO = {
    # the model now sees opponent-adjusted versions of these five; the labels stay the same
    # because the reader does not care which estimator produced the number
    "adj_net_edge_home": "Overall efficiency",
    "adj_off_diff": "Offensive efficiency",
    "adj_def_diff": "Defensive efficiency",
    "adj_pass_edge_home": "Passing matchup",
    "adj_rush_edge_home": "Running matchup",
    "points_diff_rating": "Scoring margin",
    "div_game": "Division game",
    "rest_diff": "Rest",
    "qb_epa_diff": "Quarterback play",
    "qb_rush_diff": "Quarterback rushing",
    "qb_change_diff": "Quarterback change",
    "inj_off_diff": "Offensive injuries",
    "inj_def_diff": "Defensive injuries",
    "pass_rate_diff": "Pass-run balance",
    "adot_diff": "Downfield passing",
    "pace_diff": "Tempo",
    "press_edge_home": "Pass rush vs protection",
    "prot_edge_home": "Pass protection",
    "takeaway_diff": "Takeaways",
    "havoc_diff": "Disruption",
    "int_rate_diff": "Interceptions forced",
    # a "funnel" defence is softer against one phase than the other; the fit is whether the
    # offense across from it happens to lean the way that defence gives up
    "funnel_fit_home": "Scheme fit",
    "funnel_fit_away": "Scheme fit",
    # deliberately not called "Elo": nobody outside the hobby knows the word, and what it
    # measures in plain language is a season-long record of who has beaten whom
    "elo_logit": "Track record",
}


def _g(r, c, d=np.nan):
    v = r.get(c, d)
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def detail_for(feat, r, H, A):
    """A short factual gloss on the two teams' underlying numbers for this feature."""
    f = lambda c: _g(r, c)
    try:
        # "adjusted for opponent" is said out loud because it is the whole point of these
        # numbers and it is what separates them from a raw season average anyone can look up
        if feat == "adj_off_diff":
            return (f"adjusted for opponent, {H} offense {f('home_adj_off_epa'):+.3f} EPA/play, "
                    f"{A} {f('away_adj_off_epa'):+.3f}")
        if feat == "adj_def_diff":
            return (f"adjusted for opponent, {H} defense {f('home_adj_def_epa'):+.3f} EPA/play "
                    f"suppressed, {A} {f('away_adj_def_epa'):+.3f}")
        if feat == "adj_net_edge_home":
            v = f("adj_net_edge_home")
            return f"{abs(v):.3f} EPA/play net edge to {H if v > 0 else A}, adjusted for opponent"
        if feat == "adj_pass_edge_home":
            return (f"{H} passing game vs {A} pass defense nets {f('adj_pass_edge_home'):+.3f} "
                    f"EPA/play, adjusted for opponent")
        if feat == "adj_rush_edge_home":
            return (f"{H} run game vs {A} run defense nets {f('adj_rush_edge_home'):+.3f} "
                    f"EPA/play, adjusted for opponent")
        if feat == "points_diff_rating":
            return (f"{H} {f('home_rating_points_scored') - f('home_rating_points_allowed'):+.1f} pts/gm, "
                    f"{A} {f('away_rating_points_scored') - f('away_rating_points_allowed'):+.1f}")
        if feat == "div_game":
            return "division rivals — historically closer than the ratings suggest"
        if feat == "elo_logit":
            # said as a probability, because that is the only form of a rating anyone reads
            p = f("p_elo")
            fav, q = (H, p) if p >= 0.5 else (A, 1 - p)
            return (f"on results alone, going back seasons, {fav} wins this {q:.0%} of the time"
                    if abs(p - 0.5) > 0.02 else
                    "on results alone, going back seasons, these two are a coin flip")
        if feat == "rest_diff":
            return f"{H} {f('home_rest'):.0f} days rest, {A} {f('away_rest'):.0f}"
        if feat == "qb_epa_diff":
            return (f"{r.get('home_qb_disp') or H} {f('home_qb_epa'):+.1f} passing EPA/gm, "
                    f"{r.get('away_qb_disp') or A} {f('away_qb_epa'):+.1f}")
        if feat == "qb_rush_diff":
            return (f"{r.get('home_qb_disp') or H} {f('home_qb_rush'):.0f} rush yds/gm, "
                    f"{r.get('away_qb_disp') or A} {f('away_qb_rush'):.0f}")
        if feat == "qb_change_diff":
            ch = [t for t, c in [(H, f('home_chg')), (A, f('away_chg'))] if c == 1]
            return (", ".join(ch) + " starting a different quarterback than last week") if ch else "same starters as last week"
        if feat == "inj_off_diff":
            return (f"snap-weighted offensive absences: {H} {f('home_out_off'):.2f}, {A} {f('away_out_off'):.2f}")
        if feat == "inj_def_diff":
            return (f"snap-weighted defensive absences: {H} {f('home_out_def'):.2f}, {A} {f('away_out_def'):.2f}")
        if feat == "pass_rate_diff":
            return f"{H} passes on {f('home_sch_pass_rate'):.0%} of snaps, {A} {f('away_sch_pass_rate'):.0%}"
        if feat == "adot_diff":
            return f"{H} {f('home_sch_adot'):.1f} air yds/attempt, {A} {f('away_sch_adot'):.1f}"
        if feat == "pace_diff":
            return f"{H} {f('home_sch_pace'):.0f} plays/gm, {A} {f('away_sch_pace'):.0f}"
        if feat == "press_edge_home":
            return (f"{H} pressures {f('home_sch_pressure'):.0%} of dropbacks and is sacked on "
                    f"{f('home_sch_sack_rate'):.1%}; {A} {f('away_sch_pressure'):.0%} and {f('away_sch_sack_rate'):.1%}")
        if feat == "prot_edge_home":
            return f"{H} sacked on {f('home_sch_sack_rate'):.1%} of dropbacks, {A} {f('away_sch_sack_rate'):.1%}"
        if feat == "takeaway_diff":
            return f"{H} {f('home_sch_takeaway'):.1f} takeaways/gm, {A} {f('away_sch_takeaway'):.1f}"
        if feat == "havoc_diff":
            return (f"{H} disrupts {f('home_sch_havoc'):.1%} of plays (tackles for loss, passes "
                    f"defended, forced fumbles), {A} {f('away_sch_havoc'):.1%}")
        if feat == "int_rate_diff":
            return f"{H} intercepts {f('home_sch_int_rate'):.1%} of dropbacks, {A} {f('away_sch_int_rate'):.1%}"
        if feat in ("funnel_fit_home", "funnel_fit_away"):
            d, o = ((A, H) if feat == "funnel_fit_home" else (H, A))
            fn = f("away_sch_funnel") if feat == "funnel_fit_home" else f("home_sch_funnel")
            soft = "the run" if fn > 0 else "the pass"
            return f"{d} gives up more through {soft}, and {o} leans that way"
    except Exception:
        pass
    return ""


def game_reasons(up, contribs, feats, top_n=5):
    """
    contribs: (n_games, n_feats + 1) SHAP matrix from booster.predict(pred_contribs=True).
    Returns one dict per game: ordered factors plus a short paragraph.
    """
    out = []
    for i, (_, r) in enumerate(up.iterrows()):
        c = contribs[i, :len(feats)]
        H, A = r.home_team, r.away_team
        total = np.abs(c).sum()
        order = np.argsort(-np.abs(c))

        factors = []
        for j in order[:top_n]:
            if total <= 0 or abs(c[j]) / total < 0.02:
                continue
            fname = feats[j]
            factors.append({
                "feature": fname,
                "label": FEATURE_INFO.get(fname, fname),
                "favors": H if c[j] > 0 else A,
                "share": round(float(abs(c[j]) / total), 4),
                "lo": round(float(c[j]), 4),          # signed log-odds, home side
                "detail": detail_for(fname, r, H, A),
            })

        p = float(r.p_blend)
        fav = H if p >= 0.5 else A
        dog = A if p >= 0.5 else H
        fp = max(p, 1 - p)

        pro = [f for f in factors if f["favors"] == fav]
        con = [f for f in factors if f["favors"] == dog]

        # how emphatic the language is allowed to be, given how close the number is
        if fp >= 0.72:
            lead = f"The model makes {fav} a clear favorite here, at {fp:.0%}."
        elif fp >= 0.62:
            lead = f"The model leans {fav}, at {fp:.0%}."
        elif fp >= 0.55:
            lead = f"The model gives {fav} a modest edge, {fp:.0%}."
        else:
            lead = f"This one is close to a coin flip — {fav} at {fp:.0%}."

        parts = [lead]
        if pro:
            a = pro[0]
            s = f"The biggest factor is {a['label'].lower()}"
            if a["detail"]:
                s += f" ({a['detail']})"
            if len(pro) > 1:
                b = pro[1]
                s += f", with {b['label'].lower()} adding to it"
                if b["detail"]:
                    s += f" ({b['detail']})"
            parts.append(s + ".")
        if con:
            d = con[0]
            s = f"Cutting the other way: {d['label'].lower()}"
            if d["detail"]:
                s += f" ({d['detail']})"
            parts.append(s + f", which favors {dog}.")

        # where the model sits relative to the closing line
        if pd_notna(r.get("spread_line")) and pd_notna(r.get("margin_pred")):
            e = float(r.margin_pred) - float(r.spread_line)
            side = H if e > 0 else A
            if abs(e) >= 2.5:
                parts.append(f"That lands {abs(e):.1f} points off the closing line, on {side}'s side — "
                             f"a gap that size is usually the model missing something the market knows, "
                             f"not the other way round.")
            elif abs(e) >= 1.0:
                parts.append(f"That is {abs(e):.1f} points from the closing line, leaning {side}.")
            else:
                parts.append("The model and the betting market land in essentially the same place.")

        out.append({"summary": " ".join(parts), "factors": factors})
    return out


def pd_notna(v):
    try:
        return v is not None and not pd.isna(v)
    except (TypeError, ValueError):
        return v is not None


# --------------------------------------------------------------------------- players
STAT_WORD = {
    "passing_yards": "passing yards", "passing_tds": "passing touchdowns",
    "qb_rushing_yards": "rushing yards", "rushing_yards": "rushing yards",
    "rushing_tds": "rushing touchdowns", "rb_receiving_yards": "receiving yards",
    "receiving_yards": "receiving yards", "receptions": "catches",
    "receiving_tds": "receiving touchdowns",
}


def player_reason(val, coef, intercept, form, opp_def, is_home, form_mean, opp_mean,
                  opp_rank, n_teams, usage=None, usage_mean=None):
    """
    Exact decomposition of the ridge, reported as movements away from a league-average
    context so the pieces are readable and base + form + usage + matchup + venue adds back
    up to the projection.

    The model used to be three terms (form, opponent, venue) and was, measured honestly,
    WORSE than a rolling average of the player's own recent games on four of nine stats.
    The missing ingredient was volume: a projection built from last year's yards cannot know
    a receiver's role changed, but snap share and target share can. Adding them took the
    weighted error from 13.04 to 12.71 against a 12.98 baseline — from losing to that
    baseline to beating it.

    `usage` is the block of usage coefficients and values (snap share, target share, carry
    share) collapsed into ONE reported number, because three separate near-collinear shares
    are not four readable pieces, they are noise with labels on.

    Only the numbers travel. The sentence is assembled in the page from these fields,
    because shipping the prose for every player and every stat cost 150 KB of payload to say
    what the numbers already say.
    """
    b0, b1, b2 = float(coef[0]), float(coef[1]), float(coef[2])
    fo = round(b0 * (form - form_mean), 1)
    mu = round(b1 * (opp_def - opp_mean), 1)
    ve = round(b2 * (is_home - 0.5), 1)
    us = 0.0
    if usage is not None and usage_mean is not None and len(coef) > 3:
        us = round(float(sum(float(coef[3 + i]) * (usage[i] - usage_mean[i])
                             for i in range(len(usage)))), 1)
    # The panel shows these next to the projection and invites the reader to add them up, so
    # they have to actually add up. Rounding each independently leaves drift; the residual is
    # absorbed into the baseline, the least interesting of them.
    return {
        "base": round(round(val, 1) - fo - mu - ve - us, 1),
        "form": fo, "usage": us, "matchup": mu, "venue": ve,
        "opp_rank": int(opp_rank), "n_teams": int(n_teams),
    }


# --------------------------------------------------------------------------- scheme prose
def scheme_label(pk_pass, pk_adot, pk_pace):
    """
    Three league percentiles in the words a broadcast would use. Percentiles, not raw rates:
    early in a season every team's rating is shrunk hard toward the league mean, so 54% pass
    rate can still be the 9th percentile. The ordering survives the shrinkage; the raw number
    does not, which is why the site prints the percentile alongside it.
    """
    a = ("pass-heavy" if pk_pass >= 0.70 else "run-leaning" if pk_pass <= 0.30 else "balanced")
    b = ("vertical" if pk_adot >= 0.70 else "short and quick" if pk_adot <= 0.30 else None)
    c = ("up-tempo" if pk_pace >= 0.75 else "deliberate" if pk_pace <= 0.25 else None)
    return ", ".join(x for x in [a, b, c] if x)


def defense_label(pk_pressure, pk_havoc, pk_funnel):
    """
    The defensive counterpart. `funnel` percentile high = softer against the run than the
    pass, which is what a play-caller sees as "run on them"; low = the reverse.
    """
    # 0.70 / 0.30 rather than a rounder 0.75 / 0.25: with 32 teams the percentiles land on
    # multiples of 1/32, and 23/32 = 0.71875 is a top-quarter team that a 0.72 cutoff misses
    a = ("blitz-heavy" if pk_pressure >= 0.70 else "passive rush" if pk_pressure <= 0.30 else None)
    b = ("disruptive" if pk_havoc >= 0.70 else "bend-don't-break" if pk_havoc <= 0.30 else None)
    c = ("run funnel" if pk_funnel >= 0.70 else "pass funnel" if pk_funnel <= 0.30 else None)
    parts = [x for x in [a, b, c] if x]
    return ", ".join(parts) if parts else "balanced"


def style_clash(H, A, hm, am):
    """One sentence on how the two identities meet. Only speaks when a gap is real."""
    out = []
    if abs(hm["pk_pass_rate"] - am["pk_pass_rate"]) >= 0.45:
        p, r = (H, A) if hm["pk_pass_rate"] > am["pk_pass_rate"] else (A, H)
        out.append(f"{p} throws it far more often than {r} does")
    if hm["pk_pressure"] >= 0.72 and am["pk_sack_rate"] >= 0.65:
        out.append(f"{H}'s pass rush meets an {A} line that has been giving up sacks")
    if am["pk_pressure"] >= 0.72 and hm["pk_sack_rate"] >= 0.65:
        out.append(f"{A}'s pass rush meets an {H} line that has been giving up sacks")
    if abs(hm["pk_pace"] - am["pk_pace"]) >= 0.55:
        f_, s_ = (H, A) if hm["pk_pace"] > am["pk_pace"] else (A, H)
        out.append(f"{f_} plays notably faster than {s_}")
    # an offense's lean meeting the other defense's soft side, or its strong side
    for off, deff, on, dn in [(hm, am, H, A), (am, hm, A, H)]:
        if off["pk_pass_rate"] >= 0.70 and deff["pk_funnel"] <= 0.28:
            out.append(f"{on} throws a lot into a {dn} defense that has been tougher against "
                       f"the pass than the run")
        elif off["pk_pass_rate"] >= 0.70 and deff["pk_funnel"] >= 0.72:
            out.append(f"{on} throws a lot against a {dn} defense that has been softer "
                       f"against the pass than the run")
        elif off["pk_pass_rate"] <= 0.30 and deff["pk_funnel"] >= 0.72:
            out.append(f"{on} leans on the run into a {dn} defense that has been softer "
                       f"against the pass — the ground game meets its stronger side")
    return ("; ".join(out) + ".") if out else ""


def weather_note(indoor, temp, wind, roof, surface, pk_pass_h, pk_pass_a):
    """
    Conditions in words.

    temp/wind arrive as None for every game that has not kicked off yet, and that is not a
    timing quirk — the schedule feed RECORDS the weather a game was played in, it never
    forecasts it. Measured across 2022-2025: 97% of played outdoor games carry temp and
    wind, and 0% of unplayed ones ever do, in any season. So this is a post-game record,
    and there is no point waiting for it to fill in.

    A real forecast would need an external source keyed on stadium coordinates and kickoff
    time. That is not wired up, so rather than dress the gap up as "not posted yet" this
    says what is actually true. The model median-fills these internally and never sees the
    difference, which is fine, because weather tested as noise and is not a model input.
    """
    if indoor:
        return f"Indoors ({roof}), so conditions are not a factor."
    if temp is None and wind is None:
        s = "Conditions are recorded after kickoff, not forecast, so there is nothing to show yet"
        if surface:
            s += f". {str(surface).title()} surface"
        return s + "."
    bits = []
    if temp is not None and not np.isnan(temp):
        if temp <= 25:
            bits.append(f"{temp:.0f}°F — cold enough to matter for grip and kicking")
        elif temp <= 40:
            bits.append(f"{temp:.0f}°F and cold")
        elif temp >= 88:
            bits.append(f"{temp:.0f}°F — heat becomes a conditioning issue late")
        else:
            bits.append(f"{temp:.0f}°F")
    if wind is not None and not np.isnan(wind):
        if wind >= 18:
            bits.append(f"{wind:.0f} mph wind, which is the level where deep passing and "
                        f"field goals start to suffer measurably")
        elif wind >= 12:
            bits.append(f"{wind:.0f} mph wind, briskly enough to shorten the passing game")
        elif wind > 0:
            bits.append(f"{wind:.0f} mph wind")
    s = ", ".join(bits) if bits else "no forecast posted yet"
    if surface:
        s += f". {str(surface).title()} surface"
    if wind is not None and not np.isnan(wind) and wind >= 12 and max(pk_pass_h, pk_pass_a) >= 0.65:
        s += ". Worth noting one of these teams is pass-heavy, which is the profile wind hurts most"
    return s + "."
