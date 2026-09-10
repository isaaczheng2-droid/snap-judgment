#!/usr/bin/env python3
"""
A plain Elo rating, and the reason it is here.

The project spent its life adding features — EPA splits, quarterback form, injuries,
coaching scheme — and never checked them against the dumbest possible rating system. When
it finally did, Elo won: 63.3% straight up against the feature model's 61.9%, with better
Brier and log loss.

That is not the humiliation it first looks like, and reading it as "ratings beat features"
would be wrong. Elo frozen at the start of each season, knowing only prior years, scores
60.2% — WORSE than the feature model. All of its advantage comes from updating after every
game. What it is really measuring is that a rating system tracks in-season form better than
the exponentially-weighted EPA averages do, and it does that using information the EPA
ratings deliberately throw away: the actual result, including special teams, field position,
red-zone conversion and every close-game outcome that EPA per play smooths over.

The two disagree on 24% of games and Elo takes those 53-47, so they know different things.
Handing the model Elo's log-odds as one more feature is worth more than any feature added
before it:

    model alone   SU 61.94% -> 63.83%, Brier 0.2299 -> 0.2253, margin MAE 10.42 -> 10.28
    blended       SU 66.47% -> 67.17%, log loss 0.6146 -> 0.6127

Caveats that belong next to those numbers: against the spread it went the other way
(52.59% -> 51.55%), and on the blended picks McNemar returns p = 0.136, so the blended gain
is directionally consistent — better in five of seven seasons, tied in one — but unproven.

Constants are the standard 538-style ones and were NOT tuned on this data. Tuning them
against the backtest would make every number above a selected-on number.
"""
import numpy as np
import pandas as pd

K = 20.0          # how far one game can move a rating
HFA = 55.0        # home advantage, in rating points
SCALE = 400.0     # rating points for a 10:1 odds ratio
REVERT = 0.25     # regression toward 1500 between seasons
BASE = 1500.0


def expected(rh, ra):
    return 1.0 / (1.0 + 10 ** (-((rh + HFA) - ra) / SCALE))


def ratings(df):
    """
    One pass over every game in date order, oldest first.

    Point-in-time by construction rather than by careful slicing: a game's probability is
    recorded BEFORE its result is folded into the ratings, so no game can ever influence
    its own prediction. Unplayed games are predicted and then skipped.

    Returns game_id -> pre-game home win probability, plus the final rating table.
    """
    d = df.sort_values(["season", "week", "game_id"])
    r, out, last = {}, [], None

    for row in d.itertuples():
        if last is not None and row.season != last:
            for t in r:
                r[t] = BASE + (r[t] - BASE) * (1 - REVERT)
        last = row.season

        rh = r.setdefault(row.home_team, BASE)
        ra = r.setdefault(row.away_team, BASE)
        p = expected(rh, ra)
        out.append(p)

        if pd.isna(getattr(row, "home_win", np.nan)):
            continue
        # margin-of-victory multiplier: a blowout moves the rating more, damped when the
        # favourite was already expected to win big, which is what stops good teams from
        # running away with the scale
        mov = 0.0 if pd.isna(row.home_margin) else abs(row.home_margin)
        mult = np.log(mov + 1) * (2.2 / (((rh + HFA) - ra) * 0.001 + 2.2))
        delta = K * mult * (row.home_win - p)
        r[row.home_team] = rh + delta
        r[row.away_team] = ra - delta

    res = d[["game_id"]].copy()
    res["p_elo"] = out
    return res, r


def add_elo_cols(df):
    """Attach p_elo and the log-odds the model actually consumes."""
    res, _ = ratings(df)
    out = df.merge(res, on="game_id", how="left")
    q = np.clip(out.p_elo, 1e-6, 1 - 1e-6)
    # log-odds, not the probability: rating differences are linear on this scale, and a
    # tree splitting on 0.5-vs-0.55 probability is splitting on a compressed axis
    out["elo_logit"] = np.log(q / (1 - q))
    return out


ELO_FEATS = ["elo_logit"]
