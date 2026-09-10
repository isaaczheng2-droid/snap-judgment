#!/usr/bin/env python3
"""
Team ratings that account for WHO a team played, not just how it did.

The shipped ratings (`run_pipeline.team_ratings`) are an exponentially-weighted average of
a team's own per-game EPA. Nothing in them knows that one team's three good weeks came
against the worst defenses in the league and another's came against the best. A schedule
is not random over eight games, so that gap is real and it points the same direction all
season for teams in a weak division.

Plain Elo, which has none of this project's features, beat the entire feature set. One of
the few things Elo does that the EPA ratings do not is adjust for opponent — it is built
into the update rule. That makes this the obvious thing to fix next.

The method is the standard one. For every team-game, the offense's EPA per play is modelled
as:

    epa_per_play  ~  offense_rating[team]  -  defense_rating[opponent]  +  home_field

which is a linear model with one column per team per side. Solving it by ridge regression
recovers both ratings at once, and the ridge penalty does the same job as the shrinkage in
the current code: with three games played, a team is pulled toward the league mean instead
of being trusted.

Point-in-time is enforced structurally. For each (season, week) cutoff the fit sees ONLY
games played strictly before it, so a rating used to predict week 9 cannot contain week 9.
Recent games are weighted more heavily, with an extra damp across the season boundary, so a
team is mostly its current self and slightly last year's.

Three fits per cutoff — overall, passing, rushing — which is about 570 small ridges over
the full history and costs a few seconds.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

# One rating pair per phase. The names match the shipped METRICS so the two are comparable.
PHASES = {
    "epa": ("g_off_epa_pp", "g_def_epa_pp_allowed"),
    "pass": ("g_off_pass_epa_pp", "g_def_pass_epa_pp_allowed"),
    "rush": ("g_off_rush_epa_pp", "g_def_rush_epa_pp_allowed"),
}
HALF_LIFE = 10.0     # games; a game 10 back counts half as much as the most recent one
SEASON_DAMP = 0.55   # extra multiplier per season crossed, on top of recency
ALPHA = 6.0          # ridge penalty — this is the shrinkage-toward-average knob
MIN_ROWS = 200       # below this there is nothing worth solving


def _design(rows, teams):
    """
    Sparse design matrix: [offense one-hot | defense one-hot | home] .

    Defense enters NEGATIVE, so a large defense_rating means "suppresses EPA", which reads
    the right way round and matches the sign convention of the shipped def ratings.
    """
    n, k = len(rows), len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    r = np.arange(n)
    off = rows.team.map(idx).to_numpy()
    dfn = rows.opponent_team.map(idx).to_numpy()

    data = np.concatenate([np.ones(n), -np.ones(n), rows.is_home.to_numpy(float)])
    ri = np.concatenate([r, r, r])
    ci = np.concatenate([off, k + dfn, np.full(n, 2 * k)])
    return sparse.csr_matrix((data, (ri, ci)), shape=(n, 2 * k + 1))


def _weights(rows, season, week):
    """Recency in games, with an extra penalty for every season boundary crossed."""
    age = (season - rows.season) * 18 + (week - rows.week)
    age = np.maximum(age.to_numpy(float), 0.0)
    w = 0.5 ** (age / HALF_LIFE)
    return w * (SEASON_DAMP ** np.maximum(season - rows.season.to_numpy(), 0))


def solve(long, cutoffs, teams):
    """
    long: one row per team-game with team, opponent_team, is_home, season, week and the
          six raw per-game quantities.
    Returns a frame of ratings, one row per (season, week, team).
    """
    out = []
    for season, week in cutoffs:
        prior = long[(long.season < season) | ((long.season == season) & (long.week < week))]
        if len(prior) < MIN_ROWS:
            continue
        w = _weights(prior, season, week)
        keep = w > 1e-4                      # ancient games contribute nothing but time
        prior, w = prior[keep], w[keep]
        if len(prior) < MIN_ROWS:
            continue
        X = _design(prior, teams)
        rec = {t: {"team": t, "season": season, "week": week} for t in teams}
        for ph, (ocol, _dcol) in PHASES.items():
            y = prior[ocol].to_numpy(float)
            ok = np.isfinite(y)
            if ok.sum() < MIN_ROWS:
                continue
            m = Ridge(alpha=ALPHA, fit_intercept=True, solver="sparse_cg", max_iter=2000)
            m.fit(X[ok], y[ok], sample_weight=w[ok])
            c = m.coef_
            k = len(teams)
            # centre each side so the ratings are readable as "vs a league-average opponent"
            o = c[:k] - c[:k].mean()
            d = c[k:2 * k] - c[k:2 * k].mean()
            for i, t in enumerate(teams):
                rec[t][f"adj_off_{ph}"] = float(o[i])
                rec[t][f"adj_def_{ph}"] = float(d[i])
        out.extend(rec.values())
    return pd.DataFrame(out)


def build_long(team, sched):
    """One row per team-game, with the per-play quantities the shipped ratings use."""
    tw = team[team.season_type == "REG"].copy()
    tw["off_plays"] = tw.attempts.fillna(0) + tw.sacks_suffered.fillna(0) + tw.carries.fillna(0)
    tw["pass_plays"] = tw.attempts.fillna(0) + tw.sacks_suffered.fillna(0)
    tw["rush_plays"] = tw.carries.fillna(0)
    tw["off_epa_total"] = tw.passing_epa.fillna(0) + tw.rushing_epa.fillna(0)
    tw["g_off_epa_pp"] = tw.off_epa_total / tw.off_plays.replace(0, np.nan)
    tw["g_off_pass_epa_pp"] = tw.passing_epa / tw.pass_plays.replace(0, np.nan)
    tw["g_off_rush_epa_pp"] = tw.rushing_epa / tw.rush_plays.replace(0, np.nan)

    h = sched[["game_id", "home_team"]].rename(columns={"home_team": "team"})
    h["is_home"] = 1
    long = tw[["season", "week", "game_id", "team", "opponent_team",
               "g_off_epa_pp", "g_off_pass_epa_pp", "g_off_rush_epa_pp"]].copy()
    long = long.merge(h, on=["game_id", "team"], how="left")
    long["is_home"] = long.is_home.fillna(0)
    long[["season", "week"]] = long[["season", "week"]].astype("int64")
    return long.dropna(subset=["team", "opponent_team"])


ADJ_FEATS = ["adj_off_diff", "adj_def_diff", "adj_net_edge_home",
             "adj_pass_edge_home", "adj_rush_edge_home"]


def add_adjusted_cols(df, adj):
    """Attach the five matchup features, mirroring the shipped EPA edges exactly."""
    if not len(adj):
        for f in ADJ_FEATS:
            df[f] = np.nan
        return df
    cols = [c for c in adj.columns if c.startswith("adj_")]
    for side in ["home", "away"]:
        a = adj.rename(columns={"team": f"{side}_team", **{c: f"{side}_{c}" for c in cols}})
        df = df.merge(a, on=["season", "week", f"{side}_team"], how="left")

    df["adj_off_diff"] = df.home_adj_off_epa - df.away_adj_off_epa
    df["adj_def_diff"] = df.home_adj_def_epa - df.away_adj_def_epa
    # an offense is only as good as the defense across from it; same shape as the shipped
    # net_epa_edge_home so the two can be compared directly
    df["adj_net_edge_home"] = ((df.home_adj_off_epa + df.away_adj_def_epa)
                               - (df.away_adj_off_epa + df.home_adj_def_epa))
    df["adj_pass_edge_home"] = ((df.home_adj_off_pass + df.away_adj_def_pass)
                                - (df.away_adj_off_pass + df.home_adj_def_pass))
    df["adj_rush_edge_home"] = ((df.home_adj_off_rush + df.away_adj_def_rush)
                                - (df.away_adj_off_rush + df.home_adj_def_rush))
    return df


def team_adjusted(team, sched, cur, target_week):
    """Ratings for every (season, week) that appears in the data, plus the upcoming week."""
    long = build_long(team, sched)
    teams = sorted(long.team.unique())
    cutoffs = sorted(set(map(tuple, long[["season", "week"]].to_numpy())))
    cutoffs = [(s, w) for s, w in cutoffs if s >= long.season.min() + 1]
    if (cur, target_week) not in cutoffs:
        cutoffs.append((cur, target_week))
    return solve(long, cutoffs, teams)
