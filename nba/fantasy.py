"""
Fantasy basketball scoring: configurable points formats and category (8/9-cat) formats.

Nothing here reuses NFL scoring. Percentages are handled through attempts (a 2-for-10 night
hurts FG% more than an 0-for-1), so a player's FG%/FT% "value" is expressed as the impact
relative to a league-average line at his attempt volume, not as a bare percentage.

Points formats ship with the published defaults of the named providers as of the season this
file was written; they are editable on the page and stored with the projection so a changed
setting never silently regrades old results.
"""
import numpy as np

POINTS_FORMATS = {
    "espn_points": {"label": "ESPN points (default)", "pts": 1, "reb": 1, "ast": 1, "stl": 1, "blk": 1, "tov": -1,
                    "fg3m": 1, "fgm": 2, "fga": -1, "ftm": 1, "fta": -1},
    "yahoo_points": {"label": "Yahoo points (default)", "pts": 1, "reb": 1.2, "ast": 1.5, "stl": 3, "blk": 3, "tov": -1},
    "draftkings_dfs": {"label": "DraftKings DFS", "pts": 1, "reb": 1.25, "ast": 1.5, "stl": 2, "blk": 2, "tov": -0.5, "fg3m": 0.5,
                       "dd_bonus": 1.5, "td_bonus": 3},
    "fanduel_dfs": {"label": "FanDuel DFS", "pts": 1, "reb": 1.2, "ast": 1.5, "stl": 3, "blk": 3, "tov": -1},
    "sleeper_points": {"label": "Sleeper (default)", "pts": 1, "reb": 1, "ast": 1, "stl": 1, "blk": 1, "tov": -1, "fg3m": 0.5,
                       "_note": "Sleeper league defaults are commissioner-editable; verify in your league"},
}
CATEGORIES_9 = ["pts", "reb", "ast", "stl", "blk", "fg3m", "fg_pct", "ft_pct", "tov"]
CATEGORIES_8 = [c for c in CATEGORIES_9 if c != "tov"]
LEAGUE_FG, LEAGUE_FT = 0.470, 0.780     # reference lines for impact; refreshed from the data layer per season


def score_points(sim, fmt):
    """Vectorised points score over simulation arrays (or scalars). fmt is a POINTS_FORMATS entry
    or a dict of the same shape."""
    s = 0.0
    for k in ("pts", "reb", "ast", "stl", "blk", "tov", "fg3m", "fgm", "fga", "ftm", "fta"):
        if fmt.get(k):
            s = s + fmt[k] * sim[k]
    if fmt.get("dd_bonus") or fmt.get("td_bonus"):
        cats = (sim["pts"] >= 10).astype(int) + (sim["reb"] >= 10) + (sim["ast"] >= 10) + (sim["stl"] >= 10) + (sim["blk"] >= 10)
        if fmt.get("dd_bonus"):
            s = s + fmt["dd_bonus"] * (cats >= 2)
        if fmt.get("td_bonus"):
            s = s + fmt["td_bonus"] * (cats >= 3)
    return s


def pct_impact(made, att, league):
    """Attempt-weighted percentage impact: (made - league*att). Positive helps the category."""
    return made - league * att


def category_line(sim, league_fg=LEAGUE_FG, league_ft=LEAGUE_FT):
    """Per-category expectations from a simulation, with percentage impact via attempts."""
    out = {k: float(np.mean(sim[k])) for k in ("pts", "reb", "ast", "stl", "blk", "fg3m", "tov")}
    out["fg_pct"] = float(np.sum(sim["fgm"]) / max(np.sum(sim["fga"]), 1))
    out["ft_pct"] = float(np.sum(sim["ftm"]) / max(np.sum(sim["fta"]), 1))
    out["fg_impact"] = float(np.mean(pct_impact(sim["fgm"], sim["fga"], league_fg)))
    out["ft_impact"] = float(np.mean(pct_impact(sim["ftm"], sim["fta"], league_ft)))
    out["fga"], out["fta"] = float(np.mean(sim["fga"])), float(np.mean(sim["fta"]))
    return out


def z_scores(rows, cats=CATEGORIES_9):
    """Category z-scores across a player pool (list of category_line dicts). tov is inverted;
    percentages use the impact columns so volume counts."""
    keys = {"fg_pct": "fg_impact", "ft_pct": "ft_impact"}
    out = [dict(r) for r in rows]
    for c in cats:
        k = keys.get(c, c)
        v = np.array([r[k] for r in rows], dtype=float)
        sd = v.std() or 1.0
        z = (v - v.mean()) / sd
        if c == "tov":
            z = -z
        for r, zz in zip(out, z):
            r[f"z_{c}"] = float(zz)
    for r in out:
        r["z_total"] = float(sum(r[f"z_{c}"] for c in cats))
    return out


def week_schedule(games, team_id, start, end):
    """Games for a team in [start, end] (dates), for weekly-league streaming counts."""
    g = games[((games.home_id == team_id) | (games.away_id == team_id)) & (games.tipoff_utc >= start) & (games.tipoff_utc <= end)]
    return g.sort_values("tipoff_utc")
