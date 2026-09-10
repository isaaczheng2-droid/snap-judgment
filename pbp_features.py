#!/usr/bin/env python3
"""
Team quality measures that only exist at the play level.

Everything else in this project is built from nflverse's pre-aggregated weekly team file,
which carries EPA totals but not the shape of how they were earned. These five do not
exist in that file at any price:

  success rate      share of plays that gained enough to keep the down-and-distance ahead
                    of schedule. Two teams with identical EPA can get there by being
                    steadily fine or by being mostly bad with three explosions.
  explosive rate    the other half of that same distinction: 20+ yard passes, 10+ yard runs
  early-down EPA    downs 1 and 2 only, before the situation forces anyone's hand. This is
                    the closest thing to a team's neutral-script identity.
  third-down rate   conversions over attempts, the situational skill EPA/play averages away
  red-zone EPA      inside the 20, where the field shortens and efficiency stops meaning
                    what it means everywhere else

Each is computed for a team's own offense AND for what its defense allowed, because a
matchup needs both sides. Smoothed exactly like the EPA ratings so the two are comparable:
an exponentially-weighted average of that team's PRIOR games only, shrunk toward last
season and then the league mean, so Week 2 is not pure noise.

Cost: play-by-play is ~20 MB per season against ~1 MB for the aggregates, so only the
seventeen columns actually used are read.
"""
import os
import subprocess

import numpy as np
import pandas as pd

REL = "https://github.com/nflverse/nflverse-data/releases/download"
EW_SPAN, K = 6, 3.0        # identical to run_pipeline.team_ratings, deliberately

COLS = ["season", "week", "game_id", "posteam", "defteam", "down", "ydstogo",
        "yardline_100", "epa", "success", "yards_gained", "pass", "rush",
        "play_type", "special", "penalty", "third_down_converted", "third_down_failed"]

# what gets measured, and whether a higher number is good for the offense
PBP = ["succ", "expl", "early_epa", "third", "rz_epa"]


def _fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return True
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    r = subprocess.run(["curl", "-sSL", "--max-time", "300", "-o", path, url],
                       capture_output=True)
    ok = r.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 1024
    if not ok and os.path.exists(path):
        os.remove(path)
    return ok


def load_pbp(datadir, seasons, current=None):
    """Only the columns above, and the current season is always re-fetched."""
    frames = []
    for y in seasons:
        p = f"{datadir}/pbp/play_by_play_{y}.parquet"
        if current is not None and y == current and os.path.exists(p):
            os.remove(p)
        if _fetch(f"{REL}/pbp/play_by_play_{y}.parquet", p):
            try:
                frames.append(pd.read_parquet(p, columns=COLS))
            except Exception as e:
                print(f"  skip pbp {y}: {e}", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)


def per_game(pbp):
    """
    One row per (game, team) with the five offensive measures, plus the same five as
    allowed by that team's defense, recovered by relabelling the opponent's rows.
    """
    d = pbp[(pbp.play_type.isin(["pass", "run"])) & (pbp.posteam.notna())].copy()
    # penalty-only plays have no real down-and-distance outcome to score
    d = d[d.penalty.fillna(0) == 0]
    d = d[d.special.fillna(0) == 0]

    d["is_pass"] = d["pass"].fillna(0).astype(int)
    d["explosive"] = np.where(d.is_pass == 1, d.yards_gained >= 20, d.yards_gained >= 10).astype(int)
    d["early"] = d.down.isin([1, 2]).astype(int)
    d["rz"] = (d.yardline_100 <= 20).astype(int)

    g = d.groupby(["season", "week", "game_id", "posteam"], observed=True)
    off = g.agg(
        plays=("epa", "size"),
        succ=("success", "mean"),
        expl=("explosive", "mean"),
        third_c=("third_down_converted", "sum"),
        third_f=("third_down_failed", "sum"),
    ).reset_index()

    e = d[d.early == 1].groupby(["season", "week", "game_id", "posteam"], observed=True)
    off = off.merge(e["epa"].mean().rename("early_epa").reset_index(),
                    on=["season", "week", "game_id", "posteam"], how="left")
    z = d[d.rz == 1].groupby(["season", "week", "game_id", "posteam"], observed=True)
    off = off.merge(z["epa"].mean().rename("rz_epa").reset_index(),
                    on=["season", "week", "game_id", "posteam"], how="left")

    att = off.third_c + off.third_f
    off["third"] = np.where(att > 0, off.third_c / att.replace(0, np.nan), np.nan)
    off = off.rename(columns={"posteam": "team"}).drop(columns=["third_c", "third_f"])

    # a team's defensive row is its opponent's offensive row, relabelled. game_id pairs
    # them, so this needs no assumptions about who was home.
    opp = off.rename(columns={"team": "against", **{m: f"{m}_allowed" for m in PBP}})
    pair = off[["game_id", "team"]].merge(opp, on="game_id")
    pair = pair[pair.team != pair.against]
    dfn = pair[["season", "week", "game_id", "team"] + [f"{m}_allowed" for m in PBP]]

    out = off.merge(dfn, on=["season", "week", "game_id", "team"], how="left")
    return out


MEASURES = PBP + [f"{m}_allowed" for m in PBP]


def team_pbp(pbp, sched, cur, target_week):
    """EWMA + shrinkage, the same recipe team_ratings uses, then a row for the week ahead."""
    m = per_game(pbp)
    if not len(m):
        return pd.DataFrame(columns=["team", "season", "week"] + [f"pbp_{x}" for x in MEASURES]), {}

    gd = sched[["game_id", "gameday"]].drop_duplicates("game_id")
    m = m.merge(gd, on="game_id", how="left").sort_values(["team", "season", "gameday"])
    m["gp_prior"] = m.groupby(["team", "season"]).cumcount()

    lg = m[MEASURES].mean()
    prev = m.groupby(["team", "season"])[MEASURES].mean().reset_index()
    prev = prev.rename(columns={x: f"prev_{x}" for x in MEASURES})
    prev["season"] += 1

    for x in MEASURES:
        sh = m.groupby(["team", "season"])[x].shift(1)
        m[f"ewma_{x}"] = sh.groupby([m.team, m.season]).transform(
            lambda s: s.ewm(span=EW_SPAN, min_periods=1).mean())
    m = m.merge(prev, on=["team", "season"], how="left")
    for x in MEASURES:
        m[f"prev_{x}"] = m[f"prev_{x}"].fillna(lg[x])
        m[f"ewma_{x}"] = m[f"ewma_{x}"].fillna(m[f"prev_{x}"])
        m[f"prior_{x}"] = 0.5 * lg[x] + 0.5 * m[f"prev_{x}"]
    w = m.gp_prior / (m.gp_prior + K)
    for x in MEASURES:
        m[f"pbp_{x}"] = w * m[f"ewma_{x}"] + (1 - w) * m[f"prior_{x}"]

    out = m[["team", "season", "week", "game_id"] + [f"pbp_{x}" for x in MEASURES]].copy()

    # ---- the upcoming week has no plays yet ----
    played = m[m.season == cur]
    gp = played.groupby("team").size().to_dict()
    up = sched[(sched.season == cur) & (sched.week == target_week) & (sched.game_type == "REG")]
    teams = pd.unique(pd.concat([up.home_team, up.away_team]))
    # same guard as team_ratings: once one game of the week is played its teams already
    # have a real row, and a synthetic one on top duplicates them
    have = set(out[(out.season == cur) & (out.week == target_week)].team)
    teams = [t for t in teams if t not in have]
    prev_cur = prev[prev.season == cur].set_index("team")

    rows = []
    for t in teams:
        n = gp.get(t, 0)
        tp = played[played.team == t].sort_values("gameday")
        row = {"team": t, "season": cur, "week": target_week}
        for x in MEASURES:
            p = prev_cur[f"prev_{x}"].get(t, lg[x]) if t in prev_cur.index else lg[x]
            if pd.isna(p):
                p = lg[x]
            prior = 0.5 * lg[x] + 0.5 * p
            ew = tp[x].ewm(span=EW_SPAN, min_periods=1).mean().iloc[-1] if n else prior
            if pd.isna(ew):
                ew = prior
            ww = n / (n + K)
            row[f"pbp_{x}"] = ww * ew + (1 - ww) * prior
        g = up[(up.home_team == t) | (up.away_team == t)]
        row["game_id"] = g.game_id.values[0] if len(g) else None
        rows.append(row)

    full = pd.concat([out, pd.DataFrame(rows)], ignore_index=True) if rows else out
    return full, {x: float(lg[x]) for x in MEASURES}


# Five matchup features, built the same way as the EPA edges the model already uses: an
# offense is only as good as the defense it is facing, so each is (home offense vs away
# defense) minus (away offense vs home defense). Five, not ten, because 2,500 games will
# not support one feature per raw quantity.
PBP_FEATS = ["succ_edge_home", "expl_edge_home", "early_epa_edge_home",
             "third_edge_home", "rz_edge_home"]


def add_pbp_cols(df, tp):
    if not len(tp):
        for f in PBP_FEATS:
            df[f] = np.nan
        return df
    cols = [f"pbp_{x}" for x in MEASURES]
    for side in ["home", "away"]:
        r = tp[["game_id", "team"] + cols].drop_duplicates(["game_id", "team"], keep="last")
        r = r.rename(columns={"team": f"{side}_team", **{c: f"{side}_{c}" for c in cols}})
        df = df.merge(r, on=["game_id", f"{side}_team"], how="left")

    def edge(m):
        return ((df[f"home_pbp_{m}"] - df[f"away_pbp_{m}_allowed"])
                - (df[f"away_pbp_{m}"] - df[f"home_pbp_{m}_allowed"]))

    df["succ_edge_home"] = edge("succ")
    df["expl_edge_home"] = edge("expl")
    df["early_epa_edge_home"] = edge("early_epa")
    df["third_edge_home"] = edge("third")
    df["rz_edge_home"] = edge("rz_epa")
    return df
