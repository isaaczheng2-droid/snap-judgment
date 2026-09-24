"""
NFL Model Grades and the roster/depth-chart block for the page.

Model Grade (NFL) is a 0-100 percentile within a position group, built ONLY from data we
are licensed to use: nflverse weekly box scores, snap counts, PFR advanced weekly stats
(coverage and pass rush), and play-by-play penalties. It is OUR composite model grade.
It is not a PFF grade, not Madden, not ESPN, and is never labelled as any of those.
No proprietary rating is reproduced or approximated here.

v2 (model_grade_nfl_v2) replaces the v1 box-score-only Snap Grade. What changed and why:

  * v1 refused to grade OL, DB and specialists because the weekly box score carries
    nothing for them. That was honest but unhelpful: nflverse ALSO publishes PFR advanced
    weekly defense (individual targets/completions/yards allowed, passer rating allowed,
    pressures, hurries, QB hits, missed tackles) and snap counts, and the play-by-play
    carries per-player penalties. Those are individual, public, licensed numbers, so DB
    and pass-rush grades are now real individual grades.
  * Offensive linemen have NO public individual pass-block/run-block data (pass-block win
    rate is ESPN editorial; PFF is licensed). So the OL grade is a clearly labelled
    composite: two individual inputs (penalties per snap, snap volume/availability) and
    two LINE-LEVEL inputs (the team's pressure+sack rate allowed and rushing efficiency
    over the games the player actually played, weighted by his snap share). The card says
    exactly that, and OL confidence is capped at MEDIUM because half the signal is shared
    with four teammates. This follows the product rule: grade with what legally exists,
    label the methodology, never invent precision.
  * Every grade now carries grade_source, confidence (HIGH/MEDIUM/LOW from sample size),
    sample (games + snaps/targets), and a specific reason when no grade is shown.
    "Insufficient data: 9 targets of 12 needed" replaces "N/A".

Comparison window: each player's last WINDOW regular-season games across seasons (so week
2 of a new season still has a sample), minimum MIN_GAMES. Small samples are shrunk toward
the group mean by games played. The grade is the percentile of the weighted component
rating within the position group.

Position pools and inputs (weights in COMPONENTS below):
  QB    EPA per dropback, sack rate, INT rate, rushing EPA, pressure-to-sack rate (PFR)
  RB    rushing EPA/carry, yards/carry, receiving EPA/target, fumbles, usage share
  WR/TE receiving EPA/target, yards/target, target share, first downs/target, catch rate
  OL    team pass protection*, team run blocking*, penalties per snap, snap volume
        (* line-level, attributed over the games the player played; capped at MEDIUM)
  DL    pressures, sacks, hurries+hits, TFL, run tackles, missed-tackle rate  (PFR, indiv.)
  LB    tackles, missed-tackle rate, TFL, pressures, coverage when targeted   (PFR, indiv.)
  CB/S  completion% allowed, yards/target allowed, passer rating allowed, INT+PBU/target,
        missed-tackle rate                                                    (PFR, indiv.)
  K     field-goal points over expectation by distance, PAT%
  P     not graded: punting outcomes are not in our licensed data sources (says so).

QBR note: ESPN QBR is not available from our sources and is shown as such. NFL passer
rating is computed with the official formula and shown beside the QB grade under its own
name.
"""
import os
import subprocess

import numpy as np
import pandas as pd

METHOD_VERSION = "model_grade_nfl_v2"
GRADE_SOURCE = "model"
GRADE_SOURCE_LABEL = "Model Grade — our composite from licensed public data (nflverse box scores, snap counts, PFR advanced stats, play-by-play). Not a PFF, Madden or ESPN rating."
WINDOW = 12
MIN_GAMES = 4
MIN_GROUP = 8            # a percentile against 7 peers is noise; the row says so
REL = "https://github.com/nflverse/nflverse-data/releases/download"

# 0-100 grade bands, shown with the number, never instead of it
BANDS = [(90, "Elite"), (80, "Very good"), (70, "Good"), (60, "Average"), (50, "Below average"), (0, "Poor")]

# specific position -> grading pool
POOL = {"QB": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
        "T": "OL", "OT": "OL", "G": "OL", "C": "OL", "OL": "OL",
        "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "EDGE": "DL",
        "LB": "LB", "ILB": "LB", "MLB": "LB", "OLB": "LB",
        "CB": "CB", "S": "S", "FS": "S", "SS": "S", "SAF": "S",
        "K": "K", "PK": "K", "P": "P", "LS": "LS"}
POOL_LABEL = {"QB": "quarterbacks", "RB": "running backs", "WR": "wide receivers", "TE": "tight ends",
              "OL": "offensive linemen", "DL": "defensive linemen / edge", "LB": "linebackers",
              "CB": "cornerbacks", "S": "safeties", "K": "kickers"}

# what each pool's grade is built from (name -> weight); values are per the window
COMPONENTS = {
    "QB": {"epa_per_dropback": 0.45, "sack_rate": 0.15, "int_rate": 0.15, "rush_epa_pg": 0.10, "pressure_to_sack": 0.15},
    "RB": {"rush_epa_per_carry": 0.30, "ypc": 0.15, "rec_epa_per_target": 0.25, "fumbles_lost_pg": 0.10, "touches_pg": 0.20},
    "WR": {"rec_epa_per_target": 0.30, "yards_per_target": 0.20, "target_share": 0.20, "first_downs_per_target": 0.15, "catch_rate": 0.15},
    "TE": {"rec_epa_per_target": 0.30, "yards_per_target": 0.20, "target_share": 0.20, "first_downs_per_target": 0.15, "catch_rate": 0.15},
    "OL": {"team_pass_protection": 0.35, "team_run_blocking": 0.20, "penalties_per_100_snaps": 0.20, "snap_volume": 0.25},
    "DL": {"pressures_pg": 0.30, "sacks_pg": 0.20, "hurries_hits_pg": 0.15, "tfl_pg": 0.15, "run_tackles_pg": 0.10, "missed_tackle_rate": 0.10},
    "LB": {"tackles_pg": 0.25, "missed_tackle_rate": 0.15, "tfl_pg": 0.15, "pressures_pg": 0.15, "coverage_rating_allowed": 0.30},
    "CB": {"completion_pct_allowed": 0.25, "yards_per_target_allowed": 0.25, "passer_rating_allowed": 0.20, "ball_production_per_target": 0.15, "missed_tackle_rate": 0.15},
    "S":  {"completion_pct_allowed": 0.20, "yards_per_target_allowed": 0.20, "passer_rating_allowed": 0.15, "ball_production_per_target": 0.15, "missed_tackle_rate": 0.30},
    "K":  {"fg_points_over_expected": 0.70, "pat_pct": 0.30},
}
# components that read "lower is better"; percentiles are inverted for these
INVERT = {"sack_rate", "int_rate", "fumbles_lost_pg", "penalties_per_100_snaps", "missed_tackle_rate",
          "completion_pct_allowed", "yards_per_target_allowed", "passer_rating_allowed", "pressure_to_sack"}
# line-level (not individual) inputs, named so the page can say so
TEAM_LEVEL = {"team_pass_protection", "team_run_blocking"}

# volume floor per pool: (column, minimum, unit) — below it the grade is withheld with a reason
VOLUME = {"QB": ("att", 60, "pass attempts"), "RB": ("touches", 30, "touches"), "WR": ("tgt", 20, "targets"),
          "TE": ("tgt", 15, "targets"), "OL": ("off_snaps", 120, "offensive snaps"), "DL": ("def_snaps", 100, "defensive snaps"),
          "LB": ("def_snaps", 100, "defensive snaps"), "CB": ("cov_tgt", 12, "targets in coverage"),
          "S": ("def_snaps", 100, "defensive snaps"), "K": ("fga_all", 8, "kick attempts")}

# expected make rate by distance bucket, league 2019-2025 (for FG points over expectation)
FG_EXPECT = {"0_19": 0.995, "20_29": 0.96, "30_39": 0.91, "40_49": 0.80, "50_59": 0.64, "60_": 0.35}

NOT_GRADED = {"P": "punting outcomes (gross/net/inside-20) are not in our licensed data sources",
              "LS": "long-snapping has no public per-player outcome data"}


def passer_rating(comp, att, yds, td, ints):
    """Official NFL passer rating."""
    if not att:
        return None
    a = max(0, min(2.375, ((comp / att) - 0.3) * 5))
    b = max(0, min(2.375, ((yds / att) - 3) * 0.25))
    c = max(0, min(2.375, (td / att) * 20))
    d = max(0, min(2.375, 2.375 - (ints / att) * 25))
    return round((a + b + c + d) / 6 * 100, 1)


def band(g):
    if g is None:
        return None
    for lo, name in BANDS:
        if g >= lo:
            return name
    return "Poor"


def _pct(s):
    return s.rank(pct=True) * 100


def _shrink(v, n, k=4):
    w = n / (n + k)
    return w * v + (1 - w) * v.mean()


def confidence(pool, n_games, snaps):
    """HIGH/MEDIUM/LOW from how much football the grade is built on. OL is capped at
    MEDIUM because two of its four inputs are line-level, not individual."""
    s = 0 if snaps is None or (isinstance(snaps, float) and np.isnan(snaps)) else snaps
    if n_games >= 8 and s >= 350:
        c = "HIGH"
    elif n_games >= MIN_GAMES and s >= 120:
        c = "MEDIUM"
    else:
        c = "LOW"
    if pool == "OL" and c == "HIGH":
        c = "MEDIUM"
    return c


# --------------------------------------------------------------------------- extra inputs
def fetch_pfr_def(datadir, seasons, log=print):
    """PFR advanced weekly defense (individual coverage + pass rush). Small files; cached."""
    frames = []
    d = os.path.join(datadir, "pfr")
    os.makedirs(d, exist_ok=True)
    for y in seasons:
        p = os.path.join(d, f"advstats_week_def_{y}.parquet")
        if not os.path.exists(p) or os.path.getsize(p) < 1024:
            r = subprocess.run(["curl", "-sSL", "--max-time", "90", "-o", p,
                                f"{REL}/pfr_advstats/advstats_week_def_{y}.parquet"], capture_output=True)
            if r.returncode != 0 or not os.path.exists(p) or os.path.getsize(p) < 1024:
                if os.path.exists(p):
                    os.remove(p)
                continue
    for y in seasons:
        p = os.path.join(d, f"advstats_week_def_{y}.parquet")
        try:
            frames.append(pd.read_parquet(p))
        except Exception as e:
            log(f"  grades: skip {p}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _pfr_crosswalk(rost):
    return (rost.dropna(subset=["gsis_id", "pfr_id"])[["gsis_id", "pfr_id"]]
            .drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"].to_dict())


def _norm_name(n):
    import re, unicodedata
    n = unicodedata.normalize("NFKD", str(n or "")).encode("ascii", "ignore").decode()
    n = n.lower().replace(".", "").replace("'", "").replace("-", " ")
    n = re.sub(r"[^a-z ]", "", n).strip()
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)$", "", n).strip()
    return re.sub(r"\s+", " ", n)


def _name_team_map(rost):
    """(normalised name, team) -> gsis, unique matches only. The fallback for vendors whose
    id is missing from the roster crosswalk — every offensive lineman, as it turns out:
    rosters carry pfr_id for 0% of OL, which is the root cause of OL having no snap data
    anywhere in this codebase. Ambiguous names never resolve."""
    m, dup = {}, set()
    for r in rost.dropna(subset=["gsis_id"]).itertuples():
        k = (_norm_name(getattr(r, "full_name", None)), getattr(r, "team", None))
        if k in m and m[k] != r.gsis_id:
            dup.add(k)
        m[k] = r.gsis_id
    for k in dup:
        m.pop(k, None)
    return m


def _resolve_pfr(df, rost, name_col="player"):
    """Adds a 'gsis' column to a pfr-keyed dataframe: pfr_id first, unique name+team second."""
    cw = _pfr_crosswalk(rost)
    nt = _name_team_map(rost)
    out = df.copy()
    ids = out.pfr_player_id.map(cw)
    if name_col in out.columns:
        fb = [nt.get((_norm_name(n), t)) for n, t in zip(out[name_col], out.team)]
        ids = ids.fillna(pd.Series(fb, index=out.index))
    out["gsis"] = ids
    return out.dropna(subset=["gsis"])


def _before(df, season, week):
    return df[(df.season < season) | ((df.season == season) & (df.week < week))]


def _penalties(pbp, season, week):
    """Per-player penalty counts from play-by-play (regular season, before the slate)."""
    if pbp is None or not len(pbp):
        return pd.DataFrame(columns=["player_id", "season", "week", "pen"])
    d = pbp[(pbp.get("season_type", "REG") == "REG") & (pbp.penalty == 1)].dropna(subset=["penalty_player_id"])
    d = _before(d, season, week)
    g = d.groupby(["penalty_player_id", "season", "week"]).size().reset_index(name="pen")
    return g.rename(columns={"penalty_player_id": "player_id"})


# --------------------------------------------------------------------------- main
def compute(plyr, season, week, keep_ids=None, snap=None, rost=None, pfr_def=None,
            pfr_pass=None, pbp=None, team=None, depth=None, log=print):
    """
    plyr: weekly player box scores (all seasons). Grades use games strictly before
    (season, week). The other inputs are optional: with only plyr, the box-score pools
    (QB/RB/WR/TE and a reduced DL/LB) still grade; each added input turns on the pools
    that need it, and meta.sources says which were on.

    Returns (rows: {gsis_id: row}, meta).
    """
    p = plyr[(plyr.season_type == "REG")].pipe(_before, season, week).copy().sort_values(["season", "week"])
    have = {"box_scores": True, "snap_counts": snap is not None and len(snap) > 0,
            "pfr_def": pfr_def is not None and len(pfr_def) > 0,
            "pfr_pass": pfr_pass is not None and len(pfr_pass) > 0,
            "penalties_pbp": pbp is not None and len(pbp) > 0,
            "team_stats": team is not None and len(team) > 0}

    # ---- specific position per player: snap counts are most specific, then the box score
    pos = {}
    if have["snap_counts"] and rost is not None:
        s = _resolve_pfr(snap[snap.game_type == "REG"], rost).sort_values(["season", "week"])
        for r in s.drop_duplicates("gsis", keep="last").itertuples():
            pos[r.gsis] = r.position
    box_pos = p.sort_values(["season", "week"]).drop_duplicates("player_id", keep="last")
    for r in box_pos.itertuples():
        pos.setdefault(r.player_id, r.position)
    if rost is not None and len(rost):
        rlast = rost.sort_values("week" if "week" in rost.columns else "season").drop_duplicates("gsis_id", keep="last")
        for r in rlast.dropna(subset=["gsis_id"]).itertuples():
            pos.setdefault(r.gsis_id, r.position)

    # OL slot (LT/LG/C/RG/RT) from the latest depth chart, purely descriptive
    slot = {}
    if depth is not None and len(depth):
        d = depth[depth.dt == depth.dt.max()]
        for r in d.itertuples():
            if isinstance(r.gsis_id, str) and str(r.pos_abb) in ("LT", "LG", "C", "RG", "RT"):
                slot[r.gsis_id] = str(r.pos_abb)

    # ---- per-player windowed aggregates from the box score
    last = p.groupby("player_id").tail(WINDOW)
    agg = last.groupby("player_id").agg(
        name=("player_display_name", "last"), team=("team", "last"), n=("week", "size"),
        comp=("completions", "sum"), att=("attempts", "sum"), pyds=("passing_yards", "sum"),
        ptd=("passing_tds", "sum"), ints=("passing_interceptions", "sum"), sacks=("sacks_suffered", "sum"),
        pepa=("passing_epa", "sum"), repa=("rushing_epa", "sum"), car=("carries", "sum"),
        ryds=("rushing_yards", "sum"), rfum=("rushing_fumbles_lost", "sum"), tgt=("targets", "sum"),
        rec=("receptions", "sum"), rec_yds=("receiving_yards", "sum"), rec_epa=("receiving_epa", "sum"),
        rec_fd=("receiving_first_downs", "sum"), tshare=("target_share", "mean"),
        dsk=("def_sacks", "sum"), dqb=("def_qb_hits", "sum"), dtfl=("def_tackles_for_loss", "sum"),
        dtk=("def_tackles_solo", "sum"), dta=("def_tackle_assists", "sum"), dint=("def_interceptions", "sum"),
        dpd=("def_pass_defended", "sum"),
        fgm=("fg_made", "sum"), fga=("fg_att", "sum"), patm=("pat_made", "sum"), pata=("pat_att", "sum"),
        **{f"fgm_{b}": (f"fg_made_{b}", "sum") for b in FG_EXPECT},
        **{f"fgx_{b}": (f"fg_missed_{b}", "sum") for b in FG_EXPECT},
        season=("season", "last"), week=("week", "last"))
    agg["touches"] = agg.car.fillna(0) + agg.tgt.fillna(0)
    agg["pool"] = [POOL.get(str(pos.get(i, "")).upper(), None) for i in agg.index]
    # last-seen game keys per player, for joining window-scoped extras
    gk = last[["player_id", "season", "week"]].copy()

    # ---- snap totals over the window (offense/defense), via the pfr->gsis crosswalk
    snap_tot = pd.DataFrame()
    if have["snap_counts"] and rost is not None:
        s = _resolve_pfr(snap[snap.game_type == "REG"], rost).pipe(_before, season, week)
        s = s.sort_values(["season", "week"]).groupby("gsis").tail(WINDOW)
        snap_tot = s.groupby("gsis").agg(off_snaps=("offense_snaps", "sum"), def_snaps=("defense_snaps", "sum"),
                                         off_pct=("offense_pct", "mean"), n_snap_games=("week", "size"),
                                         team_s=("team", "last"))
        # players with a snap row but no/short box-score row (all OL) join the table here
        extra = snap_tot.index.difference(agg.index)
        add = pd.DataFrame(index=extra)
        nm = s.drop_duplicates("gsis", keep="last").set_index("gsis")
        add["name"] = nm["player"]
        add["team"] = nm["team"]
        add["n"] = s.groupby("gsis").size().reindex(extra)
        add["season"] = nm["season"]; add["week"] = nm["week"]
        add["pool"] = [POOL.get(str(pos.get(i, "")).upper(), None) for i in add.index]
        agg = pd.concat([agg, add])
    for c in ("off_snaps", "def_snaps", "off_pct"):
        agg[c] = snap_tot[c].reindex(agg.index) if len(snap_tot) else np.nan
    agg["snaps_any"] = agg[["off_snaps", "def_snaps"]].max(axis=1)

    # ---- PFR individual defense over the window
    if have["pfr_def"] and rost is not None:
        d = _resolve_pfr(pfr_def[pfr_def.game_type == "REG"], rost, "pfr_player_name").pipe(_before, season, week)
        d = d.sort_values(["season", "week"]).groupby("gsis").tail(WINDOW)
        dd = d.groupby("gsis").agg(cov_tgt=("def_targets", "sum"), cov_cmp=("def_completions_allowed", "sum"),
                                   cov_yds=("def_yards_allowed", "sum"), cov_rating=("def_passer_rating_allowed", "mean"),
                                   pfr_ints=("def_ints", "sum"), press=("def_pressures", "sum"),
                                   hurr=("def_times_hurried", "sum"), hits=("def_times_hitqb", "sum"),
                                   pfr_sacks=("def_sacks", "sum"), tkl=("def_tackles_combined", "sum"),
                                   mtkl=("def_missed_tackles", "sum"), n_pfr=("week", "size"))
        for c in dd.columns:
            agg[c] = dd[c].reindex(agg.index)
    else:
        for c in ("cov_tgt", "cov_cmp", "cov_yds", "cov_rating", "pfr_ints", "press", "hurr", "hits", "pfr_sacks", "tkl", "mtkl", "n_pfr"):
            agg[c] = np.nan

    # ---- QB pressure context (PFR pass): how often he was pressured, and pressure->sack
    if have["pfr_pass"] and rost is not None:
        q = _resolve_pfr(pfr_pass[pfr_pass.game_type == "REG"], rost, "pfr_player_name").pipe(_before, season, week)
        q = q.sort_values(["season", "week"]).groupby("gsis").tail(WINDOW)
        qq = q.groupby("gsis").agg(q_press=("times_pressured", "sum"), q_sacked=("times_sacked", "sum"))
        agg["q_press"] = qq["q_press"].reindex(agg.index)
        agg["q_sacked"] = qq["q_sacked"].reindex(agg.index)
        # team pass-protection: pressures + sacks allowed per dropback, by team-week
        tq = q.groupby(["team", "season", "week"]).agg(t_press=("times_pressured", "sum"), t_sk=("times_sacked", "sum")).reset_index()
    else:
        agg["q_press"] = np.nan; agg["q_sacked"] = np.nan
        tq = pd.DataFrame(columns=["team", "season", "week", "t_press", "t_sk"])

    # ---- team line context per team-week (for OL attribution)
    tw = pd.DataFrame(columns=["team", "season", "week", "prot", "runblk"])
    if have["team_stats"]:
        t = team[team.season_type == "REG"].copy().pipe(_before, season, week)
        t["db"] = t.attempts.fillna(0) + t.sacks_suffered.fillna(0)
        t = t[["season", "week", "team", "db", "sacks_suffered", "rushing_epa", "carries"]].merge(
            tq, on=["team", "season", "week"], how="left")
        # protection: pressures (when charted) + sacks, per dropback, inverted later
        t["prot"] = -((t.t_press.fillna(t.sacks_suffered * 2.2) + t.sacks_suffered) / t.db.replace(0, np.nan))
        t["runblk"] = t.rushing_epa / t.carries.replace(0, np.nan)
        tw = t[["team", "season", "week", "prot", "runblk"]]

    # OL attribution: mean of the team's line numbers over the games the player played
    ol_ctx = pd.DataFrame()
    if have["snap_counts"] and rost is not None and len(tw):
        s = _resolve_pfr(snap[snap.game_type == "REG"], rost).pipe(_before, season, week)
        s = s[[POOL.get(str(pos.get(g, "")).upper()) == "OL" for g in s.gsis]]
        s = s.sort_values(["season", "week"]).groupby("gsis").tail(WINDOW)
        s = s.merge(tw, on=["team", "season", "week"], how="left")
        w = s.offense_pct.fillna(0)
        s["w_prot"] = s.prot * w
        s["w_run"] = s.runblk * w
        ol_ctx = s.groupby("gsis").agg(prot=("w_prot", "sum"), runblk=("w_run", "sum"), wsum=("offense_pct", "sum"))
        ol_ctx["prot"] = ol_ctx.prot / ol_ctx.wsum.replace(0, np.nan)
        ol_ctx["runblk"] = ol_ctx.runblk / ol_ctx.wsum.replace(0, np.nan)
    agg["ol_prot"] = ol_ctx["prot"].reindex(agg.index) if len(ol_ctx) else np.nan
    agg["ol_runblk"] = ol_ctx["runblk"].reindex(agg.index) if len(ol_ctx) else np.nan

    # ---- penalties per 100 offensive snaps (pbp seasons only; component drops out without it)
    pen = _penalties(pbp, season, week)
    if len(pen):
        pen_tot = pen.groupby("player_id")["pen"].sum()
        agg["pen"] = pen_tot.reindex(agg.index).fillna(0)
        pen_seasons = sorted(pbp.season.unique().tolist())
    else:
        agg["pen"] = np.nan
        pen_seasons = []
    agg["fga_all"] = agg.fga.fillna(0) + agg.pata.fillna(0)

    # --------------------------------------------------------------- component builders
    def comps_for(pool, elig):
        n = elig.n
        if pool == "QB":
            db = (elig.att + elig.sacks).replace(0, np.nan)
            raw = {"epa_per_dropback": elig.pepa / db, "sack_rate": elig.sacks / db,
                   "int_rate": elig.ints / elig.att.replace(0, np.nan), "rush_epa_pg": elig.repa / n,
                   "pressure_to_sack": (elig.q_sacked / elig.q_press.replace(0, np.nan))}
        elif pool == "RB":
            raw = {"rush_epa_per_carry": elig.repa / elig.car.replace(0, np.nan), "ypc": elig.ryds / elig.car.replace(0, np.nan),
                   "rec_epa_per_target": (elig.rec_epa / elig.tgt.replace(0, np.nan)).fillna(0),
                   "fumbles_lost_pg": elig.rfum / n, "touches_pg": elig.touches / n}
        elif pool in ("WR", "TE"):
            t = elig.tgt.replace(0, np.nan)
            raw = {"rec_epa_per_target": elig.rec_epa / t, "yards_per_target": elig.rec_yds / t,
                   "target_share": elig.tshare.fillna(0), "first_downs_per_target": elig.rec_fd / t,
                   "catch_rate": elig.rec / t}
        elif pool == "OL":
            raw = {"team_pass_protection": elig.ol_prot, "team_run_blocking": elig.ol_runblk,
                   "penalties_per_100_snaps": (elig.pen / elig.off_snaps.replace(0, np.nan)) * 100,
                   "snap_volume": elig.off_snaps * elig.off_pct.fillna(0.5).clip(lower=0.1)}
        elif pool == "DL":
            raw = {"pressures_pg": elig.press / n, "sacks_pg": elig.pfr_sacks.fillna(elig.dsk) / n,
                   "hurries_hits_pg": (elig.hurr.fillna(0) + elig.hits.fillna(0)) / n, "tfl_pg": elig.dtfl / n,
                   "run_tackles_pg": elig.tkl.fillna(elig.dtk + 0.5 * elig.dta) / n,
                   "missed_tackle_rate": elig.mtkl / elig.tkl.replace(0, np.nan)}
        elif pool == "LB":
            raw = {"tackles_pg": elig.tkl.fillna(elig.dtk + 0.5 * elig.dta) / n,
                   "missed_tackle_rate": elig.mtkl / elig.tkl.replace(0, np.nan), "tfl_pg": elig.dtfl / n,
                   "pressures_pg": elig.press.fillna(elig.dqb) / n, "coverage_rating_allowed": elig.cov_rating}
        elif pool in ("CB", "S"):
            t = elig.cov_tgt.replace(0, np.nan)
            raw = {"completion_pct_allowed": elig.cov_cmp / t, "yards_per_target_allowed": elig.cov_yds / t,
                   "passer_rating_allowed": elig.cov_rating,
                   "ball_production_per_target": (elig.pfr_ints.fillna(elig.dint) + elig.dpd.fillna(0)) / t,
                   "missed_tackle_rate": elig.mtkl / elig.tkl.replace(0, np.nan)}
        elif pool == "K":
            exp_pts = sum(FG_EXPECT[b] * (elig[f"fgm_{b}"].fillna(0) + elig[f"fgx_{b}"].fillna(0)) for b in FG_EXPECT) * 3
            made_pts = elig.fgm.fillna(0) * 3
            raw = {"fg_points_over_expected": (made_pts - exp_pts) / elig.fga.replace(0, np.nan),
                   "pat_pct": elig.patm / elig.pata.replace(0, np.nan)}
        else:
            return None, None
        return raw, COMPONENTS[pool]

    # --------------------------------------------------------------- grade every pool
    out = {}
    meta = {"methodology_version": METHOD_VERSION, "grade_type": "percentile", "grade_source": GRADE_SOURCE,
            "grade_source_label": GRADE_SOURCE_LABEL, "window_games": WINDOW, "min_games": MIN_GAMES,
            "timeframe": f"last {WINDOW} regular-season games (across seasons)",
            "components": {k: v for k, v in COMPONENTS.items()}, "invert": sorted(INVERT),
            "team_level_inputs": sorted(TEAM_LEVEL),
            "team_level_note": "OL inputs marked line-level describe the five-man line over the games this player played (snap-weighted), because no public source charts individual linemen. OL confidence is therefore capped at MEDIUM.",
            "confidence_rule": "HIGH: 8+ games and 350+ snaps in the window. MEDIUM: 4+ games and 120+ snaps. LOW: below that. Confidence reflects sample size, not model certainty.",
            "volume_floors": {k: f"{v[1]} {v[2]}" for k, v in VOLUME.items()},
            "bands": [[lo, name] for lo, name in BANDS],
            "not_graded": NOT_GRADED,
            "sources": {"box_scores": "nflverse weekly player stats",
                        "snap_counts": "nflverse snap counts" if have["snap_counts"] else None,
                        "pfr_def": "PFR advanced weekly defense (coverage, pressures, missed tackles)" if have["pfr_def"] else None,
                        "pfr_pass": "PFR advanced weekly passing (pressure context)" if have["pfr_pass"] else None,
                        "penalties": f"nflverse play-by-play penalties ({', '.join(map(str, pen_seasons))})" if pen_seasons else None,
                        "team_stats": "nflverse team weekly stats (OL line context)" if have["team_stats"] else None},
            "passer_rating_label": "NFL passer rating (official formula), same window",
            "qbr_label": "ESPN QBR: not available from our sources",
            "components_format": "[percentile within group, raw value]"}

    for pool, grp in agg.groupby("pool"):
        if pool in NOT_GRADED or pool not in COMPONENTS:
            for pid, r in grp.iterrows():
                out[pid] = _row(pid, r, pool, None, {}, f"Not graded: {NOT_GRADED.get(pool, 'no grade definition for this position')}", 0, slot)
            continue
        vol_col, vol_min, vol_unit = VOLUME[pool]
        vol = grp[vol_col].fillna(0)
        elig = grp[(grp.n >= MIN_GAMES) & (vol >= vol_min)].copy()
        # a pool that needs PFR/snap data and has none says which source is missing
        missing_src = None
        if pool in ("CB", "S", "DL", "LB") and not have["pfr_def"]:
            missing_src = "PFR advanced defense data was not available this run"
        if pool == "OL" and not (have["snap_counts"] and have["team_stats"]):
            missing_src = "snap counts / team stats were not available this run"
        if missing_src or len(elig) < MIN_GROUP:
            why = missing_src or f"Not enough data: fewer than {MIN_GROUP} qualifying {POOL_LABEL.get(pool, pool)} in the pool"
            for pid, r in grp.iterrows():
                out[pid] = _row(pid, r, pool, None, {}, f"Insufficient data: {why}", len(elig), slot)
            continue
        raw, weights = comps_for(pool, elig)
        comps = pd.DataFrame(index=elig.index)
        used_weights = {}
        for k, w in weights.items():
            v = raw[k]
            if v.notna().sum() < MIN_GROUP:      # component without data drops out, openly
                continue
            filled = v.fillna(v.median())
            comps[k] = _pct(_shrink(-filled if k in INVERT else filled, elig.n))
            used_weights[k] = w
        if not used_weights:
            for pid, r in grp.iterrows():
                out[pid] = _row(pid, r, pool, None, {}, "Insufficient data: no component had enough coverage", len(elig), slot)
            continue
        tot = sum(used_weights.values())
        rating = sum(w / tot * comps[k] for k, w in used_weights.items())
        grade = _pct(rating)
        dropped = sorted(set(weights) - set(used_weights))
        for pid, r in grp.iterrows():
            if pid in elig.index:
                c = {k: {"percentile": round(float(comps.loc[pid, k]), 1),
                         "value": None if pd.isna(raw[k].loc[pid]) else round(float(raw[k].loc[pid]), 3)}
                     for k in used_weights}
                row = _row(pid, r, pool, float(grade.loc[pid]), c, None, len(elig), slot, rating=float(rating.loc[pid]))
                if dropped:
                    row["components_dropped"] = dropped
                out[pid] = row
            else:
                v = float(vol.loc[pid])
                why = (f"Insufficient data: {int(r.n)} of {MIN_GAMES} qualifying games in the window"
                       if r.n < MIN_GAMES else
                       f"Insufficient data: {int(v)} {vol_unit} of {vol_min} needed")
                out[pid] = _row(pid, r, pool, None, {}, why, len(elig), slot)

    if keep_ids is not None:
        out = {k: v for k, v in out.items() if k in keep_ids}
    return out, meta


def _row(pid, r, pool, grade, comps, reason, n_group, slot, rating=None):
    """Compact row: shared strings (timeframe, labels, source description) live in meta."""
    n = 0 if pd.isna(r.get("n")) else int(r["n"])
    snaps = r.get("snaps_any")
    snaps = None if snaps is None or pd.isna(snaps) else int(snaps)
    row = {"name": r.get("name"), "pos": None, "group": pool, "team": r.get("team"),
           "grade": None if grade is None else round(grade),
           "band": band(None if grade is None else round(grade)),
           "confidence": None if grade is None else confidence(pool, n, snaps),
           "source": GRADE_SOURCE,
           "rating": None if rating is None else round(rating, 1),
           "reason": reason, "components": {k: [round(v["percentile"]), v["value"]] for k, v in comps.items()},
           "n_group": n_group, "n": n, "snaps": snaps,
           "through": None if pd.isna(r.get("season")) else f"{int(r['season'])} wk {int(r['week'])}"}
    if pid in slot:
        row["slot"] = slot[pid]
    if pool == "QB" and not pd.isna(r.get("att")) and r.get("att"):
        row["passer_rating"] = passer_rating(r["comp"], r["att"], r["pyds"], r["ptd"], r["ints"])
    return row


# ----------------------------------------------------------------------------- depth charts
HS_PREFIX = "https://static.www.nfl.com/image/upload/f_auto,q_auto/league/"
GROUP_LABEL = {"3WR 1TE": "offense (11 personnel base)", "Base 4-3 D": "defense (base 4-3)", "Base 3-4 D": "defense (base 3-4)", "Special Teams": "special teams"}


def depth_block(depth, rost, inj, cur, week, max_rank=3):
    """Per team: the latest depth chart (nflverse, ESPN-sourced), ranks <= max_rank per slot,
    with roster identity (jersey, headshot) and the current week's injury designation."""
    if depth is None or not len(depth):
        return {}
    d = depth.copy()
    d["dt"] = pd.to_datetime(d["dt"], utc=True, errors="coerce")
    out = {}
    r26 = rost[rost.season == cur] if rost is not None and len(rost) else pd.DataFrame()
    if len(r26):
        r26 = r26.sort_values("week").drop_duplicates("gsis_id", keep="last").set_index("gsis_id")
    wk = inj[(inj.season == cur) & (inj.week == week)] if inj is not None and len(inj) else pd.DataFrame()
    inj_map = {}
    for r in wk.itertuples():
        if pd.notna(r.gsis_id):
            inj_map[r.gsis_id] = {"status": r.report_status if pd.notna(r.report_status) else None,
                                  "injury": r.report_primary_injury if pd.notna(r.report_primary_injury) else (r.practice_primary_injury if pd.notna(r.practice_primary_injury) else None),
                                  "practice": r.practice_status if pd.notna(r.practice_status) else None}
    for team, td in d.groupby("team"):
        latest = td.dt.max()
        td = td[td.dt == latest]
        groups = {}
        for r in td.itertuples():
            if r.pos_rank > max_rank:
                continue
            gid = r.gsis_id if pd.notna(r.gsis_id) else None
            ro = r26.loc[gid] if (gid and gid in r26.index) else None
            row = {"pos": r.pos_abb, "slot": int(r.pos_slot), "rank": int(r.pos_rank), "name": r.player_name, "gsis_id": gid,
                   "jersey": (None if ro is None or pd.isna(ro.jersey_number) else int(ro.jersey_number))}
            if ro is not None and pd.notna(ro.headshot_url):
                # nfl.com headshots share one prefix; the page rebuilds the URL from the id
                row["hs"] = ro.headshot_url.replace(HS_PREFIX, "")
            if ro is not None and ro.status != "ACT":
                row["roster_status"] = ro.status
            if gid in inj_map:
                row["injury"] = inj_map[gid]
            groups.setdefault(r.pos_grp, []).append(row)
        out[team] = {"asof": latest.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(latest) else None, "groups": groups}
    out["_meta"] = {"source": "nflverse depth_charts (ESPN-sourced), ranks 1-3 per slot; headshots from nfl.com", "headshot_prefix": HS_PREFIX,
                    "group_labels": GROUP_LABEL,
                    "note": "ranks are the team's published depth chart, not a confirmed game-day lineup; formations drawn from it are illustrative"}
    return out
