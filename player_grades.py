"""
NFL Snap Grades and the roster/depth-chart block for the page.

Snap Grade (NFL) is a 0-100 DESCRIPTIVE percentile within a position group, built from the
public weekly box score (nflverse). It is not a prop probability, not a projection, and not a
PFF/Madden/ESPN rating. It is labelled "descriptive summary" until a chronological check
shows it predicts anything; that check has not been run for the NFL, so the label stays.

Comparison window: each player's last WINDOW regular-season games across seasons (so week 2
of a new season still has a sample), minimum MIN_GAMES. Position inputs:
  QB   EPA per dropback, sack rate, interception rate, rushing EPA per game
       (NFL passer rating is computed with the official formula and shown beside the grade
       under its own name; ESPN QBR is not available from our sources and is shown as such)
  RB   rushing EPA per carry, yards per carry, receiving EPA per target, fumbles lost
  WR/TE receiving EPA per target, yards per target, target share, first downs per target
  DL/LB sacks, QB hits, tackles for loss and tackles per game (box-score pass-rush and
       run-stop production only: no pressure or snap data)
  OL   not graded: no individual OL data in the box score; never inferred from team results
  DB   not graded: coverage outcomes are not in the box score; interceptions alone would
       reward volume of targets, not quality
  K/P  not graded here
Small samples are shrunk toward the group mean by games played.
"""
import numpy as np
import pandas as pd

METHOD_VERSION = "snap_grade_nfl_v1"
WINDOW = 12
MIN_GAMES = 4
NOT_GRADED = {"OL": "no individual offensive-line data in the box score; OL is never graded from team results",
              "DB": "coverage outcomes are not in the box score; not graded from interceptions alone",
              "SPEC": "kickers, punters and returners are not graded here"}


def passer_rating(comp, att, yds, td, ints):
    """Official NFL passer rating."""
    if not att:
        return None
    a = max(0, min(2.375, ((comp / att) - 0.3) * 5))
    b = max(0, min(2.375, ((yds / att) - 3) * 0.25))
    c = max(0, min(2.375, (td / att) * 20))
    d = max(0, min(2.375, 2.375 - (ints / att) * 25))
    return round((a + b + c + d) / 6 * 100, 1)


def _pct(s):
    return s.rank(pct=True) * 100


def _shrink(v, n, k=4):
    w = n / (n + k)
    return w * v + (1 - w) * v.mean()


def compute(plyr, season, week, keep_ids=None):
    """plyr: weekly player stats (all seasons). Grades use games strictly before (season, week)
    of the upcoming slate, i.e. everything already played. keep_ids limits the output to players
    the page can show (depth charts + projections); ungraded groups (OL/DB/SPEC) are described
    once in meta.not_graded rather than row by row."""
    p = plyr[(plyr.season_type == "REG") & ((plyr.season < season) | ((plyr.season == season) & (plyr.week < week)))].copy()
    p = p.sort_values(["season", "week"])
    p["pg"] = p.position_group.fillna(p.position)
    p.loc[p.position.isin(["FB"]), "pg"] = "RB"
    out, meta = {}, {"methodology_version": METHOD_VERSION, "window_games": WINDOW, "min_games": MIN_GAMES, "grade_type": "percentile",
                     "label": "descriptive summary (no predictive validation run for the NFL yet)", "not_graded": NOT_GRADED,
                     "timeframe": f"last {WINDOW} regular-season games (across seasons)", "components_format": "[percentile within group, raw value]",
                     "passer_rating_label": "NFL passer rating (official formula), same window", "qbr_label": "ESPN QBR: not available from our sources",
                     "components": {"QB": ["epa_per_dropback", "sack_rate", "int_rate", "rush_epa_pg"], "RB": ["rush_epa_per_carry", "ypc", "rec_epa_per_target", "fumbles_lost_pg"],
                                    "WR": ["rec_epa_per_target", "yards_per_target", "target_share", "first_downs_per_target"], "TE": ["rec_epa_per_target", "yards_per_target", "target_share", "first_downs_per_target"],
                                    "DL": ["sacks_pg", "qb_hits_pg", "tfl_pg", "tackles_pg"], "LB": ["sacks_pg", "qb_hits_pg", "tfl_pg", "tackles_pg"]}}
    last = p.groupby("player_id").tail(WINDOW)
    agg = last.groupby("player_id").agg(
        name=("player_display_name", "last"), pos=("position", "last"), pg=("pg", "last"), team=("team", "last"), n=("week", "size"),
        comp=("completions", "sum"), att=("attempts", "sum"), pyds=("passing_yards", "sum"), ptd=("passing_tds", "sum"), ints=("passing_interceptions", "sum"),
        sacks=("sacks_suffered", "sum"), pepa=("passing_epa", "sum"), repa=("rushing_epa", "sum"), car=("carries", "sum"), ryds=("rushing_yards", "sum"),
        rfum=("rushing_fumbles_lost", "sum"), tgt=("targets", "sum"), rec_yds=("receiving_yards", "sum"), rec_epa=("receiving_epa", "sum"),
        rec_fd=("receiving_first_downs", "sum"), tshare=("target_share", "mean"), dsk=("def_sacks", "sum"), dqb=("def_qb_hits", "sum"),
        dtfl=("def_tackles_for_loss", "sum"), dtk=("def_tackles_solo", "sum"), dta=("def_tackle_assists", "sum"), season=("season", "last"), week=("week", "last"))
    agg = agg[agg.n >= 1]
    for pg, grp in agg.groupby("pg"):
        grp = grp.copy()
        if pg in NOT_GRADED:
            continue
        comps = pd.DataFrame(index=grp.index)
        if pg == "QB":
            elig = grp[(grp.n >= MIN_GAMES) & (grp.att >= 60)]
            db = (elig.att + elig.sacks).replace(0, np.nan)
            comps = pd.DataFrame({"epa_per_dropback": _pct(_shrink(elig.pepa / db, elig.n)), "sack_rate": _pct(-_shrink(elig.sacks / db, elig.n)),
                                  "int_rate": _pct(-_shrink(elig.ints / elig.att.replace(0, np.nan), elig.n)), "rush_epa_pg": _pct(_shrink(elig.repa / elig.n, elig.n))})
            weights = {"epa_per_dropback": 0.55, "sack_rate": 0.15, "int_rate": 0.15, "rush_epa_pg": 0.15}
            raw = {"epa_per_dropback": elig.pepa / db, "sack_rate": elig.sacks / db, "int_rate": elig.ints / elig.att.replace(0, np.nan), "rush_epa_pg": elig.repa / elig.n}
        elif pg == "RB":
            elig = grp[(grp.n >= MIN_GAMES) & (grp.car + grp.tgt >= 30)]
            comps = pd.DataFrame({"rush_epa_per_carry": _pct(_shrink(elig.repa / elig.car.replace(0, np.nan), elig.n)), "ypc": _pct(_shrink(elig.ryds / elig.car.replace(0, np.nan), elig.n)),
                                  "rec_epa_per_target": _pct(_shrink((elig.rec_epa / elig.tgt.replace(0, np.nan)).fillna(0), elig.n)), "fumbles_lost_pg": _pct(-elig.rfum / elig.n)})
            weights = {"rush_epa_per_carry": 0.4, "ypc": 0.2, "rec_epa_per_target": 0.3, "fumbles_lost_pg": 0.1}
            raw = {"rush_epa_per_carry": elig.repa / elig.car.replace(0, np.nan), "ypc": elig.ryds / elig.car.replace(0, np.nan), "rec_epa_per_target": elig.rec_epa / elig.tgt.replace(0, np.nan), "fumbles_lost_pg": elig.rfum / elig.n}
        elif pg in ("WR", "TE"):
            elig = grp[(grp.n >= MIN_GAMES) & (grp.tgt >= 20)]
            t = elig.tgt.replace(0, np.nan)
            comps = pd.DataFrame({"rec_epa_per_target": _pct(_shrink(elig.rec_epa / t, elig.n)), "yards_per_target": _pct(_shrink(elig.rec_yds / t, elig.n)),
                                  "target_share": _pct(elig.tshare.fillna(0)), "first_downs_per_target": _pct(_shrink(elig.rec_fd / t, elig.n))})
            weights = {"rec_epa_per_target": 0.4, "yards_per_target": 0.2, "target_share": 0.25, "first_downs_per_target": 0.15}
            raw = {"rec_epa_per_target": elig.rec_epa / t, "yards_per_target": elig.rec_yds / t, "target_share": elig.tshare, "first_downs_per_target": elig.rec_fd / t}
        elif pg in ("DL", "LB"):
            elig = grp[grp.n >= MIN_GAMES]
            n = elig.n
            comps = pd.DataFrame({"sacks_pg": _pct(_shrink(elig.dsk / n, n)), "qb_hits_pg": _pct(_shrink(elig.dqb / n, n)), "tfl_pg": _pct(_shrink(elig.dtfl / n, n)),
                                  "tackles_pg": _pct(_shrink((elig.dtk + 0.5 * elig.dta) / n, n))})
            weights = {"sacks_pg": 0.3, "qb_hits_pg": 0.25, "tfl_pg": 0.25, "tackles_pg": 0.2} if pg == "DL" else {"sacks_pg": 0.15, "qb_hits_pg": 0.15, "tfl_pg": 0.3, "tackles_pg": 0.4}
            raw = {"sacks_pg": elig.dsk / n, "qb_hits_pg": elig.dqb / n, "tfl_pg": elig.dtfl / n, "tackles_pg": (elig.dtk + 0.5 * elig.dta) / n}
        else:
            for pid, r in grp.iterrows():
                out[pid] = _row(pid, r, None, {}, f"Not graded: position group {pg} has no grade definition", pg, 0)
            continue
        if len(elig) < 8:
            for pid, r in grp.iterrows():
                out[pid] = _row(pid, r, None, {}, "Not enough data: fewer than 8 qualifying players in the group", pg, len(elig))
            continue
        rating = sum(weights[k] * comps[k] for k in weights)
        grade = _pct(rating)
        for pid, r in grp.iterrows():
            if pid in elig.index:
                c = {k: {"percentile": round(float(comps.loc[pid, k]), 1), "value": None if pd.isna(raw[k].loc[pid]) else round(float(raw[k].loc[pid]), 3)} for k in weights}
                out[pid] = _row(pid, r, float(grade.loc[pid]), c, None, pg, len(elig), rating=float(rating.loc[pid]))
            else:
                why = f"Not enough data: {int(r.n)} of {MIN_GAMES} games" if r.n < MIN_GAMES else "Not enough data: below the volume threshold for the position"
                out[pid] = _row(pid, r, None, {}, why, pg, len(elig))
    if keep_ids is not None:
        out = {k: v for k, v in out.items() if k in keep_ids}
    return out, meta


def _row(pid, r, grade, comps, reason, pg, n_group, rating=None):
    """Compact row: shared strings (timeframe, labels, QBR note) live in the meta block."""
    row = {"name": r["name"], "pos": r.pos, "group": pg, "team": r.team,
           "grade": None if grade is None else round(grade), "rating": None if rating is None else round(rating, 1),
           "reason": reason, "components": {k: [round(v["percentile"]), v["value"]] for k, v in comps.items()},
           "n_group": n_group, "n": int(r.n), "through": f"{int(r.season)} wk {int(r.week)}"}
    if pg == "QB":
        row["passer_rating"] = passer_rating(r.comp, r.att, r.pyds, r.ptd, r.ints)
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
            if r.pos_rank <= 2 and ro is not None and pd.notna(ro.headshot_url):
                # nfl.com headshots share one prefix; the page rebuilds the URL from the id
                row["hs"] = ro.headshot_url.replace(HS_PREFIX, "")
            if ro is not None and ro.status != "ACT":
                row["roster_status"] = ro.status
            if gid in inj_map:
                row["injury"] = inj_map[gid]
            groups.setdefault(r.pos_grp, []).append(row)
        out[team] = {"asof": latest.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(latest) else None, "groups": groups}
    out["_meta"] = {"source": "nflverse depth_charts (ESPN-sourced), ranks 1-3 per slot; headshots for ranks 1-2 (nfl.com)", "headshot_prefix": HS_PREFIX,
                    "group_labels": GROUP_LABEL,
                    "note": "ranks are the team's published depth chart, not a confirmed game-day lineup; formations drawn from it are illustrative"}
    return out
