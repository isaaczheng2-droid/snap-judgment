#!/usr/bin/env python3
"""
Scheme and coaching context per team: who the staff is, and how the offence and defence
measurably play, from sources the model does NOT use (tested 2026-09-19 and rejected; see
audit/nfl-coaching-scheme-test.md). Everything here is shown, nothing here is predicted from.

Sources (all nflverse release assets, free):
  pfr_advstats  weekly passing charting: pressures, blitzes, hurries, hits per QB-game (2018+)
  ftn_charting  play-level flags: play action, motion, pass rushers, blitzers (2022+)
  pbp           early-down pass rate and 4th-down decisions (current and previous season only
                on the runner; the full history is not needed for a season-to-date card)
  coaches.json  head coach / OC / DC per team, compiled by hand with Wikipedia as the source,
                photos from Wikimedia Commons with their licences; edited when staffs change
  ESPN          the head coach ESPN lists for the team, used only to flag a stale coaches.json

Numbers on the card are season-to-date means over the team's games with the number of games
shown, ranked 1..32 within the league (1 = highest rate). With fewer than MIN_GAMES games the
card also shows last season's full-year figure, labelled as such. Nothing is shrunk or filled:
a missing source is reported as missing.
"""
import glob
import json
import os
import subprocess

import numpy as np
import pandas as pd

REL = "https://github.com/nflverse/nflverse-data/releases/download"
MIN_GAMES = 3
PFR_FIRST, FTN_FIRST = 2018, 2022
OFF = ["edp_pass", "pa_rate", "motion", "press_allow", "go4"]
DEF = ["press_made", "blitz_made", "sim_rate"]
METRICS = OFF + DEF
LABELS = {
    "edp_pass": ("Early-down pass rate", "share of 1st/2nd-down plays that were passes, quarters 1-3, score within 10", "pct"),
    "pa_rate": ("Play action", "share of dropbacks with a play-action fake (FTN charting)", "pct"),
    "motion": ("Pre-snap motion", "share of pass and run plays with a player in motion (FTN charting)", "pct"),
    "press_allow": ("Pressure allowed", "share of the QB's dropbacks under pressure (PFR charting)", "pct"),
    "go4": ("4th-and-short go rate", "went for it on 4th-and-2 or less between the 35s, quarters 1-3, score within 14", "count"),
    "press_made": ("Pressure rate", "share of opponent dropbacks where the defence got pressure (PFR charting)", "pct"),
    "blitz_made": ("Blitz rate", "share of opponent dropbacks with a blitz (PFR charting)", "pct"),
    "sim_rate": ("Sim-pressure rate", "opponent dropbacks with four or fewer rushers but at least one blitzer, i.e. a rusher dropped and someone else came (FTN charting)", "pct"),
}
TEAM_FIX = {"LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA"}


def _fetch(url, path, log=print):
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return True
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    r = subprocess.run(["curl", "-sSL", "--max-time", "120", "-o", path, url], capture_output=True)
    ok = r.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 1024
    if not ok and os.path.exists(path):
        os.remove(path)
    if not ok:
        log(f"  coaching: could not fetch {url}")
    return ok


def load_sources(d, cur, log=print):
    """Fetch what the card needs. Current-season files are refreshed every run."""
    for stale in [f"{d}/pfr/pass_{cur}.parquet", f"{d}/ftn/ftn_{cur}.parquet", f"{d}/pbp/play_by_play_{cur}.parquet"]:
        if os.path.exists(stale):
            os.remove(stale)
    for s in range(PFR_FIRST, cur + 1):
        _fetch(f"{REL}/pfr_advstats/advstats_week_pass_{s}.parquet", f"{d}/pfr/pass_{s}.parquet", log)
    for s in range(FTN_FIRST, cur + 1):
        _fetch(f"{REL}/ftn_charting/ftn_charting_{s}.parquet", f"{d}/ftn/ftn_{s}.parquet", log)
    for s in (cur - 1, cur):
        _fetch(f"{REL}/pbp/play_by_play_{s}.parquet", f"{d}/pbp/play_by_play_{s}.parquet", log)


def per_game_pbp(d, seasons, log=print):
    cols = ["game_id", "play_id", "posteam", "defteam", "down", "play_type", "qtr", "score_differential",
            "ydstogo", "yardline_100", "season", "week", "season_type"]
    rows = []
    for s in seasons:
        p = f"{d}/pbp/play_by_play_{s}.parquet"
        if not os.path.exists(p):
            continue
        try:
            x = pd.read_parquet(p, columns=cols)
        except Exception as e:
            log(f"  coaching: skip {p}: {e}")
            continue
        x = x[(x.season_type == "REG") & x.posteam.notna()]
        f = f"{d}/ftn/ftn_{s}.parquet"
        if os.path.exists(f):
            ft = pd.read_parquet(f, columns=["nflverse_game_id", "nflverse_play_id", "is_play_action", "is_motion", "n_blitzers", "n_pass_rushers"])
            x = x.merge(ft, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"], how="left")
        else:
            for c in ["is_play_action", "is_motion", "n_blitzers", "n_pass_rushers"]:
                x[c] = np.nan
        rows.append(x)
    if not rows:
        return pd.DataFrame(columns=["game_id", "team"])
    x = pd.concat(rows, ignore_index=True)
    live = x[x.play_type.isin(["pass", "run"])]
    ed = live[(live.down <= 2) & (live.qtr <= 3) & (live.score_differential.abs() <= 10)]
    edp = ed.groupby(["game_id", "posteam"]).apply(lambda g: (g.play_type == "pass").mean()).rename("edp_pass")
    fd = x[(x.down == 4) & (x.ydstogo <= 2) & x.yardline_100.between(35, 65) & (x.qtr <= 3) & (x.score_differential.abs() <= 14)
           & x.play_type.isin(["pass", "run", "punt", "field_goal"])]
    go4_go = fd.groupby(["game_id", "posteam"]).apply(lambda g: g.play_type.isin(["pass", "run"]).sum()).rename("go4_go")
    go4_n = fd.groupby(["game_id", "posteam"]).size().rename("go4_n")
    pa = live[live.play_type == "pass"].groupby(["game_id", "posteam"]).is_play_action.mean().rename("pa_rate")
    mo = live.groupby(["game_id", "posteam"]).is_motion.mean().rename("motion")
    ps = live[live.play_type == "pass"].copy()
    ps["sim"] = ((ps.n_pass_rushers <= 4) & (ps.n_blitzers >= 1)).astype(float).where(ps.n_pass_rushers.notna())
    sim = ps.groupby(["game_id", "defteam"]).sim.mean().rename("sim_rate")
    sim.index = sim.index.set_names(["game_id", "posteam"])
    out = pd.concat([edp, go4_go, go4_n, pa, mo, sim], axis=1).reset_index().rename(columns={"posteam": "team"})
    return out


def per_game_pfr(d, seasons, team, log=print):
    rows = []
    for s in seasons:
        p = f"{d}/pfr/pass_{s}.parquet"
        if os.path.exists(p):
            try:
                rows.append(pd.read_parquet(p))
            except Exception as e:
                log(f"  coaching: skip {p}: {e}")
    if not rows:
        return pd.DataFrame(columns=["game_id", "team"])
    p = pd.concat(rows, ignore_index=True)
    p = p[p.game_type == "REG"].copy()
    p["team"] = p.team.replace(TEAM_FIX)
    p["opponent"] = p.opponent.replace(TEAM_FIX)
    g = p.groupby(["game_id", "team", "opponent"])[["times_pressured", "times_blitzed"]].sum().reset_index()
    tw = team[team.season_type == "REG"][["game_id", "team", "attempts", "sacks_suffered"]].copy()
    tw["dropbacks"] = tw.attempts.fillna(0) + tw.sacks_suffered.fillna(0)
    g = g.merge(tw[["game_id", "team", "dropbacks"]], on=["game_id", "team"], how="left")
    g["press_allow"] = g.times_pressured / g.dropbacks.replace(0, np.nan)
    dd = g.rename(columns={"team": "opponent", "opponent": "team", "times_pressured": "made", "times_blitzed": "blz", "dropbacks": "db_faced"})
    dd["press_made"] = dd.made / dd.db_faced.replace(0, np.nan)
    dd["blitz_made"] = dd.blz / dd.db_faced.replace(0, np.nan)
    return g[["game_id", "team", "press_allow"]].merge(dd[["game_id", "team", "press_made", "blitz_made"]], on=["game_id", "team"], how="outer")


def per_game(d, seasons, team, log=print):
    a = per_game_pbp(d, seasons, log)
    b = per_game_pfr(d, seasons, team, log)
    return a.merge(b, on=["game_id", "team"], how="outer")


def _season_table(pg, sched, season):
    """Per team: mean of each rate over the season's games, games with the metric, 4th-down counts."""
    gd = sched[sched.game_type == "REG"][["game_id", "season", "week"]].drop_duplicates("game_id")
    m = pg.merge(gd, on="game_id", how="inner")
    m = m[m.season == season]
    rows = {}
    for t, g in m.groupby("team"):
        r = {"games": int(len(g))}
        for x in ["edp_pass", "pa_rate", "motion", "press_allow", "press_made", "blitz_made", "sim_rate"]:
            v = g[x].dropna() if x in g.columns else pd.Series(dtype=float)
            r[x] = {"value": (float(v.mean()) if len(v) else None), "n": int(len(v))}
        gg = int(g.go4_go.fillna(0).sum()) if "go4_go" in g.columns else 0
        gn = int(g.go4_n.fillna(0).sum()) if "go4_n" in g.columns else 0
        r["go4"] = {"value": (gg / gn if gn else None), "n": gn, "go": gg}
        rows[t] = r
    # league ranks: 1 = highest rate; only teams with a value
    for x in METRICS:
        vals = {t: r[x]["value"] for t, r in rows.items() if r[x]["value"] is not None and r[x]["n"] > 0}
        order = sorted(vals, key=lambda t: -vals[t])
        for i, t in enumerate(order):
            rows[t][x]["rank"] = i + 1
            rows[t][x]["of"] = len(order)
        if vals:
            med = float(np.median(list(vals.values())))
            for t in vals:
                rows[t][x]["league_median"] = med
    return rows


def _pct(v):
    return f"{v * 100:.0f}%"


def _rank_word(rank, of):
    if rank is None:
        return None
    q = rank / of
    if q <= 0.125:
        return "top"
    if q <= 0.34:
        return "high"
    if q >= 0.875:
        return "bottom"
    if q >= 0.66:
        return "low"
    return "mid"


def style_sentences(cur_row, prev_row, games):
    """Plain sentences, every one carrying the number and the rank behind it."""
    src, tag = (cur_row, "this season") if games >= MIN_GAMES else (prev_row, "last season")
    out = []
    if not src:
        return out

    def rk(x):
        m = src.get(x) or {}
        return m.get("value"), m.get("rank"), m.get("of"), m.get("n", 0)

    v, r, of, n = rk("edp_pass")
    if v is not None and r:
        w = _rank_word(r, of)
        lead = {"top": "Pass-first on early downs", "high": "Leans pass on early downs", "mid": "Balanced on early downs",
                "low": "Leans run on early downs", "bottom": "Run-first on early downs"}[w]
        out.append(f"{lead}: {_pct(v)} of neutral 1st/2nd-down plays were passes, {r}{_ord(r)} of {of} ({tag}).")
    v, r, of, n = rk("pa_rate")
    if v is not None and r:
        w = _rank_word(r, of)
        out.append(f"{'Heavy' if w in ('top', 'high') else 'Light' if w in ('low', 'bottom') else 'Average'} play-action use: {_pct(v)} of dropbacks, {r}{_ord(r)} of {of} ({tag}).")
    v, r, of, n = rk("motion")
    if v is not None and r and _rank_word(r, of) in ("top", "bottom"):
        out.append(f"{'Constant' if r <= of * 0.125 else 'Rare'} pre-snap motion: {_pct(v)} of plays, {r}{_ord(r)} of {of} ({tag}).")
    v, r, of, n = rk("press_allow")
    if v is not None and r:
        w = _rank_word(r, of)
        out.append(f"Protection: the quarterback was pressured on {_pct(v)} of dropbacks, {r}{_ord(r)}-most of {of} ({tag}).")
    g = src.get("go4") or {}
    if g.get("n"):
        out.append(f"4th-and-short in the middle of the field: went for it {g['go']} of {g['n']} times ({tag}).")
    v, r, of, n = rk("press_made")
    if v is not None and r:
        w = _rank_word(r, of)
        out.append(f"{'Elite' if w == 'top' else 'Strong' if w == 'high' else 'Middling' if w == 'mid' else 'Weak'} pass rush: pressure on {_pct(v)} of opponent dropbacks, {r}{_ord(r)} of {of} ({tag}).")
    v, r, of, n = rk("blitz_made")
    if v is not None and r:
        w = _rank_word(r, of)
        out.append(f"{'Blitz-heavy' if w in ('top', 'high') else 'Rarely blitzes' if w in ('low', 'bottom') else 'Average blitz rate'}: {_pct(v)} of opponent dropbacks, {r}{_ord(r)} of {of} ({tag}).")
    v, r, of, n = rk("sim_rate")
    if v is not None and r and _rank_word(r, of) in ("top", "high"):
        out.append(f"Uses simulated pressure (a rusher drops, someone else comes) on {_pct(v)} of opponent dropbacks, {r}{_ord(r)} of {of} ({tag}).")
    return out


def _ord(n):
    return "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def coach_records(sched, cur):
    """Regular-season record of the current head coach with this team, from the schedule."""
    s = sched[(sched.game_type == "REG") & sched.home_score.notna()]
    rows = []
    for side, opp in [("home", "away"), ("away", "home")]:
        r = s[["season", f"{side}_team", f"{side}_coach", f"{side}_score", f"{opp}_score"]].copy()
        r.columns = ["season", "team", "coach", "pf", "pa"]
        rows.append(r)
    t = pd.concat(rows)
    out = {}
    for (team, coach), g in t.groupby(["team", "coach"]):
        out[(team, coach)] = {"w": int((g.pf > g.pa).sum()), "l": int((g.pf < g.pa).sum()), "t": int((g.pf == g.pa).sum()),
                              "seasons": sorted(int(x) for x in g.season.unique())}
    return out


def espn_head_coaches(log=print):
    """{abbr: name} from ESPN's team API, used only to flag a stale coaches.json. Best effort."""
    base = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons"
    out = {}
    try:
        import urllib.request
        yr = pd.Timestamp.utcnow().year
        for tid in range(1, 35):
            try:
                with urllib.request.urlopen(f"{base}/{yr}/teams/{tid}?lang=en&region=us", timeout=8) as r:
                    t = json.load(r)
                ab = TEAM_FIX.get(t.get("abbreviation"), t.get("abbreviation"))
                if not ab or ab in ("AFC", "NFC"):
                    continue
                with urllib.request.urlopen(f"{base}/{yr}/teams/{tid}/coaches?limit=5", timeout=8) as r:
                    cl = json.load(r)
                for it in cl.get("items", []):
                    with urllib.request.urlopen(it["$ref"].replace("http://", "https://"), timeout=8) as r:
                        c = json.load(r)
                    out[ab] = f"{c.get('firstName', '')} {c.get('lastName', '')}".strip()
                    break
            except Exception:
                continue
    except Exception as e:
        log(f"  coaching: ESPN head-coach check skipped ({e})")
    return out


def build(d, sched, team, cur, coaches_path="coaches.json", log=print, espn=True):
    load_sources(d, cur, log)
    pg = per_game(d, [cur - 1, cur], team, log)
    cur_rows = _season_table(pg, sched, cur)
    prev_rows = _season_table(pg, sched, cur - 1)
    try:
        staff = json.load(open(coaches_path))
    except Exception as e:
        log(f"  coaching: no coaches.json ({e}); staff omitted")
        staff = {"teams": {}, "as_of": None}
    recs = coach_records(sched, cur)
    espn_hc = espn_head_coaches(log) if espn else {}
    reg = sched[(sched.season == cur) & (sched.game_type == "REG")]
    teams = sorted(set(reg.home_team) | set(reg.away_team))
    out = {}
    for t in teams:
        st = dict((staff.get("teams") or {}).get(t) or {})
        alerts = []
        hc = st.get("HC")
        if hc:
            hc = dict(hc)
            rec = recs.get((t, hc["name"]))
            if rec:
                hc["with_team"] = rec
            if espn_hc.get(t) and espn_hc[t].lower() != hc["name"].lower():
                alerts.append(f"ESPN lists {espn_hc[t]} as head coach; the staff file (dated {staff.get('as_of')}) says {hc['name']}. Staff may be out of date.")
            st["HC"] = hc
        cr = cur_rows.get(t, {"games": 0})
        pr = prev_rows.get(t)
        games = cr.get("games", 0)
        block = {
            "staff": st, "games": games, "alerts": alerts,
            "off": {x: cr.get(x) for x in OFF} if games else {},
            "def": {x: cr.get(x) for x in DEF} if games else {},
            "prev": {"season": cur - 1, "off": {x: pr.get(x) for x in OFF}, "def": {x: pr.get(x) for x in DEF}, "games": pr.get("games", 0)} if pr else None,
            "sentences": style_sentences(cr if games else None, pr, games),
            "basis": "this season" if games >= MIN_GAMES else "last season",
        }
        out[t] = block
    have = {x: any((cur_rows.get(t, {}).get(x) or {}).get("n") for t in teams) for x in METRICS}
    return {
        "season": cur, "staff_as_of": staff.get("as_of"), "staff_sources": staff.get("sources"),
        "min_games": MIN_GAMES, "teams": out, "definitions": {k: {"label": v[0], "detail": v[1], "kind": v[2]} for k, v in LABELS.items()},
        "sources_present": have,
        "note": ("Context only. These factors were tested in the game and player models on 2026-09-19 and none cleared the bar "
                 "(pressure, blitz, early-down pass rate, play action, motion, 4th-down rate, coaching change: all within a quarter "
                 "of a point of the shipped model). Shown so a reader can see how each side plays; not used to make the pick."),
    }


if __name__ == "__main__":
    import run_pipeline as rp
    sched, team, plyr, inj, snap, rost, depth, cur = rp.load_all("data", None)
    b = build("data", sched, team, cur, log=rp.log, espn=False)
    json.dump(b, open("data/coaching_block.json", "w"), indent=1)
    t = b["teams"]["SEA"]
    print(json.dumps({k: v for k, v in t.items() if k != "staff"}, indent=1)[:2500])
    print(len(json.dumps(b)), "bytes")
