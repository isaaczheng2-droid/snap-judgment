#!/usr/bin/env python3
"""
Live injury status from ESPN, overlaid on the official nflverse injury report.

WHY THIS EXISTS
---------------
The nflverse `injuries` file is the NFL's official injury report: practice participation
Wednesday through Friday, plus a final game status. It is authoritative, it goes back to
2009, and it is what every historical injury feature in this model was built on.

It also cannot, even in principle, contain an injury that happened during a game. The next
report lands the following Wednesday.

That is not a hypothetical gap. On 2026-09-09 New England played Seattle; A.J. Brown hurt
an ankle and Sam Darnold hurt a hip, both during the game. ESPN had Brown flagged by 23:37
UTC the next day with "expected to miss four weeks", and Darnold at Doubtful by 18:23 UTC.
The official report had neither, and would not until Wednesday the 16th. For four days the
site would have been publishing a Seattle number that assumed a healthy starting quarterback.

So: nflverse stays the base layer and the only thing history is built from. ESPN is layered
on top of the CURRENT week only, and only for teams whose game has not kicked off yet.

WHAT ESPN GIVES AND WHAT IT COSTS
---------------------------------
`site.api.espn.com/.../nfl/injuries` returns 32 teams and is timestamped to the minute.

It caps at exactly 25 entries per team, and `?limit=` does not raise it. That sounds
dangerous and is not, because the entries come back newest-first: the cap drops the
STALEST rows, which are precisely the ones the official report already covers in full.
Fresh-on-top of complete-underneath is the right way round. It would be the wrong way round
if ESPN were the only source, which is the main reason it is not.

Three traps, all handled below:

  1. "Coach's Decision" is not an injury. ESPN files healthy scratches and gameday
     inactives under the same `status: Out` as a torn ACL. Aaron Donald shows up as
     "Out - Coach's Decision"; he is fine. These are dropped from anything that reaches the
     model, because the official report never listed healthy scratches and the injury
     features were trained on a world where it did not.

  2. Team codes differ. ESPN says LAR and WSH; nflverse says LA and WAS. Everything else
     matches. A silent mismatch here would drop two teams' injuries entirely.

  3. Suspensions are not injuries either, and are dropped for the same reason as (1).

Statuses are mapped to the official report's own vocabulary rather than a richer one of our
own, so that a row sourced from ESPN and a row sourced from nflverse mean the same thing to
the model. Injured Reserve collapses to Out. Doubtful stays Doubtful, which means it does
NOT count toward the injury features -- the features only ever counted Out, and quietly
promoting Doubtful would change what the feature measures relative to every season it was
fitted on.

FAILURE BEHAVIOUR
-----------------
Every entry point returns None or an unchanged frame on any failure. ESPN going down, or
changing its JSON, must never stop the site from publishing; it just falls back to the
official report, which is what the site ran on until now.
"""
import io
import json
import os
import re
import subprocess

import numpy as np
import pandas as pd

URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"

# ESPN -> nflverse. Only the two that actually differ; anything absent passes through.
TEAM_FIX = {"LAR": "LA", "WSH": "WAS"}

# ESPN status -> the official report's vocabulary. None means "on the list but playing".
STATUS = {
    "out": "Out",
    "injured reserve": "Out",
    "doubtful": "Doubtful",
    "questionable": "Questionable",
    "active": None,
    "suspension": None,      # not an injury; the official report never carried these
    "physically unable to perform": "Out",
    "practice squad": None,
}

# Reasons that are not injuries. ESPN files these under the same status as a real one.
NOT_AN_INJURY = {"coach's decision", "coaches decision", "suspension", "not injury related"}

COLS = ["espn_id", "team", "position", "full_name", "espn_status", "updated",
        "injury_type", "return_date", "scratch"]

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\.?$", re.I)
_PUNCT = re.compile(r"[^a-z ]")


def _norm(name):
    """Loose name key for the fallback join: no punctuation, no generational suffix."""
    n = str(name or "").lower().replace(".", "").replace("'", "").replace("-", " ")
    n = _PUNCT.sub("", n).strip()
    n = _SUFFIX.sub("", n).strip()
    return re.sub(r"\s+", " ", n)


# --------------------------------------------------------------------------------- fetch
def fetch(url=URL, timeout=25, path=None):
    """
    Pull the feed. Returns the parsed JSON, or None if anything at all goes wrong.

    curl rather than requests so this has no dependency the workflow does not already have,
    and so a hung connection is bounded by --max-time rather than by hope.
    """
    if path and os.path.exists(path):                 # offline fixture, for tests
        try:
            return json.load(open(path))
        except Exception:
            return None
    try:
        r = subprocess.run(
            ["curl", "-sSL", "--max-time", str(timeout), "-H", "Accept: application/json", url],
            capture_output=True, timeout=timeout + 10)
        if r.returncode != 0 or not r.stdout:
            return None
        return json.loads(r.stdout)
    except Exception:
        return None


# --------------------------------------------------------------------------------- parse
def parse(raw):
    """Flatten ESPN's nested payload into one row per listed player. Never raises."""
    if not raw:
        return pd.DataFrame(columns=COLS)
    rows = []
    for team_block in (raw.get("injuries") or []):
        for it in (team_block.get("injuries") or []):
            try:
                a = it.get("athlete") or {}
                det = it.get("details") or {}
                eid = a.get("id")
                if not eid:
                    # some payloads omit athlete.id; the player-card link always has it
                    href = " ".join(l.get("href", "") for l in (a.get("links") or []))
                    m = re.search(r"/id/(\d+)", href)
                    eid = m.group(1) if m else None
                team = (a.get("team") or {}).get("abbreviation") or team_block.get("abbreviation")
                typ = (det.get("type") or "").strip()
                rows.append({
                    "espn_id": str(eid) if eid else None,
                    "team": TEAM_FIX.get(str(team).upper(), str(team).upper()),
                    "position": ((a.get("position") or {}).get("abbreviation") or "").upper(),
                    "full_name": a.get("displayName"),
                    "espn_status": (it.get("status") or "").strip(),
                    "updated": it.get("date"),
                    "injury_type": typ,
                    "return_date": det.get("returnDate"),
                    "note": (it.get("shortComment") or it.get("longComment") or "").strip(),
                    "scratch": typ.lower() in NOT_AN_INJURY,
                })
            except Exception:
                continue
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COLS)
    df["updated"] = pd.to_datetime(df.updated, errors="coerce", utc=True)
    # newest first, one row per player: ESPN can carry more than one note per athlete
    df = df.sort_values("updated", ascending=False).drop_duplicates(
        subset=["espn_id", "team"], keep="first")
    return df.reset_index(drop=True)


def parse_psv(path):
    """Read the captured test fixture. Same shape as parse(), for offline testing."""
    df = pd.read_csv(path, sep="|", header=None, dtype=str,
                     names=["espn_id", "team", "position", "full_name", "espn_status",
                            "updated", "injury_type", "return_date", "note"])
    df["note"] = df.note.fillna("")
    df["team"] = df.team.str.upper().map(lambda t: TEAM_FIX.get(t, t))
    df["injury_type"] = df.injury_type.fillna("")
    df["scratch"] = df.injury_type.str.lower().isin(NOT_AN_INJURY)
    df["updated"] = pd.to_datetime(df.updated, errors="coerce", utc=True)
    return df


# ---------------------------------------------------------------------------- id mapping
def attach_gsis(df, rost, season):
    """
    Map ESPN athletes onto gsis_id, the id everything else in this pipeline keys on.

    nflverse rosters carry `espn_id`, so the primary join is exact. It covers about two
    thirds of the roster file -- the misses are mostly practice-squad and deep-bench players
    -- so a normalised name+team join picks up the rest. Name joins are the kind of thing
    that quietly attaches the wrong player, so it is used only where the id is absent, and
    only when the name+team pair is unique on both sides.
    """
    out = df.copy()
    out["gsis_id"] = None
    if rost is None or not len(rost):
        return out

    r = rost.copy()
    if "season" in r.columns:
        r = r[r.season == season]
    r = r.dropna(subset=["gsis_id"])
    if not len(r):
        return out

    if "espn_id" in r.columns:
        m = (r.dropna(subset=["espn_id"]).assign(espn_id=lambda x: x.espn_id.astype(str))
             .drop_duplicates("espn_id").set_index("espn_id")["gsis_id"])
        out["gsis_id"] = out.espn_id.map(m)

    miss = out.gsis_id.isna()
    if miss.any():
        r = r.assign(_k=r.full_name.map(_norm) + "|" + r.team.astype(str))
        uniq = r.drop_duplicates("_k", keep=False).set_index("_k")["gsis_id"]
        key = out.loc[miss, "full_name"].map(_norm) + "|" + out.loc[miss, "team"].astype(str)
        out.loc[miss, "gsis_id"] = key.map(uniq)
    return out


# ------------------------------------------------------------------------------- overlay
def to_report(df, season, week, teams=None):
    """
    Turn parsed ESPN rows into rows shaped like the official injury report.

    Scratches, suspensions and players who map to no status are dropped, so what comes back
    means the same thing as a row that came from nflverse.
    """
    cols = ["season", "week", "team", "gsis_id", "position", "full_name",
            "report_status", "report_primary_injury", "game_type", "date_modified"]
    if df is None or not len(df):
        return pd.DataFrame(columns=cols)
    d = df[~df.scratch.fillna(False)].copy()
    d["report_status"] = d.espn_status.str.lower().str.strip().map(STATUS)
    d = d[d.report_status.notna() & d.gsis_id.notna()]
    if teams is not None:
        d = d[d.team.isin(set(teams))]
    # ESPN's `type` is the body part, which is the same thing the official report's
    # `report_primary_injury` holds — so the site's existing wording ("Ruled out — hip")
    # works on an ESPN row without knowing where it came from.
    d["report_primary_injury"] = d.injury_type.fillna("").str.strip().replace(
        {"": None, "Undisclosed": None})
    # the official report carries a `date_modified`; ESPN's per-row timestamp is the same
    # idea and belongs in the same column rather than being dropped on the floor
    d["date_modified"] = pd.to_datetime(d.updated, errors="coerce", utc=True)
    d["season"], d["week"], d["game_type"] = int(season), int(week), "REG"
    return d[cols].reset_index(drop=True)


def overlay(inj, espn_rows, season, week):
    """
    Splice ESPN's view of one week over the official report's rows for that same week.

    PER PLAYER, not per team, and the difference matters. ESPN returns at most 25 entries
    per team; the official report can be longer. Replacing a team's whole block would mean
    that any player the cap trimmed -- which is to say the ones whose news is OLDEST, and
    therefore the ones most likely to be a long-standing Out -- silently vanished from the
    injury features. The live feed is there to be fresher, not shorter, so a player ESPN
    did not mention keeps whatever the official report said about him.

    Strictly scoped: rows outside (season, week) are untouched, so everything the accuracy
    audit reads is historical and cannot move because of this function. That is the
    property that makes a live feed safe to use at all.
    """
    if espn_rows is None or not len(espn_rows):
        return inj
    keep_cols = list(inj.columns)
    in_week = (inj.season.astype("int64") == int(season)) & \
              (inj.week.astype("int64") == int(week))
    # only the specific players ESPN spoke about are displaced
    hit = in_week & inj.gsis_id.isin(set(espn_rows.gsis_id))
    base = inj[~hit]
    add = espn_rows.reindex(columns=keep_cols)
    for c in keep_cols:
        if c in espn_rows.columns:
            continue
        # A plain None here is not enough. The official report carries tz-aware datetime
        # columns, and concatenating an object column of None into one of those raises
        # inside pandas rather than filling with NaT. Ask for a missing value OF THE RIGHT
        # TYPE, and fall back to None for anything that cannot represent one.
        try:
            add[c] = pd.array([None] * len(add), dtype=inj[c].dtype)
        except (TypeError, ValueError):
            add[c] = None
    if "season_type" in keep_cols:
        add["season_type"] = "REG"
    out = pd.concat([base, add], ignore_index=True)
    assert len(out[(out.season.astype("int64") == int(season)) &
                   (out.week.astype("int64") == int(week))]) >= len(espn_rows)
    return out


def live_overlay(inj, rost, season, week, teams=None, url=URL, fixture=None, log=print):
    """
    The whole thing, end to end, and it cannot raise.

    Returns (injuries_frame, parsed_espn_or_None). A None second element means the site is
    running on the official report alone, which is what it did before this module existed.
    """
    try:
        raw = fetch(url) if not fixture else None
        df = parse_psv(fixture) if fixture else parse(raw)
        if not len(df):
            log("  espn: no rows returned, falling back to the official report")
            return inj, None
        df = attach_gsis(df, rost, season)
        rep = to_report(df, season, week, teams=teams)
        # Mark which rows actually made it into the report. The rest are still worth
        # keeping -- the "reported after kickoff" block reads them -- but the page must not
        # claim a status came off the live feed when the overlay skipped that team and the
        # status on screen is the official one.
        df["applied"] = df.gsis_id.isin(set(rep.gsis_id)) if len(rep) else False
        matched = int(df.gsis_id.notna().sum())
        log(f"  espn: {len(df)} listed, {matched} mapped to a player id, "
            f"{len(rep)} usable for week {week}"
            + (f" across {rep.team.nunique()} teams" if len(rep) else ""))
        return overlay(inj, rep, season, week), df
    except Exception as e:
        log(f"  espn: unavailable ({type(e).__name__}: {e}); using the official report")
        return inj, None
