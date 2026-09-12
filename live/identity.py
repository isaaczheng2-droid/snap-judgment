"""
Identity resolution: one internal_player_id per human being, whatever a vendor calls him.

The internal id is the NFL GSIS id (`00-0034381`), because it is what every historical
table in this project keys on and what nflverse rosters carry alongside every other vendor
id. The crosswalk is rebuilt from the roster file each run and written to
live/data/source_mapping.json so the mapping is auditable.

Name matching is the last resort, never the first: it is applied only when no vendor id
matches, only within a team, and only when the normalised name is unique on both sides.
Suffixes (Jr., III), hyphens, apostrophes and diacritics are normalised; a name that still
collides is flagged as unresolved rather than guessed.
"""
import hashlib
import json
import os
import re
import unicodedata

from . import store

VENDOR_ID_COLS = {"espn": "espn_id", "pfr": "pfr_id", "sportradar": "sportradar_id",
                  "sportsdataio": "fantasy_data_id", "yahoo": "yahoo_id", "rotowire": "rotowire_id",
                  "sleeper": "sleeper_id", "pff": "pff_id", "esb": "esb_id"}

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\.?$", re.I)


def norm_name(name):
    n = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    n = n.lower().replace(".", "").replace("'", "").replace("-", " ")
    n = re.sub(r"[^a-z ]", "", n).strip()
    n = _SUFFIX.sub("", n).strip()
    return re.sub(r"\s+", " ", n)


class Crosswalk:
    def __init__(self, rows):
        # rows: list of dicts with internal_player_id, player_name, team, position, vendor ids
        self.rows = rows
        self.by_vendor = {v: {} for v in VENDOR_ID_COLS}
        self.by_name_team = {}
        dup = set()
        for r in rows:
            for v, col in VENDOR_ID_COLS.items():
                vid = r.get(col)
                if vid not in (None, "", "None", "nan"):
                    self.by_vendor[v][str(vid).split(".")[0]] = r["internal_player_id"]
            k = (norm_name(r.get("player_name")), r.get("team"))
            if k in self.by_name_team:
                dup.add(k)
            self.by_name_team[k] = r["internal_player_id"]
        for k in dup:                       # ambiguous names never resolve by name
            self.by_name_team.pop(k, None)
        self.meta = {r["internal_player_id"]: r for r in rows}

    @classmethod
    def from_roster(cls, rost, season=None):
        r = rost
        if season is not None and "season" in r.columns:
            r = r[r.season == season]
        r = r.dropna(subset=["gsis_id"])
        rows = []
        for t in r.itertuples():
            d = {"internal_player_id": t.gsis_id, "player_name": getattr(t, "full_name", None),
                 "team": getattr(t, "team", None), "position": getattr(t, "position", None),
                 "roster_status": getattr(t, "status", None)}
            for v, col in VENDOR_ID_COLS.items():
                val = getattr(t, col, None)
                d[col] = None if val is None or str(val) in ("nan", "None", "") else str(val).split(".")[0]
            rows.append(d)
        return cls(rows)

    def resolve(self, vendor=None, vendor_id=None, name=None, team=None):
        """-> (internal_player_id or None, how). Vendor id first, unique name+team second."""
        if vendor and vendor_id not in (None, ""):
            hit = self.by_vendor.get(vendor, {}).get(str(vendor_id).split(".")[0])
            if hit:
                return hit, f"{vendor}_id"
        if name and team:
            hit = self.by_name_team.get((norm_name(name), team))
            if hit:
                return hit, "name+team"
        return None, "unresolved"

    def write(self, path=None):
        path = path or os.path.join(store.ROOT, "source_mapping.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        keep = ("internal_player_id", "player_name", "team", "position", "espn_id", "pfr_id", "sportradar_id", "esb_id")
        slim = [{k: r.get(k) for k in keep if r.get(k) is not None} for r in self.rows]
        body = json.dumps({"n": len(slim), "players": slim}, separators=(",", ":"), sort_keys=True)
        digest = hashlib.sha256(body.encode()).hexdigest()[:16]
        try:
            if json.load(open(path)).get("digest") == digest:
                return path                       # unchanged roster: do not churn the file
        except Exception:
            pass
        json.dump({"generated_at": store.now_iso(), "digest": digest, "n": len(slim), "players": slim},
                  open(path, "w"), separators=(",", ":"))
        return path
