"""
League scoring and lineup settings.

Full PPR is the default. Everything a projection can be scored on is a setting; everything
a projection cannot see is listed in NOT_PROJECTED so the page can say what the number
leaves out (fumbles, two-point conversions, return yards) rather than pretend it is there.
The actual results the site grades against are nflverse's fantasy_points_ppr, which DO
include those, so the residual ranges absorb them honestly.
"""
from dataclasses import dataclass, asdict, field

# projection keys (run_pipeline.PTARGETS + fantasy.engine.EXTRA_TARGETS) -> scoring buckets
STAT_BUCKET = {
    "passing_yards": "pass_yd", "passing_tds": "pass_td", "passing_interceptions": "int",
    "qb_rushing_yards": "rush_yd", "qb_rushing_tds": "rush_td",
    "rushing_yards": "rush_yd", "rushing_tds": "rush_td",
    "rb_receptions": "rec", "rb_receiving_yards": "rec_yd", "rb_receiving_tds": "rec_td",
    "receptions": "rec", "receiving_yards": "rec_yd", "receiving_tds": "rec_td",
}
NOT_PROJECTED = ["fumbles lost", "two-point conversions", "return yards and touchdowns",
                 "rushing yards by wide receivers and tight ends", "receiving by quarterbacks"]


@dataclass
class LeagueSettings:
    name: str = "Full PPR"
    rec: float = 1.0                # points per reception
    pass_td: float = 4.0
    pass_yd: float = 0.04           # 1 per 25
    int_: float = -2.0
    rush_yd: float = 0.1            # 1 per 10
    rush_td: float = 6.0
    rec_yd: float = 0.1
    rec_td: float = 6.0
    te_rec_bonus: float = 0.0       # TE premium leagues add to `rec` for tight ends
    # lineup: slots that decide which positions compete for the same spot
    slots: dict = field(default_factory=lambda: {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1})
    flex_positions: tuple = ("RB", "WR", "TE")
    superflex: bool = False         # a second QB-eligible slot

    def to_dict(self):
        d = asdict(self)
        d["int"] = d.pop("int_")
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        if "int" in d:
            d["int_"] = d.pop("int")
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "slots" in known and isinstance(known["slots"], dict):
            known["slots"] = {k: int(v) for k, v in known["slots"].items()}
        if "flex_positions" in known:
            known["flex_positions"] = tuple(known["flex_positions"])
        return cls(**known)


PRESETS = {
    "full_ppr": LeagueSettings(),
    "half_ppr": LeagueSettings(name="Half PPR", rec=0.5),
    "standard": LeagueSettings(name="Standard (no PPR)", rec=0.0),
    "six_pt_pass_td": LeagueSettings(name="Full PPR, 6-pt passing TD", pass_td=6.0),
}
DEFAULT = "full_ppr"


def points(stats, settings, position=None):
    """
    Fantasy points from a dict of projection keys -> values. Missing keys count as zero
    because the projection did not produce them (a QB has no receptions key), which is a
    different thing from an actual result being missing; results are never scored here.
    """
    s = settings if isinstance(settings, LeagueSettings) else LeagueSettings.from_dict(settings)
    rec_pts = s.rec + (s.te_rec_bonus if position == "TE" else 0.0)
    w = {"pass_yd": s.pass_yd, "pass_td": s.pass_td, "int": s.int_, "rush_yd": s.rush_yd,
         "rush_td": s.rush_td, "rec": rec_pts, "rec_yd": s.rec_yd, "rec_td": s.rec_td}
    total = 0.0
    for key, val in (stats or {}).items():
        b = STAT_BUCKET.get(key)
        if b is None or val is None:
            continue
        total += w[b] * float(val)
    return round(total, 2)


def actual_points(row, settings, position=None):
    """
    Points from a weekly box-score row (nflverse columns) under the given settings. Used to
    grade against custom scoring; under Full PPR nflverse's own fantasy_points_ppr is the
    reference and includes what the projection cannot (fumbles, two-point conversions).
    """
    s = settings if isinstance(settings, LeagueSettings) else LeagueSettings.from_dict(settings)
    g = lambda k: float(row.get(k) or 0.0) if row.get(k) == row.get(k) else 0.0   # NaN-safe
    rec_pts = s.rec + (s.te_rec_bonus if position == "TE" else 0.0)
    return round(s.pass_yd * g("passing_yards") + s.pass_td * g("passing_tds") + s.int_ * g("passing_interceptions")
                 + s.rush_yd * g("rushing_yards") + s.rush_td * g("rushing_tds")
                 + rec_pts * g("receptions") + s.rec_yd * g("receiving_yards") + s.rec_td * g("receiving_tds")
                 - 2.0 * (g("sack_fumbles_lost") + g("rushing_fumbles_lost") + g("receiving_fumbles_lost"))
                 + 2.0 * (g("passing_2pt_conversions") + g("rushing_2pt_conversions") + g("receiving_2pt_conversions")), 2)


def lineup_slots(settings):
    """Which positions are eligible for each slot, in the order a lineup is filled."""
    s = settings if isinstance(settings, LeagueSettings) else LeagueSettings.from_dict(settings)
    out = []
    for slot, n in s.slots.items():
        elig = (list(s.flex_positions) if slot == "FLEX" else
                ["QB"] + list(s.flex_positions) if slot == "SUPERFLEX" else [slot])
        for _ in range(int(n)):
            out.append({"slot": slot, "eligible": elig})
    if s.superflex and not any(x["slot"] == "SUPERFLEX" for x in out):
        out.append({"slot": "SUPERFLEX", "eligible": ["QB"] + list(s.flex_positions)})
    return out
