"""
Stadium database and roof logic.

Coordinates are the stadium's own (cross-checked between greerreNFL/stadiums and Wikipedia
infoboxes; they agree to the third decimal), never a city centroid. Keyed by nflverse
`stadium_id`, which is what games.csv carries for every game including international ones.
"""
import json
import os

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stadiums.json")
_DB = None


def load():
    global _DB
    if _DB is None:
        _DB = json.load(open(_PATH))["stadiums"]
    return _DB


def get(stadium_id):
    return load().get(stadium_id)


def for_team(team):
    for s in load().values():
        if team in s.get("teams", []):
            return s
    return None


def roof_status(stadium, game_roof=None):
    """
    What is known about the roof for THIS game.

      indoor          fixed dome; outside weather does not reach the field
      open            open-air stadium
      closed / open   retractable, and an authoritative source (nflverse games.csv `roof`,
                      filled once the game is played or the league announces it) says which
      pending         retractable, state not yet known. Never assumed either way.
    """
    if not stadium:
        return {"status": "unknown", "label": "Venue unknown", "outdoor_weather_applies": None}
    rt = stadium.get("roof_type")
    if rt == "fixed_dome":
        return {"status": "indoor", "label": "Indoor / climate controlled", "outdoor_weather_applies": False}
    if rt == "retractable":
        gr = str(game_roof or "").strip().lower()
        if gr == "closed":
            return {"status": "closed", "label": "Roof closed (reported)", "outdoor_weather_applies": False}
        if gr == "open":
            return {"status": "open", "label": "Roof open (reported)", "outdoor_weather_applies": True}
        return {"status": "pending", "label": "Roof status pending", "outdoor_weather_applies": None}
    return {"status": "open", "label": "Open-air stadium", "outdoor_weather_applies": True}
