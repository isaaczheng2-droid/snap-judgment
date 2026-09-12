"""
Weather: two providers, one internal schema, game-window analysis, thresholds, impact.

  PRIMARY   Open-Meteo forecast API. Free for non-commercial use, no key, 10,000 calls/day,
            CC-BY 4.0 (attribution shown on the site). Hourly, 16-day horizon, model runs
            refreshed every 1-6 h. Only the variables the site uses are requested.
  FALLBACK  National Weather Service (api.weather.gov). Public domain, free, needs a
            User-Agent, unpublished rate limit, 7-day hourly horizon, and the only source of
            official watches/warnings. Grid metadata is cached per stadium.
  OBSERVED  Open-Meteo archive API after the game (about a five-day lag) so forecasts can be
            scored against what actually happened.

Forecasts from the two providers are stored separately and never averaged. When they
disagree by more than the tolerance the game is marked FORECAST UNCERTAINTY.

Times: the schedule gives kickoff in US Eastern; the stadium row gives the venue timezone;
everything is converted to UTC here and the window is expressed in UTC.
"""
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from . import http, stadiums, store

OM_URL = "https://api.open-meteo.com/v1/forecast"
OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OM_VARS = ["temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation_probability",
           "precipitation", "rain", "snowfall", "weather_code", "wind_speed_10m", "wind_gusts_10m",
           "wind_direction_10m", "visibility"]
NWS = "https://api.weather.gov"

WINDOW_BEFORE_H = 1     # analyse from one hour before kickoff ...
WINDOW_AFTER_H = 4      # ... to four hours after (a 1 PM game: noon to 5 PM)

WMO = {0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Freezing fog",
       51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle", 56: "Freezing drizzle", 57: "Freezing drizzle",
       61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain", 67: "Freezing rain",
       71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains", 80: "Rain showers",
       81: "Rain showers", 82: "Violent rain showers", 85: "Snow showers", 86: "Heavy snow showers",
       95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with hail"}

# ------------------------------------------------------------------ thresholds
# Evidence (see live/README.md): per 10 mph of wind, passing yards fall ~7% and completion
# rate ~2.4 points (Claremont, 3,133 games); the widely used practitioner bands put the
# first visible passing decline at 10-15 mph and a step change above 20; any rain costs
# roughly a tenth of passing production, heavy snow far more; cold below about 25°F is where
# the temperature effect stops being negligible. These bands classify DISPLAY impact only.
# Nothing here changes a prediction: the game model tested weather as an input and it added
# nothing, so the honest use is context, tracking, and the backtest that will decide.
IMPACT_ORDER = ["NONE", "LOW", "MODERATE", "HIGH"]


def _bump(level, n=1):
    return IMPACT_ORDER[min(len(IMPACT_ORDER) - 1, IMPACT_ORDER.index(level) + n)]


def classify(summary, alerts=None):
    """-> (impact level, reasons list, potential effects list) from a window summary."""
    if not summary:
        return "NONE", [], []
    lvl, why, eff = "NONE", [], set()
    w, g = summary.get("max_window_wind"), summary.get("max_window_gust")
    if w is not None:
        if w >= 20: lvl, _ = "HIGH", why.append(f"sustained wind {w:.0f} mph")
        elif w >= 15: lvl, _ = max(lvl, "MODERATE", key=IMPACT_ORDER.index), why.append(f"sustained wind {w:.0f} mph")
        elif w >= 10: lvl, _ = max(lvl, "LOW", key=IMPACT_ORDER.index), why.append(f"wind {w:.0f} mph")
        if w >= 10: eff |= {"Passing efficiency", "Deep passing", "Kicking"}
    if g is not None and g >= 30:
        lvl = _bump(lvl) if g < 40 else "HIGH"
        why.append(f"gusts to {g:.0f} mph"); eff |= {"Deep passing", "Kicking", "Punting"}
    pp, pa, sn = summary.get("precipitation_probability"), summary.get("forecast_precipitation"), summary.get("snowfall")
    if sn:
        lvl = "HIGH" if sn >= 1.0 else max(lvl, "MODERATE", key=IMPACT_ORDER.index)
        why.append(f"snow {sn:.1f} in"); eff |= {"Passing efficiency", "Ball security", "Footing", "Kicking"}
    elif pp is not None and pp >= 60 and (pa or 0) >= 0.1:
        lvl = "HIGH" if (pa or 0) >= 0.25 else max(lvl, "MODERATE", key=IMPACT_ORDER.index)
        why.append(f"rain {pp:.0f}% likely, {pa:.2f} in"); eff |= {"Passing efficiency", "Ball security", "Kicking"}
    elif pp is not None and pp >= 50:
        lvl = max(lvl, "LOW", key=IMPACT_ORDER.index); why.append(f"rain {pp:.0f}% likely")
    t, fl = summary.get("kickoff_temperature"), summary.get("kickoff_feels_like")
    if t is not None:
        if t <= 10: lvl, _ = "HIGH", why.append(f"{t:.0f}°F")
        elif t <= 25: lvl, _ = max(lvl, "MODERATE", key=IMPACT_ORDER.index), why.append(f"{t:.0f}°F")
        if t <= 25: eff |= {"Passing efficiency", "Kicking"}
        if (fl or t) >= 95: lvl, _ = max(lvl, "LOW", key=IMPACT_ORDER.index), why.append(f"feels like {(fl or t):.0f}°F")
    vis = summary.get("visibility_min_mi")
    if vis is not None and vis < 1:
        lvl = max(lvl, "MODERATE", key=IMPACT_ORDER.index); why.append(f"visibility {vis:.1f} mi")
    for a in alerts or []:
        if a.get("severity") in ("Extreme", "Severe") or re.search(r"blizzard|high wind|winter storm|ice storm|severe thunderstorm|tornado|flash flood|hurricane|tropical storm", a.get("event", ""), re.I):
            lvl = "HIGH"; why.append(f"NWS {a.get('event')}"); eff |= {"Game conditions", "Kicking", "Passing efficiency"}
        elif a.get("event"):
            lvl = max(lvl, "LOW", key=IMPACT_ORDER.index); why.append(f"NWS {a.get('event')}")
    return lvl, why, sorted(eff)


# ------------------------------------------------------------------ time helpers
def kickoff_utc(gameday, gametime):
    """games.csv gameday 'YYYY-MM-DD' + gametime 'HH:MM' (US Eastern) -> aware UTC datetime."""
    if not gameday:
        return None
    try:
        hh, mm = [int(x) for x in str(gametime or "13:00").split(":")[:2]]
        local = datetime.fromisoformat(str(gameday)[:10]).replace(hour=hh, minute=mm, tzinfo=ZoneInfo("America/New_York"))
        return local.astimezone(timezone.utc)
    except Exception:
        return None


def window(kick):
    return kick - timedelta(hours=WINDOW_BEFORE_H), kick + timedelta(hours=WINDOW_AFTER_H)


def _iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ Open-Meteo
def openmeteo_url(lat, lon, tz, days):
    return (f"{OM_URL}?latitude={lat}&longitude={lon}&hourly={','.join(OM_VARS)}"
            f"&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
            f"&timezone={tz}&forecast_days={min(16, max(1, days))}")


def parse_openmeteo(data, tz):
    """Provider JSON -> list of hourly rows in the internal schema (times in UTC)."""
    h = (data or {}).get("hourly") or {}
    times = h.get("time") or []
    z = ZoneInfo(tz)
    rows = []
    for i, t in enumerate(times):
        try:
            dt = datetime.fromisoformat(t).replace(tzinfo=z).astimezone(timezone.utc)
        except Exception:
            continue
        g = lambda k: (h.get(k) or [None] * len(times))[i]
        rows.append({"time": _iso(dt), "temperature": g("temperature_2m"), "feels_like": g("apparent_temperature"),
                     "humidity": g("relative_humidity_2m"), "precip_probability": g("precipitation_probability"),
                     "precipitation": g("precipitation"), "rain": g("rain"), "snowfall": g("snowfall"),
                     "weather_code": g("weather_code"), "wind": g("wind_speed_10m"), "gust": g("wind_gusts_10m"),
                     "wind_direction": g("wind_direction_10m"),
                     "visibility_mi": (g("visibility") / 5280.0) if g("visibility") is not None else None})
    return rows


# ------------------------------------------------------------------ NWS
_DIR = {"N": 0, "NNE": 22, "NE": 45, "ENE": 67, "E": 90, "ESE": 112, "SE": 135, "SSE": 157, "S": 180,
        "SSW": 202, "SW": 225, "WSW": 247, "W": 270, "WNW": 292, "NW": 315, "NNW": 337}


def _mph(s):
    m = re.findall(r"\d+", str(s or ""))
    return max(int(x) for x in m) if m else None


def parse_nws_hourly(data):
    per = ((data or {}).get("properties") or {}).get("periods") or []
    rows = []
    for p in per:
        try:
            dt = datetime.fromisoformat(p["startTime"]).astimezone(timezone.utc)
        except Exception:
            continue
        pop = (p.get("probabilityOfPrecipitation") or {}).get("value")
        rh = (p.get("relativeHumidity") or {}).get("value")
        rows.append({"time": _iso(dt), "temperature": p.get("temperature"), "feels_like": None, "humidity": rh,
                     "precip_probability": pop, "precipitation": None, "rain": None, "snowfall": None,
                     "weather_code": None, "condition": p.get("shortForecast"), "wind": _mph(p.get("windSpeed")),
                     "gust": None, "wind_direction": _DIR.get(str(p.get("windDirection") or "").upper()),
                     "visibility_mi": None})
    return rows


def parse_nws_alerts(data):
    out = []
    for f in (data or {}).get("features") or []:
        p = f.get("properties") or {}
        if p.get("status") not in (None, "Actual"):
            continue
        out.append({"alert_id": p.get("id") or f.get("id"), "event": p.get("event"), "severity": p.get("severity"),
                    "certainty": p.get("certainty"), "urgency": p.get("urgency"), "headline": p.get("headline"),
                    "onset": p.get("onset"), "ends": p.get("ends") or p.get("expires"), "expires": p.get("expires"),
                    "sender": p.get("senderName"), "area": p.get("areaDesc")})
    return out


# ------------------------------------------------------------------ window analysis
def summarize(rows, kick):
    """Hourly rows -> the per-game summary the site stores and displays."""
    lo, hi = window(kick)
    lo = lo.replace(minute=0, second=0, microsecond=0)     # the hour containing the window start counts
    lo_s, hi_s = _iso(lo), _iso(hi)
    inw = [r for r in rows if lo_s <= r["time"] <= hi_s]
    if not inw:
        return None
    ks = _iso(kick.replace(minute=0, second=0))
    at = min(inw, key=lambda r: abs(datetime.fromisoformat(r["time"].replace("Z", "+00:00")) - kick))
    vals = lambda k: [r[k] for r in inw if r.get(k) is not None]
    mx = lambda k: max(vals(k)) if vals(k) else None
    sm = lambda k: round(sum(vals(k)), 3) if vals(k) else None
    mn = lambda k: min(vals(k)) if vals(k) else None
    code = at.get("weather_code")
    return {"window_start": lo_s, "window_end": hi_s, "kickoff_hour": ks, "hours": len(inw),
            "kickoff_temperature": at.get("temperature"), "kickoff_feels_like": at.get("feels_like"),
            "kickoff_wind": at.get("wind"), "kickoff_gust": at.get("gust"), "kickoff_wind_direction": at.get("wind_direction"),
            "max_window_wind": mx("wind"), "max_window_gust": mx("gust"),
            "precipitation_probability": mx("precip_probability"), "forecast_precipitation": sm("precipitation"),
            "snowfall": sm("snowfall"), "humidity": at.get("humidity"), "visibility_min_mi": mn("visibility_mi"),
            "weather_code": code, "condition": at.get("condition") or WMO.get(code),
            "hourly": [{k: r.get(k) for k in ("time", "temperature", "wind", "gust", "precip_probability", "precipitation", "snowfall", "weather_code")} for r in inw]}


# ------------------------------------------------------------------ comparison
TOL = {"wind": 6.0, "gust": 10.0, "precip_probability": 30.0, "temperature": 10.0}


def compare(primary, secondary):
    """Do two providers meaningfully disagree about the game window? -> (uncertain, notes)."""
    if not primary or not secondary:
        return False, []
    notes = []
    for a, b, k, tol in [("max_window_wind", "max_window_wind", "wind", TOL["wind"]),
                         ("precipitation_probability", "precipitation_probability", "rain chance", TOL["precip_probability"]),
                         ("kickoff_temperature", "kickoff_temperature", "temperature", TOL["temperature"])]:
        x, y = primary.get(a), secondary.get(b)
        if x is not None and y is not None and abs(x - y) > tol:
            notes.append(f"{k}: Open-Meteo {x:.0f}, NWS {y:.0f}")
    return bool(notes), notes


# ------------------------------------------------------------------ change detection
CHANGE = {"max_window_wind": 6.0, "max_window_gust": 10.0, "precipitation_probability": 25.0,
          "kickoff_temperature": 12.0, "snowfall": 0.1}


def diff(prev, cur):
    """Meaningful forecast changes between two summaries of the same game/provider."""
    if not prev or not cur:
        return []
    out = []
    for k, tol in CHANGE.items():
        a, b = prev.get(k), cur.get(k)
        if a is None or b is None:
            if (a or 0) != (b or 0) and k == "snowfall":
                out.append({"field": k, "before": a, "after": b})
            continue
        if abs(b - a) >= tol:
            out.append({"field": k, "before": a, "after": b})
    la, _, _ = classify(prev)
    lb, _, _ = classify(cur)
    if la != lb:
        out.append({"field": "impact", "before": la, "after": lb})
    return out


# ------------------------------------------------------------------ fetch for one game
def fetch_game(game, stadium, fixtures=None, log=print, state=None):
    """
    Pull both providers for one game and return a dict of forecasts plus alerts.
    `fixtures` = {"openmeteo": path, "nws_points": path, "nws_hourly": path, "nws_alerts": path}
    `state` caches NWS grid metadata per stadium across runs.
    """
    fixtures = fixtures or {}
    kick = kickoff_utc(game.get("gameday_iso"), game.get("kickoff"))
    out = {"game_id": game["game_id"], "stadium_id": stadium["stadium_id"], "kickoff_utc": _iso(kick) if kick else None,
           "forecasts": {}, "alerts": [], "errors": []}
    if not kick:
        out["errors"].append("no kickoff time"); return out
    days = max(1, int((kick - datetime.now(timezone.utc)).total_seconds() // 86400) + 2)
    lat, lon, tz = stadium["latitude"], stadium["longitude"], stadium["timezone"]

    # Open-Meteo
    data, info = http.get_json(openmeteo_url(lat, lon, tz, days), "openmeteo", "forecast", fixture=fixtures.get("openmeteo"), log=log)
    if data:
        s = summarize(parse_openmeteo(data, tz), kick)
        if s:
            out["forecasts"]["openmeteo"] = {**s, "provider": "openmeteo", "forecast_created_at": info["completed_at"],
                                             "fetched_at": info["completed_at"], "horizon_days": days}
        else:
            out["errors"].append("openmeteo: kickoff outside forecast horizon")
    else:
        out["errors"].append(f"openmeteo: {info.get('error')}")

    # NWS (US stadiums only)
    if stadium.get("country") in (None, "United States", "USA", "US"):
        cache = (state or {}).setdefault("nws_grid", {})
        grid = cache.get(stadium["stadium_id"])
        if not grid:
            pts, pinfo = http.get_json(f"{NWS}/points/{lat:.4f},{lon:.4f}", "nws", "points", fixture=fixtures.get("nws_points"), log=log)
            props = (pts or {}).get("properties") or {}
            if props.get("forecastHourly"):
                grid = {"forecastHourly": props["forecastHourly"], "gridId": props.get("gridId"),
                        "gridX": props.get("gridX"), "gridY": props.get("gridY"), "cached_at": store.now_iso()}
                cache[stadium["stadium_id"]] = grid
            else:
                out["errors"].append(f"nws points: {pinfo.get('error')}")
        if grid:
            hourly, hinfo = http.get_json(grid["forecastHourly"], "nws", "forecast_hourly", fixture=fixtures.get("nws_hourly"), log=log)
            if hourly:
                s = summarize(parse_nws_hourly(hourly), kick)
                if s:
                    props = (hourly.get("properties") or {})
                    out["forecasts"]["nws"] = {**s, "provider": "nws", "forecast_created_at": props.get("updateTime") or props.get("generatedAt"),
                                               "fetched_at": hinfo["completed_at"], "horizon_days": 7}
            else:
                out["errors"].append(f"nws hourly: {hinfo.get('error')}")
        al, ainfo = http.get_json(f"{NWS}/alerts/active?point={lat:.4f},{lon:.4f}", "nws", "alerts", fixture=fixtures.get("nws_alerts"), log=log)
        if al is not None:
            out["alerts"] = [{**a, "game_id": game["game_id"], "stadium_id": stadium["stadium_id"], "source": "nws",
                              "fetched_at": ainfo["completed_at"]} for a in parse_nws_alerts(al)]
        else:
            out["errors"].append(f"nws alerts: {ainfo.get('error')}")
    return out


def observed_url(lat, lon, tz, date):
    return (f"{OM_ARCHIVE}?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}"
            f"&hourly=temperature_2m,precipitation,snowfall,wind_speed_10m,wind_gusts_10m,weather_code"
            f"&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch&timezone={tz}")
