"""
Canonical team identifiers and vendor mappings.

The canonical code is nflverse's (LA, WAS, ...). Every vendor abbreviation is translated at
the ingestion boundary through `to_canonical`; nothing downstream sees a vendor code.
"""
CANONICAL = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
             "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ",
             "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"]

# vendor -> {vendor_code: canonical}. Only the codes that differ need listing; identical codes
# pass through. Keep every vendor here, even the ones not integrated yet, so the day one is
# added the mapping is already in one place.
VENDOR = {
    "espn":         {"LAR": "LA", "WSH": "WAS"},
    "nflverse":     {},
    "sportsdataio": {"LAR": "LA", "WAS": "WAS", "JAC": "JAX"},
    "sportradar":   {"LAR": "LA", "WAS": "WAS", "JAC": "JAX"},
    "pfr":          {"RAM": "LA", "RAI": "LV", "SDG": "LAC", "OTI": "TEN", "CLT": "IND",
                     "CRD": "ARI", "HTX": "HOU", "NWE": "NE", "GNB": "GB", "KAN": "KC",
                     "NOR": "NO", "SFO": "SF", "TAM": "TB", "RAV": "BAL"},
    "nfl":          {"LAR": "LA", "WSH": "WAS", "JAC": "JAX"},
}
# older canonical spellings that still appear in historical data
LEGACY = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "WSH": "WAS", "JAC": "JAX"}

NAMES = {"ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
         "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
         "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
         "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
         "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
         "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
         "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
         "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
         "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
         "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
         "TEN": "Tennessee Titans", "WAS": "Washington Commanders"}


def to_canonical(code, vendor="nflverse"):
    """Vendor abbreviation -> canonical code, or None if it is not an NFL team."""
    if code is None:
        return None
    c = str(code).strip().upper()
    c = VENDOR.get(vendor, {}).get(c, c)
    c = LEGACY.get(c, c)
    return c if c in CANONICAL else None


def is_canonical(code):
    return code in CANONICAL
