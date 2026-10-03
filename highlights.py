#!/usr/bin/env python3
"""
Write media/highlights.json: the reel the intro screen plays.

The reel is the teams' own cinematic film: hype videos, game trailers and "Sights and
Sounds" pieces, shot from the sideline and the field rather than cut from the broadcast,
so there is no score bug on them. 2020 onward only.

Nothing is downloaded or re-hosted; the page embeds YouTube's own player. The catch is that
a lot of NFL footage on YouTube reports itself as embeddable and then refuses to play on
any outside site ("blocked from display on this website"). The league's own channels are
blocked wholesale, a few clubs block everything, most block some, and nothing in a listing
says which. So every candidate is checked against the embedded player itself, with an
outside referrer, exactly as a visitor's browser would, and only videos that really play
are listed, two a team at most, newest first. The page opens on the newest and mixes the rest.

No API key. Any .mp4/.webm dropped into media/ is listed too, and the page plays those
instead of the YouTube reel while they are there.

A bad day upstream never blanks the screen: if too little is found, the existing reel stays.
"""
import datetime, glob, json, os, re, sys, urllib.error, urllib.parse, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "media", "highlights.json")
# nickname -> channel handles to try; a handle only counts if the channel's own name carries the nickname
TEAMS = {
    "Cardinals": ["AZCardinals"], "Falcons": ["AtlantaFalcons"], "Ravens": ["BaltimoreRavens", "Ravens"],
    "Bills": ["BuffaloBills"], "Panthers": ["Panthers", "CarolinaPanthers"], "Bears": ["ChicagoBears"],
    "Bengals": ["Bengals"], "Browns": ["Browns"], "Cowboys": ["DallasCowboys"], "Broncos": ["Broncos", "DenverBroncos"],
    "Lions": ["DetroitLionsNFL", "DetroitLions"], "Packers": ["packers"], "Texans": ["HoustonTexans"], "Colts": ["Colts"],
    "Jaguars": ["Jaguars"], "Chiefs": ["KansasCityChiefs"], "Raiders": ["Raiders"], "Chargers": ["chargers"],
    "Rams": ["LARams", "RamsNFL"], "Dolphins": ["MiamiDolphins"], "Vikings": ["vikings"], "Patriots": ["Patriots"],
    "Saints": ["NewOrleansSaints", "Saints"], "Giants": ["NewYorkGiants", "nygiants", "Giants"], "Jets": ["nyjets"],
    "Eagles": ["Eagles"], "Steelers": ["steelers"], "49ers": ["49ers"], "Seahawks": ["Seahawks"],
    "Buccaneers": ["Buccaneers"], "Titans": ["Titans"], "Commanders": ["Commanders"],
}
QUERIES = ["hype video", "sights and sounds", "game trailer"]
SINCE = 2020                                   # nothing uploaded before this season
PER_TEAM = 2                                   # playable videos kept per team: the reel is about variety
CHECKS_PER_TEAM = 10                           # candidates tried per team before moving on
MAX_ITEMS = 90
MIN_ITEMS = 8                                  # fewer than this means the run went wrong: keep the old reel
MIN_SECONDS, MAX_SECONDS = 25, 30 * 60
WANT = re.compile(r"hype|trailer|cinematic|sights\s*(?:and|&|x|\+|n)\s*sounds|all.?access", re.I)
SKIP = re.compile(r"practice|camp\b|\bota\b|minicamp|draft|press|interview|podcast|react|madden|cheer|combine|schedule release|"
                  r"uniform|jersey|high school|\blive\b|#shorts|booth|radio", re.I)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
      "Accept-Language": "en-US,en;q=0.9", "Referer": "https://example.com/"}
RESULT = re.compile(r'"videoRenderer":\{"videoId":"([\w-]{11})".{0,2500}?"title":\{"runs":\[\{"text":"((?:[^"\\]|\\.)*)"\}'
                    r'.{0,2500}?"lengthText":\{.{0,200}?"simpleText":"([\d:]+)"', re.S)
AGE = re.compile(r'"publishedTimeText":\{"simpleText":"(?:Streamed )?(\d+)\s*([a-z]+) ago"')   # "5y ago", "9mo ago"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


_plain = urllib.request.build_opener(_NoRedirect)


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def playable(vid):
    """Ask the embedded player, with an outside referrer, exactly as a visitor's browser would."""
    m = re.search(r"previewPlayabilityStatus\W+status\W+(\w+)", get("https://www.youtube.com/embed/" + vid))
    return bool(m) and m.group(1) == "OK"


def vertical(vid):
    """A phone-format video answers at /shorts/<id>; an ordinary one redirects away to /watch."""
    try:
        _plain.open(urllib.request.Request("https://www.youtube.com/shorts/" + vid, headers=UA, method="HEAD"), timeout=20)
        return True
    except urllib.error.HTTPError as e:
        return e.code not in (301, 302, 303, 307, 308)


def seconds(text):
    n = 0
    for part in text.split(":"):
        n = n * 60 + int(part)
    return n


def uploaded(block, title, today):
    """Roughly when it went up, as "YYYY-MM", from YouTube's "3y ago" / "9mo ago"; None if before SINCE or unknown."""
    years = [int(y) for y in re.findall(r"\b(20[0-2]\d)\b", title)]
    if years and max(years) < SINCE:
        return None
    age = AGE.search(block)
    if not age:
        return f"{max(years)}-01" if years else None    # no upload age shown: only trust a title that names its year
    n, unit = int(age.group(1)), age.group(2)
    days = n * (365 if unit.startswith("y") else 30 if unit.startswith("mo") else 7 if unit.startswith("w") else 1 if unit.startswith("d") else 0)
    when = today - datetime.timedelta(days=days)
    return when.strftime("%Y-%m") if when.year >= SINCE else None


def channel(team, handles):
    """The first handle that really is this team's channel."""
    for handle in handles:
        try:
            html = get(f"https://www.youtube.com/@{handle}/search?query=" + urllib.parse.quote(QUERIES[0]))
        except urllib.error.HTTPError:
            continue
        name = re.search(r"<title>(.*?)</title>", html)
        if name and team.lower() in name.group(1).lower():
            return handle, html
    return None, None


def team_videos(team, handles, today):
    handle, first = channel(team, handles)
    if not handle:
        return None, []
    kept, tried, seen = [], 0, set()
    per_query = max(1, PER_TEAM // len(QUERIES)) + 1    # spread a team's picks across the kinds of film
    for n, query in enumerate(QUERIES):
        html = first if n == 0 else get(f"https://www.youtube.com/@{handle}/search?query=" + urllib.parse.quote(query))
        got = 0
        for m in RESULT.finditer(html):
            vid, length = m.group(1), seconds(m.group(3))
            try:
                title = json.loads('"' + m.group(2) + '"').strip()
            except ValueError:
                continue
            # this result's own text: from its start to wherever the next result begins
            nxt = html.find('"videoRenderer"', m.end())
            block = html[m.start():nxt if nxt != -1 else m.end() + 3000]
            when = uploaded(block, title, today)
            if vid in seen or not MIN_SECONDS <= length <= MAX_SECONDS or not WANT.search(title) or SKIP.search(title) or not when:
                continue
            seen.add(vid)
            tried += 1
            if playable(vid) and not vertical(vid):
                kept.append({"type": "youtube", "id": vid, "title": title, "source": team, "seconds": length, "uploaded": when})
                got += 1
            if got >= per_query or len(kept) >= PER_TEAM or tried >= CHECKS_PER_TEAM:
                break
        if len(kept) >= PER_TEAM or tried >= CHECKS_PER_TEAM:
            break
    return handle, kept


def youtube_items():
    per_team, today = [], datetime.date.today()
    for team, handles in TEAMS.items():
        try:
            handle, kept = team_videos(team, handles, today)
        except Exception as e:                  # one team down is not a reason to drop the others
            print(f"  {team}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        print(f"  {team}: {'no channel found' if handle is None else f'{len(kept)} playable (@{handle})'}")
        if kept:
            per_team.append(kept)
    # newest first: the page opens on the most recent film and mixes the rest
    out = sorted((v for kept in per_team for v in kept), key=lambda v: v["uploaded"], reverse=True)
    return out[:MAX_ITEMS]


def local_items():
    files = sorted(f for ext in ("*.mp4", "*.webm") for f in glob.glob(os.path.join(ROOT, "media", ext)))
    for f in files:
        mb = os.path.getsize(f) / 1e6
        if mb > 95:
            print(f"warning: {os.path.basename(f)} is {mb:.0f} MB; GitHub rejects files over 100 MB", file=sys.stderr)
    local = [{"type": "file", "src": "media/" + os.path.basename(f),
              "title": os.path.splitext(os.path.basename(f))[0].replace("_", " ").replace("-", " ")} for f in files]
    # media/pinned.json: clips hosted elsewhere that we are free to play. Each is {"src", "title"} plus an
    # optional window: "tail" (last N seconds), "start"/"end" (seconds), "length" (seconds from the start point).
    try:
        pinned = [dict(p, type="file") for p in json.load(open(os.path.join(ROOT, "media", "pinned.json"), encoding="utf-8"))
                  if isinstance(p, dict) and str(p.get("src", "")).startswith("https://")]
    except (OSError, ValueError):
        pinned = []
    return local + pinned


def main():
    try:
        old = json.load(open(OUT, encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    yt = youtube_items()
    if len(yt) < MIN_ITEMS:
        print(f"only {len(yt)} playable videos found; keeping the existing reel", file=sys.stderr)
        yt = [i for i in old.get("items", []) if i.get("type") == "youtube"] or yt
    new = {"items": local_items() + yt}
    if old.get("items") == new["items"]:
        print(f"highlights unchanged ({len(new['items'])} items)")
        return
    new["updated"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(new, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {OUT}: {len(new['items'])} items from {len({i.get('source') for i in yt})} teams")


if __name__ == "__main__":
    main()
