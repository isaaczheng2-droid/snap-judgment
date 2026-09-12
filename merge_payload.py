#!/usr/bin/env python3
"""
Splice a fresh payload into the published page without disturbing its design or its
historical analysis.

  python3 merge_payload.py app.html payload.json [-o app.html]

The page carries its data as:   const DATA = /*PAYLOAD_START*/{...}/*PAYLOAD_END*/;
Only the keys the daily run regenerates are replaced (games, players, live scoring,
feature importance, timestamp). Everything static — the 2019-2025 accuracy audit — is
carried over from whatever is already in the page, so a refresh can never wipe it.
"""
import json, re, sys, argparse

# Every key the daily run regenerates. A key missing from this list is NOT an omission the
# page survives -- it silently keeps whatever was baked in at build time. That is how the
# Bets tab ended up reporting "FanDuel lines are not connected" while payload.json held 212
# live props: `games` was listed so the props themselves came through, but `props_meta` was
# not, and the tab keys off the metadata to decide whether a feed exists at all.
FRESH_KEYS = ["generated", "season", "week", "games", "players", "feature_importance",
               "backtest", "live", "scheme_league", "tracker",
               "props_meta", "prop_audit", "prop_range", "live_meta"]
START, END = "/*PAYLOAD_START*/", "/*PAYLOAD_END*/"


def extract(html):
    i, j = html.find(START), html.find(END)
    if i != -1 and j != -1:
        return html[i + len(START):j], i + len(START), j
    m = re.search(r"const DATA = (\{.*?\});\s*\n", html, re.S)   # pages published before sentinels
    if not m:
        raise SystemExit("could not locate the DATA payload in the page")
    return m.group(1), m.start(1), m.end(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html")
    ap.add_argument("payload")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()

    html = open(a.html).read()
    raw, i, j = extract(html)
    try:
        old = json.loads(raw)
    except json.JSONDecodeError:
        old = json.loads(raw.replace("NaN", "null"))
    new = json.load(open(a.payload))

    merged = dict(old)
    for k in FRESH_KEYS:
        if k in new:
            merged[k] = new[k]

    # `</script` inside a JSON string ends the script element, whatever the quoting, so the
    # rest of the application would be parsed as page text. Injury notes are third-party
    # wire copy and can contain anything. Escaping `<` is valid JSON and parses back to the
    # identical string; U+2028/9 are string-legal but are line terminators to JavaScript.
    # See build_site.embed() for the full story -- this is the same fix on the path the
    # hourly workflow actually takes.
    blob = (json.dumps(merged)
            .replace("<", "\\u003c")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029"))
    out = html[:i] + blob + html[j:]
    open(a.out or a.html, "w").write(out)
    print(f"merged {len(new.get('games', []))} games, {len(new.get('players', []))} players "
          f"into {a.out or a.html}; kept {sorted(set(old) - set(FRESH_KEYS))}")


if __name__ == "__main__":
    main()
