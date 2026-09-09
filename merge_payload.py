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

FRESH_KEYS = ["generated", "season", "week", "games", "players", "feature_importance",
               "backtest", "live", "scheme_league", "tracker"]
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

    out = html[:i] + json.dumps(merged) + html[j:]
    open(a.out or a.html, "w").write(out)
    print(f"merged {len(new.get('games', []))} games, {len(new.get('players', []))} players "
          f"into {a.out or a.html}; kept {sorted(set(old) - set(FRESH_KEYS))}")


if __name__ == "__main__":
    main()
