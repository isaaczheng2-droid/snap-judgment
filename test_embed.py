#!/usr/bin/env python3
"""
The payload is JSON pasted inside a <script> block, and that is not the same as JSON.

An HTML parser scanning a script element stops at the literal characters `</script`,
without caring that they sit inside a quoted JSON string. One such string in the payload
therefore ends the script early and the rest of the application gets parsed as page text.

This was a live vulnerability, found by accident. The injury notes now published on the
page are wire copy from a third party. A test note containing a <script> tag broke the
build immediately: the page half-rendered with raw `${...}` template source on screen and
two "Invalid or unexpected token" errors in the console. Escaping the note before display
did not help and could not have -- the failure happens in the HTML parser, before a single
line of JavaScript runs.

The hostile string lives HERE and not in the ESPN fixture, because that fixture feeds real
builds and nothing fabricated should be able to reach a published page.

Two layers are checked, because either alone is insufficient:
  - embed()/merge_payload escaping, which keeps the document parseable at all
  - escHtml() in the page, which keeps note text from becoming markup once it renders
"""
import json
import subprocess
import sys
import tempfile
import os

import build_site

HOSTILE = {
    "script_close": 'Jones </script><script>window.__pwned=1</script> is questionable.',
    "case_variant": "Smith </SCRIPT> is out.",
    "entities": 'Brown is "day-to-day" & <b>limited</b>.',
    "sep": "Davis is out. Line separator here. And a paragraph one.",
}
fails = []


def check(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"   {d}" if d else ""))
    if not c:
        fails.append(n)


def main():
    print("\nEMBEDDING -- the payload must not be able to close its own script tag")
    blob = build_site.embed({"notes": HOSTILE})
    check("no literal '</script' survives", "</script" not in blob.lower(), blob[:60])
    check("no bare '<' survives at all", "<" not in blob)
    check("U+2028 is escaped", " " not in blob)
    check("U+2029 is escaped", " " not in blob)

    print("\n...while still being the same data")
    back = json.loads(blob)
    check("it is valid JSON", isinstance(back, dict))
    check("every string round-trips byte-identically", back["notes"] == HOSTILE,
          str(back["notes"])[:60])

    print("\nTHE PRODUCTION PATH -- merge_payload.py, which the hourly workflow runs")
    page = ('<html><body><script>const DATA = /*PAYLOAD_START*/{"a":1}/*PAYLOAD_END*/;'
            '\nwindow.__ok = true;</script></body></html>')
    with tempfile.TemporaryDirectory() as d:
        hp, pp = os.path.join(d, "p.html"), os.path.join(d, "p.json")
        open(hp, "w").write(page)
        json.dump({"games": [{"note": HOSTILE["script_close"]}]}, open(pp, "w"))
        r = subprocess.run([sys.executable, "merge_payload.py", hp, pp],
                           capture_output=True, text=True)
        check("merge_payload ran", r.returncode == 0, r.stderr[-160:])
        out = open(hp).read()
        body = out[out.index("/*PAYLOAD_START*/"):out.index("/*PAYLOAD_END*/")]
        check("the merged payload cannot close the script tag",
              "</script" not in body.lower())
        check("the script tag is still closed exactly once",
              out.lower().count("</script>") == 1, str(out.lower().count("</script>")))
        check("the code after the payload is still inside the script",
              "window.__ok = true;" in out.split("</script>")[0])

    print("\nEND TO END -- render a page carrying hostile note text")
    try:
        import asyncio
        from playwright.async_api import async_playwright
    except Exception:
        print("  SKIP  playwright unavailable")
        print(f"\n{len(fails)} failed" if fails else "\nall checks passed")
        return 1 if fails else 0

    src = json.load(open("data/dashboard_payload.json"))
    hit = 0
    for g in src.get("games", []):
        for side in ("home", "away"):
            for p in g.get("injuries", {}).get(side, []):
                p["note"] = HOSTILE["script_close"]
                p["src"], p["updated"] = "espn", "2026-09-11T00:00Z"
                hit += 1
    check("found injury rows to poison", hit > 0, f"{hit} rows")
    html = (open("build/app.html").read()
            .replace("__PAYLOAD__", build_site.embed(src))
            .replace("__TEAM_META__", build_site.embed(json.load(open("data/team_meta.json")))))

    async def run():
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "hostile.html")
            open(f, "w").write(html)
            async with async_playwright() as pw:
                b = await pw.chromium.launch()
                pg = await b.new_page()
                errs = []
                pg.on("pageerror", lambda e: errs.append(str(e)))
                await pg.goto("file://" + f)
                await pg.wait_for_timeout(1500)
                # the page opens on the home screen; the game cards are on #/games
                await pg.evaluate("location.hash = '#/games'")
                await pg.wait_for_timeout(400)
                cards = await pg.evaluate("document.querySelectorAll('.gcard').length")
                # v2 routes: the injury report lives on the matchup page's Injuries tab
                await pg.evaluate(
                    "location.hash = '#/games/' + DATA.games[0].game_id + '/injuries'")
                await pg.wait_for_timeout(600)
                pwned = await pg.evaluate("!!window.__pwned")
                ntag = await pg.evaluate("document.querySelectorAll('.injquote script').length")
                shown = await pg.evaluate(
                    "[...document.querySelectorAll('.injquote')].map(e=>e.textContent).join('')")
                await b.close()
                return errs, pwned, ntag, shown, cards

    errs, pwned, ntag, shown, cards = asyncio.run(run())
    check("the page still parsed and rendered", cards > 0, f"{cards} cards")
    check("no JavaScript errors", not errs, "; ".join(errs[:1]))
    check("the injected script did NOT execute", not pwned)
    check("no script element was built from note text", ntag == 0)
    check("the note is displayed as literal text", "</script>" in shown, shown[:70])

    print(f"\n{len(fails)} failed" if fails else "\nall checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
