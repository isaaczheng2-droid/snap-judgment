#!/usr/bin/env python3
"""Splice build/v2/intro.html and build/v2/theme.html into index.html.

Each is one self-contained block (its own style, markup and script) so it can be added to,
updated in, or removed from the deployed page without a rebuild and without touching the
payload markers the hourly refresh rewrites. The intro goes right after <body>, so it is up
before the page is parsed; the theme goes right before </body>, so its styles and its home
page come after everything they replace. Run this again after editing either file; the block
between its markers is replaced, never duplicated.

    python build/v2/add_intro.py            # add or update both
    python build/v2/add_intro.py --remove   # take both out
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "..", "..", "index.html")
BLOCKS = [("INTRO", "intro.html", "<body>", "after"), ("THEME", "theme.html", "</body>", "before")]

page = open(PAGE, encoding="utf-8", newline="").read()
for name, src, anchor, side in BLOCKS:
    start, end = f"<!-- {name}_START -->", f"<!-- {name}_END -->"
    page = re.sub(re.escape(start) + r".*?" + re.escape(end) + r"\n?", "", page, flags=re.S)
    if "--remove" in sys.argv:
        continue
    block = open(os.path.join(HERE, src), encoding="utf-8").read().strip()
    assert "PAYLOAD_START" not in block and start not in block
    assert page.count(anchor) == 1, f"expected exactly one {anchor}"
    wrapped = start + "\n" + block + "\n" + end
    page = page.replace(anchor, anchor + "\n" + wrapped if side == "after" else wrapped + "\n" + anchor, 1)
assert "PAYLOAD_START" in page and "renderCharts" in page
open(PAGE, "w", encoding="utf-8", newline="").write(page)
print(f"{'removed blocks from' if '--remove' in sys.argv else 'intro and theme spliced into'} {os.path.normpath(PAGE)}: {len(page):,} bytes")
