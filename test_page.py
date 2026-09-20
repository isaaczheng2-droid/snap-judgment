#!/usr/bin/env python3
"""
Page checks, run headless against site/index.html:
  1. every displayed probability, score and margin on a game page equals the forecast record
     in the embedded payload (presentation changes cannot alter values)
  2. the game page order is matchup, standalone probability, labelled score, data status
  3. the market comparison is present, labelled, and secondary
  4. the mascot's state is exactly what the frozen tracker rows imply (recomputed here)
  5. every route renders with no JavaScript errors, on desktop and on a phone viewport
  6. keyboard: Tab reaches the nav and the first game link; focus is visible; skip nothing
  7. no market blend in the page code; reduced-motion rules exist; every image has alt text
  8. contrast of ink on ivory and the accent on ivory clears WCAG AA
"""
import asyncio, json, math, re, sys
from playwright.async_api import async_playwright

URL = "file:///home/claude/nfl_model/site/index.html"
FAILS = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def lum(hexs):
    r, g, b = [int(hexs[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(a, b):
    la, lb = lum(a), lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def mood_py(payload):
    """the same rule as js_home.moodSummary, written independently"""
    T = payload["tracker"]; S = T["by_season"][str(T["season"])]["games"]
    rows = [r for r in S["rows"] if r.get("ok") is not None and r.get("pm") is not None and r.get("pk") is not None and r.get("hs") is not None]
    if payload.get("health", {}).get("ok") is False:
        return "maintenance"
    if len(rows) < 16:
        return "studying"
    win = rows[-32:]
    y = [1 if r["hs"] > r["as"] else 0 for r in win]
    d = [(r["pk"] - yy) ** 2 - (r["pm"] - yy) ** 2 for r, yy in zip(win, y)]
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1))
    se = sd / math.sqrt(len(d))
    if m - 1.96 * se > 0: return "celebrating"
    if m + 1.96 * se < 0: return "reviewing"
    return "calm"


async def main():
    html = open("site/index.html").read()
    payload = json.loads(html.split("/*PAYLOAD_START*/")[1].split("/*PAYLOAD_END*/")[0])
    check("p_blend" not in html.split("/*PAYLOAD_END*/")[1], "no p_blend reference survives in the page code")
    check("BLEND_W" not in html and "0.2 *" not in html.split("/*PAYLOAD_END*/")[1], "no blend weight in the page code")
    check("prefers-reduced-motion" in html, "reduced-motion rules are present")
    check(contrast("#17213a", "#f3eddf") >= 7 and contrast("#185c58", "#f3eddf") >= 4.5 and contrast("#5b6480", "#f3eddf") >= 4.5,
          f"contrast: ink {contrast('#17213a', '#f3eddf'):.1f}:1, accent {contrast('#185c58', '#f3eddf'):.1f}:1, muted {contrast('#5b6480', '#f3eddf'):.1f}:1 on ivory")
    check(contrast("#f1ebdc", "#0f1726") >= 7 and contrast("#7fcbc2", "#0f1726") >= 4.5, "contrast in dark mode clears AA")

    async with async_playwright() as p:
        b = await p.chromium.launch()
        errs = []
        ctx = await b.new_context(viewport={"width": 1360, "height": 900})
        pg = await ctx.new_page()
        pg.on("pageerror", lambda e: errs.append(str(e)))

        # 1-3: every game page against its record
        bad = []
        order_ok = True
        for g in payload["games"]:
            f = g["forecast"]
            await pg.goto(URL + f"#/games/{g['game_id']}"); await pg.wait_for_timeout(150)
            hero = await pg.inner_text(".mhero")
            facts = await pg.inner_text(".facts")
            ph, pa = round(f["p_home"] * 100), round(f["p_away"] * 100)
            hs, as_ = round(f["expected_home_score"]), round(f["expected_away_score"])
            if f"{ph}%" not in hero or f"{pa}%" not in hero: bad.append((g["game_id"], "prob"))
            if f"{g['away_team']} {as_} – {g['home_team']} {hs}" not in hero.replace("–", "–"): bad.append((g["game_id"], "score"))
            m = f["margin_home"]
            mt = "even" if abs(m) < 0.05 else f"{g['home_team'] if m > 0 else g['away_team']} by {abs(m):.1f}"
            if f"Expected margin: {mt}" not in facts: bad.append((g["game_id"], "margin", mt))
            if f["market"]["p_home_novig"] is not None and f"{round(f['market']['p_home_novig'] * 100)}%" not in facts: bad.append((g["game_id"], "market"))
            if abs(ph + pa - 100) > 1: bad.append((g["game_id"], "sum"))
            # order: band, probability, score, then strip, then facts
            body = await pg.inner_text("#matchView")
            i_prob, i_score, i_strip, i_fact = body.find("%"), body.find("EXPECTED SCORE, MODEL MEAN"), body.find("FORECAST"), body.find("WIN PROBABILITY, OUR MODEL")
            if not (0 <= i_prob < i_score < i_strip < i_fact): order_ok = False
        check(not bad, f"every game page shows exactly its forecast record: probabilities, rounded expected score, margin, market ({bad[:4]})")
        check(order_ok, "game page order: matchup, standalone probability, labelled expected score, forecast status, then analysis")
        await pg.goto(URL + "#/games/2026_02_JAX_DEN"); await pg.wait_for_timeout(150)
        facts = await pg.inner_text(".facts")
        check("MARKET, FOR COMPARISON ONLY" in facts and "WIN PROBABILITY, OUR MODEL" in facts, "market panel is labelled as comparison only, next to the model's own number")
        mk_font = await pg.evaluate("getComputedStyle(document.querySelector('.fact.market .v')).fontSize")
        md_font = await pg.evaluate("getComputedStyle(document.querySelector('.facts .fact .v')).fontSize")
        check(float(mk_font[:-2]) < float(md_font[:-2]), f"market number is visually secondary ({mk_font} vs {md_font})")
        jd = [g for g in payload["games"] if g["game_id"] == "2026_02_JAX_DEN"][0]
        check("JAX by 2.4" in facts and "toward JAX" in facts and jd["forecast"]["margin_home"] < 0 and jd["forecast"]["market"]["margin_edge_home"] < 0,
              "JAX-DEN reproduction: margin and edge both point to JAX in words, and both are negative on the home side in the record")

        # 4: mascot
        await pg.goto(URL + "#/home"); await pg.wait_for_timeout(200)
        label = await pg.get_attribute(".mascot", "aria-label")
        want = {"studying": "studying", "calm": "watching", "celebrating": "pleased", "reviewing": "reviewing", "maintenance": "maintenance"}[mood_py(payload)]
        check(label.endswith(want), f"mascot state '{want}' matches an independent recomputation from the frozen tracker rows")
        home = await pg.inner_text("#homeBody")
        check("guarantee" not in home.lower() and "lock of the week" not in home.lower(), "home page makes no guaranteed-winner claim")
        check("does not beat the market" in home or "does NOT beat" in home, "home page states the market comparison plainly")

        # 5: routes
        routes = ["#/home", "#/games", "#/games/picks", "#/games/coaches", "#/games/2026_02_JAX_DEN/analytics", "#/games/2026_02_JAX_DEN/coaching",
                  "#/games/2026_02_JAX_DEN/players", "#/games/2026_02_JAX_DEN/props", "#/games/2026_02_JAX_DEN/injuries", "#/fantasy", "#/players", "#/props",
                  "#/performance", "#/performance/history", "#/performance/props", "#/performance/health", "#/about", "#/about/method", "#/about/definitions", "#/nonsense"]
        for r in routes:
            await pg.goto(URL + r); await pg.wait_for_timeout(120)
            vis = await pg.evaluate("[...document.querySelectorAll('section[data-panel]')].filter(s=>!s.hidden).length")
            if vis != 1: errs.append(f"{r}: {vis} panels visible")
        check(not errs, f"every route renders one panel with no JavaScript errors on desktop ({errs[:3]})")
        # tabs preserved
        await pg.goto(URL + "#/games/2026_02_JAX_DEN"); await pg.wait_for_timeout(120)
        tabs = await pg.eval_on_selector_all(".mtabs a", "els => els.map(e => e.textContent.replace(/\\d+$/, '').trim())")
        check(tabs == ["Overview", "Team analytics", "Coaching", "Player projections", "Props", "Injuries"], f"matchup tabs kept: {tabs}")
        nav = await pg.eval_on_selector_all(".nav a", "els => els.map(e => e.textContent.trim())")
        check(nav == ["Home", "Games", "Fantasy", "Players", "Props", "Model Performance", "About"], f"top navigation: {nav}")

        # 6: keyboard
        await pg.goto(URL + "#/games"); await pg.wait_for_timeout(120)
        await pg.keyboard.press("Tab")
        first = await pg.evaluate("document.activeElement.className + '|' + document.activeElement.textContent.trim()")
        reached = False
        for _ in range(30):
            await pg.keyboard.press("Tab")
            cls = await pg.evaluate("document.activeElement.className")
            if "glink" in cls: reached = True; break
        outline = await pg.evaluate("getComputedStyle(document.activeElement).outlineStyle")
        check(first.startswith("wordmark") and reached, f"Tab order starts at the wordmark and reaches a game card ({first[:30]})")
        check(outline != "none", f"focused game card shows a visible focus outline ({outline})")
        imgs = await pg.evaluate("[...document.images].filter(i => !i.hasAttribute('alt')).length")
        check(imgs == 0, f"every image has an alt attribute ({imgs} without)")
        await ctx.close()

        # phone
        errs2 = []
        ctx = await b.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
        pg = await ctx.new_page(); pg.on("pageerror", lambda e: errs2.append(str(e)))
        wide = []
        for r in ["#/home", "#/games", "#/games/2026_02_JAX_DEN", "#/performance/history", "#/games/coaches", "#/props"]:
            await pg.goto(URL + r); await pg.wait_for_timeout(200)
            sw = await pg.evaluate("document.documentElement.scrollWidth")
            if sw > 392: wide.append((r, sw))
            bn = await pg.evaluate("getComputedStyle(document.querySelector('.bottomnav')).display")
            if bn == "none": errs2.append(f"{r}: no bottom nav")
        check(not errs2 and not wide, f"phone: every route renders, bottom nav present, no horizontal overflow ({errs2[:2]} {wide[:3]})")
        await ctx.close(); await b.close()

    print("\n" + ("all checks passed" if not FAILS else f"{len(FAILS)} FAILED"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
