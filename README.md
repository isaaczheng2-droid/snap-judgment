# Snap Judgment

NFL game and player projections built on [nflverse](https://nflreadr.nflverse.com) data,
with a published accuracy audit. The site is a single self-contained `index.html` — all
markup, styles, script, and prediction data in one file. No build step, no server.

GitHub Actions refreshes it every morning and pushes the result back here, so GitHub Pages
redeploys on its own. Once it's set up, nothing needs to be run by hand.

## Setup

**1. Create the repo.** On GitHub, make a new repository — public if you want Pages and
Actions free. Then, from this folder:

```bash
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git branch -M main
git push -u origin main
```

**2. Turn on Pages.** Repo → Settings → Pages → Source: *Deploy from a branch*, branch
`main`, folder `/ (root)`. Your site appears at
`https://YOUR-USERNAME.github.io/YOUR-REPO/` within a minute or two.

**3. Let Actions push.** Repo → Settings → Actions → General → Workflow permissions →
*Read and write permissions*. Without this the daily job builds fine but cannot commit.

**4. Test it.** Repo → Actions → "Refresh predictions" → *Run workflow*. It takes about
three minutes. When it finishes, the commit list should show a "Refresh predictions" commit
and the site's "Data refreshed" stamp should show the current time.

That's it. It now runs itself at 11:00 UTC (7am ET) daily.

## Using your own domain

Settings → Pages → Custom domain. Add a `CNAME` record at your registrar pointing to
`YOUR-USERNAME.github.io`, then tick *Enforce HTTPS* once the certificate provisions.

## What the daily job does

`.github/workflows/refresh.yml` runs `run_pipeline.py`, which re-downloads nflverse data
(injury reports, weather, betting lines, announced starters, and box scores for games that
have finished), rebuilds the model's features, retrains, predicts the next unplayed week,
and scores the model against completed games of the current season. `merge_payload.py` then
splices the result into `index.html`.

The merge only touches the data between the `/*PAYLOAD_START*/` and `/*PAYLOAD_END*/`
markers, so design edits and the historical accuracy analysis survive a refresh. The job
also refuses to publish if the payload comes back empty or the page fails its size and
content checks — a bad data day leaves the last good version up rather than breaking the site.

`payload.json` is committed too, so `https://YOUR-USERNAME.github.io/YOUR-REPO/payload.json`
is a usable JSON feed of the current predictions.

## Running it by hand

```bash
pip install -r requirements.txt
python run_pipeline.py --outdir . --datadir data    # ~3 min, writes payload.json
python merge_payload.py index.html payload.json     # updates index.html in place
```

`data/` is gitignored — it's about 40 MB of nflverse parquet files, re-downloaded each run.

## Editing the site

Everything lives in `index.html`. Design tokens are CSS custom properties in the `:root`
block at the top (light theme), redefined twice below for dark mode — change a colour once
there and it changes everywhere. Content is the HTML in the middle, behaviour the single
`<script>` at the bottom.

## Two things that will eventually bite you

GitHub **disables scheduled workflows after 60 days without repo activity**, and emails you
when it does. The daily commits count as activity, so this only matters if the job is
already failing. Re-enable from the Actions tab.

GitHub's **scheduler is best-effort** — scheduled runs are frequently 10–30 minutes late and
occasionally skipped entirely under load. For a daily data refresh that's harmless; don't
build anything time-critical on it.

## About the model

It does not beat the betting market: 61.2% picking winners against the market's 66.6%, and
50.4% against the spread, which is a coin flip. The Accuracy tab shows the full working —
calibration, per-season results, head-to-head against the closing line, and predicted vs.
actual team records across 1,855 games from 2019–2025, every one of them scored
walk-forward on a model that had only seen earlier seasons.

Keep that page up if you publish this. A prediction site that hides its own scorecard is
the thing this one is deliberately not.

Not betting advice.
