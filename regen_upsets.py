#!/usr/bin/env python3
"""
When the model calls an upset, how often is it right?

An "upset call" here means the model's own probability (before the market blend) favours the
team the closing line has as the underdog. The published pick is the 20/80 blend with the
line, so it rarely disagrees with the market; the model underneath disagrees on about a
fifth of games, and the site should say what happened the last time it did.

Same walk-forward as the audit (regen_accuracy.walk: 8-seed ensemble, test season S trained
on seasons < S), on the shipped feature set only. Writes:

  data/upsets.json and ./upsets.json    the record by band, read by run_pipeline into the
                                        payload (`upsets`), so every game card can quote it
  data/game_backtest_oof.parquet        the shipped model's out-of-fold predictions

Every number is the underdog's win rate on the games where the model called for it, next
to what the market price already implied. Reported whichever way it comes out.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

import run_pipeline as rp
from adjusted_ratings import add_adjusted_cols, team_adjusted
from elo import add_elo_cols
from scheme_features import team_scheme, add_scheme_cols
from regen_accuracy import walk, SHIPPED

MODEL_BANDS = [(0.50, 0.55, "50-55%"), (0.55, 0.60, "55-60%"), (0.60, 0.65, "60-65%"), (0.65, 1.01, "65%+")]
FAV_BANDS = [(0.50, 0.55, "near pick'em (favourite under 55%)"), (0.55, 0.65, "modest favourite (55-65%)"),
             (0.65, 0.75, "clear favourite (65-75%)"), (0.75, 1.01, "heavy favourite (75%+)")]


def band_of(p, bands):
    for lo, hi, lab in bands:
        if lo <= p < hi:
            return lab
    return None


def block(sel, dog_won, p_mkt_dog):
    n = int(sel.sum())
    if n == 0:
        return {"n": 0}
    w = int(dog_won[sel].sum())
    lo, hi = stats.binomtest(w, n, 0.5).proportion_ci(0.95) if n else (0.0, 1.0)
    return {"n": n, "dog_won": round(w / n, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "market_said": round(float(p_mkt_dog[sel].mean()), 4),
            # flat one-unit bet on the dog at the market's no-vig price, before the book's cut
            "fair_roi": round(float((dog_won[sel] / p_mkt_dog[sel]).mean() - 1), 4)}


def main():
    sched, team, plyr, inj, snap, rost, depth, cur = rp.load_all("data", None)
    reg = sched[(sched.season == cur) & (sched.game_type == "REG")]
    un = reg[reg.home_score.isna()]
    tw = int(un.week.min()) if len(un) else int(reg.week.max())
    ratings, _ = rp.team_ratings(team, sched, cur, tw)
    pw = rp.player_form(plyr, sched, ratings)
    ctx = rp.context_features(sched, pw, inj, snap, rost, depth, cur)
    df = rp.build_games(sched, ratings, ctx)
    scheme, _ = team_scheme(team, sched, cur, tw)
    df = add_scheme_cols(df, scheme)
    df = add_adjusted_cols(df, team_adjusted(team, sched, cur, tw))
    df = add_elo_cols(df)
    d = df.dropna(subset=["home_win", "home_margin"] + SHIPPED)
    tests = list(range(2019, int(cur)))

    p, m, y, mk, sp, am, sn, te = walk(d, SHIPPED, tests, keep=True)
    blend = np.where(~np.isnan(mk), rp.BLEND_W * p + (1 - rp.BLEND_W) * mk, p)
    oof = pd.DataFrame({"game_id": te.game_id.values, "season": sn, "week": te.week.values,
                        "home_team": te.home_team.values, "away_team": te.away_team.values,
                        "p_model": p, "p_market": mk, "p_blend": blend, "margin_pred": m,
                        "home_win": y, "home_margin": am, "spread_line": sp})
    oof.to_parquet("data/game_backtest_oof.parquet", index=False)

    has = ~np.isnan(mk)
    o = oof[has].copy()
    mkt_home_fav = o.p_market > 0.5
    o["p_model_dog"] = np.where(mkt_home_fav, 1 - o.p_model, o.p_model)
    o["p_blend_dog"] = np.where(mkt_home_fav, 1 - o.p_blend, o.p_blend)
    o["p_mkt_dog"] = np.minimum(o.p_market, 1 - o.p_market)
    o["dog_won"] = np.where(mkt_home_fav, o.home_win == 0, o.home_win == 1).astype(float)
    upset = (o.p_model_dog > 0.5).values
    dog_won, p_mkt_dog = o.dog_won.values, o.p_mkt_dog.values

    out = {
        "definition": "The model's own probability (before the market blend) favours the team the closing "
                      "line has as the underdog. Walk-forward, 8-seed ensemble, test season trained on earlier seasons only.",
        "n_games": int(len(o)), "n_upset_calls": int(upset.sum()), "share_of_games": round(float(upset.mean()), 4),
        "all_games_dog_won": round(float(dog_won.mean()), 4),
        "overall": block(upset, dog_won, p_mkt_dog),
        "published": block((o.p_blend_dog > 0.5).values, dog_won, p_mkt_dog),
        "by_model_band": [], "by_favorite": [], "by_season": [], "calibration": [],
        "seasons": [int(tests[0]), int(tests[-1])],
        "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }
    for lo, hi, lab in MODEL_BANDS:
        sel = upset & (o.p_model_dog.values >= lo) & (o.p_model_dog.values < hi)
        out["by_model_band"].append({"band": lab, "lo": lo, "hi": hi, **block(sel, dog_won, p_mkt_dog),
                                     "claimed": round(float(o.p_model_dog.values[sel].mean()), 4) if sel.sum() else None})
    fav = 1 - o.p_mkt_dog.values
    for lo, hi, lab in FAV_BANDS:
        sel = upset & (fav >= lo) & (fav < hi)
        base = (fav >= lo) & (fav < hi)
        out["by_favorite"].append({"band": lab, "lo": lo, "hi": hi, **block(sel, dog_won, p_mkt_dog),
                                   "all_games_dog_won": round(float(dog_won[base].mean()), 4) if base.sum() else None})
    for s in tests:
        sel = upset & (o.season.values == s)
        out["by_season"].append({"season": int(s), **block(sel, dog_won, p_mkt_dog)})
    out["calibration"] = [{"band": b["band"], "claimed": b["claimed"], "won": b.get("dog_won"), "n": b["n"]}
                          for b in out["by_model_band"]]

    for path in ["data/upsets.json", "upsets.json"]:
        json.dump(out, open(path, "w"), indent=1)
    ov = out["overall"]
    print(f"\n{out['n_upset_calls']} upset calls in {out['n_games']} games ({out['share_of_games']:.1%}); "
          f"the dog won {ov['dog_won']:.1%} (95% CI {ov['ci95'][0]:.1%}-{ov['ci95'][1]:.1%}), the market said {ov['market_said']:.1%}; "
          f"all dogs {out['all_games_dog_won']:.1%}")
    for b in out["by_model_band"]:
        print(f"  model dog {b['band']:<7} n={b['n']:4d}  won {b.get('dog_won', 0):.1%}  market said {b.get('market_said', 0):.1%}  fair ROI {b.get('fair_roi', 0):+.1%}")
    for b in out["by_favorite"]:
        print(f"  {b['band']:<40} n={b['n']:4d}  won {b.get('dog_won', 0):.1%}  (all games in band {b['all_games_dog_won']:.1%})")
    pb = out["published"]
    print(f"  published pick (blend) against the market: n={pb['n']} dog won {pb.get('dog_won', 0):.1%}")
    print("wrote data/upsets.json, upsets.json, data/game_backtest_oof.parquet")


if __name__ == "__main__":
    main()
