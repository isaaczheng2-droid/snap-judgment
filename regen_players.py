#!/usr/bin/env python3
"""
Re-score the player projections and emit the two blocks the site reads, so the published
numbers describe the model that actually runs.

This exists because `regen_accuracy.py` never touched them: `player_backtest` and
`accuracy.player_accuracy` were static blocks carried forward from an old run, and they
survived a change to the player model unaltered. That is exactly how a site ends up
reporting numbers for code it no longer runs.

Two baselines are reported for every stat, and the second is the one that matters:

  flat        that position's league average every week. Easy to beat, and it is the only
              one the site used to show.
  naive       a rolling average of that player's own recent games. This is the first thing
              anyone would try, and until usage features were added the model LOST to it on
              four of nine stats.

Walk-forward by season: for test season S the ridge sees only seasons < S.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import run_pipeline as rp

PRETTY = {
    "passing_yards": ("Passing yards", 50), "passing_tds": ("Passing TDs", 1),
    "qb_rushing_yards": ("QB rushing yards", 15), "rushing_yards": ("Rushing yards", 25),
    "rushing_tds": ("Rushing TDs", 1), "rb_receiving_yards": ("RB receiving yards", 20),
    "receiving_yards": ("Receiving yards", 25), "receptions": ("Receptions", 2),
    "receiving_tds": ("Receiving TDs", 1),
}


def main():
    sched, team, plyr, inj, snap, rost, depth, cur = rp.load_all("data", None)
    reg = sched[(sched.season == cur) & (sched.game_type == "REG")]
    un = reg[reg.home_score.isna()]
    tw = int(un.week.min()) if len(un) else int(reg.week.max())
    ratings, _ = rp.team_ratings(team, sched, cur, tw)
    pw = rp.player_form(plyr, sched, ratings)      # now carries tgt_share / car_share
    tests = list(range(2019, int(cur)))

    acc, back = [], []
    for key, cfg in rp.PTARGETS.items():
        oc, pc = rp.OPPCOL[cfg["opp"]], f"proj_{cfg['stat']}"
        feats = [pc, oc, "is_home"] + rp.USAGE
        sub = pw[pw.position.isin(cfg["pos"])].dropna(subset=[cfg["stat"]] + feats)
        sub = sub[sub[cfg["vol"]] >= cfg["mn"]]
        if len(sub) < 500:
            continue

        pred, naive, act = [], [], []
        for s in tests:
            tr, te = sub[sub.season < s], sub[sub.season == s]
            if len(tr) < 300 or not len(te):
                continue
            m = Ridge(alpha=5.0).fit(tr[feats], tr[cfg["stat"]])
            pred.append(m.predict(te[feats]))
            naive.append(te[pc].values)            # the player's own prior-games mean
            act.append(te[cfg["stat"]].values)
        if not pred:
            continue
        p, nv, a = map(np.concatenate, (pred, naive, act))
        flat = np.full_like(a, a.mean(), dtype=float)   # league average every week

        label, band = PRETTY.get(key, (key.replace("_", " ").capitalize(), 1))
        mae = float(np.abs(p - a).mean())
        flat_mae = float(np.abs(flat - a).mean())
        acc.append({
            "stat": label, "n": int(len(a)), "actual_mean": float(a.mean()),
            "mae": mae, "naive_mae": float(np.abs(nv - a).mean()), "flat_mae": flat_mae,
            "rmse": float(np.sqrt(((p - a) ** 2).mean())),
            "corr": float(np.corrcoef(p, a)[0, 1]),
            "bias": float((p - a).mean()),
            "within_band": float((np.abs(p - a) <= band).mean()), "band": band,
            "lift_vs_flat": float((flat_mae - mae) / flat_mae),
        })
        back.append({"stat": key, "adjusted_model": mae,
                     "naive_rolling_avg": float(np.abs(nv - a).mean())})
        beat = (back[-1]["naive_rolling_avg"] - mae) / back[-1]["naive_rolling_avg"]
        print(f"  {label:<20} n={len(a):>6}  model {mae:>7.3f}  naive {back[-1]['naive_rolling_avg']:>7.3f}"
              f"  vs naive {beat:+.2%}   vs flat {acc[-1]['lift_vs_flat']:+.1%}")

    json.dump({"player_accuracy": acc, "player_backtest": back},
              open("player_patch.json", "w"), indent=1)
    n_beat = sum(1 for b in back if b["naive_rolling_avg"] > b["adjusted_model"])
    print(f"\nbeats a rolling average on {n_beat} of {len(back)} stats")
    print("wrote player_patch.json")


if __name__ == "__main__":
    main()
