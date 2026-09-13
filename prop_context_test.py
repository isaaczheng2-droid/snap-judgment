#!/usr/bin/env python3
"""
Do football-context features improve the player projections on games they have not seen?

Same walk-forward harness as the shipped models (Ridge, trained on seasons < s, scored on
season s), same rows, one difference at a time:

  base      projection = f(own rolling form, opponent split defense, home, usage shares)   [shipped]
  +script   + implied team total, own spread (favoured/underdog), game total
  +offense  + own team offensive pass/rush efficiency rating
  +both

Selection is done on 2019-2024. 2025 is reported separately and was not used to pick
anything. The metric is MAE relative to the shipped model, and the share of rows where the
new projection would take the other side of the rolling-median line (so we know whether it
changes decisions, not only errors).
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import run_pipeline as rp
from test_prop_edge import half_point, NOT_OVER_UNDER

OU = [k for k in rp.PTARGETS if k not in NOT_OVER_UNDER]


def main():
    sched, team, plyr, inj, snap, rost, depth, cur = rp.load_all("data", None)
    reg = sched[(sched.season == cur) & (sched.game_type == "REG")]
    un = reg[reg.home_score.isna()]
    tw = int(un.week.min()) if len(un) else int(reg.week.max())
    ratings, _ = rp.team_ratings(team, sched, cur, tw)
    pw = rp.player_form(plyr, sched, ratings).sort_values(["player_id", "season", "week"])
    # game script inputs from the schedule (closing spread/total; both exist well before kickoff)
    g = sched[["game_id", "home_team", "spread_line", "total_line"]].copy()
    pw = pw.merge(g, on=["game_id", "home_team"], how="left")
    home = pw.team == pw.home_team
    pw["own_spread"] = np.where(home, pw.spread_line, -pw.spread_line)          # + = favoured by this many
    pw["implied_total"] = np.where(home, (pw.total_line + pw.spread_line) / 2, (pw.total_line - pw.spread_line) / 2)
    pw["game_total"] = pw.total_line
    # own offense ratings (pre-game, from the ratings table)
    own = ratings[["game_id", "team", "rating_g_off_pass_epa_pp", "rating_g_off_rush_epa_pp"]].rename(
        columns={"rating_g_off_pass_epa_pp": "own_off_pass", "rating_g_off_rush_epa_pp": "own_off_rush"})
    pw = pw.merge(own, on=["game_id", "team"], how="left")
    for c in ["own_spread", "implied_total", "game_total", "own_off_pass", "own_off_rush"]:
        pw[c] = pw[c].fillna(pw[c].median())
    for key, cfg in rp.PTARGETS.items():
        st = cfg["stat"]
        pw[f"med_{st}"] = pw.groupby(["player_id", "season"])[st].transform(lambda x: x.shift(1).expanding().median())

    SETS = {
        "base": [],
        "+script": ["implied_total", "own_spread", "game_total"],
        "+offense": ["own_off_pass", "own_off_rush"],
        "+both": ["implied_total", "own_spread", "game_total", "own_off_pass", "own_off_rush"],
    }
    tests = list(range(2019, int(cur)))
    out = {}
    for key in OU:
        cfg = rp.PTARGETS[key]
        oc, pc, med = rp.OPPCOL[cfg["opp"]], f"proj_{cfg['stat']}", f"med_{cfg['stat']}"
        base_feats = [pc, oc, "is_home"] + rp.USAGE
        sub = pw[pw.position.isin(cfg["pos"])].dropna(subset=[cfg["stat"], med] + base_feats)
        sub = sub[sub[cfg["vol"]] >= cfg["mn"]]
        res = {}
        preds = {}
        for name, extra in SETS.items():
            feats = base_feats + extra
            rows = []
            for s in tests:
                tr, te = sub[sub.season < s], sub[sub.season == s]
                if len(tr) < 300 or not len(te):
                    continue
                m = Ridge(alpha=5.0).fit(tr[feats], tr[cfg["stat"]])
                rows.append(pd.DataFrame({"season": s, "proj": m.predict(te[feats]), "actual": te[cfg["stat"]].values,
                                          "line": half_point(te[med].values), "idx": te.index.values}))
            preds[name] = pd.concat(rows, ignore_index=True)
        base = preds["base"].set_index("idx")
        for name, d in preds.items():
            d = d.set_index("idx")
            sel = d[d.season <= 2024]; hold = d[d.season == 2025]
            b_sel = base.loc[sel.index]; b_hold = base.loc[hold.index]
            flips = float((np.sign(d.proj - d.line) != np.sign(base.loc[d.index].proj - base.loc[d.index].line)).mean())
            res[name] = {
                "mae_2019_2024": float((sel.proj - sel.actual).abs().mean()),
                "mae_2025": float((hold.proj - hold.actual).abs().mean()),
                "rel_2019_2024": float((sel.proj - sel.actual).abs().mean() / (b_sel.proj - b_sel.actual).abs().mean() - 1),
                "rel_2025": float((hold.proj - hold.actual).abs().mean() / (b_hold.proj - b_hold.actual).abs().mean() - 1),
                "corr_2025": float(np.corrcoef(hold.proj, hold.actual)[0, 1]),
                "side_flips_vs_base": flips, "n_sel": int(len(sel)), "n_2025": int(len(hold)),
            }
        out[key] = res
        rp.log(f"  {key}: " + "  ".join(f"{n} {r['rel_2019_2024']:+.2%}/{r['rel_2025']:+.2%}" for n, r in res.items()))
    json.dump(out, open("data/prop_context_test.json", "w"), indent=1)
    print("\nMAE relative to the shipped features (negative = better); selection 2019-2024 / untouched 2025")
    print(f"  {'stat':<20}" + "".join(f"{n:>22}" for n in SETS))
    for key, res in out.items():
        print(f"  {key:<20}" + "".join(f"{res[n]['rel_2019_2024']:>+10.2%}/{res[n]['rel_2025']:>+9.2%}" for n in SETS))
    print("wrote data/prop_context_test.json")


if __name__ == "__main__":
    sys.exit(main())
