#!/usr/bin/env python3
"""
Do teammate absences and snap share improve the player projections on games they have not seen?

The shipped player model knows a player's own form and his own prior share of the team's
targets and carries. It does not know that the WR1 is out this week, so it projects the WR2
at his usual number. This test adds two things, one at a time, on the same walk-forward
harness as the shipped model (Ridge, trained on seasons < s, scored on season s):

  shipped    form + opponent defence + venue + own target/carry share          [what is live]
  +snap      + prior offensive snap share (expanding mean of prior games)
  +absence   own shares rescaled by the share the absent teammates leave behind, plus the
             lost share itself as a feature
  +both

"Absent" is point-in-time: listed Out or Doubtful on this week's injury report and seen in one
of the team's last two games, so the absence is new. Nothing after kickoff is used.

Selection on 2019-2024; 2025 reported separately and not used to choose anything. Reported:
MAE relative to shipped, MAE on the rows where an absence actually mattered (lost share
>= 10%), and how often the new projection would take the other side of the rolling-median
line. Plus the raw empirical question the feature rests on: when a teammate is out, how
much does a remaining player's share really move?
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import run_pipeline as rp
from test_prop_edge import half_point, NOT_OVER_UNDER

OU = [k for k in rp.PTARGETS if k not in NOT_OVER_UNDER]
SKILL = {"QB", "RB", "WR", "TE", "FB"}
GRP = {"WR": "rec", "TE": "rec", "RB": "rb", "FB": "rb", "QB": "qb"}
NEW_GAP = 2             # the absentee appeared within this many weeks, so his share is still in the priors
MATTERS = 0.10          # lost share that counts as "an absence that mattered"


def prior_snap_share(snap, rost):
    s = snap[snap.game_type == "REG"].copy()
    cw = rost.dropna(subset=["gsis_id", "pfr_id"])[["gsis_id", "pfr_id"]].drop_duplicates("pfr_id")
    s = s.merge(cw, left_on="pfr_player_id", right_on="pfr_id", how="left").dropna(subset=["gsis_id"])
    s = s.sort_values(["gsis_id", "season", "week"])
    s["prior_snap"] = s.groupby(["gsis_id", "season"])["offense_pct"].transform(lambda x: x.shift(1).expanding().mean())
    u = s[["gsis_id", "season", "week", "prior_snap"]].rename(columns={"gsis_id": "player_id"})
    u[["season", "week"]] = u[["season", "week"]].astype("int64")
    return u.drop_duplicates(["player_id", "season", "week"])


def absence_features(pw, inj, log=print):
    """
    Per (team, season, week): the prior target and carry share of the teammates who are absent,
    and each present player's share rescaled to the remaining pie.
    """
    pw = pw.copy()
    pw[["season", "week"]] = pw[["season", "week"]].astype("int64")
    skill = pw[pw.position.isin(SKILL)].copy()
    # share of the team's targets/carries in each game the player appeared in
    for col, out in [("targets", "sh_tgt"), ("carries", "sh_car")]:
        tm = skill.groupby(["team", "season", "week"])[col].transform("sum")
        skill[out] = (skill[col] / tm.replace(0, np.nan)).fillna(0)
    skill = skill.sort_values(["player_id", "season", "week"])
    # expanding mean INCLUDING this game; merged as-of strictly before a week it gives the
    # mean over the games before that week, for players who did not appear that week too
    for c in ["sh_tgt", "sh_car"]:
        skill[f"cum_{c}"] = skill.groupby(["player_id", "season"])[c].transform(lambda x: x.expanding().mean())
    played = skill[["player_id", "team", "season", "week", "cum_sh_tgt", "cum_sh_car"]].copy()

    # absentees per team-week: listed Out or Doubtful on this week's report, and seen in one of
    # the team's last two games, so the absence is NEW relative to the teammates' priors. A
    # first cut also flagged any regular missing for two-plus games (an IR proxy); that fired
    # on 60% of rows and its "lost" share was mostly already absorbed into everyone else's
    # expanding mean. Recorded in data/prop_absence_test.json as the rejected variant.
    absent = inj[(inj.game_type == "REG") & (inj.report_status.isin(["Out", "Doubtful"]))][
        ["season", "week", "team", "gsis_id", "position"]].dropna(subset=["gsis_id"]).rename(columns={"gsis_id": "player_id"})
    absent["grp"] = absent.position.map(GRP)
    absent = absent.dropna(subset=["grp"])
    absent[["season", "week"]] = absent[["season", "week"]].astype("int64")
    absent["how"] = "report"
    absent = absent.drop_duplicates(["season", "week", "team", "player_id"]).sort_values("week")
    # the absentee's share over his last three appearances, as of strictly before this week
    for c in ["sh_tgt", "sh_car"]:
        skill[f"rec_{c}"] = skill.groupby(["player_id", "season"])[c].transform(lambda x: x.rolling(3, min_periods=1).mean())
    pl = skill[["player_id", "season", "week", "rec_sh_tgt", "rec_sh_car"]].rename(
        columns={"week": "last_week", "rec_sh_tgt": "cum_sh_tgt", "rec_sh_car": "cum_sh_car"}).sort_values("last_week")
    absent = pd.merge_asof(absent, pl, left_on="week", right_on="last_week", by=["player_id", "season"],
                           direction="backward", allow_exact_matches=False)
    absent = absent.dropna(subset=["cum_sh_tgt"])
    absent = absent[(absent.week - absent.last_week) <= NEW_GAP]
    # lost share by the absentee's position group: WR1 targets go to the other receivers,
    # RB1 carries (and his targets) go to the other backs. Summed per team-week.
    lost = absent.groupby(["team", "season", "week", "grp"]).agg(t=("cum_sh_tgt", "sum"), c=("cum_sh_car", "sum")).reset_index()
    lost = lost.pivot_table(index=["team", "season", "week"], columns="grp", values=["t", "c"], fill_value=0)
    lost.columns = [f"lost_{a}_{b}" for a, b in lost.columns]           # lost_t_rec, lost_c_rb, ...
    lost = lost.reset_index()
    for c in ["lost_t_rec", "lost_t_rb", "lost_c_rb", "lost_t_qb", "lost_c_qb"]:
        if c not in lost.columns:
            lost[c] = 0.0
    self_abs = absent[["team", "season", "week", "player_id", "cum_sh_tgt", "cum_sh_car"]].rename(
        columns={"cum_sh_tgt": "self_tgt", "cum_sh_car": "self_car"})
    pw = pw.merge(lost, on=["team", "season", "week"], how="left").merge(self_abs, on=["team", "season", "week", "player_id"], how="left")
    for c in ["lost_t_rec", "lost_t_rb", "lost_c_rb", "lost_t_qb", "lost_c_qb", "self_tgt", "self_car"]:
        pw[c] = pw[c].fillna(0)
    pw["grp"] = pw.position.map(GRP)
    # the pie the player competes for: his own group's targets (receivers, backs) or carries (backs)
    own_t = np.where(pw.grp == "rec", pw.lost_t_rec, np.where(pw.grp == "rb", pw.lost_t_rb, 0.0))
    own_c = np.where(pw.grp == "rb", pw.lost_c_rb, 0.0)
    pw["lost_tgt"] = np.clip(own_t - pw.self_tgt, 0, 0.9)
    pw["lost_car"] = np.clip(own_c - pw.self_car, 0, 0.9)
    pw["lost_any_tgt"] = np.clip(pw.lost_t_rec + pw.lost_t_rb - pw.self_tgt, 0, 0.9)   # what a QB loses
    pw["tgt_share_adj"] = (pw.tgt_share / (1 - pw.lost_tgt)).clip(upper=1)
    pw["car_share_adj"] = (pw.car_share / (1 - pw.lost_car)).clip(upper=1)
    pw["boost_tgt"] = pw.lost_tgt / (1 - pw.lost_tgt)          # proportional uplift factor - 1
    pw["boost_car"] = pw.lost_car / (1 - pw.lost_car)
    log(f"  absentees with a recent share: {len(absent)}; team-weeks with one: {len(lost)}; "
        f"rows with own-group lost target share >= {MATTERS:.0%}: {(pw.lost_tgt >= MATTERS).mean():.1%}, carries: {(pw.lost_car >= MATTERS).mean():.1%}")
    return pw


def empirical_uplift(pw):
    """When teammates are out, how much does a present player's ACTUAL share move vs his prior?"""
    out = {}
    for stat, sh, lost in [("targets", "tgt_share", "lost_tgt"), ("carries", "car_share", "lost_car")]:
        d = pw[(pw.position.isin(SKILL)) & (pw[sh] >= 0.05) & (pw.gp_prior >= 2)].copy()
        tm = d.groupby(["team", "season", "week"])[stat].transform("sum")
        d["actual"] = (d[stat] / tm.replace(0, np.nan)).fillna(0)
        d["delta"] = d.actual - d[sh]
        rows = {}
        for lo, hi, lab in [(0, 0.001, "none"), (0.001, 0.10, "<10%"), (0.10, 0.20, "10-20%"), (0.20, 1.0, ">=20%")]:
            s = d[(d[lost] >= lo) & (d[lost] < hi)]
            if len(s) < 50:
                continue
            # proportional redistribution predicts delta = prior * lost / (1 - lost)
            pred = (s[sh] * s[lost] / (1 - s[lost])).mean()
            rows[lab] = {"n": int(len(s)), "prior_share": round(float(s[sh].mean()), 4),
                         "actual_share": round(float(s.actual.mean()), 4), "delta": round(float(s.delta.mean()), 4),
                         "proportional_pred": round(float(pred), 4)}
        out[stat] = rows
    return out


def main():
    sched, team, plyr, inj, snap, rost, depth, cur = rp.load_all("data", None)
    reg = sched[(sched.season == cur) & (sched.game_type == "REG")]
    un = reg[reg.home_score.isna()]
    tw = int(un.week.min()) if len(un) else int(reg.week.max())
    ratings, _ = rp.team_ratings(team, sched, cur, tw)
    pw = rp.player_form(plyr, sched, ratings).sort_values(["player_id", "season", "week"])
    pw = absence_features(pw, inj, log=rp.log)
    pw = pw.merge(prior_snap_share(snap, rost), on=["player_id", "season", "week"], how="left")
    rp.log(f"  prior_snap coverage: {pw.prior_snap.notna().mean():.1%} of rows")
    pw["prior_snap"] = pw.prior_snap.fillna(pw.prior_snap.median())
    for key, cfg in rp.PTARGETS.items():
        st = cfg["stat"]
        pw[f"med_{st}"] = pw.groupby(["player_id", "season"])[st].transform(lambda x: x.shift(1).expanding().median())

    uplift = empirical_uplift(pw)
    print("\nWhen teammates are absent, the present players' actual share vs their prior share")
    for stat, rows in uplift.items():
        print(f"  {stat}: " + "  ".join(f"{k}: n={v['n']} prior {v['prior_share']:.3f} -> actual {v['actual_share']:.3f} "
                                       f"(delta {v['delta']:+.3f}, proportional says {v['proportional_pred']:+.3f})" for k, v in rows.items()))

    SETS = {
        "shipped":  ([], ["tgt_share", "car_share"]),
        "+snap":    (["prior_snap"], ["tgt_share", "car_share"]),
        "+absence": (["lost_tgt", "lost_car"], ["tgt_share_adj", "car_share_adj"]),
        "+abs*form": (["lost_tgt", "lost_car", "form_x_boost"], ["tgt_share", "car_share"]),
        "+both":    (["prior_snap", "lost_tgt", "lost_car", "form_x_boost"], ["tgt_share", "car_share"]),
    }
    tests = list(range(2019, int(cur)))
    out = {"uplift": uplift, "stats": {}}
    print(f"\nMAE relative to shipped (negative = better); selection 2019-2024 / untouched 2025 / rows where lost share >= {MATTERS:.0%}")
    print(f"  {'stat':<20}" + "".join(f"{n:>30}" for n in SETS))
    for key in OU:
        cfg = rp.PTARGETS[key]
        oc, pc, med = rp.OPPCOL[cfg["opp"]], f"proj_{cfg['stat']}", f"med_{cfg['stat']}"
        base = [pc, oc, "is_home"]
        need = sorted(set(base + ["tgt_share", "car_share", "tgt_share_adj", "car_share_adj", "prior_snap", "lost_tgt", "lost_car", cfg["stat"], med]))
        sub = pw[pw.position.isin(cfg["pos"])].dropna(subset=need).copy()
        sub = sub[sub[cfg["vol"]] >= cfg["mn"]]
        # the interaction: how many more of the stat the form implies if the lost share flows
        # proportionally. Backs on carries, everyone else on targets; a QB sees what his
        # receivers lost (expected sign negative).
        if cfg["pos"] == ["QB"]:
            sub["form_x_boost"] = sub[pc] * sub.lost_any_tgt
            lostcol = "lost_any_tgt"
        elif cfg["stat"] in ("rushing_yards", "rushing_tds"):
            sub["form_x_boost"] = sub[pc] * sub.boost_car
            lostcol = "lost_car"
        else:
            sub["form_x_boost"] = sub[pc] * sub.boost_tgt
            lostcol = "lost_tgt"
        sub["lost_tgt"] = sub[lostcol] if cfg["pos"] == ["QB"] else sub.lost_tgt
        preds = {}
        for name, (extra, shares) in SETS.items():
            feats = base + shares + extra
            rows = []
            for s in tests:
                tr, te = sub[sub.season < s], sub[sub.season == s]
                if len(tr) < 300 or not len(te):
                    continue
                m = Ridge(alpha=5.0).fit(tr[feats], tr[cfg["stat"]])
                rows.append(pd.DataFrame({"season": s, "proj": m.predict(te[feats]), "actual": te[cfg["stat"]].values,
                                          "line": half_point(te[med].values), "lost": te[lostcol].values, "idx": te.index.values}))
            preds[name] = pd.concat(rows, ignore_index=True).set_index("idx")
        base_p = preds["shipped"]
        res = {}
        for name, d in preds.items():
            sel, hold = d[d.season <= 2024], d[d.season == 2025]
            b_sel, b_hold = base_p.loc[sel.index], base_p.loc[hold.index]
            mat = d[d.lost >= MATTERS]; b_mat = base_p.loc[mat.index]
            flips = float((np.sign(d.proj - d.line) != np.sign(base_p.loc[d.index].proj - base_p.loc[d.index].line)).mean())
            res[name] = {
                "rel_2019_2024": float((sel.proj - sel.actual).abs().mean() / (b_sel.proj - b_sel.actual).abs().mean() - 1),
                "rel_2025": float((hold.proj - hold.actual).abs().mean() / (b_hold.proj - b_hold.actual).abs().mean() - 1),
                "rel_matters": float((mat.proj - mat.actual).abs().mean() / (b_mat.proj - b_mat.actual).abs().mean() - 1) if len(mat) else None,
                "mae_2025": float((hold.proj - hold.actual).abs().mean()),
                "side_flips_vs_shipped": flips, "n_sel": int(len(sel)), "n_2025": int(len(hold)), "n_matters": int(len(mat)),
            }
        out["stats"][key] = res
        print(f"  {key:<20}" + "".join(f"{r['rel_2019_2024']:>+9.2%}/{r['rel_2025']:>+8.2%}/{(r['rel_matters'] if r['rel_matters'] is not None else 0):>+8.2%}"
                                     for r in res.values()))
    json.dump(out, open("data/prop_absence_test.json", "w"), indent=1)
    print("wrote data/prop_absence_test.json")


if __name__ == "__main__":
    sys.exit(main())
