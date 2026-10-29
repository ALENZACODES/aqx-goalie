"""Full pipeline: rates, optimal pull times with uncertainty, backtests, team tables,
charts and the JSON that powers the dashboard. Run after scripts/fetch_games.py."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pullgoalie.analysis import load_all, rate_table, rates_dict, posterior_draws
from pullgoalie.model import solve, optimal_pull_times, policy_grid, DS, D_MIN

PROC = ROOT / "data" / "processed"; PROC.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "docs" / "data"; OUT.mkdir(parents=True, exist_ok=True)
TRAIN = [20212022, 20222023, 20232024, 20242025]
TEST = [20252026, 20262027]
SEASON_LABEL = lambda s: f"{str(s)[:4]}-{str(s)[6:]}"
H = 600
TS = np.arange(0, H + 1)

def cached():
    f = [PROC / n for n in ("exposure.csv.gz", "pulls.csv.gz", "states.csv.gz")]
    if all(x.exists() for x in f) and "--rebuild" not in sys.argv:
        return [pd.read_csv(x) for x in f]
    E, P, S = load_all()
    for df, x in zip((E, P, S), f):
        df.to_csv(x, index=False)
    return E, P, S

def ix(d): return int(d) - D_MIN

def intended_pull(S, P):
    """Seconds left at which the team planned to pull. Observed when they pulled. If they tied
    before pulling, use that team's median pull time for the deficit that season (league median
    as fallback). If they never pulled and never tied, they effectively chose 0."""
    med_team = P.groupby(["season", "team", "diff"]).pull_remaining.median()
    med_lg = P.groupby(["season", "diff"]).pull_remaining.median()
    vals = []
    for r in S.itertuples():
        if r.pull_remaining > 0:
            v = r.pull_remaining
        elif r.caught_up_first:
            v = med_team.get((r.season, r.team, r.diff), med_lg.get((r.season, max(r.diff, -3)), 120))
        else:
            v = 0.0
        vals.append(min(v, r.checkpoint))
    return np.array(vals)

def brier(p, y): return float(np.mean((p - y) ** 2))
def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def main():
    E, P, S = cached()
    S["T_intent"] = intended_pull(S, P)
    E_train = E[E.season.isin(TRAIN)]
    rt_all, rt_train = rate_table(E), rate_table(E_train)
    r_all, r_train = rates_dict(rt_all), rates_dict(rt_train)
    ot = P[P.went_ot.astype(bool)]
    p_ot_emp = float(ot.won.mean()); p_ot = 0.5
    res = {"meta": {"seasons": sorted(int(s) for s in E.season.unique()),
                    "games": int(E.game.nunique()), "pull_events": int(len(P)),
                    "train": TRAIN, "test": TEST, "p_ot_used": p_ot, "p_ot_observed": p_ot_emp,
                    "p_ot_n": int(len(ot))}}
    res["rates"] = rt_all.round(4).to_dict("records")
    res["rates_js"] = {f"{d},{p}": list(v) for (d, p), v in r_all.items()}

    # optimal pull times
    opt = {}
    for mode in ("points", "win", "tie"):
        _, PULL = solve(r_all, mode, p_ot)
        opt[mode] = {str(k): v for k, v in optimal_pull_times(PULL).items() if k >= -3}
    draws = posterior_draws(rt_all, n=400)
    dT = {"-1": [], "-2": []}
    for dr in draws:
        for d in (0, 1, 2): dr.setdefault((d, 0), dr[(0, 0)])
        _, PULL = solve(dr, "points", p_ot)
        o = optimal_pull_times(PULL); dT["-1"].append(o[-1]); dT["-2"].append(o[-2])
    opt["points_ci90"] = {k: [float(np.percentile(v, 5)), float(np.percentile(v, 50)), float(np.percentile(v, 95))] for k, v in dT.items()}
    res["optimal"] = opt
    Tstar = {-1: opt["points"]["-1"], -2: opt["points"]["-2"]}

    # sensitivity
    sens = []
    def add(label, rates, mode="points", pot=p_ot):
        _, PULL = solve(rates, mode, pot); o = optimal_pull_times(PULL)
        sens.append({"case": label, "down1": o[-1], "down2": o[-2]})
    add("Main: all seasons, standings points, OT 50/50", r_all)
    add("Playoff mode: win probability", r_all, "win")
    add("Overtime win rate 45 percent for the team that came back", r_all, "points", 0.45)
    add("Overtime win rate 55 percent", r_all, "points", 0.55)
    add("Train seasons only (2021-22 to 2024-25)", r_train)
    add("Trailing 5 on 5 rates from the whole third period", rates_dict(rate_table(E, late_only_trailing=False)))
    add("All manpower states (penalties folded in)", rates_dict(rate_table(E, strengths="all")))
    recent = E[E.season.isin([20242025, 20252026, 20262027])]
    add("Recent seasons only (2024-25 onward)", rates_dict(rate_table(recent)))
    res["sensitivity"] = sens

    # actual pull behaviour
    trend = []
    for s, g in P.groupby("season"):
        row = {"season": SEASON_LABEL(s)}
        for d in (-1, -2):
            x = g[g["diff"] == d].pull_remaining
            row[f"median_down{-d}"] = float(x.median()); row[f"n_down{-d}"] = int(len(x))
            row[f"share_optimal_down{-d}"] = float((x >= Tstar[d] - 10).mean())
        trend.append(row)
    res["trend"] = trend
    res["hist_down1"] = np.histogram(P[P["diff"] == -1].pull_remaining.clip(0, 420), bins=np.arange(0, 430, 15))[0].tolist()
    res["hist_down2"] = np.histogram(P[P["diff"] == -2].pull_remaining.clip(0, 600), bins=np.arange(0, 615, 15))[0].tolist()
    res["median_pull"] = {"-1": float(P[P["diff"] == -1].pull_remaining.median()), "-2": float(P[P["diff"] == -2].pull_remaining.median())}

    # value curves (points at 10:00 left) with posterior band
    G = policy_grid(r_all, TS, "points", p_ot, H)
    curves = {}
    sub = draws[:120]
    band = {d: [] for d in (-1, -2)}
    for dr in sub:
        g2 = policy_grid(dr, TS[::5], "points", p_ot, H)
        for d in (-1, -2): band[d].append(g2[H, :, ix(d)] - g2[H, :, ix(d)].max())
    for d in (-1, -2):
        y = G[H, ::5, ix(d)]
        b = np.array(band[d])
        curves[str(d)] = {"T": TS[::5].tolist(), "points": y.round(5).tolist(),
                          "regret_lo": np.percentile(b, 5, axis=0).round(5).tolist(),
                          "regret_hi": np.percentile(b, 95, axis=0).round(5).tolist()}
    res["curves"] = curves

    # backtest A: calibration on held out seasons
    Gt_tie = policy_grid(r_train, TS, "tie", p_ot, H)
    Gt_pts = policy_grid(r_train, TS, "points", p_ot, H)
    CK = [600, 480, 360, 300, 240, 180, 120, 60]
    test = S[S.season.isin(TEST) & S.checkpoint.isin(CK) & S["diff"].isin([-1, -2])].copy()
    train = S[S.season.isin(TRAIN) & S.checkpoint.isin(CK) & S["diff"].isin([-1, -2])]
    base = train.groupby(["checkpoint", "diff"]).reached_ot_or_won.mean()
    lg_med = P[P.season.isin(TRAIN)].groupby("diff").pull_remaining.median()
    c = test.checkpoint.values.astype(int); d = test["diff"].values.astype(int)
    Ti = test.T_intent.round().astype(int).values
    Tm = np.minimum(np.array([lg_med[x] for x in d]).round().astype(int), c)
    test["p_model"] = Gt_tie[c, Ti, d - D_MIN]
    test["p_model_median_pull"] = Gt_tie[c, Tm, d - D_MIN]
    Gt_tie_all = policy_grid(rates_dict(rate_table(E_train, strengths="all")), TS, "tie", p_ot, H)
    test["p_model_all"] = Gt_tie_all[c, Ti, d - D_MIN]
    test["p_base"] = [base[(a, b)] for a, b in zip(c, d)]
    test["pts_model"] = Gt_pts[c, Ti, d - D_MIN]
    y = test.reached_ot_or_won.values
    from sklearn.metrics import roc_auc_score
    bt = {"n_states": int(len(test)), "n_games": int(test.game.nunique()),
          "observed_rate": float(y.mean()), "predicted_rate": float(test.p_model.mean())}
    for k in ("p_model", "p_model_all", "p_model_median_pull", "p_base"):
        bt[k] = {"brier": brier(test[k].values, y), "logloss": logloss(test[k].values, y), "auc": float(roc_auc_score(y, test[k]))}
    bt["points_observed"] = float(test.points.mean()); bt["points_predicted"] = float(test.pts_model.mean())
    test["bin"] = pd.qcut(test.p_model, 10, duplicates="drop")
    test["bin_all"] = pd.qcut(test.p_model_all, 10, duplicates="drop")
    bt["calibration_all"] = test.groupby("bin_all", observed=True).agg(pred=("p_model_all", "mean"), obs=("reached_ot_or_won", "mean"), n=("p_model_all", "size")).reset_index(drop=True).round(4).to_dict("records")
    bt["predicted_rate_all"] = float(test.p_model_all.mean())
    cal = test.groupby("bin", observed=True).agg(pred=("p_model", "mean"), obs=("reached_ot_or_won", "mean"), n=("p_model", "size")).reset_index(drop=True)
    bt["calibration"] = cal.round(4).to_dict("records")
    by_ck = test.groupby(["checkpoint", "diff"]).agg(pred=("p_model", "mean"), obs=("reached_ot_or_won", "mean"), n=("p_model", "size")).reset_index()
    bt["by_checkpoint"] = by_ck.round(4).to_dict("records")
    res["backtest"] = bt

    # backtest B: does pulling earlier actually pay? (state: down 1 with 3:00 left, all seasons and test only)
    Ga_tie = policy_grid(r_all, TS, "tie", p_ot, H)
    def buckets(df, ck, dd, edges, labels):
        x = df[(df.checkpoint == ck) & (df["diff"] == dd)].copy()
        x["bucket"] = pd.cut(x.T_intent, edges, labels=labels, include_lowest=True, right=False)
        x["pred"] = Ga_tie[ck, x.T_intent.round().astype(int).values, dd - D_MIN]
        g = x.groupby("bucket", observed=True).agg(n=("pred", "size"), obs=("reached_ot_or_won", "mean"), pred=("pred", "mean"),
                                                    pts=("points", "mean")).reset_index()
        g["bucket"] = g.bucket.astype(str)
        g["se"] = np.sqrt(g.obs * (1 - g.obs) / g.n)
        return g.round(4).to_dict("records")
    res["buckets_down1"] = buckets(S, 180, -1, [0, 60, 90, 120, 150, 181], ["under 1:00 or never", "1:00 to 1:29", "1:30 to 1:59", "2:00 to 2:29", "2:30 to 3:00"])
    res["buckets_down2"] = buckets(S, 300, -2, [0, 120, 180, 240, 301], ["under 2:00 or never", "2:00 to 2:59", "3:00 to 3:59", "4:00 to 5:00"])

    # backtest C: standings points left on the table, every team, every season
    Gpts = G
    Vopt, _ = solve(r_all, "points", p_ot, horizon=H)
    rows = []
    for dd in (-1, -2):
        ck = int(round(Tstar[dd] / 10) * 10)
        late = S[(S.checkpoint == ck) & (S["diff"] == dd)]
        for r in late.itertuples():
            Ta = int(round(min(r.T_intent, ck)))
            rows.append(dict(season=r.season, team=r.team, game=r.game, diff=dd, kind="late",
                             T=Ta, loss=float(Vopt[ck, ix(dd)] - Gpts[ck, Ta, ix(dd)])))
        early = P[(P["diff"] == dd) & (P.pull_remaining > ck + 5)]
        for r in early.itertuples():
            Ta = int(round(min(r.pull_remaining, H)))
            rows.append(dict(season=r.season, team=r.team, game=r.game, diff=dd, kind="early",
                             T=Ta, loss=float(Vopt[Ta, ix(dd)] - Gpts[Ta, Ta, ix(dd)])))
    L = pd.DataFrame(rows)
    L.to_csv(PROC / "decision_losses.csv.gz", index=False)
    per_season = L.groupby("season").loss.sum()
    n_games_season = E.groupby("season").game.nunique()
    full = [s for s in per_season.index if n_games_season[s] >= 1300]
    res["points_left"] = {
        "league_per_full_season": float(per_season[full].mean()),
        "per_team_per_full_season": float(per_season[full].mean() / 32),
        "by_season": {SEASON_LABEL(s): {"points": float(v), "games": int(n_games_season[s])} for s, v in per_season.items()},
        "late_share": float(L[L.kind == "late"].loss.sum() / L.loss.sum()),
        "n_decisions": int(len(L)),
    }
    # team table: last two full seasons plus this season
    recent_s = [20242025, 20252026, 20262027]
    team = []
    Er = E[E.season.isin(recent_s)]
    lg = {k: v for k, v in r_all.items()}
    K = 8.0  # prior strength in hours of 6 on 5 play
    for t, g in Er.groupby("team"):
        rt = rate_table(g)
        rr = dict(lg)
        for _, x in rt.iterrows():
            key = (int(x.d), int(x.pulled))
            if key in [(-1, 1), (-2, 1)]:
                pf, pa = lg[key]
                rr[key] = ((x.gf + pf * K) / (x.hours + K), (x.ga + pa * K) / (x.hours + K))
        _, PULL = solve(rr, "points", p_ot); o = optimal_pull_times(PULL)
        Pt = P[(P.team == t) & P.season.isin(recent_s)]
        Lt = L[(L.team == t) & L.season.isin(recent_s)]
        x1 = rt[(rt.d == -1) & (rt.pulled == 1)]
        team.append(dict(team=t, opt_down1=o[-1], opt_down2=o[-2],
                         med_down1=float(Pt[Pt["diff"] == -1].pull_remaining.median()) if (Pt["diff"] == -1).any() else None,
                         med_down2=float(Pt[Pt["diff"] == -2].pull_remaining.median()) if (Pt["diff"] == -2).any() else None,
                         pulls=int(len(Pt)),
                         rates_override={"-1,1": [float(v) for v in rr[(-1, 1)]], "-2,1": [float(v) for v in rr[(-2, 1)]]},
                         sixv5_gf60=float(rr[(-1, 1)][0]), sixv5_ga60=float(rr[(-1, 1)][1]),
                         sixv5_hours=float(x1.hours.sum()) if len(x1) else 0.0,
                         points_left=float(Lt.loss.sum()),
                         points_left_per_82=float(Lt.loss.sum() / max(1, g.game.nunique()) * 82)))
    res["teams"] = sorted(team, key=lambda r: -r["points_left_per_82"])
    res["team_seasons"] = [SEASON_LABEL(s) for s in recent_s]
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("meta", "optimal", "sensitivity", "median_pull", "points_left")}, indent=1, default=float))
    print(json.dumps(res["backtest"], indent=1, default=float)[:3000])
    print(res["buckets_down1"]); print(res["buckets_down2"]); print(res["trend"])
    print(pd.DataFrame(res["teams"]).round(2).to_string())

if __name__ == "__main__":
    main()
