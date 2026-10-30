"""Redraw the full chart set in one style. Every number comes from
docs/data/results.json at runtime. Output goes to docs/img/v2/.

Run from the repo root:  python scripts/make_charts_v2.py

Charts
  hist_down1.png, hist_down2.png   where coaches pull vs the best window
  curve_down1.png, curve_down2.png points lost per 100 games by pull time
  cost_curves.png                  both curves in one figure (small multiples)
  trend.png                        median pull by season vs the best time
  buckets_down1.png, buckets_down2.png  real vs model comeback share by pull time
  calibration.png                  out-of-sample calibration, with skill scores
  teams.png                        points left on the table per 82 games
  sensitivity.png                  best time under alternative assumptions
  rates.png                        goals for and against per 60, goalie in vs out
Each image also gets a <name>.alt.txt file and all alt text is collected in
docs/img/v2/alt.json for index.html.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import chartstyle as cs

OUT = cs.ROOT / "docs" / "img" / "v2"
ALT: dict[str, str] = {}
Z = 1.959964  # two-sided 95% normal quantile


def save(fig, name: str, alt: str, **kw) -> None:
    ALT[name] = " ".join(alt.split())
    cs.save(fig, OUT / name, ALT[name], **kw)


def new_fig(w: float = 12, h: float = 6.75, *, top: float = 0.78, bottom: float = 0.14,
            left: float = 0.07, right: float = 0.97):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right)
    return fig, ax


def down_word(d: int) -> str:
    return {1: "one", 2: "two", 3: "three"}[d]


# ----------------------------------------------------------------------------
# 1. Where coaches pull: histogram against the best window
# ----------------------------------------------------------------------------
def hist_facts(R: dict, k: int) -> dict:
    counts = np.asarray(R[f"hist_down{k}"], dtype=int)
    edges = np.arange(0, 15 * (len(counts) + 1), 15)
    lo, best, hi = R["optimal"]["points_ci90"][f"-{k}"]
    best = R["optimal"]["points"][f"-{k}"]
    med = R["median_pull"][f"-{k}"]
    tr = R["trend"]
    n_tr = sum(t[f"n_down{k}"] for t in tr)
    on_time = sum(t[f"share_optimal_down{k}"] * t[f"n_down{k}"] for t in tr)
    return dict(k=k, counts=counts, edges=edges, lo=lo, hi=hi, best=best, med=med,
                gap=best - med, n=int(counts.sum()), n_on_time=int(round(on_time)),
                share_on_time=on_time / n_tr, overflow_edge=edges[-2])


def chart_hist(R: dict, k: int) -> None:
    F = hist_facts(R, k)
    counts, edges = F["counts"], F["edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    ymax = counts.max() * 1.32
    fig, ax = new_fig(top=0.80, bottom=0.15)

    ax.axvspan(F["hi"], F["lo"], color=cs.ICE, alpha=0.32, lw=0, zorder=0)
    ax.axvline(F["best"], color=cs.NAVY, lw=1.4, ls=(0, (4, 3)), zorder=3)
    colors = [cs.NAVY if c > F["best"] else cs.BAR for c in centers]
    ax.bar(centers, counts, width=13.2, color=colors, lw=0, zorder=2)
    ax.bar(centers[-1], counts[-1], width=13.2, facecolor="none", edgecolor=cs.GREY,
           hatch="////", lw=0.8, zorder=3)
    ax.axvline(F["med"], color=cs.RED, lw=2, zorder=4)
    ax.set_ylim(0, ymax)
    cs.clock_ticks(ax, 0, edges[-1], 60)
    ax.set_xlabel("Time left in regulation when the goalie came out", labelpad=8)
    ax.text(0, 1.02, "pulls", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, color=cs.GREY)

    y_lab = ymax * 0.985
    ax.text(F["hi"] + 4, y_lab, f"Best window\n{cs.fmt_clock(F['hi'])} to {cs.fmt_clock(F['lo'])}",
            ha="right", va="top", fontsize=11, color=cs.NAVY, fontweight="semibold", linespacing=1.15)
    ax.text(F["best"] - 6, ymax * 0.64, f"Best: {cs.fmt_clock(F['best'])}", ha="left", va="center",
            fontsize=13, color=cs.NAVY, fontweight="bold", family=cs.DISPLAY)
    ax.text(F["med"] - 4, y_lab, f"Typical NHL pull: {cs.fmt_clock(F['med'])}", ha="left", va="top",
            fontsize=13, color=cs.RED, fontweight="bold", family=cs.DISPLAY)
    y_arrow = ymax * 0.875
    ax.add_patch(FancyArrowPatch((F["best"], y_arrow), (F["med"], y_arrow), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.8, color=cs.RED, shrinkA=0, shrinkB=2, zorder=5))
    ax.text((F["best"] + F["med"]) / 2, y_arrow + ymax * 0.03,
            f"{cs.fmt_clock(F['gap'])} later than the model says", ha="center", va="bottom",
            fontsize=11.5, color=cs.RED, fontweight="semibold")
    share = F["share_on_time"]
    share_txt = f"{share * 100:.1f}%" if share < 0.10 else f"{share * 100:.0f}%"
    ax.text(edges[-1] - 14, ymax * 0.50,
            f"{F['n_on_time']} of {F['n']:,} pulls ({share_txt})\ncame within 10 s of the best time, or earlier",
            ha="left", va="center", fontsize=11, color=cs.GREY, linespacing=1.2)
    ax.annotate(f"{counts[-1]} pulls at {cs.fmt_clock(F['overflow_edge'])} or earlier\n(all shown in this bin)",
                xy=(centers[-1], counts[-1]), xytext=(centers[-1] - 22, ymax * 0.22), ha="left",
                va="center", fontsize=9.5, color=cs.GREY,
                arrowprops=dict(arrowstyle="-", color=cs.GREY_LIGHT, lw=0.8,
                                connectionstyle="arc3,rad=-0.2"))

    cs.title_block(fig, f"Down {k}, almost every pull comes after the best window",
                   f"{F['n']:,} real pulls while trailing by {down_word(k)}, in 15-second bins. "
                   f"Shaded: the model's 90% range for the best pull time.")
    cs.footer(fig, cs.source_line(R))
    save(fig, f"hist_down{k}.png",
         f"Histogram of {F['n']:,} NHL goalie pulls while trailing by {down_word(k)}, on a game clock "
         f"from {cs.fmt_clock(edges[-1])} to 0:00. The model's best window, {cs.fmt_clock(F['hi'])} to "
         f"{cs.fmt_clock(F['lo'])} left, is shaded and nearly empty; the typical pull comes at "
         f"{cs.fmt_clock(F['med'])}, {cs.fmt_clock(F['gap'])} after the best time of {cs.fmt_clock(F['best'])}. "
         f"{F['n_on_time']} pulls ({share_txt}) came within 10 seconds of the best time, or earlier. The last bar on the left "
         f"holds every pull at {cs.fmt_clock(F['overflow_edge'])} or earlier.")


# ----------------------------------------------------------------------------
# 2. Cost curves
# ----------------------------------------------------------------------------
def curve_facts(R: dict, k: int) -> dict:
    c = R["curves"][f"-{k}"]
    T = np.asarray(c["T"], dtype=float)
    pts = np.asarray(c["points"], dtype=float)
    best_i = int(np.argmax(pts))
    loss = (pts[best_i] - pts) * 100                       # points lost per 100 games
    lo = -np.asarray(c["regret_lo"]) * 100                 # worse end of the 90% band
    hi = -np.asarray(c["regret_hi"]) * 100                 # better end
    med = R["median_pull"][f"-{k}"]
    # the curve is on a 5-second grid; the best time itself comes from the
    # second-by-second solve in optimal.points, and the cost at the median is
    # interpolated between the two grid points around it
    best_T = R["optimal"]["points"][f"-{k}"]
    if abs(T[best_i] - best_T) > 5:
        print(f"WARNING: curves[-{k}] peaks at {T[best_i]:.0f}s but optimal.points says {best_T}s; "
              f"results.json predates the per-deficit policy fix, re-run scripts/run_analysis.py",
              file=sys.stderr)
    med_loss = float(np.interp(med, T, loss))
    return dict(k=k, T=T, loss=loss, lo=lo, hi=hi, best_T=best_T, best_pts=pts[best_i],
                med=med, med_loss=med_loss, horizon=T[-1],
                ci=R["optimal"]["points_ci90"][f"-{k}"])


def draw_curve(ax, F: dict, *, label_size: float = 12) -> None:
    ax.fill_between(F["T"], F["hi"], F["lo"], color=cs.ICE, alpha=0.45, lw=0,
                    label="90% range")
    ax.plot(F["T"], F["loss"], color=cs.NAVY, lw=2.4)
    ax.axvline(F["best_T"], color=cs.NAVY, lw=1.2, ls=(0, (4, 3)))
    ax.axvline(F["med"], color=cs.RED, lw=2)
    top = max(F["lo"].max(), F["loss"].max()) * 1.12
    ax.set_ylim(0, top)
    cs.clock_ticks(ax, 0, F["horizon"], 60)
    ax.set_xlabel("Time left in regulation when the goalie is pulled", labelpad=8)
    ax.text(0, 1.02, "standings points lost per 100 games", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10, color=cs.GREY)
    # direct labels
    ax.text(F["best_T"] + 5, top * 0.93, f"Best: {cs.fmt_clock(F['best_T'])}", ha="right", va="top",
            fontsize=label_size + 1, color=cs.NAVY, fontweight="bold", family=cs.DISPLAY)
    ax.text(F["best_T"] + 5, top * 0.84, f"90% range {cs.fmt_clock(F['ci'][2])} to {cs.fmt_clock(F['ci'][0])}",
            ha="right", va="top", fontsize=label_size - 2, color=cs.GREY)
    ax.annotate(f"NHL median {cs.fmt_clock(F['med'])}\ncosts {F['med_loss']:.1f} pts per 100 games",
                xy=(F["med"], F["med_loss"]), xytext=(F["med"] + 12, top * 0.62), ha="right", va="center",
                fontsize=label_size, color=cs.RED, fontweight="semibold", linespacing=1.2,
                arrowprops=dict(arrowstyle="-", color=cs.RED, lw=0.9, shrinkB=4))
    ax.plot([F["med"]], [F["med_loss"]], "o", ms=7, color=cs.RED, mec="white", mew=1.5, zorder=5)


def chart_curve(R: dict, k: int) -> None:
    F = curve_facts(R, k)
    fig, ax = new_fig(top=0.80, bottom=0.15)
    draw_curve(ax, F)
    cs.title_block(fig, f"Down {k} with {cs.fmt_clock(F['horizon'])} left: every second of waiting past "
                        f"{cs.fmt_clock(F['best_T'])} costs points",
                   f"Expected standings points lost per 100 such games for each pull time, against the best time. "
                   f"Band: 90% range from resampled goal rates.")
    cs.footer(fig, cs.source_line(R))
    save(fig, f"curve_down{k}.png",
         f"U-shaped curve of standings points lost per 100 games by goalie pull time when trailing by "
         f"{down_word(k)} with {cs.fmt_clock(F['horizon'])} left, with a 90% uncertainty band. The minimum is at "
         f"{cs.fmt_clock(F['best_T'])} left (90% range {cs.fmt_clock(F['ci'][2])} to {cs.fmt_clock(F['ci'][0])}). "
         f"The NHL median pull at {cs.fmt_clock(F['med'])} costs about {F['med_loss']:.1f} points per 100 games.")


def chart_cost_curves(R: dict) -> None:
    F1, F2 = curve_facts(R, 1), curve_facts(R, 2)
    fig, axes = plt.subplots(2, 1, figsize=(12, 11), dpi=200, sharex=False)
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.07, right=0.97, hspace=0.42)
    for ax, F in zip(axes, (F1, F2)):
        draw_curve(ax, F, label_size=11.5)
        ax.set_title(f"Trailing by {down_word(F['k'])}", loc="left", fontsize=15, fontweight="bold",
                     family=cs.DISPLAY, color=cs.NAVY, pad=26)
    axes[0].set_xlabel("")
    cs.title_block(fig, "The cost of waiting, down 1 and down 2",
                   f"Standings points lost per 100 games by pull time, from {cs.fmt_clock(F1['horizon'])} left. "
                   f"Best: {cs.fmt_clock(F1['best_T'])} down 1, {cs.fmt_clock(F2['best_T'])} down 2. "
                   f"NHL medians: {cs.fmt_clock(F1['med'])} and {cs.fmt_clock(F2['med'])}.",
                   y=0.975, sub_gap=0.04)
    cs.footer(fig, cs.source_line(R), y=0.012)
    save(fig, "cost_curves.png",
         f"Two stacked curves of standings points lost per 100 games by pull time. Down one the best time is "
         f"{cs.fmt_clock(F1['best_T'])} and the NHL median {cs.fmt_clock(F1['med'])} costs {F1['med_loss']:.1f} "
         f"points per 100 games. Down two the best time is {cs.fmt_clock(F2['best_T'])} and the median "
         f"{cs.fmt_clock(F2['med'])} costs {F2['med_loss']:.1f}. Both carry 90% bands.")


# ----------------------------------------------------------------------------
# 3. Trend by season
# ----------------------------------------------------------------------------
def chart_trend(R: dict) -> None:
    tr = R["trend"]
    seasons = [t["season"] for t in tr]
    x = np.arange(len(tr))
    m1 = np.array([t["median_down1"] for t in tr])
    m2 = np.array([t["median_down2"] for t in tr])
    b1, b2 = R["optimal"]["points"]["-1"], R["optimal"]["points"]["-2"]
    n_last1, n_last2 = tr[-1]["n_down1"], tr[-1]["n_down2"]

    fig, ax = new_fig(top=0.80, bottom=0.12, right=0.80)
    for b, col, lab in ((b1, cs.RED, "best, down 1"), (b2, cs.NAVY, "best, down 2")):
        ax.axhline(b, color=col, lw=1.1, ls=(0, (4, 3)), alpha=0.9)
        ax.text(len(tr) - 0.55, b + 6, f"{lab}: {cs.fmt_clock(b)}", ha="left", va="bottom",
                fontsize=11, color=col, fontweight="semibold")
    for m, col, lab in ((m1, cs.RED, "Typical pull, down 1"), (m2, cs.NAVY, "Typical pull, down 2")):
        ax.plot(x, m, color=col, lw=2.4, marker="o", ms=7, mec="white", mew=1.5)
        ax.text(len(tr) - 0.55, m[-1], f"{lab}: {cs.fmt_clock(m[-1])}", ha="left", va="center",
                fontsize=11, color=col, fontweight="semibold")
    # gap brackets at the last full season
    i = len(tr) - 2
    for m, b, col, dx, ha in ((m1, b1, cs.RED, -0.14, "right"), (m2, b2, cs.NAVY, 0.14, "left")):
        ax.annotate("", xy=(i + dx, b), xytext=(i + dx, m[i]),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, alpha=0.7, shrinkA=6, shrinkB=2))
        ax.text(i + dx * 1.5, (m[i] + b) / 2, f"{cs.fmt_clock(b - m[i])} gap", ha=ha, va="center",
                fontsize=10.5, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([s + ("\n(to date)" if j == len(tr) - 1 else "") for j, s in enumerate(seasons)])
    yt = np.arange(0, max(b2, m2.max()) + 61, 60)
    ax.set_yticks(yt)
    ax.set_yticklabels([cs.fmt_clock(t) for t in yt])
    ax.set_ylim(0, yt[-1] + 30)
    ax.set_xlim(-0.4, len(tr) - 0.6)
    ax.text(0, 1.02, "minutes left at the pull", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, color=cs.GREY)
    ax.text(len(tr) - 1, 12, f"n = {n_last1} and {n_last2} pulls so far", ha="center", va="bottom",
            fontsize=9.5, color=cs.GREY_LIGHT)
    cs.title_block(fig, f"Coaches have barely moved since {seasons[0]}: the gap is still minutes wide",
                   f"Median time left when the goalie came out, by season, against the model's best time.\n"
                   f"Down 1 the typical pull went from {cs.fmt_clock(m1[0])} to {cs.fmt_clock(m1[-1])}; "
                   f"down 2 from {cs.fmt_clock(m2[0])} to {cs.fmt_clock(m2[-1])}.", sub_gap=0.07)
    cs.footer(fig, cs.source_line(R))
    save(fig, "trend.png",
         f"Line chart of the NHL median goalie pull time by season from {seasons[0]} to {seasons[-1]}. Down one it "
         f"moved from {cs.fmt_clock(m1[0])} to {cs.fmt_clock(m1[-1])} against a best time of {cs.fmt_clock(b1)}; "
         f"down two from {cs.fmt_clock(m2[0])} to {cs.fmt_clock(m2[-1])} against {cs.fmt_clock(b2)}. "
         f"The last season is partial.")


# ----------------------------------------------------------------------------
# 4. Buckets: real vs model
# ----------------------------------------------------------------------------
def clock_to_sec(txt: str) -> int:
    m, s = txt.split(":")
    return int(m) * 60 + int(s)


def chart_buckets(R: dict, k: int) -> None:
    B = list(reversed(R[f"buckets_down{k}"]))   # earliest pulls at the top
    # the checkpoint these teams were measured at is the latest time named in the bucket labels
    start = max(clock_to_sec(w) for b in B for w in b["bucket"].split() if ":" in w)
    y = np.arange(len(B))
    obs = np.array([b["obs"] for b in B]) * 100
    se = np.array([b["se"] for b in B]) * 100
    pred = np.array([b["pred"] for b in B]) * 100
    n = [b["n"] for b in B]

    fig, ax = new_fig(12, 6.5, top=0.78, bottom=0.14, left=0.22)
    ax.grid(axis="x", color=cs.LINE)
    ax.grid(axis="y", visible=False)
    for yi, o, s, p in zip(y, obs, se, pred):
        ax.plot([p, o], [yi, yi], color=cs.LINE, lw=3, zorder=1)
    fmt = (lambda v: f"{v:.1f}%") if max(obs.max(), pred.max()) < 10 else (lambda v: f"{v:.0f}%")
    lower = np.minimum(Z * se, obs)                      # a share cannot go below zero
    ax.errorbar(obs, y, xerr=[lower, Z * se], fmt="o", color=cs.RED, ms=10, mec="white", mew=1.5,
                elinewidth=2, capsize=4, zorder=3)
    ax.plot(pred, y, "s", ms=10, mfc="white", mec=cs.NAVY, mew=2.2, zorder=2)
    for yi, o, p in zip(y, obs, pred):
        ax.text(o, yi - 0.30, fmt(o), ha="center", va="top", fontsize=10.5, color=cs.RED,
                fontweight="semibold")
        ax.text(p, yi + 0.30, fmt(p), ha="center", va="bottom", fontsize=10.5, color=cs.NAVY)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{b['bucket']}\nn = {b['n']:,}" for b in B], fontsize=11)
    for lab in ax.get_yticklabels():
        lab.set_color(cs.NAVY)
    ax.set_ylim(-0.7, len(B) - 0.3)
    ax.invert_yaxis()
    xmax = max((obs + Z * se).max(), pred.max()) * 1.25
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Share that tied the game or won (%)", labelpad=8)
    ax.spines["left"].set_visible(False)
    # one-time key, written as text next to the first row
    ax.text(max(obs[0] + Z * se[0], pred[0]) + xmax * 0.03, 0, "real outcome with 95% range",
            ha="left", va="center", fontsize=10.5, color=cs.RED)
    ax.text(max(obs[0] + Z * se[0], pred[0]) + xmax * 0.03, 0.42, "model", ha="left", va="center",
            fontsize=10.5, color=cs.NAVY)
    ax.text(-0.27, 1.02, "When the goalie came out", ha="left", va="bottom", fontsize=10, color=cs.GREY,
            transform=ax.transAxes, clip_on=False)

    worst = B[-1]
    cs.title_block(fig, f"Down {k} with {cs.fmt_clock(start)} left: the teams that pulled latest came back least",
                   f"Teams trailing by {down_word(k)} at {cs.fmt_clock(start)}, grouped by when they pulled. "
                   f"Red: what happened, with 95% ranges. Navy: what the model expected.\n"
                   f"In the latest group reality ({fmt(worst['obs'] * 100)}) sits below the model "
                   f"({fmt(worst['pred'] * 100)}): the model and the data disagree there.", sub_gap=0.07)
    cs.footer(fig, cs.source_line(R))
    save(fig, f"buckets_down{k}.png",
         f"Dot plot of the share of teams trailing by {down_word(k)} with {cs.fmt_clock(start)} left that tied or won, "
         f"grouped by when the goalie came out, with 95% ranges, against the model's expectation. "
         + " ".join(f"{b['bucket']}: {fmt(b['obs'] * 100)} real, {fmt(b['pred'] * 100)} model, n={b['n']}." for b in B))


# ----------------------------------------------------------------------------
# 5. Calibration
# ----------------------------------------------------------------------------
def chart_calibration(R: dict) -> None:
    bt = R["backtest"]
    cal = bt["calibration_all"]
    pred = np.array([c["pred"] for c in cal])
    obs = np.array([c["obs"] for c in cal])
    n = np.array([c["n"] for c in cal])
    se = np.sqrt(obs * (1 - obs) / n)
    m, b = bt["p_model_all"], bt["p_base"]
    test = [cs.season_label(s) for s in R["meta"]["test"]]
    train = [cs.season_label(s) for s in R["meta"]["train"]]

    fig, ax = plt.subplots(figsize=(8.5, 9), dpi=200)
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.12, right=0.95)
    lim = max(pred.max(), (obs + Z * se).max()) * 1.1
    ax.plot([0, lim], [0, lim], color=cs.GREY_LIGHT, lw=1, ls=":")
    ax.errorbar(pred, obs, yerr=Z * se, fmt="o", color=cs.NAVY, ms=9, mec="white", mew=1.5,
                elinewidth=1.6, capsize=3, zorder=3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(axis="both", color=cs.LINE)
    ax.set_xlabel("Model's chance of getting at least a point", labelpad=8)
    ax.set_ylabel("Share that actually got a point", labelpad=8)
    ax.text(lim * 0.90, lim * 0.905, "perfect calibration", ha="center", va="bottom", fontsize=10,
            color=cs.GREY_LIGHT, rotation=45, rotation_mode="anchor")
    ax.text(0.03, 0.97,
            f"{len(cal)} equal-size bins, {int(n.sum()):,} game states\nfrom {bt['n_games']:,} games; "
            f"bars are 95% ranges",
            transform=ax.transAxes, ha="left", va="top", fontsize=10.5, color=cs.GREY, linespacing=1.3)
    cs.title_block(fig, "On games it never saw, the model is well calibrated",
                   f"Fit on {train[0]} to {train[-1]}, scored on {' and '.join(test)}.\n"
                   f"Predicted {bt['predicted_rate_all'] * 100:.1f}% of late-trailing teams would get a point; "
                   f"{bt['observed_rate'] * 100:.1f}% did.\n"
                   f"Ranking skill is only slightly above a lookup table of past comeback\nrates: "
                   f"Brier {m['brier']:.4f} vs {b['brier']:.4f}, AUC {m['auc']:.3f} vs {b['auc']:.3f}.",
                   title_size=22, sub_size=11.5, sub_gap=0.045)
    cs.footer(fig, cs.source_line(R), y=0.012)
    save(fig, "calibration.png",
         f"Calibration plot of predicted against observed chance of getting at least a point, {len(cal)} bins "
         f"with 95% ranges, all close to the diagonal from {pred.min():.2f} to {pred.max():.2f}. Fit on "
         f"{train[0]} to {train[-1]}, tested on {' and '.join(test)}. Predicted {bt['predicted_rate_all'] * 100:.1f}%, "
         f"observed {bt['observed_rate'] * 100:.1f}%. Brier {m['brier']:.4f} against a baseline of {b['brier']:.4f}; "
         f"AUC {m['auc']:.3f} against {b['auc']:.3f}.")


# ----------------------------------------------------------------------------
# 6. Teams
# ----------------------------------------------------------------------------
def chart_teams(R: dict) -> None:
    T = sorted(R["teams"], key=lambda t: t["points_left_per_82"], reverse=True)
    names = [t["team"] for t in T]
    v = np.array([t["points_left_per_82"] for t in T])
    mean = v.mean()
    y = np.arange(len(T))
    seasons = R["team_seasons"]

    fig, ax = plt.subplots(figsize=(9, 13), dpi=200)
    fig.subplots_adjust(top=0.875, bottom=0.085, left=0.09, right=0.93)
    ax.grid(axis="x", color=cs.LINE)
    ax.grid(axis="y", visible=False)
    colors = [cs.NAVY if i < 3 else (cs.BAR if i < len(T) - 3 else cs.ICE) for i in range(len(T))]
    ax.barh(y, v, color=colors, height=0.72, lw=0, zorder=2)
    for yi, val in zip(y, v):
        ax.text(val + 0.006, yi, f"{val:.2f}", ha="left", va="center", fontsize=9.5, color=cs.GREY)
    ax.axvline(mean, color=cs.RED, lw=1.4, ls=(0, (4, 3)), zorder=3)
    ax.text(mean + 0.006, -0.62, f"average of the {len(T)} teams: {mean:.2f}", ha="left", va="center",
            fontsize=10.5, color=cs.RED, fontweight="semibold")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11, color=cs.NAVY, family=cs.DISPLAY, fontweight="semibold")
    ax.invert_yaxis()
    ax.set_ylim(len(T) - 0.3, -1.0)
    ax.set_xlim(0, v.max() * 1.15)
    ax.set_xlabel("Standings points left on the table per 82 games", labelpad=8)
    ax.spines["left"].set_visible(False)
    cs.title_block(fig, "What late pulls cost each team",
                   f"Expected standings points lost against the best pull time, per 82 games, "
                   f"{seasons[0]} to {seasons[-1]}.\nEach team is judged against its own best time, "
                   f"built from its own 6 on 5 scoring rates blended with the league's.",
                   y=0.975, title_size=24, sub_gap=0.03)
    cs.footer(fig, cs.source_line(R), y=0.012)
    save(fig, "teams.png",
         f"Horizontal bar chart ranking all {len(T)} NHL teams by standings points lost to late goalie pulls per "
         f"82 games, {seasons[0]} to {seasons[-1]}. {names[0]} loses the most at {v[0]:.2f}, {names[-1]} the "
         f"least at {v[-1]:.2f}; the league average is {mean:.2f}.")


# ----------------------------------------------------------------------------
# 7. Sensitivity
# ----------------------------------------------------------------------------
def chart_sensitivity(R: dict) -> None:
    S = R["sensitivity"]
    main = S[0]
    ci = R["optimal"]["points_ci90"]
    y = np.arange(len(S))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), dpi=200, sharey=True)
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.33, right=0.97, wspace=0.12)
    for ax, k, col in ((axes[0], 1, cs.RED), (axes[1], 2, cs.NAVY)):
        key = f"down{k}"
        vals = np.array([s[key] for s in S], dtype=float)
        lo, best, hi = ci[f"-{k}"]
        ax.axvspan(hi, lo, color=cs.ICE, alpha=0.35, lw=0)
        ax.axvline(best, color=col, lw=1.2, ls=(0, (4, 3)))
        ax.grid(axis="x", color=cs.LINE)
        ax.grid(axis="y", visible=False)
        ax.plot(vals, y, "o", ms=10, color=col, mec="white", mew=1.5, zorder=3)
        span = vals.max() - vals.min()
        lo_x = min(vals.min(), lo) - max(30, span * 0.3)
        hi_x = max(vals.max(), hi) + max(30, span * 0.3)
        for yi, val in zip(y, vals):
            ax.text(val - (hi_x - lo_x) * 0.03, yi, cs.fmt_clock(val), ha="left", va="center",
                    fontsize=10, color=col, bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.85), zorder=4)
        step = 30 if (hi_x - lo_x) <= 240 else 60
        ticks = np.arange(np.ceil(lo_x / step) * step, hi_x + 1, step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([cs.fmt_clock(t) for t in ticks])
        ax.set_xlim(hi_x, lo_x)          # clock runs left to right
        ax.set_title(f"Down {k}: {cs.fmt_clock(vals.min())} to {cs.fmt_clock(vals.max())} across all cases",
                     loc="left", fontsize=13, fontweight="bold", family=cs.DISPLAY, color=col, pad=10)
        ax.set_xlabel("Best time to pull (time left)", labelpad=6)
        ax.spines["left"].set_visible(False)
        ax.text(hi - 2, len(S) - 0.45, f"90% range {cs.fmt_clock(hi)} to {cs.fmt_clock(lo)}", ha="left",
                va="bottom", fontsize=9.5, color=cs.GREY)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([s["case"] for s in S], fontsize=10.5, color=cs.NAVY)
    axes[0].invert_yaxis()
    axes[0].set_ylim(len(S) - 0.3, -0.7)
    cs.title_block(fig, "The answer barely moves under other assumptions",
                   f"Best pull time in the main case and {len(S) - 1} alternatives. Main case: {cs.fmt_clock(main['down1'])} down 1, "
                   f"{cs.fmt_clock(main['down2'])} down 2. Shaded: the main case's 90% range.")
    cs.footer(fig, cs.source_line(R))
    d1 = [s["down1"] for s in S]
    d2 = [s["down2"] for s in S]
    save(fig, "sensitivity.png",
         f"Dot plot of the best goalie pull time in the main case and {len(S) - 1} alternative assumptions. Down one it stays between "
         f"{cs.fmt_clock(min(d1))} and {cs.fmt_clock(max(d1))}; down two between {cs.fmt_clock(min(d2))} and "
         f"{cs.fmt_clock(max(d2))}. Cases: " + "; ".join(f"{s['case']}: {cs.fmt_clock(s['down1'])} and "
                                                        f"{cs.fmt_clock(s['down2'])}" for s in S) + ".")


# ----------------------------------------------------------------------------
# 8. Goal rates, goalie in vs out
# ----------------------------------------------------------------------------
def chart_rates(R: dict) -> None:
    rows = {(r["d"], r["pulled"]): r for r in R["rates"]}
    ds = [-1, -2, -3]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200, sharey=True)
    fig.subplots_adjust(top=0.75, bottom=0.16, left=0.10, right=0.97, wspace=0.10)
    y = np.arange(len(ds))
    xmax = max(r[k] for r in R["rates"] if r["d"] < 0 for k in ("gf60", "ga60")) * 1.18   # one scale for both panels
    for ax, key, title in ((axes[0], "gf60", "Goals scored per 60"), (axes[1], "ga60", "Goals allowed per 60")):
        ax.grid(axis="x", color=cs.LINE)
        ax.grid(axis="y", visible=False)
        for yi, d in zip(y, ds):
            a, b = rows[(d, 0)][key], rows[(d, 1)][key]
            ax.plot([a, b], [yi, yi], color=cs.LINE, lw=4, zorder=1, solid_capstyle="round")
            ax.plot([a], [yi], "o", ms=11, color=cs.BAR, mec="white", mew=1.5, zorder=3)
            ax.plot([b], [yi], "o", ms=11, color=cs.NAVY if key == "gf60" else cs.RED, mec="white", mew=1.5, zorder=3)
            ax.text(a, yi + 0.26, f"{a:.1f}", ha="center", va="top", fontsize=10.5, color=cs.GREY)
            ax.text(b, yi + 0.26, f"{b:.1f}", ha="center", va="top", fontsize=10.5,
                    color=cs.NAVY if key == "gf60" else cs.RED, fontweight="semibold")
            ax.text(b + xmax * 0.025, yi, f"×{b / a:.1f}", ha="left", va="center", fontsize=10.5, color=cs.GREY)
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", family=cs.DISPLAY, color=cs.NAVY, pad=10)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("goals per 60 minutes", labelpad=6)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"Down {-d}" for d in ds], fontsize=12, color=cs.NAVY, fontweight="semibold")
    axes[0].invert_yaxis()
    axes[0].set_ylim(len(ds) - 0.4, -0.7)
    axes[0].text(rows[(-1, 0)]["gf60"], -0.42, "goalie in", ha="center", va="bottom", fontsize=10, color=cs.GREY)
    axes[0].text(rows[(-1, 1)]["gf60"], -0.42, "goalie out", ha="center", va="bottom", fontsize=10, color=cs.NAVY)
    axes[1].text(rows[(-1, 0)]["ga60"], -0.42, "goalie in", ha="center", va="bottom", fontsize=10, color=cs.GREY)
    axes[1].text(rows[(-1, 1)]["ga60"], -0.42, "goalie out", ha="center", va="bottom", fontsize=10, color=cs.RED)
    g_in, g_out = rows[(-1, 0)]["gf60"], rows[(-1, 1)]["gf60"]
    a_in, a_out = rows[(-1, 0)]["ga60"], rows[(-1, 1)]["ga60"]
    cs.title_block(fig, "Pulling the goalie multiplies both rates, and that is the point",
                   f"Third-period goals per 60 minutes by score, goalie in net versus pulled. Down 1, scoring goes "
                   f"from {g_in:.1f} to {g_out:.1f} and goals against from {a_in:.1f} to {a_out:.1f}.\n"
                   f"Losing by two is worth the same zero points as losing by one, so the extra goals against cost "
                   f"little; the tying goal is worth a point plus an overtime chance.",
                   sub_gap=0.075)
    cs.footer(fig, cs.source_line(R))
    save(fig, "rates.png",
         "Two dumbbell charts of third-period goals scored and allowed per 60 minutes with the goalie in net versus "
         "pulled, for teams trailing by one, two and three. " +
         " ".join(f"Down {-d}: scored {rows[(d, 0)]['gf60']:.1f} to {rows[(d, 1)]['gf60']:.1f}, allowed "
                  f"{rows[(d, 0)]['ga60']:.1f} to {rows[(d, 1)]['ga60']:.1f}." for d in ds))


def main() -> None:
    cs.setup()
    R = cs.load()
    OUT.mkdir(parents=True, exist_ok=True)
    chart_hist(R, 1)
    chart_hist(R, 2)
    chart_curve(R, 1)
    chart_curve(R, 2)
    chart_cost_curves(R)
    chart_trend(R)
    chart_buckets(R, 1)
    chart_buckets(R, 2)
    chart_calibration(R)
    chart_teams(R)
    chart_sensitivity(R)
    chart_rates(R)
    (OUT / "alt.json").write_text(json.dumps(ALT, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", (OUT / "alt.json").relative_to(cs.ROOT))


if __name__ == "__main__":
    main()
