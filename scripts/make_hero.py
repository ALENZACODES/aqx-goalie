"""Hero chart: where NHL coaches actually pull the goalie when down 1,
against the model's best window. One picture, one message.

Outputs docs/img/hero.png (1600x900), docs/img/hero_mobile.png (1080x1350)
and docs/img/hero.svg. Every number is read from docs/data/results.json.
Run from the repo root:  python scripts/make_hero.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import chartstyle as cs

OUT = cs.ROOT / "docs" / "img"


def facts(R: dict) -> dict:
    """Pull every number the hero needs out of results.json."""
    best = R["optimal"]["points"]["-1"]
    lo, _, hi = R["optimal"]["points_ci90"]["-1"]
    med = R["median_pull"]["-1"]
    counts = np.asarray(R["hist_down1"], dtype=int)
    edges = np.arange(0, 15 * (len(counts) + 1), 15)        # 0..420 by 15
    n = int(counts.sum())
    # share of down-1 pulls at the best time or earlier, pooled over seasons
    # (the same pooled figure app.js shows in the hero card)
    tr = R["trend"]
    n_tr = sum(t["n_down1"] for t in tr)
    on_time = sum(t["share_optimal_down1"] * t["n_down1"] for t in tr)
    return dict(best=best, lo=lo, hi=hi, med=med, gap=best - med,
                counts=counts, edges=edges, n=n,
                n_on_time=int(round(on_time)), share_on_time=on_time / n_tr,
                overflow_edge=edges[-2], first_season=cs.season_label(R["meta"]["seasons"][0]))


def draw_distribution(ax, F: dict, *, portrait: bool = False) -> None:
    """The histogram, the best window, the median and, in landscape, the gap arrow.
    The portrait version is viewed at about a third of its pixel size on a phone,
    so its type is 2.3x larger and it carries fewer labels."""
    s = 2.3 if portrait else 1.15
    counts, edges = F["counts"], F["edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    ymax = counts.max() * 1.30

    # best window (90% range) and best time
    ax.axvspan(F["hi"], F["lo"], color=cs.ICE, alpha=0.32, lw=0, zorder=0)
    ax.axvline(F["best"], color=cs.NAVY, lw=1.6 * s ** 0.5, ls=(0, (4, 3)), zorder=3)

    # what coaches did; bins after the best time in navy, the overflow bin hatched
    colors = [cs.NAVY if c > F["best"] else cs.BAR for c in centers]
    ax.bar(centers, counts, width=13.2, color=colors, zorder=2, lw=0)
    ax.bar(centers[-1], counts[-1], width=13.2, facecolor="none", edgecolor=cs.GREY,
           hatch="////", lw=0.8, zorder=3)

    # the typical pull
    ax.axvline(F["med"], color=cs.RED, lw=2.2 * s ** 0.5, zorder=4)

    ax.set_ylim(0, ymax)
    cs.clock_ticks(ax, 0, edges[-1], 60, size=11.5 * s)
    ax.set_yticks([0, 100, 200, 300, 400])
    ax.tick_params(axis="y", labelsize=10.5 * s)
    ax.tick_params(axis="x", pad=6 * s)
    ax.spines["bottom"].set_color(cs.GREY_LIGHT)
    ax.set_xlabel("Time left in regulation when the goalie came out",
                  fontsize=12 * s, color=cs.GREY, labelpad=8 * s)
    ax.text(0, 1.015, "pulls", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10.5 * s, color=cs.GREY)

    y_lab = ymax * 0.985
    ax.text(F["hi"] + 4, y_lab, f"Best window\n{cs.fmt_clock(F['hi'])} to {cs.fmt_clock(F['lo'])}",
            ha="right", va="top", fontsize=12 * s, color=cs.NAVY, fontweight="semibold", linespacing=1.15)
    ax.text(F["best"] - 7, ymax * (0.70 if portrait else 0.66), f"Best: {cs.fmt_clock(F['best'])}",
            ha="left", va="center", fontsize=14 * s, color=cs.NAVY, fontweight="bold", family=cs.DISPLAY)
    if portrait:
        ax.text(F["med"] + 5, y_lab, f"Typical NHL\npull: {cs.fmt_clock(F['med'])}", ha="right", va="top",
                fontsize=14 * s, color=cs.RED, fontweight="bold", family=cs.DISPLAY, linespacing=1.05)
    else:
        ax.text(F["med"] - 4, y_lab, f"Typical NHL pull: {cs.fmt_clock(F['med'])}", ha="left", va="top",
                fontsize=14 * s, color=cs.RED, fontweight="bold", family=cs.DISPLAY)
        y_arrow = ymax * 0.875
        ax.add_patch(FancyArrowPatch((F["best"], y_arrow), (F["med"], y_arrow), arrowstyle="-|>",
                                     mutation_scale=18, lw=2, color=cs.RED, shrinkA=0, shrinkB=2, zorder=5))
        ax.text((F["best"] + F["med"]) / 2, y_arrow + ymax * 0.03,
                f"{cs.fmt_clock(F['gap'])} later than the model says", ha="center", va="bottom",
                fontsize=12.5 * s, color=cs.RED, fontweight="semibold")

    share = F["share_on_time"]
    share_txt = f"{share * 100:.1f}%" if share < 0.10 else f"{share * 100:.0f}%"
    ax.text(edges[-1] - 14, ymax * (0.44 if portrait else 0.50),
            f"{F['n_on_time']} of {F['n']:,} pulls ({share_txt})\ncame within 10 s of\nthe best time, or earlier"
            if portrait else
            f"{F['n_on_time']} of {F['n']:,} pulls ({share_txt})\ncame within 10 s of the best time, or earlier",
            ha="left", va="center", fontsize=11.5 * s, color=cs.GREY, linespacing=1.2)

    ov = counts[-1]
    ax.annotate(f"{ov} pulls at {cs.fmt_clock(F['overflow_edge'])}\nor earlier (all in this bin)" if portrait
                else f"{ov} pulls at {cs.fmt_clock(F['overflow_edge'])} or earlier\n(all shown in this bin)",
                xy=(centers[-1], ov), xytext=(centers[-1] - 22, ymax * 0.22), ha="left", va="center",
                fontsize=10 * s, color=cs.GREY, linespacing=1.15,
                arrowprops=dict(arrowstyle="-", color=cs.GREY_LIGHT, lw=0.9,
                                connectionstyle="arc3,rad=-0.2"))


def make(R: dict, F: dict, *, portrait: bool) -> plt.Figure:
    title = f"NHL coaches pull the goalie {cs.fmt_clock(F['gap'])} too late"
    if portrait:
        fig = plt.figure(figsize=(10.8, 13.5), dpi=100)
        fig.text(0.05, 0.975, "NHL coaches pull the goalie", ha="left", va="top", fontsize=50,
                 fontweight="bold", family=cs.DISPLAY, color=cs.NAVY)
        fig.text(0.05, 0.922, f"{cs.fmt_clock(F['gap'])} too late", ha="left", va="top", fontsize=50,
                 fontweight="bold", family=cs.DISPLAY, color=cs.RED)
        fig.text(0.05, 0.862, f"Best time to pull, down 1: {cs.fmt_clock(F['best'])} left.\n"
                              f"Coaches typically wait until {cs.fmt_clock(F['med'])}.",
                 ha="left", va="top", fontsize=26, color=cs.GREY, linespacing=1.25)
        ax = fig.add_axes([0.12, 0.165, 0.85, 0.575])
        draw_distribution(ax, F, portrait=True)
        cs.footer(fig, cs.source_line(R) + "\nShaded: 90% range of the best time across resampled goal rates",
                  x=0.05, y=0.018, size=17)
    else:
        fig = plt.figure(figsize=(16, 9), dpi=100)
        cs.title_block(fig, title, None, x=0.05, y=0.95, title_size=44)
        sub = (f"Every goalie pull while trailing by one since {F['first_season']}, "
               f"{F['n']:,} pulls. A model of the last ten minutes says pull with "
               f"{cs.fmt_clock(F['best'])} left; coaches typically wait until {cs.fmt_clock(F['med'])}.")
        fig.text(0.05, 0.862, sub, ha="left", va="top", fontsize=16.5, color=cs.GREY)
        ax = fig.add_axes([0.065, 0.15, 0.90, 0.60])
        draw_distribution(ax, F)
        cs.footer(fig, cs.credit(R, "Window: 90% range of the best time across resampled goal rates"),
                  x=0.05, y=0.03, size=11)
    return fig


def main() -> None:
    cs.setup()
    R = cs.load()
    F = facts(R)
    alt = (f"Histogram of {F['n']:,} NHL goalie pulls while trailing by one goal, "
           f"{F['first_season']} onward, on a game clock from {cs.fmt_clock(F['edges'][-1])} to 0:00. "
           f"The model's best window, {cs.fmt_clock(F['hi'])} to {cs.fmt_clock(F['lo'])} left with the "
           f"best time at {cs.fmt_clock(F['best'])}, is shaded and almost empty. The pulls pile up "
           f"around the typical NHL pull at {cs.fmt_clock(F['med'])}, {cs.fmt_clock(F['gap'])} later. "
           f"{F['n_on_time']} pulls, about {F['share_on_time'] * 100:.1f} percent, came within 10 seconds of the best time, or earlier.")
    cs.save(make(R, F, portrait=False), OUT / "hero.png", alt, dpi=100, svg=True,
            metadata_title="Pull the Goalie Sooner? hero chart")
    cs.save(make(R, F, portrait=True), OUT / "hero_mobile.png", alt, dpi=100,
            metadata_title="Pull the Goalie Sooner? hero chart, portrait")


if __name__ == "__main__":
    main()
