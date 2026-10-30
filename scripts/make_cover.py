"""Devpost cover (1500x1000) and thumbnail (600x400).

A navy card: the question, the two numbers that answer it, and the real
distribution of down-1 pulls drawn as a silhouette on a game clock. Every
number is read from docs/data/results.json.
Run from the repo root:  python scripts/make_cover.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import chartstyle as cs

OUT = cs.ROOT / "docs" / "img" / "v2"


def facts(R: dict) -> dict:
    best = R["optimal"]["points"]["-1"]
    lo, _, hi = R["optimal"]["points_ci90"]["-1"]
    med = R["median_pull"]["-1"]
    counts = np.asarray(R["hist_down1"], dtype=int)
    edges = np.arange(0, 15 * (len(counts) + 1), 15)
    return dict(best=best, lo=lo, hi=hi, med=med, gap=best - med, counts=counts, edges=edges,
                n=int(counts.sum()), games=R["meta"]["games"], pulls=R["meta"]["pull_events"],
                points=R["points_left"]["league_per_full_season"],
                best2=R["optimal"]["points"]["-2"], med2=R["median_pull"]["-2"],
                first=cs.season_label(R["meta"]["seasons"][0]))


def make(R: dict) -> plt.Figure:
    F = facts(R)
    fig = plt.figure(figsize=(15, 10), dpi=100, facecolor=cs.NAVY)
    fig.patch.set_facecolor(cs.NAVY)

    # question
    fig.text(0.06, 0.93, "AQX SPORTS ANALYTICS DATA BOWL 4.0  ·  NHL", ha="left", va="top", fontsize=15,
             color=cs.ICE, family=cs.TEXT, fontweight="semibold")
    fig.text(0.06, 0.865, "Pull the goalie sooner?", ha="left", va="top", fontsize=78, color="white",
             family=cs.DISPLAY, fontweight="bold")
    fig.text(0.06, 0.735, f"Yes. Coaches pull {cs.fmt_clock(F['gap'])} too late.", ha="left", va="top",
             fontsize=40, color=cs.RED, family=cs.DISPLAY, fontweight="bold")

    # the two numbers
    for x, val, lab, col in ((0.06, cs.fmt_clock(F["best"]), "best time to pull, down 1", "white"),
                             (0.36, cs.fmt_clock(F["med"]), "when coaches actually pull", cs.RED)):
        fig.text(x, 0.615, val, ha="left", va="top", fontsize=96, color=col, family=cs.DISPLAY, fontweight="bold")
        fig.text(x, 0.465, lab, ha="left", va="top", fontsize=20, color=cs.ICE, family=cs.TEXT)
    fig.text(0.66, 0.615, f"{F['points']:.0f}", ha="left", va="top", fontsize=96, color="white",
             family=cs.DISPLAY, fontweight="bold")
    fig.text(0.66, 0.465, "standings points the league\nleaves on the ice each season", ha="left", va="top",
             fontsize=20, color=cs.ICE, family=cs.TEXT, linespacing=1.2)

    # the distribution as a silhouette on a game clock
    ax = fig.add_axes([0.06, 0.14, 0.88, 0.23])
    ax.set_facecolor(cs.NAVY)
    counts, edges = F["counts"], F["edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    ax.axvspan(F["hi"], F["lo"], color=cs.ICE, alpha=0.25, lw=0)
    ax.bar(centers, counts, width=13.5, color=cs.ICE, alpha=0.9, lw=0)
    ax.axvline(F["best"], color="white", lw=2.5, ls=(0, (4, 3)))
    ax.axvline(F["med"], color=cs.RED, lw=4)
    ax.set_xlim(edges[-1], 0)
    ax.set_ylim(0, counts.max() * 1.35)
    ticks = list(range(0, int(edges[-1]) + 1, 60))
    ax.set_xticks(ticks)
    ax.set_xticklabels([cs.fmt_clock(t) for t in ticks], fontsize=15, color=cs.ICE)
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(cs.ICE)
    ax.tick_params(axis="x", colors=cs.ICE, length=0, pad=8)
    ax.text(F["best"] - 5, counts.max() * 1.25, f"model: {cs.fmt_clock(F['best'])}", ha="left", va="top",
            fontsize=17, color="white", family=cs.TEXT, fontweight="semibold")
    ax.text(F["med"] - 6, counts.max() * 1.25, f"NHL median: {cs.fmt_clock(F['med'])}", ha="left", va="top",
            fontsize=17, color=cs.RED, family=cs.TEXT, fontweight="bold")
    ax.text(edges[-1] - 6, counts.max() * 1.25, f"{F['n']:,} real pulls while down 1,\nby time left when the goalie came out",
            ha="left", va="top", fontsize=15, color=cs.ICE, family=cs.TEXT, linespacing=1.2)

    # footer
    last = cs.season_label(R["meta"]["seasons"][-1])
    fig.text(0.06, 0.04, f"{F['games']:,} games  ·  {F['pulls']:,} goalie pulls  ·  {F['first']} to {last}  ·  "
                         f"backtested on seasons the model never saw", ha="left", va="bottom", fontsize=15,
             color=cs.ICE, family=cs.TEXT)
    fig.text(0.94, 0.04, "John Khan  ·  Toronto", ha="right", va="bottom", fontsize=15, color=cs.ICE,
             family=cs.TEXT)
    return fig


def main() -> None:
    cs.setup()
    R = cs.load()
    F = facts(R)
    alt = (f"Navy cover card. Title: Pull the goalie sooner? Answer: yes, coaches pull {cs.fmt_clock(F['gap'])} "
           f"too late. Three big numbers: {cs.fmt_clock(F['best'])} best time to pull down 1, "
           f"{cs.fmt_clock(F['med'])} when coaches actually pull, {F['points']:.0f} standings points the league "
           f"leaves on the ice each season. Below, the distribution of {F['n']:,} real down-1 pulls on a game clock, "
           f"piled around the median and far from the model's time. {F['games']:,} games, {F['pulls']:,} pulls.")
    fig = make(R)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "cover.png", dpi=100, facecolor=cs.NAVY,
                metadata={"Title": "Pull the Goalie Sooner? cover", "Author": "John Khan", "Description": alt})
    fig.savefig(OUT / "thumb.png", dpi=40, facecolor=cs.NAVY,
                metadata={"Title": "Pull the Goalie Sooner? thumbnail", "Author": "John Khan", "Description": alt})
    (OUT / "cover.alt.txt").write_text(alt + "\n", encoding="utf-8")
    plt.close(fig)
    print("wrote docs/img/v2/cover.png and docs/img/v2/thumb.png")


if __name__ == "__main__":
    main()
