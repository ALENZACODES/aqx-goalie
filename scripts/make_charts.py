"""Static charts for the README and the Devpost gallery, from docs/data/results.json."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "docs/data/results.json").read_text())
IMG = ROOT / "docs/img"; IMG.mkdir(exist_ok=True, parents=True)
NAVY, RED, ICE, GREY = "#0B1B3B", "#C8102E", "#7FB3D5", "#8A94A6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 150})
mmss = lambda s: f"{int(s)//60}:{int(round(s))%60:02d}"
META = {"Title": "Pull the Goalie Sooner?", "Author": "John Khan", "Software": None}

def save(fig, name):
    fig.tight_layout(); fig.savefig(IMG / name, metadata=META); plt.close(fig)

opt = R["optimal"]; ci = opt["points_ci90"]; med = R["median_pull"]

# 1. value curves
for d, start_label in (("-1", "down 1"), ("-2", "down 2")):
    c = R["curves"][d]; T = np.array(c["T"]); y = np.array(c["points"])
    loss = (y.max() - y) * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(T / 60, -np.array(c["regret_hi"]) * 100, -np.array(c["regret_lo"]) * 100, color=ICE, alpha=.35, label="90% range from data uncertainty")
    ax.plot(T / 60, loss, color=NAVY, lw=2.5, label="Points lost per 100 games vs the best pull time")
    ax.axvline(opt["points"][d] / 60, color=NAVY, ls="--", lw=1.2)
    ax.axvline(med[d] / 60, color=RED, lw=2)
    ax.text(opt["points"][d] / 60, ax.get_ylim()[1] * .55, f"  Best: {mmss(opt['points'][d])} left", color=NAVY, fontweight="bold")
    ax.text(med[d] / 60, ax.get_ylim()[1] * .42, f"  NHL median: {mmss(med[d])}", color=RED, fontweight="bold")
    ax.invert_xaxis(); ax.set_xlabel("Minutes left in regulation when the goalie is pulled")
    ax.set_ylabel("Standings points lost per 100 games")
    ax.set_title(f"Trailing {start_label} with 10:00 left: when should the goalie come out?")
    ax.legend(loc="upper center", frameon=False, fontsize=9)
    save(fig, f"curve_down{d[1]}.png")

# 2. histogram of real pull times, down 1
h = np.array(R["hist_down1"]); edges = np.arange(0, 430, 15)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(edges[:-1] / 60 + 0.125, h, width=0.23, color=GREY)
ax.axvspan(ci["-1"][0] / 60, ci["-1"][2] / 60, color=ICE, alpha=.35, label="Model's best window (90% range)")
ax.axvline(opt["points"]["-1"] / 60, color=NAVY, ls="--", lw=1.5, label=f"Best: {mmss(opt['points']['-1'])}")
ax.axvline(med["-1"] / 60, color=RED, lw=2, label=f"NHL median: {mmss(med['-1'])}")
ax.invert_xaxis(); ax.set_xlabel("Minutes left when the goalie came out (trailing by 1)"); ax.set_ylabel("Pulls, 2021-22 to now")
ax.set_title(f"{sum(h):,} real pulls while down 1: almost all come too late"); ax.legend(frameon=False)
save(fig, "hist_down1.png")

# 3. trend
tr = R["trend"]; xs = [t["season"] for t in tr]
fig, ax = plt.subplots(figsize=(8, 4.2))
for d, col in (("1", RED), ("2", NAVY)):
    ax.plot(xs, [t[f"median_down{d}"] / 60 for t in tr], "o-", color=col, lw=2, label=f"NHL median pull, down {d}")
    ax.axhline(opt["points"][f"-{d}"] / 60, color=col, ls="--", lw=1, alpha=.8)
    ax.text(len(xs) - .6, opt["points"][f"-{d}"] / 60 + .07, f"best, down {d}", color=col, fontsize=9, ha="right")
ax.set_ylabel("Minutes left at the pull"); ax.set_ylim(0, 7.5)
ax.set_title("Coaches are pulling a little earlier, but not nearly enough"); ax.legend(frameon=False, loc="center left")
ax.text(len(xs) - 1, 0.25, "2026-27 to date", ha="center", fontsize=8, color=GREY)
save(fig, "trend.png")

# 4. calibration
cal = R["backtest"]["calibration_all"]
fig, ax = plt.subplots(figsize=(5.2, 5))
p = [c["pred"] for c in cal]; o = [c["obs"] for c in cal]; n = [c["n"] for c in cal]
se = [np.sqrt(max(oo * (1 - oo), 1e-4) / nn) for oo, nn in zip(o, n)]
ax.plot([0, .4], [0, .4], color=GREY, ls=":")
ax.errorbar(p, o, yerr=1.96 * np.array(se), fmt="o", color=NAVY, capsize=3)
ax.set_xlim(0, .4); ax.set_ylim(0, .4)
ax.set_xlabel("Model's chance of getting at least a point"); ax.set_ylabel("What actually happened")
ax.set_title("Backtest on 2025-26 and 2026-27\n(model trained on earlier seasons only)", fontsize=11)
save(fig, "calibration.png")

# 5. buckets
b = R["buckets_down1"]
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(b))
ax.bar(x - .2, [r["obs"] * 100 for r in b], .4, yerr=[1.96 * r["se"] * 100 for r in b], color=RED, capsize=3, label="Real outcome")
ax.bar(x + .2, [r["pred"] * 100 for r in b], .4, color=NAVY, label="Model")
ax.set_xticks(x, [f"{r['bucket']}\n(n={r['n']})" for r in b], fontsize=9)
ax.set_ylabel("Tied it up or won (%)"); ax.set_xlabel("When the goalie came out, for teams down 1 with 3:00 left")
ax.set_title("Down 1 at 3:00: teams that waited past 1:30 rarely came back"); ax.legend(frameon=False)
save(fig, "buckets_down1.png")

# 6. teams
T = sorted(R["teams"], key=lambda r: r["points_left_per_82"])
fig, ax = plt.subplots(figsize=(7, 8))
ax.barh([t["team"] for t in T], [t["points_left_per_82"] for t in T], color=NAVY)
ax.set_xlabel("Standings points left on the table per 82 games")
ax.set_title(f"Cost of late pulls by team, {R['team_seasons'][0]} to now")
save(fig, "teams.png")
# 7. cover card for the Devpost gallery (3:2)
fig = plt.figure(figsize=(9, 6)); fig.patch.set_facecolor(NAVY)
fig.text(.06, .86, "PULL THE GOALIE SOONER?", color="white", fontsize=30, fontweight="bold")
fig.text(.06, .79, f"{R['meta']['games']:,} NHL games, {R['meta']['pull_events']:,} goalie pulls, one Markov model", color=ICE, fontsize=14)
for i, (big, lab, sub) in enumerate([(mmss(opt["points"]["-1"]), "best pull, down 1", f"NHL median {mmss(med['-1'])}"),
                                     (mmss(opt["points"]["-2"]), "best pull, down 2", f"NHL median {mmss(med['-2'])}"),
                                     (f"{R['points_left']['league_per_full_season']:.0f}", "standings points lost", "league wide, every season")]):
    x = .06 + i * .31
    fig.text(x, .50, big, color="white", fontsize=50, fontweight="bold")
    fig.text(x, .42, lab, color="white", fontsize=14)
    fig.text(x, .37, sub, color=ICE, fontsize=12)
fig.text(.06, .16, "Backtested on 2025-26 and 2026-27 games the model never saw.", color="white", fontsize=13)
fig.text(.06, .10, "John Khan  ·  Toronto, Ontario  ·  AQX Sports Analytics Data Bowl 4.0", color=ICE, fontsize=11)
fig.savefig(IMG / "cover.png", metadata=META, facecolor=NAVY); plt.close(fig)
print("charts:", sorted(p.name for p in IMG.glob("*.png")))
