"""Shared look for every chart: palette, fonts, helpers.

Imported by make_hero.py, make_charts_v2.py and make_cover.py. Every number
drawn by those scripts is read from docs/data/results.json through load().
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "data" / "results.json"

# Palette (site identity). Navy is ink, red is the one accent, ice blue is
# only ever a fill under a direct label, never the sole carrier of meaning.
NAVY = "#0B1B3B"
RED = "#C8102E"
ICE = "#7FB3D5"
ICE_LIGHT = "#EAF3FA"
GREY = "#5B6475"
GREY_LIGHT = "#8A94A6"
LINE = "#d9e1ea"
BAR = "#8A94A6"      # neutral grey for "what the league does"
WHITE = "#FFFFFF"
NAVY_2 = "#16336b"   # second stop of the site's hero gradient


def _first_available(candidates: list[str], fallback: str = "DejaVu Sans") -> str:
    have = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in have:
            return c
    return fallback


# Display face for titles and big numbers, text face for everything else.
DISPLAY = _first_available(["Barlow Condensed", "Oswald", "Roboto Condensed"])
TEXT = _first_available(["Barlow", "Inter", "Source Sans 3", "IBM Plex Sans", "Roboto"])


def setup() -> None:
    plt.rcParams.update({
        "font.family": TEXT,
        "font.size": 12,
        "axes.edgecolor": GREY_LIGHT,
        "axes.labelcolor": GREY,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": LINE,
        "grid.linewidth": 0.8,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "ytick.left": False,
        "legend.frameon": False,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "svg.fonttype": "none",   # keep text as text in SVG
        "pdf.fonttype": 42,
    })


def load() -> dict:
    with open(RESULTS, encoding="utf-8") as f:
        return json.load(f)


def season_label(s: int) -> str:
    """20212022 -> '2021-22'."""
    s = int(s)
    return f"{s // 10000}-{str(s % 10000)[-2:]}"


def fmt_clock(sec: float) -> str:
    """Seconds left -> m:ss."""
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def source_line(R: dict) -> str:
    """Source caption built from meta. If run_analysis.py writes
    meta.data_through (an ISO date), the exact date is shown; otherwise the
    last season in meta.seasons with 'to date'."""
    seasons = R["meta"]["seasons"]
    first = season_label(seasons[0])
    through = R["meta"].get("data_through")
    if through:
        d = date.fromisoformat(str(through)[:10])
        end = d.strftime("%b %d, %Y").replace(" 0", " ")
    else:
        end = f"{season_label(seasons[-1])} season to date"
    return f"Source: NHL public play-by-play, {first} to {end}"


def credit(R: dict, extra: str | None = None) -> str:
    s = source_line(R)
    return f"{s}  ·  {extra}" if extra else s


def title_block(fig, title: str, subtitle: str | None = None, *,
                x: float = 0.04, y: float = 0.965, title_size: float = 24,
                sub_size: float = 13, color: str = NAVY, sub_color: str = GREY,
                sub_gap: float = 0.062) -> None:
    fig.text(x, y, title, ha="left", va="top", fontsize=title_size,
             fontweight="bold", family=DISPLAY, color=color)
    if subtitle:
        fig.text(x, y - sub_gap, subtitle, ha="left", va="top",
                 fontsize=sub_size, color=sub_color, family=TEXT)


def footer(fig, text: str, *, x: float = 0.04, y: float = 0.02,
           size: float = 9.5, color: str = GREY_LIGHT) -> None:
    fig.text(x, y, text, ha="left", va="bottom", fontsize=size, color=color,
             family=TEXT)


def clock_ticks(ax, lo: float, hi: float, step: int = 60, reverse: bool = True,
                size: float | None = None) -> None:
    """Game-clock x axis: seconds left, drawn so time runs left to right."""
    ticks = list(range(int(lo), int(hi) + 1, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt_clock(t) for t in ticks], fontsize=size)
    ax.set_xlim(hi, lo) if reverse else ax.set_xlim(lo, hi)


def alt_text_file(path: Path, alt: str) -> None:
    """Write the alt text next to the image so index.html can carry it."""
    path.with_suffix(".alt.txt").write_text(alt.strip() + "\n", encoding="utf-8")


def save(fig, path: Path, alt: str, *, dpi: int = 200, svg: bool = False,
         metadata_title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"Title": metadata_title or path.stem, "Author": "John Khan",
            "Description": alt.strip()}
    fig.savefig(path, dpi=dpi, metadata=meta)
    if svg:
        # glyphs as paths so the SVG looks the same on machines without the fonts
        with plt.rc_context({"svg.fonttype": "path"}):
            fig.savefig(path.with_suffix(".svg"), metadata={"Title": meta["Title"]})
    alt_text_file(path, alt)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))
