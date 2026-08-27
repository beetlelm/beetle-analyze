"""
plot_style.py — Publication-quality figure style for beetle-analyze.

Every plotting module in ``analyze/`` imports from this file so every figure
has identical typography, palette, sizing, DPI, and artefact set (PDF +
PNG + optional SVG).

Design targets
--------------
* **ACL main body**: 3.3″ single-column / 6.9″ double-column, Times-serif,
  LaTeX-embeddable Type-42 fonts, 300 DPI vector PDF.
* **Journal fallback**: optional SVG for Nature / PLoS workflows; optional
  EPS for IEEE via ``savefig_pub(..., formats=("pdf","png","eps"))``.
* **Grayscale fallback**: hatch patterns on bars, markers on lines; the
  palette passes simulated deuteranopia / protanopia contrast checks
  (ColorBrewer Dark2 base + Paul Tol's bright set for baselines).
* **Consistency**: no figure module calls ``plt.rcParams`` directly; no
  figure calls ``fig.savefig`` directly — use ``savefig_pub()``.

Semantic axes are orthogonal so a single panel can encode all four:
    colour     → curriculum condition (B1..B5, mono)
    marker     → continual-learning variant (ewc, lamol, lr, maml, episodic)
    linestyle  → model family (humanscale / fineweb / beetlelm_legacy)
    hatch      → grayscale fallback (mirrors colour, for B&W printing)

Usage
-----
    from analyze.plot_style import apply_acl_style, fig_single, savefig_pub
    apply_acl_style()
    fig, ax = fig_single(height=2.4)
    ax.plot(...)
    style_axes(ax, xlabel="L2 tokens", ylabel="PPL")
    savefig_pub(fig, Path("results/figures/main/fig1_foo"))

"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_log = logging.getLogger("plot_style")


# ─────────────────────────────────────────────────────────────────────────────
# Size constants (ACL / EMNLP column widths)
# ─────────────────────────────────────────────────────────────────────────────

DPI              = 300
WIDTH_SINGLE     = 3.3   # inches (ACL single column, matches \columnwidth)
WIDTH_DOUBLE     = 6.9   # inches (ACL double column, matches \textwidth)
WIDTH_ONE_HALF   = 5.0   # inches (intermediate "1.5-column")

# Font sizes calibrated so ACL-rendered PDFs match surrounding body text.
SIZE_TITLE       = 9
SIZE_LABEL       = 8
SIZE_TICK        = 7
SIZE_LEGEND      = 7
SIZE_ANNOTATION  = 6.5
SIZE_PANEL_LABEL = 9

# Line / marker defaults.
LW_AXIS          = 0.6    # axis spines
LW_GRID          = 0.4    # grid
LW_PLOT          = 1.25   # primary data lines
LW_ERROR         = 0.8    # error-bar lines
MARKER_SIZE      = 3.5    # primary markers
MARKER_SIZE_SMALL= 2.5

# Colour defaults.
COLOR_AXIS       = "#222222"
COLOR_GRID       = "#d0d0d0"
COLOR_GRID_DARK  = "#b0b0b0"
COLOR_ANNOTATION = "#555555"
COLOR_ZEROLINE   = "#888888"


# ─────────────────────────────────────────────────────────────────────────────
# Palettes
# ─────────────────────────────────────────────────────────────────────────────

#: Condition palette — ColorBrewer Dark2 (colourblind-safe).
#: Keys use paper codes so figure authors don't have to think about English.
PALETTE_ACL: dict[str, str] = {
    "mono":    "#525252",   # neutral grey
    "B1":      "#1b9e77",   # teal — balanced (50:50)
    "B2":      "#d95f02",   # orange — simultaneous
    "B3":      "#7570b3",   # violet — sequential (33:67)
    "B4":      "#e7298a",   # magenta — classroom
    "B5":      "#66a61e",   # green — late
    "B6":      "#a6761d",   # olive — heritage
    "unknown": "#b3b3b3",
}

#: Legacy condition-name → B-code aliases.
LEGACY_ALIASES: dict[str, str] = {
    "balanced":     "B1",
    "simultaneous": "B2",
    "sequential":   "B3",
    "part_time":    "B4",
    "classroom":    "B4",
    "late":         "B5",
    "heritage":     "B6",
}

#: Continual-learning variant markers — orthogonal to colour.
CL_VARIANT_MARKERS: dict[str, str] = {
    "none":     "o",
    "ewc":      "^",
    "lamol":    "s",
    "lr":       "D",
    "maml":     "v",
    "episodic": "*",
}

#: Family linestyle — lets paper figures mix families on one panel.
FAMILY_LINESTYLES: dict[str, tuple | str] = {
    "humanscale":       "-",
    "fineweb":          "--",
    "beetlelm_legacy":  ":",
    "unknown":          (0, (1, 1)),
}

#: Hatch patterns for B&W / grayscale fallback (paired with colour).
CONDITION_HATCHES: dict[str, str] = {
    "mono": "",
    "B1":   "",
    "B2":   "///",
    "B3":   "\\\\\\",
    "B4":   "xxx",
    "B5":   "...",
    "B6":   "+++",
    "unknown": "",
}

#: Per-language palette — Paul Tol's bright qualitative scheme.
#: Colourblind-friendly; keep stable so paper figures across chapters match.
LANG_PALETTE: dict[str, str] = {
    "eng": "#4477AA",    # blue
    "deu": "#EE6677",    # red
    "nld": "#228833",    # green
    "zho": "#CCBB44",    # yellow
    "fra": "#66CCEE",    # cyan
    "fas": "#AA3377",    # purple
    "bul": "#BBBBBB",    # grey
    "spa": "#009988",
    "ita": "#CC3311",
    "jpn": "#EECC66",
}

#: Baseline-group palette for MECO / LLM comparisons.
BASELINE_PALETTE: dict[str, str] = {
    "small_bilingual":  "#1f77b4",
    "bgpt":             "#ff7f0e",
    "bilingual_llm":    "#2ca02c",
    "bloom_ladder":     "#d62728",
    "eurollm":          "#9467bd",
    "multilingual_llm": "#8c564b",
    "unknown":          "#7f7f7f",
}

#: Diverging colourmap for heatmaps (ΔPPL, Δaccuracy). Uses RdBu_r so
#: positive = better under the convention that higher is better.
DIVERGING_CMAP   = "RdBu_r"
SEQUENTIAL_CMAP  = "viridis"


# ─────────────────────────────────────────────────────────────────────────────
# rcParams application
# ─────────────────────────────────────────────────────────────────────────────

def _best_serif_font() -> str:
    """Pick the best available serif font for publication text.

    Prefers Times New Roman → Times → STIXGeneral → CMU Serif → DejaVu.
    Logs a warning if only DejaVu Serif is found (matplotlib fallback).
    """
    preferences = [
        "Times New Roman", "Times", "Nimbus Roman",
        "STIXGeneral", "CMU Serif", "Liberation Serif",
        "DejaVu Serif",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferences:
        if name in available:
            if name == "DejaVu Serif":
                _log.info(
                    "plot_style: Times-family font not installed; "
                    "falling back to DejaVu Serif. Install "
                    "`msttcorefonts` or `texlive-fonts-extra` for ACL-grade text."
                )
            return name
    return "serif"


def apply_acl_style(use_tex: Optional[bool] = None) -> None:
    """Set matplotlib rcParams for ACL / EMNLP / NeurIPS main-body quality.

    Parameters
    ----------
    use_tex : Optional[bool]
        If True, enable ``text.usetex`` for full LaTeX text rendering (best
        quality but requires a working TeX install). If None, uses the
        ``BEETLE_USETEX`` env var; if that is unset, defaults to False so
        the library works on headless nodes without TeX.
    """
    import os
    if use_tex is None:
        use_tex = os.environ.get("BEETLE_USETEX", "0") == "1"

    serif = _best_serif_font()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mpl.rcParams.update({
            # ── Typography ─────────────────────────────────────────────
            "font.family":       "serif",
            "font.serif":        [serif, "Times", "DejaVu Serif"],
            "mathtext.fontset":  "stix",
            "text.usetex":       bool(use_tex),
            "font.size":         SIZE_LABEL,

            # ── Axes / ticks ───────────────────────────────────────────
            "axes.titlesize":    SIZE_TITLE,
            "axes.labelsize":    SIZE_LABEL,
            "axes.linewidth":    LW_AXIS,
            "axes.edgecolor":    COLOR_AXIS,
            "axes.labelcolor":   COLOR_AXIS,
            "axes.titlepad":     4.0,
            "axes.labelpad":     2.0,
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "axes.grid":         False,            # turn on per-plot via style_axes
            "axes.grid.which":   "major",
            "axes.axisbelow":    True,             # data always on top of grid
            "axes.prop_cycle":   mpl.cycler(color=[
                PALETTE_ACL["B1"], PALETTE_ACL["B2"], PALETTE_ACL["B3"],
                PALETTE_ACL["B4"], PALETTE_ACL["B5"], PALETTE_ACL["mono"],
            ]),
            "xtick.labelsize":   SIZE_TICK,
            "ytick.labelsize":   SIZE_TICK,
            "xtick.direction":   "out",
            "ytick.direction":   "out",
            "xtick.major.width": LW_AXIS,
            "ytick.major.width": LW_AXIS,
            "xtick.major.size":  2.5,
            "ytick.major.size":  2.5,
            "xtick.minor.width": LW_AXIS * 0.75,
            "ytick.minor.width": LW_AXIS * 0.75,
            "xtick.minor.size":  1.4,
            "ytick.minor.size":  1.4,
            "xtick.color":       COLOR_AXIS,
            "ytick.color":       COLOR_AXIS,

            # ── Lines / markers ───────────────────────────────────────
            "lines.linewidth":    LW_PLOT,
            "lines.markersize":   MARKER_SIZE,
            "lines.markeredgewidth": 0.6,
            "patch.linewidth":    LW_AXIS,
            "patch.edgecolor":    COLOR_AXIS,

            # ── Legend ────────────────────────────────────────────────
            "legend.fontsize":    SIZE_LEGEND,
            "legend.frameon":     False,
            "legend.handlelength":1.6,
            "legend.handletextpad":0.6,
            "legend.borderaxespad":0.3,
            "legend.labelspacing":0.3,
            "legend.columnspacing":1.0,

            # ── Grid ──────────────────────────────────────────────────
            "grid.color":         COLOR_GRID,
            "grid.linewidth":     LW_GRID,
            "grid.alpha":         0.8,
            "grid.linestyle":     "-",

            # ── Figure / save ─────────────────────────────────────────
            "figure.dpi":         150,       # screen preview
            "figure.facecolor":   "white",
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.04,
            "figure.constrained_layout.w_pad": 0.04,
            "savefig.dpi":        DPI,
            "savefig.bbox":       "tight",
            "savefig.pad_inches": 0.02,
            "savefig.facecolor":  "white",
            "savefig.transparent":False,

            # ── LaTeX-friendly embedded fonts ─────────────────────────
            "pdf.fonttype":       42,        # Type-42 (TrueType) embedded
            "ps.fonttype":        42,
            "pdf.compression":    6,
            "svg.fonttype":       "none",    # keep text editable in SVG

            # ── Miscellaneous ─────────────────────────────────────────
            "errorbar.capsize":   2.0,
            "image.cmap":         SEQUENTIAL_CMAP,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Figure builders
# ─────────────────────────────────────────────────────────────────────────────

def fig_single(height: float = 2.4, ncols: int = 1, nrows: int = 1,
               **subplot_kw):
    """Create a single-column figure (3.3″ wide)."""
    return plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(WIDTH_SINGLE, height),
        **subplot_kw,
    )


def fig_one_half(height: float = 2.8, ncols: int = 1, nrows: int = 1,
                 **subplot_kw):
    """Create a 1.5-column figure (5.0″ wide)."""
    return plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(WIDTH_ONE_HALF, height),
        **subplot_kw,
    )


def fig_double(height: float = 3.2, ncols: int = 1, nrows: int = 1,
               **subplot_kw):
    """Create a double-column figure (6.9″ wide)."""
    return plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(WIDTH_DOUBLE, height),
        **subplot_kw,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Axis / annotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def style_axes(
    ax,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    grid: bool = True,
    grid_axis: str = "y",
    minor_ticks: bool = True,
    zero_line: bool = False,
) -> None:
    """Apply the house axis style to a single axes instance.

    Idempotent — safe to call multiple times.
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if grid:
        ax.grid(True, which="major", axis=grid_axis,
                linestyle="-", color=COLOR_GRID,
                linewidth=LW_GRID, alpha=0.8, zorder=0)
        if minor_ticks:
            ax.minorticks_on()
            ax.grid(True, which="minor", axis=grid_axis,
                    linestyle=":", color=COLOR_GRID,
                    linewidth=LW_GRID * 0.6, alpha=0.4, zorder=0)

    if zero_line:
        ax.axhline(0, color=COLOR_ZEROLINE, linewidth=LW_AXIS,
                   linestyle="--", zorder=1)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(LW_AXIS)
        ax.spines[spine].set_color(COLOR_AXIS)


def add_panel_label(ax, label: str, loc: str = "top-left",
                    dx: float = 0.0, dy: float = 0.0) -> None:
    """Add a bold panel label (``A``, ``B``, …) to a subplot corner.

    Uses axes-fraction coordinates so it tracks the axes, not the figure.
    """
    positions = {
        "top-left":     (0.0 + dx, 1.02 + dy, "left",  "bottom"),
        "top-right":    (1.0 + dx, 1.02 + dy, "right", "bottom"),
        "bottom-left":  (0.0 + dx, -0.12 + dy, "left",  "top"),
        "bottom-right": (1.0 + dx, -0.12 + dy, "right", "top"),
    }
    x, y, ha, va = positions.get(loc, positions["top-left"])
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=SIZE_PANEL_LABEL,
        fontweight="bold",
        ha=ha, va=va,
    )


def sig_stars(p: float) -> str:
    """Map a p-value to conventional significance stars (``**``, ``*``, n.s.)."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def annotate_significance(
    ax, x0: float, x1: float, y: float,
    p: float | str,
    height_frac: float = 0.02,
    color: str = COLOR_ANNOTATION,
) -> None:
    """Draw a significance bracket between two x-positions at height ``y``.

    ``p`` may be a float (auto-converted via ``sig_stars``) or any string
    (e.g. ``'p=0.03'``).
    """
    label = sig_stars(p) if isinstance(p, (int, float)) else str(p)
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * height_frac
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y],
            color=color, linewidth=LW_AXIS)
    ax.text((x0 + x1) / 2, y + h, label,
            ha="center", va="bottom",
            fontsize=SIZE_ANNOTATION, color=color)


def despine(ax, sides: Sequence[str] = ("top", "right")) -> None:
    """Hide one or more axes spines (default: top and right)."""
    for side in sides:
        ax.spines[side].set_visible(False)


def compact_legend(ax, ncol: int = 1, loc: str = "best",
                   title: Optional[str] = None, **kw):
    """Add a compact, frameless legend with consistent spacing."""
    leg = ax.legend(
        frameon=False,
        fontsize=SIZE_LEGEND,
        ncol=ncol,
        loc=loc,
        title=title,
        handlelength=1.6,
        handletextpad=0.6,
        columnspacing=1.0,
        labelspacing=0.3,
        **kw,
    )
    if title is not None and leg is not None:
        leg.get_title().set_fontsize(SIZE_LEGEND)
    return leg


# ─────────────────────────────────────────────────────────────────────────────
# Figure saver
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_FORMATS: tuple[str, ...] = ("pdf", "png")


def savefig_pub(
    fig,
    stem: Path,
    close: bool = True,
    formats: Iterable[str] = _DEFAULT_FORMATS,
    metadata: Optional[dict] = None,
    transparent: bool = False,
) -> dict[str, Path]:
    """Save ``fig`` in every requested format at publication settings.

    Parameters
    ----------
    fig : matplotlib Figure
    stem : Path
        Output path without suffix. Parents are created automatically.
    close : bool
        Close the figure after saving (default True).
    formats : iterable of {"pdf", "png", "svg", "eps"}
        Default: ("pdf", "png"). PDF is the primary vector artefact; PNG
        doubles as a 300-DPI preview. SVG keeps text editable (useful for
        journals that require figure re-edit); EPS for IEEE.
    metadata : dict, optional
        Title / Author / Creator fields to embed in the PDF.
    transparent : bool
        Transparent background (default: False — white for print).

    Returns
    -------
    dict[str, Path]
        Mapping of format → written path.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    meta = {"Creator": "beetle-analyze"}
    if metadata:
        meta.update(metadata)

    written: dict[str, Path] = {}
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        kw: dict = {"bbox_inches": "tight", "pad_inches": 0.02,
                    "transparent": transparent}
        if fmt == "pdf":
            kw["metadata"] = meta
        if fmt == "png":
            kw["dpi"] = DPI
        fig.savefig(path, **kw)
        written[fmt] = path

    if close:
        plt.close(fig)
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Lookups
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_condition(cond: str) -> str:
    if cond in PALETTE_ACL:
        return cond
    return LEGACY_ALIASES.get(cond, "unknown")


def color_for_condition(cond: Optional[str]) -> str:
    if not cond:
        return PALETTE_ACL["unknown"]
    return PALETTE_ACL[_normalise_condition(cond)]


def hatch_for_condition(cond: Optional[str]) -> str:
    if not cond:
        return CONDITION_HATCHES["unknown"]
    return CONDITION_HATCHES.get(_normalise_condition(cond), "")


def marker_for_cl(variant: Optional[str]) -> str:
    if not variant:
        return CL_VARIANT_MARKERS["none"]
    return CL_VARIANT_MARKERS.get(variant, CL_VARIANT_MARKERS["none"])


def linestyle_for_family(family: Optional[str]):
    if not family:
        return FAMILY_LINESTYLES["unknown"]
    return FAMILY_LINESTYLES.get(family, FAMILY_LINESTYLES["unknown"])


def color_for_lang(lang: Optional[str]) -> str:
    if not lang:
        return "#444444"
    return LANG_PALETTE.get(lang, "#444444")


def color_for_baseline(group: Optional[str]) -> str:
    if not group:
        return BASELINE_PALETTE["unknown"]
    return BASELINE_PALETTE.get(group, BASELINE_PALETTE["unknown"])


# ─────────────────────────────────────────────────────────────────────────────
# Regression / error-bar helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_with_ci(
    ax, x, y, ci_lower, ci_upper,
    *, color: str = None, label: Optional[str] = None,
    linestyle: str = "-", marker: str = "o",
    alpha_fill: float = 0.18,
) -> None:
    """Line-plot with a shaded 95 % CI band. Use in place of ``ax.plot``."""
    line, = ax.plot(
        x, y, color=color, label=label,
        linestyle=linestyle, marker=marker,
        linewidth=LW_PLOT, markersize=MARKER_SIZE,
        markeredgewidth=0.6, zorder=3,
    )
    ax.fill_between(
        x, ci_lower, ci_upper,
        color=color or line.get_color(),
        alpha=alpha_fill, linewidth=0, zorder=2,
    )


def errorbar_pub(
    ax, x, y, yerr,
    *, color: str = None, label: Optional[str] = None,
    marker: str = "o", capsize: float = 2.0,
) -> None:
    """Error-bar plot with house defaults."""
    ax.errorbar(
        x, y, yerr=yerr,
        color=color, ecolor=color,
        label=label, marker=marker,
        linewidth=LW_PLOT, elinewidth=LW_ERROR,
        markersize=MARKER_SIZE, capsize=capsize,
        zorder=3,
    )


__all__ = [
    # Sizes
    "DPI", "WIDTH_SINGLE", "WIDTH_ONE_HALF", "WIDTH_DOUBLE",
    "SIZE_TITLE", "SIZE_LABEL", "SIZE_TICK", "SIZE_LEGEND",
    "SIZE_ANNOTATION", "SIZE_PANEL_LABEL",
    "LW_AXIS", "LW_GRID", "LW_PLOT", "LW_ERROR",
    "MARKER_SIZE", "MARKER_SIZE_SMALL",
    "COLOR_AXIS", "COLOR_GRID", "COLOR_GRID_DARK",
    "COLOR_ANNOTATION", "COLOR_ZEROLINE",
    # Palettes
    "PALETTE_ACL", "CL_VARIANT_MARKERS", "FAMILY_LINESTYLES",
    "LANG_PALETTE", "BASELINE_PALETTE", "LEGACY_ALIASES",
    "CONDITION_HATCHES", "DIVERGING_CMAP", "SEQUENTIAL_CMAP",
    # Entry points
    "apply_acl_style",
    "fig_single", "fig_one_half", "fig_double",
    "savefig_pub",
    "style_axes", "add_panel_label", "compact_legend",
    "sig_stars", "annotate_significance", "despine",
    "plot_with_ci", "errorbar_pub",
    # Lookups
    "color_for_condition", "hatch_for_condition",
    "marker_for_cl", "linestyle_for_family",
    "color_for_lang", "color_for_baseline",
]
