"""
analyze.meco_detailed._plot — shared visual idiom for the eight D-figures.

Design goals (from the EMNLP review of the v1 figures)
------------------------------------------------------
1. **Crosslinguistic differences must be visible.** L1 always appears on an
   ordered axis (typological distance to English), with a *single* consistent
   palette across all eight figures.
2. **Curriculum differences must be visible.** B1..B5 + mono get a single
   consistent palette across all figures, ordered by L2-onset (early → late).
3. **Incomplete cells must remain readable.** When a (L1, curriculum, scale)
   cell isn't populated, draw it in greyscale with diagonal hatching and the
   string "n/a" — never an empty bar that looks like ΔlogLik=0.
4. **A coverage badge** appears in every figure's footer summarising how
   many of the *expected* cells are populated. Reviewers can see at a glance
   whether a figure is making a complete or partial claim.

Public surface
--------------
  L1_ORDER, L1_PRETTY, L1_PALETTE
  CURRICULUM_ORDER, CURRICULUM_LABEL, CURRICULUM_PALETTE
  SCALE_ORDER, SCALE_MARKERS
  attach_model_metadata(df, model_col="model")
  expected_cell_grid(metadata_df, dims=("l1","curriculum"))
  mark_missing_cell(ax, x, y, w=1, h=1, label="n/a")
  coverage_badge(fig, df, expected_total)
  facet_grid_l1_x_curriculum(...)
  l1_legend_handles(used_l1s), curriculum_legend_handles(used_curr)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# L1 ordering (typological distance to English, ascending)
# ─────────────────────────────────────────────────────────────────────────────

L1_ORDER: tuple[str, ...] = (
    "du",   # Dutch        Germanic
    "ge",   # German       Germanic
    "no",   # Norwegian    Germanic
    "it",   # Italian      Romance
    "sp",   # Spanish      Romance
    "gr",   # Greek        Hellenic
    "ru",   # Russian      Slavic
    "ee",   # Estonian     Uralic
    "fi",   # Finnish      Uralic
    "he",   # Hebrew       Semitic
    "tr",   # Turkish      Turkic
    "ch_s", # Chinese (S)  Sino-Tibetan
    "en",   # English      L1 native control
)

L1_PRETTY: dict[str, str] = {
    "du": "Dutch", "ge": "German", "no": "Norwegian", "it": "Italian",
    "sp": "Spanish", "gr": "Greek", "ru": "Russian", "ee": "Estonian",
    "fi": "Finnish", "he": "Hebrew", "tr": "Turkish", "ch_s": "Chinese",
    "en": "English (L1)",
}

# Paul Tol's "muted" qualitative palette extended to 13 colors. Deuteranopia-
# safe; ordered to keep adjacent typological neighbours visually similar
# (Germanic cluster cool, Romance warm, isolates pink/brown).
L1_PALETTE: dict[str, str] = {
    "du":   "#332288",  # Germanic — indigo
    "ge":   "#117733",  # Germanic — green
    "no":   "#44AA99",  # Germanic — teal
    "it":   "#999933",  # Romance  — olive
    "sp":   "#DDCC77",  # Romance  — sand
    "gr":   "#88CCEE",  # Hellenic — sky
    "ru":   "#CC6677",  # Slavic   — rose
    "ee":   "#AA4499",  # Uralic   — magenta
    "fi":   "#882255",  # Uralic   — wine
    "he":   "#E69F00",  # Semitic  — amber
    "tr":   "#D55E00",  # Turkic   — vermilion
    "ch_s": "#000000",  # Sinitic  — black
    "en":   "#888888",  # control  — grey
}

# Map common alternate L1 codes (deu, nld, zho, ...) onto MECO codes.
L1_ALIASES: dict[str, str] = {
    "nld": "du", "deu": "ge", "zho": "ch_s", "ces": "du",  # rare
    "rus": "ru", "spa": "sp", "tur": "tr", "fin": "fi",
    "ita": "it", "nor": "no", "heb": "he", "ell": "gr",
    "eng": "en", "est": "ee",
}


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum ordering — early L2 onset to late
# ─────────────────────────────────────────────────────────────────────────────

CURRICULUM_ORDER: tuple[str, ...] = ("mono", "B1", "B2", "B3", "B4", "B5")

CURRICULUM_LABEL: dict[str, str] = {
    "mono":     "mono",
    "B1":       "B1 balanced",
    "B2":       "B2 simul.",
    "B3":       "B3 sequential",
    "B4":       "B4 classroom-20",
    "B5":       "B5 late-80",
    "unknown":  "unknown",
    "baseline": "baseline LLM",
}

# Sequential palette: cool blues for early L2, warm reds for late L2.
# Helps reviewers see at a glance that the gradient reflects L2 dose.
CURRICULUM_PALETTE: dict[str, str] = {
    "mono":     "#444444",
    "B1":       "#3B528B",  # purple-blue   (balanced from the start)
    "B2":       "#21908C",  # teal          (simultaneous)
    "B3":       "#5DC863",  # green         (sequential, classic)
    "B4":       "#FDE725",  # yellow        (classroom-20, small L1)
    "B5":       "#D33F6A",  # red           (late, heavy L1)
    "unknown":  "#BBBBBB",
    "baseline": "#888888",
}


# ─────────────────────────────────────────────────────────────────────────────
# Scale ordering — fewer to more tokens
# ─────────────────────────────────────────────────────────────────────────────

SCALE_ORDER: tuple[str, ...] = ("humanscale", "fineweb-100m", "fineweb-2b",
                                 "fineweb", "fineweb-24b", "fineweb2-24b")

SCALE_LABEL: dict[str, str] = {
    "humanscale":   "100 M (HumanScale)",
    "fineweb-100m": "100 M (FineWeb)",
    "fineweb-2b":   "2 B  (FineWeb)",
    "fineweb":      "24 B (FineWeb)",
    "fineweb-24b":  "24 B (FineWeb)",
    "fineweb2-24b": "24 B (FineWeb-v2)",
    "unknown":      "unknown",
    "baseline":     "baseline",
}

# Marker shapes per scale (used in scatter plots; size scales with magnitude).
SCALE_MARKERS: dict[str, str] = {
    "humanscale":   "o",
    "fineweb-100m": "o",
    "fineweb-2b":   "s",
    "fineweb":      "D",
    "fineweb-24b":  "D",
    "fineweb2-24b": "D",
    "unknown":      "x",
    "baseline":     "*",
}


# ─────────────────────────────────────────────────────────────────────────────
# Metadata attachment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _MetaRow:
    l1_code: str
    curriculum: str
    scale: str
    direction: str  # "L1→L2", "L2→L1", or "n/a"
    is_beetle: bool


def _parse_one(model: str) -> _MetaRow:
    """Heuristic parser layered on top of ``analyze.model_naming.parse_repo``.

    Returns ``("baseline", "baseline", "baseline", "n/a", False)`` for
    non-beetle baselines so they survive the same plotting code path. The
    canonical fallback handles legacy model names that ``parse_repo`` can't
    interpret.
    """
    if not isinstance(model, str) or not model:
        return _MetaRow("unknown", "unknown", "unknown", "n/a", False)
    name = model.lower()
    if "beetle" not in name and "beetlelm" not in name:
        return _MetaRow("baseline", "baseline", "baseline", "n/a", False)

    # Try the canonical parser first.
    try:
        from analyze.model_naming import parse_repo
        meta = parse_repo(model)
        l1 = meta.l1 or "unknown"
        l1_code = L1_ALIASES.get(l1, l1)
        if l1_code not in L1_PRETTY:
            l1_code = "unknown"
        cond = meta.condition if meta.condition in CURRICULUM_ORDER else "unknown"
        scale = meta.family if meta.family in SCALE_ORDER else "unknown"
        # Direction: if l2 is given and != l1, encode L1→L2 vs L2→L1
        if meta.l2 and meta.l1:
            if meta.l1 == "eng":
                direction = "L2→L1"
            elif meta.l2 == "eng":
                direction = "L1→L2"
            else:
                direction = "n/a"
        else:
            direction = "n/a"
        return _MetaRow(l1_code, cond, scale, direction, True)
    except Exception:
        pass

    # Heuristic fallback for legacy slugs.
    cond = "unknown"
    for c in CURRICULUM_ORDER:
        if c.lower() in name:
            cond = c
            break
    l1_code = "unknown"
    for code, alias in L1_ALIASES.items():
        if f"-{code}-eng" in name or f"_{code}_l1" in name or f"-{code}-" in name:
            l1_code = alias
            break
    scale = "unknown"
    for s in SCALE_ORDER:
        if s in name:
            scale = s
            break
    return _MetaRow(l1_code, cond, scale, "n/a", True)


def attach_model_metadata(
    df: pd.DataFrame,
    model_col: str = "model",
) -> pd.DataFrame:
    """Return ``df`` with ``l1_code, curriculum, scale, direction, is_beetle``
    columns attached (joined by ``model_col``).

    No-op for empty frames or frames missing the model column.
    """
    if df.empty or model_col not in df.columns:
        return df
    out = df.copy()
    uniq = sorted(out[model_col].dropna().astype(str).unique())
    parsed = {m: _parse_one(m) for m in uniq}
    out["l1_code"]    = out[model_col].map(lambda m: parsed.get(str(m), _parse_one(str(m))).l1_code)
    out["curriculum"] = out[model_col].map(lambda m: parsed.get(str(m), _parse_one(str(m))).curriculum)
    out["scale"]      = out[model_col].map(lambda m: parsed.get(str(m), _parse_one(str(m))).scale)
    out["direction"]  = out[model_col].map(lambda m: parsed.get(str(m), _parse_one(str(m))).direction)
    out["is_beetle"]  = out[model_col].map(lambda m: parsed.get(str(m), _parse_one(str(m))).is_beetle)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Coverage helpers
# ─────────────────────────────────────────────────────────────────────────────

def expected_cell_grid(
    df: pd.DataFrame,
    *,
    dims: Sequence[str] = ("l1_code", "curriculum"),
    l1_filter: Optional[Sequence[str]] = None,
    curriculum_filter: Optional[Sequence[str]] = None,
) -> tuple[set, set]:
    """Return (expected_cells, observed_cells) as sets of dim-tuples.

    ``expected`` is the cartesian product of the dim domains drawn from the
    ordered ``L1_ORDER`` / ``CURRICULUM_ORDER`` / ``SCALE_ORDER`` sequences;
    ``observed`` is whatever combinations are present in ``df``. Either set
    can be passed to ``coverage_badge`` for the footer summary.
    """
    if df.empty:
        return set(), set()
    domains: list[Sequence[str]] = []
    for d in dims:
        if d == "l1_code":
            domains.append(l1_filter or L1_ORDER)
        elif d == "curriculum":
            domains.append(curriculum_filter or [c for c in CURRICULUM_ORDER if c != "mono"])
        elif d == "scale":
            domains.append(SCALE_ORDER)
        else:
            domains.append(sorted(df[d].dropna().unique().tolist()))
    from itertools import product
    expected = set(product(*domains))
    if all(d in df.columns for d in dims):
        observed = set(map(tuple, df[list(dims)].dropna().astype(str).to_numpy()))
    else:
        observed = set()
    return expected, observed


def mark_missing_cell(ax, x, y, *, w: float = 1.0, h: float = 1.0,
                       label: str = "n/a",
                       facecolor: str = "#EFEFEF",
                       hatch: str = "//",
                       textcolor: str = "#888888") -> None:
    """Overlay a hatched grey rectangle + small label at (x, y) on ``ax``.

    Use for missing-cell annotation in grid / heatmap plots so an empty
    cell is visually distinct from a zero-valued cell. Coordinates and
    ``w/h`` are in the axes' data units.
    """
    import matplotlib.patches as patches
    rect = patches.Rectangle((x - w / 2, y - h / 2), w, h,
                             facecolor=facecolor, edgecolor="white",
                             hatch=hatch, linewidth=0.6, alpha=0.85,
                             zorder=0)
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=6, color=textcolor, alpha=0.9, zorder=1)


def axis_label(name: str, *, better: Optional[str] = None,
                units: Optional[str] = None) -> str:
    """Format a publication-ready axis label that includes the metric, its
    units (optional), and an explicit direction arrow indicating which way
    is "better".

    Examples
    --------
        axis_label("ΔlogLik",                 better="higher")
            → "ΔlogLik  (↑ better)"
        axis_label("ICC(β)",                  better="lower")
            → "ICC(β)  (↓ better)"
        axis_label("β(surprisal → comp acc)", better="more negative")
            → "β(surprisal → comp acc)  (↓ better)"
        axis_label("Reading time", units="ms")
            → "Reading time (ms)"
    """
    parts: list[str] = [name]
    if units:
        parts.append(f"({units})")
    if better:
        arrow = {"higher": "↑", "lower": "↓",
                 "more negative": "↓", "more positive": "↑",
                 "positive": "↑", "negative": "↓"}.get(better, "→")
        parts.append(f" ({arrow} better)")
    return "  ".join(parts).replace("  (", " (")


def direction_badge(
    fig,
    metric: str,
    *,
    better: str = "higher",
    zero_means: Optional[str] = None,
    loc: tuple[float, float] = (0.99, 0.985),
) -> None:
    """Stamp a small direction-indicator badge in the figure corner.

    Tells the reader *immediately* which direction is "better" on this
    plot's primary scale. Examples:

        direction_badge(fig, "ΔlogLik",                    better="higher")
        direction_badge(fig, "ICC(β)",                     better="lower")
        direction_badge(fig, "β(surprisal → comp)",        better="more negative",
                          zero_means="no comprehension link")
        direction_badge(fig, "L2 − L1 gap",                better="higher",
                          zero_means="advantage is generic, not L2-specific")

    Renders as a small text block in the top-right by default. Set
    ``loc=(0.01, 0.985)`` for top-left placement.
    """
    arrow = {"higher": "↑", "lower": "↓",
             "more negative": "↓", "more positive": "↑",
             "positive": "↑", "negative": "↓"}.get(better, "→")
    text = f"{metric}\n{arrow} {better} = better"
    if zero_means:
        text += f"\nzero ≈ {zero_means}"
    fig.text(loc[0], loc[1], text,
             ha="right" if loc[0] > 0.5 else "left",
             va="top",
             fontsize=6.5,
             color="#2c5e2e",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0f8f0",
                        edgecolor="#2c5e2e", linewidth=0.6, alpha=0.92))


def coverage_badge(fig, df: pd.DataFrame, *,
                    dims: Sequence[str] = ("l1_code", "curriculum"),
                    text_y: float = -0.02) -> None:
    """Stamp a small coverage badge across the figure footer.

    Example: "coverage: 18/72 (L1 × curriculum) cells populated; missing
    cells shown hatched."
    """
    expected, observed = expected_cell_grid(df, dims=dims)
    n_e = max(len(expected), 1)
    n_o = sum(1 for c in observed if c in expected)
    pct = 100 * n_o / n_e
    dim_label = " × ".join({"l1_code": "L1", "curriculum": "curriculum",
                            "scale": "scale"}.get(d, d) for d in dims)
    fig.text(0.5, text_y,
             f"coverage: {n_o}/{n_e} ({pct:.0f}%) of expected "
             f"{dim_label} cells populated; missing shown hatched",
             ha="center", va="top", fontsize=6.5, color="#666666")


# ─────────────────────────────────────────────────────────────────────────────
# Legend helpers
# ─────────────────────────────────────────────────────────────────────────────

def l1_legend_handles(used_l1s: Iterable[str]):
    import matplotlib.patches as patches
    used = [l for l in L1_ORDER if l in set(used_l1s)]
    return [patches.Patch(facecolor=L1_PALETTE.get(l, "#888"),
                          label=L1_PRETTY.get(l, l), edgecolor="white")
            for l in used]


def curriculum_legend_handles(used_curricula: Iterable[str]):
    import matplotlib.patches as patches
    used = [c for c in CURRICULUM_ORDER if c in set(used_curricula)]
    return [patches.Patch(facecolor=CURRICULUM_PALETTE.get(c, "#888"),
                          label=CURRICULUM_LABEL.get(c, c), edgecolor="white")
            for c in used]


# ─────────────────────────────────────────────────────────────────────────────
# Facet-grid helper for "L1 × curriculum" canonical layout
# ─────────────────────────────────────────────────────────────────────────────

def facet_grid_l1_x_curriculum(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (8.0, 5.0),
    l1_filter: Optional[Sequence[str]] = None,
    curriculum_filter: Optional[Sequence[str]] = None,
):
    """Return (fig, axes_2d, used_l1s, used_curr).

    Layout: rows = L1 (typological order, top to bottom), cols = curriculum
    (early-L2 to late-L2 left to right). Empty cells are kept (axis exists)
    so callers can stamp ``mark_missing_cell`` on them.
    """
    import matplotlib.pyplot as plt
    used_l1s = [l for l in (l1_filter or L1_ORDER)
                if l in set(df.get("l1_code", []))]
    used_curr = [c for c in (curriculum_filter
                              or [c for c in CURRICULUM_ORDER if c != "mono"])
                 if c in set(df.get("curriculum", []))]
    if not used_l1s:
        used_l1s = list(l1_filter or L1_ORDER[:3])
    if not used_curr:
        used_curr = list(curriculum_filter or ["B1", "B2", "B3", "B4", "B5"])

    fig, axes = plt.subplots(len(used_l1s), len(used_curr),
                              figsize=figsize, sharex=False, sharey=True,
                              squeeze=False)
    for r, l1 in enumerate(used_l1s):
        for c, curr in enumerate(used_curr):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if r == 0:
                ax.set_title(CURRICULUM_LABEL.get(curr, curr),
                              fontsize=7, color=CURRICULUM_PALETTE.get(curr, "#000"))
            if c == 0:
                ax.set_ylabel(L1_PRETTY.get(l1, l1), fontsize=7, rotation=0,
                               ha="right", va="center",
                               color=L1_PALETTE.get(l1, "#000"))
    fig.tight_layout(rect=(0.03, 0.04, 0.98, 0.95))
    return fig, axes, used_l1s, used_curr
