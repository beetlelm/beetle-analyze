#!/usr/bin/env python3
"""
scripts/priming_paper_figures.py — Paper-quality figures for a single
priming-tier output bundle.

Reads:
    {tier_dir}/comparison/priming_combined.tsv      (beetle + bgpt long)
    {tier_dir}/comparison/priming_effects_per_model.csv

Writes (300 DPI):
    {tier_dir}/figures/fig_priming_by_dataset.{pdf,png}
    {tier_dir}/figures/fig_priming_by_curriculum.{pdf,png}
    {tier_dir}/figures/fig_priming_direction_asymmetry.{pdf,png}
    {tier_dir}/figures/fig_priming_trajectory.{pdf,png}    # only if multi-checkpoint
    {tier_dir}/figures/fig_caption_metadata.json           # provenance for paper

Usage:
    python scripts/priming_paper_figures.py --tier-dir results/evals/priming/tier0_sanity

The script silently skips figures whose required columns are absent (e.g.
trajectory figure when only `main` revisions were evaluated). Figure choice
is data-driven, not tier-prescribed, so the same script works on every tier.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s")
logger = logging.getLogger("priming-figs")

# ── EMNLP-style rcParams ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size":    9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#222",
    "axes.labelcolor": "#222",
    "xtick.color": "#222",
    "ytick.color": "#222",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,        # TrueType in PDFs (camera-ready safe)
    "ps.fonttype":  42,
})

# Palette: deuteranopia-safe; bgpt vs beetle are visually distinct
COLORS = {
    "bgpt":   "#737373",
    "beetle": "#1f4f8a",
    "b1":     "#2c7bb6",
    "b2":     "#abd9e9",
    "b3":     "#fdae61",
    "b4":     "#d7191c",
    "b5":     "#762a83",
}

CURRICULUM_LABEL = {
    "b1": "B1 balanced",
    "b2": "B2 simultaneous",
    "b3": "B3 sequential",
    "b4": "B4 classroom",
    "b5": "B5 late",
}

DATASET_ORDER = ["bernolet", "schoonbaert", "hartsuiker", "kotzochampou", "fleischer"]

# Headline comparison: Beetle FineWeb canonical (~24B) nld-eng simultaneous +
# sequential vs BGPT nl_en simultaneous (5050) + sequential (7525). This is
# the only matched comparison: BGPT was trained on FineWeb-canonical-scale
# data with Dutch-as-L1, sim/seq curricula. The Beetle 100M and 2B repos are
# different scales; Beetle eng-nld is the opposite L1; both go in supplementary.
HEADLINE_BGPT_MODELS = {
    "BGPT-sim": "nl_en_5050",
    "BGPT-seq": "nl_en_7525",
}

# Beetle FineWeb-canonical (24B), nld-eng direction, all five curricula.
HEADLINE_BEETLE_PATTERNS = {
    "B1": re.compile(r"beetle-bilingual-balanced-b1-fineweb-nld-eng$"),
    "B2": re.compile(r"beetle-bilingual-l2-50-simultaneous-b2-fineweb-nld-eng$"),
    "B3": re.compile(r"beetle-bilingual-l2-50-sequential-\d+-\d+-b3-fineweb-nld-eng$"),
    "B4": re.compile(r"beetle-bilingual-(?:balanced|l2-50-classroom-20)-b4-fineweb-nld-eng$"),
    "B5": re.compile(r"beetle-bilingual-(?:balanced|l2-80-late)-b5-fineweb-nld-eng$"),
}
HEADLINE_CONDITION_ORDER = ["B1", "B2", "B3", "B4", "B5", "BGPT-sim", "BGPT-seq"]

# Short x-tick labels (one line each — keeps them readable without rotation).
HEADLINE_TICK_LABEL = {
    "B1":       "B1",
    "B2":       "B2",
    "B3":       "B3",
    "B4":       "B4",
    "B5":       "B5",
    "BGPT-sim": "sim",
    "BGPT-seq": "seq",
}

# Beetle bars use a gradient cueing curriculum order; BGPT bars are muted
# gray with a hatch so they read as the reference baseline.
BEETLE_CURRICULUM_PALETTE = {
    "B1": "#2c7bb6",
    "B2": "#5d96c6",
    "B3": "#88b1d6",
    "B4": "#b8cce3",
    "B5": "#dde6ef",
}

HEADLINE_DATASETS = ["bernolet", "schoonbaert"]   # the two Dutch datasets


def _detect_curriculum(model: str) -> str | None:
    m = re.search(r"-b([1-5])-", model)
    return f"b{m.group(1)}" if m else None


def _detect_direction(model: str) -> str | None:
    """Return 'eng-X' if the model name's lang pair starts with eng, 'X-eng'
    if it ends with eng, else None."""
    m = re.search(r"-([a-z]{3})-([a-z]{3})(?:-|$)", model)
    if not m: return None
    l1, l2 = m.group(1), m.group(2)
    if l1 == "eng": return "eng-X"
    if l2 == "eng": return "X-eng"
    return None


def _detect_scale(model: str) -> str | None:
    if "fineweb-2b"   in model: return "FineWeb-2B"
    if "fineweb-100m" in model: return "FineWeb-100M"
    if "fineweb-"     in model: return "FineWeb"
    if "humanscale"   in model: return "HumanScale"
    return None


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                  rng: np.random.Generator | None = None) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    if len(values) < 2:
        return float("nan"), float("nan")
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


# ── Headline figure: Beetle 24B nld-eng vs BGPT nl_en, sim/seq, Dutch ─────

def fig_headline_24b_nld_curriculum(eff: pd.DataFrame, out_dir: Path) -> Path | None:
    """Headline panel: Beetle FineWeb-24B nld-eng B1–B5 plus BGPT (nl_en)
    sim / seq, on the two Dutch datasets (bernolet, schoonbaert).

    For each (dataset, condition) cell we show the bar height = mean of the
    two cross-lingual directions (Dutch→English, English→Dutch), with the
    individual direction values overlaid as small dots. With n=2 per cell
    a bootstrap CI is misleading; the dots are an honest representation of
    the spread.

    BGPT data is restricted to its largest-numeric checkpoint per model so
    we compare end-of-training to end-of-training.
    """
    if eff.empty: return None

    def _lang(s):
        s = str(s); return s.replace("_Prime", "").replace("_Target", "")
    eff = eff.assign(_PL=eff.PrimeLanguage.map(_lang),
                     _TL=eff.TargetLanguage.map(_lang))
    cross = eff[eff._PL != eff._TL].copy()
    if cross.empty: return None

    cross["_ckpt_num"] = pd.to_numeric(cross["checkpoint"], errors="coerce")
    bgpt_finals = (
        cross[cross.model_source == "bgpt"]
        .dropna(subset=["_ckpt_num"])
        .groupby("model")["_ckpt_num"].max()
        .to_dict()
    )
    bgpt_mask = cross.model_source == "bgpt"
    keep_bgpt = bgpt_mask & cross.apply(
        lambda r: bgpt_finals.get(r["model"]) == r["_ckpt_num"], axis=1)
    cross = pd.concat([cross[~bgpt_mask], cross[keep_bgpt]], ignore_index=True)

    def _condition_rows(ds: str, cond: str) -> pd.DataFrame:
        if cond.startswith("BGPT"):
            model = HEADLINE_BGPT_MODELS[cond]
            return cross[(cross.dataset == ds) & (cross.model_source == "bgpt") &
                         (cross.model == model)]
        pat = HEADLINE_BEETLE_PATTERNS[cond]
        return cross[(cross.dataset == ds) & (cross.model_source == "beetle") &
                     cross.model.astype(str).str.contains(pat, regex=True)]

    direction_color = {"Dutch": "#d7191c", "English": "#222222"}

    fig, axes = plt.subplots(
        1, len(HEADLINE_DATASETS),
        figsize=(7.0, 3.0),
        sharey=True, squeeze=False,
        constrained_layout=True,
    )
    axes = axes.ravel()
    x = np.arange(len(HEADLINE_CONDITION_ORDER))

    for ax, ds in zip(axes, HEADLINE_DATASETS):
        for i, cond in enumerate(HEADLINE_CONDITION_ORDER):
            sub = _condition_rows(ds, cond)
            if sub.empty:
                continue
            mean = float(sub.priming_effect.mean())
            color = (BEETLE_CURRICULUM_PALETTE.get(cond)
                     if not cond.startswith("BGPT") else COLORS["bgpt"])
            hatch = "//" if cond.startswith("BGPT") else ""
            ax.bar(x[i], mean, width=0.72,
                   color=color, hatch=hatch,
                   edgecolor="white", linewidth=0.5, alpha=0.95, zorder=1)
            for _, r in sub.iterrows():
                ax.scatter(x[i], r.priming_effect,
                           s=14, marker="o",
                           color=direction_color.get(r._PL, "#888"),
                           edgecolor="white", linewidth=0.4, zorder=3)
        # Subtle background distinguishes Beetle (5 curricula) from BGPT (2).
        ax.axvspan(-0.5, 4.5, alpha=0.06, color=COLORS["beetle"], zorder=0)
        ax.axhline(0, color="#222", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([HEADLINE_TICK_LABEL[c] for c in HEADLINE_CONDITION_ORDER],
                           rotation=0)
        ax.set_title(ds, pad=2, fontsize=10)
        ax.tick_params(axis="x", which="major", pad=2)

    axes[0].set_ylabel("Cross-lingual priming effect\n(Δ relative probability)")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=BEETLE_CURRICULUM_PALETTE["B2"], edgecolor="white",
              label="Beetle 24B (B1–B5)"),
        Patch(facecolor=COLORS["bgpt"], edgecolor="white", hatch="//",
              label="BGPT (sim, seq)"),
        Line2D([0], [0], marker="o", linestyle="", markersize=5,
               markerfacecolor=direction_color["Dutch"], markeredgecolor="white",
               label="Dutch → English"),
        Line2D([0], [0], marker="o", linestyle="", markersize=5,
               markerfacecolor=direction_color["English"], markeredgecolor="white",
               label="English → Dutch"),
    ]
    fig.legend(handles=handles, frameon=False, loc="outside lower center",
               ncol=4, fontsize=7.5)
    fig.suptitle("Beetle covers 5 bilingual curricula; BGPT covers 2",
                 fontsize=10)

    out = out_dir / "fig_headline_24b_nld_curriculum"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


# ── Between-condition (Beetle only) by scale × curriculum, Dutch ──────────

# BGPT has fewer training tokens than Beetle 100M/2B, so it is not a fair
# baseline for those scales. The fair Beetle-vs-BGPT cell is the 24B
# FineWeb canonical model only (handled in fig_headline_24b_nld_curriculum).
# The 100M and 2B FineWeb scales are best understood by comparing them to
# each other — i.e. between Beetle conditions, no BGPT.
SCALE_FROM_FAMILY = {
    "humanscale":   "HumanScale",
    "fineweb":      "FineWeb-24B",
    "fineweb-100m": "FineWeb-100M",
    "fineweb-2b":   "FineWeb-2B",
}
SCALE_ORDER = ["HumanScale", "FineWeb-24B", "FineWeb-100M", "FineWeb-2B"]
SCALE_COLOR = {
    "HumanScale":   "#abd9e9",
    "FineWeb-24B":  "#1f4f8a",
    "FineWeb-100M": "#fdae61",
    "FineWeb-2B":   "#762a83",
}

def _beetle_scale_from_model(model: str) -> str | None:
    if "fineweb-2b"   in model: return "FineWeb-2B"
    if "fineweb-100m" in model: return "FineWeb-100M"
    if "humanscale"   in model: return "HumanScale"
    if "fineweb"      in model: return "FineWeb-24B"
    return None


def fig_beetle_between_condition_dutch(eff: pd.DataFrame, out_dir: Path) -> Path | None:
    """Between-condition Beetle-only comparison on the two Dutch datasets.

    Layout: 2 panels (bernolet, schoonbaert); x-axis = curriculum (B1..B5);
    grouped bars colored by training-data scale. Both directions are
    averaged within each (scale, curriculum) cell. Cross-lingual rows only.
    No BGPT — the BGPT–Beetle apples-to-apples cell is the 24B headline.
    """
    if eff.empty: return None

    def _lang(s):
        s = str(s); return s.replace("_Prime", "").replace("_Target", "")
    eff = eff.assign(_PL=eff.PrimeLanguage.map(_lang),
                     _TL=eff.TargetLanguage.map(_lang))
    cross = eff[(eff.model_source == "beetle") & (eff._PL != eff._TL)].copy()
    cross = cross[cross.dataset.isin(HEADLINE_DATASETS)]
    if cross.empty: return None

    cross["scale"]      = cross["model"].map(_beetle_scale_from_model)
    cross["curriculum"] = cross["model"].map(_detect_curriculum)
    cross = cross[cross.scale.notna() & cross.curriculum.notna()]

    scales = [s for s in SCALE_ORDER if s in cross.scale.unique()]
    curricula = sorted(cross.curriculum.unique(), key=lambda c: c[1])
    if not scales or not curricula:
        return None

    fig, axes = plt.subplots(1, len(HEADLINE_DATASETS),
                             figsize=(7.0, 3.0),
                             sharey=True, squeeze=False,
                             constrained_layout=True)
    axes = axes.ravel()
    width = 0.78 / max(len(scales), 1)
    x = np.arange(len(curricula))

    for ax, ds in zip(axes, HEADLINE_DATASETS):
        sub = cross[cross.dataset == ds]
        for i, sc in enumerate(scales):
            sub_sc = sub[sub.scale == sc]
            means = []
            for c in curricula:
                v = sub_sc[sub_sc.curriculum == c]["priming_effect"].dropna().values
                means.append(float(np.mean(v)) if len(v) else np.nan)
                # Plot raw points as faint dots — honest at small n.
            offset = (i - (len(scales) - 1) / 2) * width
            ax.bar(x + offset, means, width=width,
                   color=SCALE_COLOR.get(sc, "#888"), label=sc,
                   edgecolor="white", linewidth=0.4)
            for c_i, c in enumerate(curricula):
                v = sub_sc[sub_sc.curriculum == c]["priming_effect"].dropna().values
                if len(v) == 0:
                    continue
                ax.scatter(np.full_like(v, x[c_i] + offset, dtype=float),
                           v, s=8, color="#222", alpha=0.7,
                           edgecolor="white", linewidth=0.3, zorder=3)
        ax.axhline(0, color="#222", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels([c.upper() for c in curricula])
        ax.set_title(ds, pad=2, fontsize=10)
        ax.set_xlabel("Curriculum")
    axes[0].set_ylabel("Cross-lingual priming effect\n(Δ relative probability)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside lower center",
               ncol=len(scales), fontsize=7.5)
    fig.suptitle("Beetle between-condition comparison — Dutch datasets",
                 fontsize=10)

    out = out_dir / "fig_beetle_between_condition_dutch"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


# ── Supplementary: priming effect by dataset, beetle vs bgpt ─────────────────

def fig_by_dataset(eff: pd.DataFrame, out_dir: Path) -> Path | None:
    """Bar chart: priming effect per dataset, BGPT vs Beetle (mean across all
    models within each source)."""
    if eff.empty:
        return None
    rows = []
    for (src, ds), g in eff.groupby(["model_source", "dataset"]):
        vals = g["priming_effect"].dropna().values
        lo, hi = _bootstrap_ci(vals)
        rows.append(dict(source=src, dataset=ds, n_models=g["model"].nunique(),
                         n_obs=len(vals), mean=float(np.mean(vals)),
                         ci_lo=lo, ci_hi=hi))
    df = pd.DataFrame(rows)
    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
    df = df.sort_values(["dataset", "source"])

    datasets = [d for d in DATASET_ORDER if d in df.dataset.values]
    sources  = ["beetle", "bgpt"]
    width = 0.36
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(5.5, 2.3))
    for i, src in enumerate(sources):
        sub = df[df.source == src].set_index("dataset")
        means = [float(sub.loc[d, "mean"]) if d in sub.index else np.nan for d in datasets]
        los   = [float(sub.loc[d, "ci_lo"]) if d in sub.index else np.nan for d in datasets]
        his   = [float(sub.loc[d, "ci_hi"]) if d in sub.index else np.nan for d in datasets]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width=width,
               yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
               color=COLORS.get(src, "#888"), label=src.upper() if src == "bgpt" else "Beetle",
               edgecolor="white", linewidth=0.4, capsize=2.4)
    ax.axhline(0, color="#222", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=0)
    ax.set_ylabel("Priming effect\n(Δ relative probability)")
    ax.set_title("Cross-lingual structural priming by dataset", pad=4)
    ax.legend(frameon=False, loc="upper right")

    out = out_dir / "fig_priming_by_dataset"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


# ── Figure 2: by curriculum ────────────────────────────────────────────────

def fig_by_curriculum(eff: pd.DataFrame, out_dir: Path) -> Path | None:
    beetle = eff[eff.model_source == "beetle"].copy()
    if beetle.empty:
        return None
    beetle["curriculum"] = beetle["model"].map(_detect_curriculum)
    beetle = beetle[beetle.curriculum.notna()]
    if beetle.empty:
        return None

    datasets = [d for d in DATASET_ORDER if d in beetle.dataset.values]
    curricula = sorted(beetle.curriculum.unique(), key=lambda c: c[1])
    if not datasets or not curricula:
        return None

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(min(8.0, 1.8 * len(datasets)), 2.6),
        sharey=True, squeeze=False, constrained_layout=True,
    )
    axes = axes.ravel()
    for ax, ds in zip(axes, datasets):
        sub = beetle[beetle.dataset == ds]
        rows = []
        for c in curricula:
            v = sub[sub.curriculum == c]["priming_effect"].dropna().values
            rows.append((c, float(np.mean(v)) if len(v) else np.nan, v))
        c_x = np.arange(len(curricula))
        means = [r[1] for r in rows]
        ax.bar(c_x, means,
               color=[COLORS.get(c, "#888") for c in curricula],
               edgecolor="white", linewidth=0.4)
        for ci, (_, _, v) in enumerate(rows):
            if len(v) == 0: continue
            ax.scatter(np.full_like(v, c_x[ci], dtype=float),
                       v, s=8, color="#222", alpha=0.7,
                       edgecolor="white", linewidth=0.3, zorder=3)
        ax.axhline(0, color="#222", linewidth=0.5)
        ax.set_xticks(c_x)
        ax.set_xticklabels([c.upper() for c in curricula])
        ax.set_title(ds, pad=2, fontsize=10)
    axes[0].set_ylabel("Priming effect\n(Δ relative probability)")
    fig.suptitle("Beetle priming effect by curriculum",
                 fontsize=10)

    out = out_dir / "fig_priming_by_curriculum"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


# ── Figure 3: direction asymmetry ──────────────────────────────────────────

def fig_direction_asymmetry(eff: pd.DataFrame, out_dir: Path) -> Path | None:
    beetle = eff[eff.model_source == "beetle"].copy()
    if beetle.empty:
        return None
    beetle["curriculum"] = beetle["model"].map(_detect_curriculum)
    beetle["direction"]  = beetle["model"].map(_detect_direction)
    beetle = beetle[beetle.curriculum.notna() & beetle.direction.notna()]
    if beetle.direction.nunique() < 2:
        return None  # only one direction in this tier; figure not informative

    curricula = sorted(beetle.curriculum.unique(), key=lambda c: c[1])
    fig, ax = plt.subplots(figsize=(5.0, 2.6), constrained_layout=True)
    width = 0.36
    x = np.arange(len(curricula))
    for i, direction in enumerate(["eng-X", "X-eng"]):
        sub = beetle[beetle.direction == direction]
        means = []
        all_pts: list[tuple[float, float]] = []
        for ci, c in enumerate(curricula):
            v = sub[sub.curriculum == c]["priming_effect"].dropna().values
            means.append(float(np.mean(v)) if len(v) else np.nan)
            for val in v: all_pts.append((ci, val))
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width=width,
               color=COLORS.get("b1" if i == 0 else "b3"),
               edgecolor="white", linewidth=0.4, label=direction)
        if all_pts:
            ax.scatter([p[0] + offset for p in all_pts], [p[1] for p in all_pts],
                       s=8, color="#222", alpha=0.7,
                       edgecolor="white", linewidth=0.3, zorder=3)
    ax.axhline(0, color="#222", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([c.upper() for c in curricula])
    ax.set_ylabel("Priming effect\n(Δ relative probability)")
    ax.set_title("Direction asymmetry × curriculum")
    ax.legend(frameon=False, loc="best", fontsize=7.5)

    out = out_dir / "fig_priming_direction_asymmetry"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


# ── Figure 4: trajectory over training (if multi-checkpoint) ────────────────

def fig_trajectory(eff: pd.DataFrame, out_dir: Path) -> Path | None:
    beetle = eff[eff.model_source == "beetle"].copy()
    if beetle.empty:
        return None
    beetle = beetle[beetle.checkpoint.astype(str) != "main"]
    beetle["step"] = pd.to_numeric(beetle["checkpoint"], errors="coerce")
    beetle = beetle[beetle.step.notna()]
    if beetle.step.nunique() < 3:
        return None  # no real trajectory

    beetle["curriculum"] = beetle["model"].map(_detect_curriculum)
    beetle = beetle[beetle.curriculum.notna()]

    fig, ax = plt.subplots(figsize=(6.0, 2.8), constrained_layout=True)
    for c, sub in beetle.groupby("curriculum"):
        agg = sub.groupby("step")["priming_effect"].agg(["mean", "count"]).reset_index()
        agg = agg.sort_values("step")
        ax.plot(agg["step"], agg["mean"], "-o",
                color=COLORS.get(c, "#888"),
                markersize=3, linewidth=1.2,
                label=CURRICULUM_LABEL.get(c, c))
    ax.axhline(0, color="#222", linewidth=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Priming effect\n(Δ relative probability)")
    ax.set_title("Developmental trajectory of cross-lingual priming")
    ax.legend(frameon=False, loc="best", ncol=2, fontsize=7.5)

    out = out_dir / "fig_priming_trajectory"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


# ── Driver ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-dir", required=True, type=Path)
    ap.add_argument("--out-subdir", default="figures")
    args = ap.parse_args()

    tier_dir: Path = args.tier_dir.resolve()
    if not tier_dir.exists():
        logger.error("tier dir %s does not exist", tier_dir); return 2

    eff_csv = tier_dir / "comparison" / "priming_effects_per_model.csv"
    if not eff_csv.exists():
        logger.error("no comparison/priming_effects_per_model.csv at %s", eff_csv)
        return 3

    eff = pd.read_csv(eff_csv)
    if eff.empty:
        logger.error("empty effects CSV at %s", eff_csv); return 4

    # Repair NaN-only checkpoint column (mirror compare_priming_to_bgpt fix)
    if "checkpoint" in eff.columns and eff["checkpoint"].isna().all():
        eff["checkpoint"] = "main"

    out_dir = tier_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("tier=%s rows=%d sources=%s", tier_dir.name, len(eff),
                sorted(eff.model_source.unique()))

    written: list[str] = []
    for fn, label in (
        (fig_headline_24b_nld_curriculum,    "headline_24b_nld_curriculum"),     # paper
        (fig_beetle_between_condition_dutch, "beetle_between_condition_dutch"),  # paper
        (fig_by_curriculum,         "by_curriculum"),                            # supp
        (fig_direction_asymmetry,   "direction_asymmetry"),                      # supp
        (fig_trajectory,            "trajectory"),                               # main if present
    ):
        try:
            p = fn(eff, out_dir)
            if p is None:
                logger.info("  skipped %s (no data)", label)
                continue
            written.append(p.name)
            logger.info("  wrote %s.{pdf,png}", p.name)
        except Exception as e:
            logger.exception("  %s figure failed: %s", label, e)

    meta = {
        "tier_dir":          str(tier_dir),
        "n_models_beetle":   int(eff[eff.model_source == "beetle"]["model"].nunique()),
        "n_models_bgpt":     int(eff[eff.model_source == "bgpt"]["model"].nunique()),
        "datasets":          sorted(eff.dataset.unique()),
        "checkpoints":       sorted(map(str, eff.checkpoint.dropna().unique()))[:50],
        "figures":           written,
    }
    (out_dir / "fig_caption_metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info("done: %d figure(s)", len(written))
    return 0 if written else 5


if __name__ == "__main__":
    sys.exit(main())
