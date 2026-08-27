#!/usr/bin/env python3
"""
plot_results.py — Generate publication-quality figures from BeetleLM eval CSVs.

Figures produced (saved to --output_dir as 300 dpi PNGs)
---------------------------------------------------------
  01_learning_curves_{benchmark}.png  — accuracy vs step, one line per model,
                                        faceted by eval language, coloured by
                                        bilingual type. multiblimp and xcomps
                                        use a compact 2-col grid with one shared
                                        legend; all others use a single-column
                                        stack with legend well below x-axis.
  02_benchmark_by_language.png        — compact grouped bar chart sized for ACL
                                        double-column. x=curricula (abbreviated),
                                        rows=language (tight, no gap), colour=
                                        benchmark. Single y-axis label on left.
  03_barplot_by_type_{benchmark}.png  — per-benchmark accuracy by curricula.
                                        Horizontal bars for most benchmarks;
                                        vertical for blimpnl / zhoblimp (with
                                        panel titles left-aligned to avoid
                                        suptitle collision).
                                        xcomps / xnli are skipped (already done).
  04_scatter_l1_l2.png                — L1 vs L2 accuracy scatter
  05_phenomenon_breakdown.png         — MultiBLiMP per-phenomenon breakdown

Usage
-----
  python plot_results.py --results_dir results/ --output_dir figures/
"""

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Visual style ──────────────────────────────────────────────────────────────
FONT_SIZE   = 9
LABEL_SIZE  = 8
TITLE_SIZE  = 10
DPI         = 300
FIGW_COL    = 3.5    # single ACL column  (inches)
FIGW_FULL   = 7.0    # double ACL column  (inches)

TYPE_PALETTE = {
    "mono"        : "#333333",
    "balanced"    : "#1f77b4",
    "simultaneous": "#ff7f0e",
    "sequential"  : "#2ca02c",
    "part_time"   : "#d62728",
    "late"        : "#9467bd",
    "heritage"    : "#8c564b",
    "unknown"     : "#7f7f7f",
}
TYPE_ORDER = ["mono", "balanced", "simultaneous", "sequential",
              "part_time", "late", "heritage"]

# Abbreviated curricula labels for x-axes (avoids crowding)
TYPE_SHORT = {
    "mono"        : "mono",
    "balanced"    : "bal.",
    "simultaneous": "simul.",
    "sequential"  : "seq.",
    "part_time"   : "part-t.",
    "late"        : "late",
    "heritage"    : "herit.",
    "unknown"     : "unk.",
}

matplotlib.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : FONT_SIZE,
    "axes.titlesize"   : TITLE_SIZE,
    "axes.labelsize"   : LABEL_SIZE,
    "xtick.labelsize"  : LABEL_SIZE,
    "ytick.labelsize"  : LABEL_SIZE,
    "legend.fontsize"  : LABEL_SIZE,
    "figure.dpi"       : DPI,
    "savefig.dpi"      : DPI,
    "savefig.bbox"     : "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

# Benchmarks whose learning curves are laid out in a compact 2-col grid
_GRID_LC_BENCHMARKS = {"multiblimp", "xcomps", "xnli"}

# Benchmarks that use vertical bars in fig 03
_VERTICAL_BENCHMARKS = {"blimpnl", "zhoblimp"}

# Benchmarks whose fig-03 barplots are skipped (xcomps only — already perfect)
_SKIP_BARPLOT = {"xcomps"}


# ── Language normalisation ────────────────────────────────────────────────────
# zh / Chinese  →  "Chinese"
# de / German   →  "German"

_LANG_MAP = {
    "zh": "Chinese", "zh-cn": "Chinese", "zh-tw": "Chinese", "chinese": "Chinese",
    "de": "German",  "de-de": "German",  "german": "German",  "deutsch": "German",
    "fr": "French", "fr-fr": "French", "french": "French", "français": "French",
}

def normalise_language(lang: str) -> str:
    return _LANG_MAP.get(str(lang).strip().lower(), lang)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_results(results_dir: str, benchmark_filter=None,
                     lang_group_filter=None) -> pd.DataFrame:
    dfs = []
    for p in Path(results_dir).glob("*.csv"):
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"Warning: could not read {p}: {e}")

    if not dfs:
        sys.exit(f"No CSVs found in {results_dir}")

    df = pd.concat(dfs, ignore_index=True).drop_duplicates()

    # Normalise language names before anything else
    df["eval_language"] = df["eval_language"].apply(normalise_language)

    def parse_step(ckpt):
        if isinstance(ckpt, str) and ckpt.startswith("step-"):
            return int(ckpt.split("-")[1])
        return -1

    df["step"]    = df["checkpoint"].apply(parse_step)
    df["is_main"] = df["checkpoint"] == "main"

    max_step = df.loc[df["step"] >= 0, "step"].max() if (df["step"] >= 0).any() else 0
    df.loc[df["is_main"], "step"] = max_step + 1

    df["bilingual_type"] = df["bilingual_type"].fillna("unknown")

    if benchmark_filter:
        df = df[df["benchmark"] == benchmark_filter]
    if lang_group_filter:
        df = df[df["lang_pair"].str.contains(lang_group_filter)]

    return df


def short_name(repo: str) -> str:
    return repo.split("/")[-1].replace("beetlelm_", "")


def _get_final(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    bdf = df[(df["benchmark"] == benchmark) & (df["is_main"])].copy()
    if bdf.empty:
        bdf = (df[df["benchmark"] == benchmark]
               .sort_values("step")
               .groupby(["model", "eval_language"])
               .last().reset_index())
    return bdf


def _fmt_steps(steps):
    unique = sorted(set(steps))
    n = len(unique)
    if n == 0:
        return [], []
    idx    = np.linspace(0, n - 1, min(4, n), dtype=int)
    ticks  = [unique[i] for i in idx]
    labels = [f"{v // 1000}k" if v >= 1000 else str(v) for v in ticks]
    return ticks, labels


# ── Figure 1 — Learning curves ────────────────────────────────────────────────

def _draw_lc_panel(ax, ldf, types, all_steps_sink):
    """Populate one learning-curve axes panel."""
    panel_steps = []
    for bil_type in types:
        tdf = ldf[ldf["bilingual_type"] == bil_type].sort_values("step")
        if tdf.empty:
            continue
        grouped = tdf.groupby("step")["accuracy"]
        means   = grouped.mean()
        stds    = grouped.std().fillna(0)
        steps   = means.index.values
        panel_steps.extend(steps.tolist())

        ax.plot(steps, means.values,
                color=TYPE_PALETTE[bil_type],
                label=bil_type.replace("_", "-"),
                linewidth=1.3, marker="o", markersize=2.2, zorder=3)
        ax.fill_between(steps, means - stds, means + stds,
                        color=TYPE_PALETTE[bil_type], alpha=0.12, zorder=2)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.5, zorder=1)
    ax.set_ylim(0.35, 1.02)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.4, color="grey")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", pad=3, labelsize=LABEL_SIZE)

    if panel_steps:
        ticks, tlabels = _fmt_steps(panel_steps)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tlabels, fontsize=LABEL_SIZE)

    all_steps_sink.extend(panel_steps)


def plot_learning_curves(df: pd.DataFrame, benchmark: str, out_dir: str):
    bdf = df[df["benchmark"] == benchmark].copy()
    if bdf.empty:
        return

    langs = sorted(bdf["eval_language"].unique())
    types = [t for t in TYPE_ORDER if t in bdf["bilingual_type"].unique()]

    use_grid = benchmark.lower() in _GRID_LC_BENCHMARKS

    if use_grid:
        # ── 2-col compact grid ─────────────────────────────────────────────
        ncols   = 2
        nrows   = (len(langs) + ncols - 1) // ncols
        panel_h = 2.1
        leg_h   = 0.85   # generous bottom margin — legend clears x-axis ticks
        title_h = 0.50   # top margin for suptitle

        total_h = nrows * panel_h + leg_h + title_h
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(FIGW_FULL, total_h),
            sharey=True, sharex=False, squeeze=False
        )
        fig.subplots_adjust(
            hspace=0.75,          # vertical gap between rows — title clears ticks
            wspace=0.18,
            left=0.09, right=0.98,
            top=1.0 - title_h / total_h,
            bottom=leg_h / total_h
        )
        flat = axes.flatten()
        all_steps = []

        for i, ax in enumerate(flat):
            if i >= len(langs):
                ax.set_visible(False)
                continue
            _draw_lc_panel(ax, bdf[bdf["eval_language"] == langs[i]], types, all_steps)
            ax.set_title(langs[i], pad=6, fontsize=LABEL_SIZE + 0.5,
                         fontweight="semibold", loc="left")
            if i % ncols == 0:
                ax.set_ylabel("Accuracy", labelpad=4)
            if i >= (nrows - 1) * ncols:
                ax.set_xlabel("Training step", labelpad=5)

    else:
        # ── single-column stack ────────────────────────────────────────────
        panel_h = 2.3
        leg_h   = 0.95   # generous — legend sits well below bottom x-axis ticks
        title_h = 0.50   # top margin for suptitle

        total_h = len(langs) * panel_h + leg_h + title_h
        fig, axes = plt.subplots(
            len(langs), 1,
            figsize=(FIGW_FULL * 0.82, total_h),
            sharey=True, sharex=False, squeeze=False
        )
        fig.subplots_adjust(
            hspace=0.80,          # larger gap so panel titles clear x-axis ticks
            left=0.11, right=0.97,
            top=1.0 - title_h / total_h,
            bottom=leg_h / total_h
        )
        flat = axes.flatten()
        all_steps = []

        for i, (ax, lang) in enumerate(zip(flat, langs)):
            _draw_lc_panel(ax, bdf[bdf["eval_language"] == lang], types, all_steps)
            ax.set_title(lang, pad=6, fontsize=LABEL_SIZE + 0.5,
                         fontweight="semibold", loc="left")
            ax.set_ylabel("Accuracy", labelpad=4)

        flat[-1].set_xlabel("Training step", labelpad=5)

    # Shared legend anchored to figure bottom — below every x-axis
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(types), 4),
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        title="Curricula",
        title_fontsize=LABEL_SIZE,
        fontsize=LABEL_SIZE - 0.5,
        borderpad=0.3,
    )

    fig.suptitle(
        f"{benchmark.upper()} — Learning curves",
        fontsize=TITLE_SIZE + 1, fontweight="bold",
        x=0.5, ha="center",
        y=1.0 - 0.08 / fig.get_size_inches()[1]
    )

    out = os.path.join(out_dir, f"01_learning_curves_{benchmark}.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 2 — Final accuracy: x=curricula, rows/cols=language, colour=benchmark
#
#  Landscape ACL figure. Subplots laid out in a grid, 3 columns per row.
#  · x-axis: curricula (abbreviated), shared across rows, labelled on bottom row
#  · y-axis: accuracy, shared across all panels, labelled on left column only
#  · Colour: benchmark
#  · Language name as panel title (left-aligned)
#  · Single legend below

def plot_benchmark_by_language(df: pd.DataFrame, out_dir: str):
    mdf = df[df["is_main"]].copy()
    if mdf.empty:
        mdf = (df.sort_values("step")
                 .groupby(["model", "eval_language", "benchmark"])
                 .last().reset_index())
    if mdf.empty:
        return

    langs      = sorted(mdf["eval_language"].unique())
    benchmarks = sorted(mdf["benchmark"].unique())
    types      = [t for t in TYPE_ORDER if t in mdf["bilingual_type"].unique()]

    n_langs = len(langs)
    n_bm    = len(benchmarks)
    n_types = len(types)

    _bm_palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728",
                   "#9467bd","#8c564b","#e377c2","#17becf"]
    bm_colors = {bm: _bm_palette[i % len(_bm_palette)]
                 for i, bm in enumerate(benchmarks)}

    agg = (mdf.groupby(["benchmark", "bilingual_type", "eval_language"])["accuracy"]
              .agg(mean="mean",
                   sem=lambda x: float(x.sem()) if len(x) > 1 else 0.0)
              .reset_index())

    # ── grid geometry ─────────────────────────────────────────────────────
    ncols   = 4
    nrows   = (n_langs + ncols - 1) // ncols
    panel_w = FIGW_FULL / ncols          # ~2.33" per panel
    panel_h = max(0.6, 0.1 * n_bm * n_types + 0.45)
    leg_h   = 0.50
    xlbl_h  = 0.52   # below bottom row for rotated labels
    title_h = 0.35

    fig_w = FIGW_FULL * 1.30   # landscape: wider than standard double-col
    fig_h = nrows * panel_h + leg_h + xlbl_h + title_h

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        sharex=True, sharey=True,
        squeeze=False
    )

    fig.subplots_adjust(
        hspace=0.25,            # small vertical gap — titles + bottom ticks
        wspace=0.12,
        left=0.07,
        right=0.98,
        top=1.0 - title_h / fig_h,
        bottom=(leg_h + xlbl_h) / fig_h,
    )

    # Bar geometry
    total_width = 0.78
    bar_w   = total_width / n_bm
    offsets = np.linspace(-(n_bm - 1) / 2, (n_bm - 1) / 2, n_bm) * bar_w
    x       = np.arange(n_types)

    flat = axes.flatten()
    for idx, ax in enumerate(flat):
        row_i = idx // ncols
        col_i = idx % ncols
        is_bottom = (row_i == nrows - 1) or (idx + ncols >= n_langs)
        is_left   = (col_i == 0)

        if idx >= n_langs:
            ax.set_visible(False)
            continue

        lang = langs[idx]
        ldf  = agg[agg["eval_language"] == lang]

        # Alternating column shading
        for ci in range(n_types):
            if ci % 2 == 0:
                ax.axvspan(ci - 0.48, ci + 0.48,
                           color="#f5f5f5", zorder=0, linewidth=0)

        ax.axhline(0.5, color="#bbbbbb", linestyle="--",
                   linewidth=0.65, alpha=0.8, zorder=1)

        for bi, bm in enumerate(benchmarks):
            bdf_bm = ldf[ldf["benchmark"] == bm]
            means, sems = [], []
            for t in types:
                row = bdf_bm[bdf_bm["bilingual_type"] == t]
                if row.empty:
                    means.append(np.nan); sems.append(0.0)
                else:
                    means.append(row["mean"].values[0])
                    sems.append(row["sem"].values[0])

            means = np.array(means, dtype=float)
            sems  = np.array(sems,  dtype=float)
            xs    = x + offsets[bi]

            ax.bar(xs, means, width=bar_w * 0.88,
                   color=bm_colors[bm], alpha=0.88,
                   edgecolor="white", linewidth=0.3, zorder=2)
            ax.errorbar(xs, means, yerr=sems,
                        fmt="none", ecolor="#555", elinewidth=0.5,
                        capsize=1.2, capthick=0.5, zorder=3)

        ax.set_ylim(0.35, 1.05)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.yaxis.grid(True, linestyle=":", linewidth=0.4,
                      alpha=0.45, color="grey", zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)

        # y-tick labels only on left column
        if is_left:
            ax.tick_params(axis="y", labelsize=LABEL_SIZE - 0.5, pad=2)
            ax.set_ylabel("Accuracy", labelpad=4, fontsize=LABEL_SIZE)
        else:
            ax.tick_params(axis="y", length=0, labelleft=False)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [TYPE_SHORT.get(t, t) for t in types],
            fontsize=LABEL_SIZE - 1, rotation=40, ha="right"
        )

        ax.set_title(lang, pad=4, fontsize=LABEL_SIZE + 0.5,
                     fontweight="semibold", loc="left")

    # Benchmark legend below figure
    bm_handles = [mpatches.Patch(color=bm_colors[bm], label=bm)
                  for bm in benchmarks]
    fig.legend(
        handles=bm_handles,
        loc="lower center",
        ncol=min(n_bm, 6),
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        title="Benchmark",
        title_fontsize=LABEL_SIZE,
        fontsize=LABEL_SIZE - 0.5,
        handlelength=0.9, handleheight=0.85,
        handletextpad=0.4, columnspacing=0.8,
    )

    out = os.path.join(out_dir, "02_benchmark_by_language.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 3 — Barplot by curricula per benchmark ─────────────────────────────
#
#  · xcomps / xnli: skipped entirely (already perfect)
#  · blimpnl / zhoblimp: vertical bars, panel titles left-aligned
#  · all others: horizontal bars

def plot_barplot_by_type(df: pd.DataFrame, benchmark: str, out_dir: str):
    if benchmark.lower() in _SKIP_BARPLOT:
        return

    bdf = _get_final(df, benchmark)
    if bdf.empty:
        return

    langs = sorted(bdf["eval_language"].unique())
    types = [t for t in TYPE_ORDER if t in bdf["bilingual_type"].unique()]

    agg = (bdf.groupby(["bilingual_type", "eval_language"])["accuracy"]
              .agg(mean="mean", sem=lambda x: x.sem() if len(x) > 1 else 0)
              .reset_index())

    bm_lower      = benchmark.lower()
    use_vertical  = bm_lower in _VERTICAL_BENCHMARKS

    ncols = min(len(langs), 3)
    nrows = (len(langs) + ncols - 1) // ncols

    if use_vertical:
        # ── vertical bars (blimpnl, zhoblimp) ──────────────────────────────
        title_h = 0.50
        panel_h = 2.6

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(FIGW_FULL, nrows * panel_h + title_h),
            sharey=True
        )
        fig.subplots_adjust(
            hspace=0.70, wspace=0.28,
            left=0.09, right=0.97,
            top=1.0 - title_h / (nrows * panel_h + title_h),
            bottom=0.12
        )
        axes = np.array(axes).flatten()

        for ax, lang in zip(axes, langs):
            ldf   = (agg[agg["eval_language"] == lang]
                     .set_index("bilingual_type").reindex(types))
            means = ldf["mean"].values
            sems  = ldf["sem"].fillna(0).values
            x_pos = np.arange(len(types))

            ax.bar(x_pos, means, yerr=sems, capsize=2.5,
                   color=[TYPE_PALETTE[t] for t in types],
                   edgecolor="white", linewidth=0.6, width=0.68, zorder=2)
            ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7,
                       alpha=0.55, zorder=1)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([TYPE_SHORT.get(t, t) for t in types],
                               fontsize=LABEL_SIZE - 0.5,
                               rotation=35, ha="right")
            # Left-aligned title keeps it away from suptitle in centre
            ax.set_title(lang, pad=5, fontsize=LABEL_SIZE + 0.5,
                         fontweight="semibold", loc="left")
            ax.set_ylabel("Accuracy", labelpad=4)
            ax.set_ylim(0.35, 1.08)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
            ax.yaxis.grid(True, linestyle=":", linewidth=0.4,
                          alpha=0.5, color="grey")
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", pad=3)

        for ax in axes[len(langs):]:
            ax.set_visible(False)

    else:
        # ── horizontal bars (all remaining benchmarks) ──────────────────────
        title_h = 0.50
        panel_h = max(2.0, len(types) * 0.40 + 0.55)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(FIGW_FULL, nrows * panel_h + title_h),
            sharex=True
        )
        fig.subplots_adjust(
            hspace=0.62, wspace=0.28,
            left=0.18, right=0.97,
            top=1.0 - title_h / (nrows * panel_h + title_h),
            bottom=0.10
        )
        axes = np.array(axes).flatten()

        for ax, lang in zip(axes, langs):
            ldf   = (agg[agg["eval_language"] == lang]
                     .set_index("bilingual_type").reindex(types))
            means = ldf["mean"].values
            sems  = ldf["sem"].fillna(0).values
            y_pos = np.arange(len(types))

            ax.barh(y_pos, means, xerr=sems, capsize=2.5,
                    color=[TYPE_PALETTE[t] for t in types],
                    edgecolor="white", linewidth=0.6, height=0.62, zorder=2)
            ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.7,
                       alpha=0.55, zorder=1)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([t.replace("_", "-") for t in types],
                               fontsize=LABEL_SIZE)
            ax.set_title(lang, pad=5, fontsize=LABEL_SIZE + 0.5,
                         fontweight="semibold", loc="left")
            ax.set_xlabel("Accuracy", labelpad=4)
            ax.set_xlim(0.35, 1.05)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
            ax.xaxis.grid(True, linestyle=":", linewidth=0.4,
                          alpha=0.5, color="grey")
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", pad=3)

        for ax in axes[len(langs):]:
            ax.set_visible(False)

    fig.suptitle(
        f"{benchmark.upper()}",
        fontsize=TITLE_SIZE + 1, fontweight="bold",
        x=0.5, ha="center",
        y=1.0 - 0.07 / (nrows * panel_h + title_h)
    )

    out = os.path.join(out_dir, f"03_barplot_by_type_{benchmark}.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 4 — L1 vs L2 accuracy scatter ─────────────────────────────────────

def plot_l1_l2_scatter(df: pd.DataFrame, out_dir: str):
    mdf = df[df["is_main"]].copy()
    if mdf.empty:
        return

    l1l2_models = mdf[mdf["model"].str.contains("_L1|_L2", regex=True)].copy()
    if l1l2_models.empty:
        print("No L1/L2-labelled models found for scatter plot — skipping.")
        return

    rows = []
    for repo, grp in l1l2_models.groupby("model"):
        name = repo.split("/")[-1]
        m_l1 = re.search(r"([a-z]{3})_L1", name)
        m_l2 = re.search(r"_L2_([a-z]{3})", name)
        if not (m_l1 and m_l2):
            continue
        l1_code = m_l1.group(1)
        l2_code = m_l2.group(1)
        l1_rows = grp[grp["eval_language"].str.lower().str[:2] == l1_code[:2]]
        l2_rows = grp[grp["eval_language"].str.lower().str[:2] == l2_code[:2]]
        if l1_rows.empty or l2_rows.empty:
            continue
        rows.append({"model": short_name(repo),
                     "l1_acc": l1_rows["accuracy"].mean(),
                     "l2_acc": l2_rows["accuracy"].mean(),
                     "bilingual_type": grp["bilingual_type"].iloc[0],
                     "l1": l1_code, "l2": l2_code})

    if not rows:
        print("Could not extract L1/L2 pairs — skipping scatter plot.")
        return

    sdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(FIGW_FULL * 0.72, FIGW_FULL * 0.72))
    fig.subplots_adjust(left=0.13, right=0.96, top=0.91, bottom=0.12)

    for bil_type, grp in sdf.groupby("bilingual_type"):
        ax.scatter(grp["l1_acc"], grp["l2_acc"],
                   color=TYPE_PALETTE.get(bil_type, "#999"),
                   label=bil_type.replace("_", "-"),
                   s=35, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)

    lim = [0.4, 1.0]
    ax.plot(lim, lim, "--", color="grey", linewidth=0.8, alpha=0.6, zorder=1)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("L1 accuracy", labelpad=6)
    ax.set_ylabel("L2 accuracy", labelpad=6)
    ax.set_title("L1 vs L2 accuracy (main checkpoint, all benchmarks)",
                 fontweight="bold")
    ax.xaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="grey")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="grey")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, title="Curricula", ncol=2,
              fontsize=7, title_fontsize=7)
    ax.tick_params(axis="both", pad=4)

    out = os.path.join(out_dir, "04_scatter_l1_l2.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 5 — Per-phenomenon breakdown (MultiBLiMP) ─────────────────────────

def plot_phenomenon_breakdown(df: pd.DataFrame, out_dir: str):
    if "phenomenon" not in df.columns:
        print("No 'phenomenon' column — skipping breakdown plot.")
        return

    mdf = df[(df["benchmark"] == "multiblimp") & (df["is_main"])].copy()
    if mdf.empty:
        return

    agg = (mdf.groupby(["phenomenon", "bilingual_type", "eval_language"])
              ["accuracy"].mean().reset_index())

    langs = sorted(agg["eval_language"].unique())
    types = [t for t in TYPE_ORDER if t in agg["bilingual_type"].unique()]

    for lang in langs:
        ldf       = agg[agg["eval_language"] == lang]
        phenomena = sorted(ldf["phenomenon"].unique())
        pivot = (ldf.pivot_table(index="phenomenon", columns="bilingual_type",
                                 values="accuracy", aggfunc="mean")
                    .reindex(columns=[t for t in types if t in ldf["bilingual_type"].unique()])
                    .reindex(phenomena))

        fig, ax = plt.subplots(figsize=(FIGW_FULL,
                                        max(3.2, len(phenomena) * 0.38)))
        fig.subplots_adjust(left=0.28, right=0.97, top=0.91, bottom=0.10)

        x     = np.arange(len(phenomena))
        width = 0.8 / len(pivot.columns)
        for i, col in enumerate(pivot.columns):
            ax.barh(x + i * width, pivot[col].values,
                    height=width * 0.9,
                    color=TYPE_PALETTE.get(col, "#999"),
                    label=col.replace("_", "-"), zorder=2)
        ax.set_yticks(x + width * (len(pivot.columns) - 1) / 2)
        ax.set_yticklabels(phenomena, fontsize=6)
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.7,
                   alpha=0.55, zorder=1)
        ax.set_xlabel("Accuracy", labelpad=5)
        ax.set_title(f"MultiBLiMP — Phenomenon breakdown ({lang})",
                     fontweight="bold")
        ax.legend(frameon=False, ncol=2, fontsize=6)
        ax.set_xlim(0.3, 1.0)
        ax.xaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="grey")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", pad=4)

        out = os.path.join(out_dir, f"05_phenomenon_{lang.lower()}.png")
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved: {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--output_dir",  default="figures")
    p.add_argument("--benchmark",   default=None)
    p.add_argument("--lang_group",  default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_all_results(args.results_dir, args.benchmark, args.lang_group)
    print(f"Loaded {len(df):,} rows from {args.results_dir}")
    print(f"Benchmarks : {sorted(df['benchmark'].unique())}")
    print(f"Languages  : {sorted(df['eval_language'].unique())}")
    print(f"Models     : {df['model'].nunique()}")
    print()

    for bm in sorted(df["benchmark"].unique()):
        print(f"── {bm} ──────────────────────────────")
        plot_learning_curves(df, bm, args.output_dir)
        plot_barplot_by_type(df, bm, args.output_dir)   # skips xcomps/xnli

    plot_benchmark_by_language(df, args.output_dir)
    plot_l1_l2_scatter(df, args.output_dir)
    plot_phenomenon_breakdown(df, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()