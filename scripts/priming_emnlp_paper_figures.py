#!/usr/bin/env python3
"""
scripts/priming_emnlp_paper_figures.py — paper-quality figures for the EMNLP
priming-results section, drawing from BOTH tier1_final and
tier2_headline_trajectory under results/evals/priming/.

All numbers in these figures are computed from BETWEEN-LANGUAGE pairs only
(Dutch_Prime/English_Target averaged with English_Prime/Dutch_Target). This
matches the headline narrative numbers in paper/emnlp/PRIMING_RESULTS.md.

Outputs (300 DPI, fonttype-42 PDF + PNG):
    results/evals/priming/tier2_headline_trajectory/figures/
        fig1_trajectory_headline.{pdf,png}
        fig2_trajectory_per_curriculum.{pdf,png}
        fig3_finalckpt_bars.{pdf,png}
        fig4_curriculum_coverage.{pdf,png}
        fig5_episodic_memory.{pdf,png}
        fig6_direction_asymmetry_bernolet.{pdf,png}
        fig7_direction_asymmetry_full.{pdf,png}
        fig8_humanscale_consistency.{pdf,png}
        fig9_bgpt_peak_decay.{pdf,png}
        fig10_cl_variant_caveat.{pdf,png}
        fig_summary_metadata.json

Usage:
    python scripts/priming_emnlp_paper_figures.py
"""
from __future__ import annotations
import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s")
logger = logging.getLogger("priming-emnlp")

# ── EMNLP-style rcParams ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Serif",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "axes.linewidth":   0.6,
    "axes.edgecolor":   "#222",
    "axes.labelcolor":  "#222",
    "xtick.color":      "#222",
    "ytick.color":      "#222",
    "xtick.major.width":0.6,
    "ytick.major.width":0.6,
    "xtick.direction":  "out",
    "ytick.direction":  "out",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "savefig.bbox":     "tight",
    "savefig.dpi":      300,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# Colour palette (deuteranopia-safe).
PAL = {
    "bgpt_seq":    "#404040",   # very dark grey for sequential
    "bgpt_sim":    "#9e9e9e",   # mid grey for simultaneous
    "beetle":      "#1f4f8a",
    "B1":          "#2c7bb6",
    "B2":          "#4daf4a",
    "B3":          "#fdae61",
    "B4":          "#984ea3",
    "B5":          "#762a83",
    "episodic":    "#d7191c",
    "ewc":         "#fb9a99",
    "lamol":       "#fdbf6f",
    "lr":          "#cab2d6",
}

# ── Data loading ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
TIERS = ("tier1_final", "tier2_headline_trajectory")


def _classify(model: str) -> str:
    """Return a short curriculum label for grouping/legends."""
    s = model.lower()
    if "classroom" in s:                     return "B4-classroom"
    if "balanced-b4" in s:                   return "B4-balanced"
    if "late-b5" in s or "balanced-b5" in s: return "B5"
    if "sequential-33-67-b3" in s or "sequential-0-100-b3" in s: return "B3-seq"
    if "simultaneous-b2" in s and "ewc"      in s: return "B2-EWC"
    if "simultaneous-b2" in s and "lamol"    in s: return "B2-LAMOL"
    if "simultaneous-b2" in s and "lr-b2"    in s: return "B2-LRshrink"
    if "simultaneous-b2" in s:               return "B2-sim"
    if "balanced-b1" in s and "episodic" in s: return "B1+episodic"
    if "balanced-b1" in s:                   return "B1-bal"
    if "monolingual" in s:                   return "mono"
    return "other"


def _scale(model: str) -> str:
    s = model.lower()
    if "fineweb-2b"   in s: return "fineweb-2B"
    if "fineweb-100m" in s: return "fineweb-100M"
    if "fineweb"      in s: return "fineweb-24B"
    if "humanscale"   in s: return "humanscale"
    return "bgpt"


def _bgpt_label(m: str) -> str:
    """Pretty label for BGPT models."""
    return m.replace("_", "→").replace("→en", "→en").replace("nl→en", "Nl→En")


def load_data() -> pd.DataFrame:
    frames = []
    for tier in TIERS:
        p = REPO_ROOT / "results/evals/priming" / tier / "comparison/priming_effects_per_model.csv"
        if not p.exists():
            logger.warning("missing %s", p); continue
        d = pd.read_csv(p); d["tier"] = tier
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["ckpt_n"] = pd.to_numeric(df["checkpoint"], errors="coerce")
    df["plang"] = df["PrimeLanguage"].astype(str).str.split("_").str[0]
    df["tlang"] = df["TargetLanguage"].astype(str).str.split("_").str[0]
    df["btw"]   = df["plang"] != df["tlang"]
    df["scale"] = df["model"].map(_scale)
    df["cur"]   = df["model"].map(_classify)
    df["direction"] = np.where(
        (df["plang"] == "Dutch")   & (df["tlang"] == "English"), "D→E",
        np.where(
            (df["plang"] == "English") & (df["tlang"] == "Dutch"), "E→D",
            df["plang"].astype(str) + "→" + df["tlang"].astype(str)
        )
    )
    return df


def between_lang(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["btw"]].copy()


def per_model_dataset_btw(df: pd.DataFrame) -> pd.DataFrame:
    """Mean priming over both between-language directions, per (model, dataset, ckpt)."""
    btw = between_lang(df)
    return (btw.groupby(["model_source", "model", "scale", "cur",
                         "dataset", "tier", "ckpt_n", "checkpoint"], dropna=False)
                ["priming_effect"].mean()
                .reset_index())


def final_ckpt(df_btw_agg: pd.DataFrame) -> pd.DataFrame:
    """Per-model final checkpoint row (numerically max ckpt; fall back to 'main')."""
    out = []
    for (m, ds), sub in df_btw_agg.groupby(["model", "dataset"]):
        if sub["ckpt_n"].notna().any():
            row = sub.loc[sub["ckpt_n"].idxmax()]
        else:
            # 'main' rows
            row = sub.iloc[-1]
        out.append(row)
    return pd.DataFrame(out)


# ── Helpers ────────────────────────────────────────────────────────────────

def save(fig: plt.Figure, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    logger.info("  wrote %s.{pdf,png}", name)


def annotate(ax, txt, xy, xytext=None, fs=8, **kw):
    ax.annotate(txt, xy=xy, xytext=xytext or xy, fontsize=fs,
                ha="center", va="bottom", color="#222",
                arrowprops=dict(arrowstyle="-", color="#666", lw=0.5),
                **kw)


# ── Figures ────────────────────────────────────────────────────────────────

def _smooth(y: pd.Series, win: int = 5) -> pd.Series:
    """Rolling mean for trajectory plots (denoises noisy 150-ckpt curves)."""
    return y.rolling(win, min_periods=1, center=True).mean()


def fig1_trajectory_headline(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Headline 2-panel: Schoonbaert + Bernolet. BGPT trajectories vs Beetle FineWeb-24B.

    X-axis = fraction of total training (each model normalised to its own max ckpt),
    so BGPT (~128k steps) and Beetle (~68k) are directly comparable.
    """
    btw = per_model_dataset_btw(df)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=False)

    panels = [("schoonbaert", "Schoonbaert"), ("bernolet", "Bernolet")]
    for ax, (ds, title) in zip(axes, panels):
        sub = btw[(btw.dataset == ds) & btw.ckpt_n.notna()]

        # BGPT trajectories (sim + seq, Nl-En)
        for m, color, lbl in [
            ("nl_en_5050", PAL["bgpt_sim"], "BGPT sim (50/50)"),
            ("nl_en_7525", PAL["bgpt_seq"], "BGPT seq (75/25)"),
        ]:
            d = sub[sub.model == m].sort_values("ckpt_n")
            if not len(d): continue
            x = d.ckpt_n / d.ckpt_n.max()
            ax.plot(x, _smooth(d.priming_effect, 3),
                    color=color, lw=1.8, label=lbl, zorder=4)
            ax.scatter([x.iloc[-1]], [d.priming_effect.iloc[-1]],
                       s=32, color=color, edgecolor="white", linewidth=0.7, zorder=5)

        # Beetle FineWeb-24B trajectories — B1, B2, B3, B5 nld-eng (smoothed)
        beetle_trajs = [
            ("balanced-b1-fineweb-nld-eng",         PAL["B1"], "Beetle B1"),
            ("simultaneous-b2-fineweb-nld-eng",     PAL["B2"], "Beetle B2"),
            ("sequential-33-67-b3-fineweb-nld-eng", PAL["B3"], "Beetle B3"),
            ("balanced-b5-fineweb-nld-eng",         PAL["B5"], "Beetle B5"),
        ]
        for pat, color, lbl in beetle_trajs:
            d = sub[sub.model.str.contains(pat)].sort_values("ckpt_n")
            if not len(d): continue
            x = d.ckpt_n / d.ckpt_n.max()
            ax.plot(x, d.priming_effect, color=color, lw=0.4, alpha=0.25, zorder=2)
            ax.plot(x, _smooth(d.priming_effect, 7),
                    color=color, lw=1.2, alpha=0.95, label=lbl, zorder=3)

        ax.axhline(0, color="#999", lw=0.5, ls=":", zorder=1)
        ax.set_title(title)
        ax.set_xlabel("fraction of training")
        ax.set_xlim(0, 1.0)
        ax.set_ylabel("between-language priming")
        ax.grid(alpha=0.25, lw=0.4)

    handles = [Line2D([0], [0], color=PAL["bgpt_sim"], lw=1.8, label="BGPT sim (50/50)"),
               Line2D([0], [0], color=PAL["bgpt_seq"], lw=1.8, label="BGPT seq (75/25)"),
               Line2D([0], [0], color=PAL["B1"],  lw=1.2, label="Beetle B1 (balanced)"),
               Line2D([0], [0], color=PAL["B2"],  lw=1.2, label="Beetle B2 (sim 50/50)"),
               Line2D([0], [0], color=PAL["B3"],  lw=1.2, label="Beetle B3 (seq 33/67)"),
               Line2D([0], [0], color=PAL["B5"],  lw=1.2, label="Beetle B5 (balanced)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.18))
    fig.suptitle("Priming as transient developmental peak (training-time normalised)",
                 y=1.02, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, out_dir, "fig1_trajectory_headline")
    return "fig1_trajectory_headline"


def fig2_trajectory_per_curriculum(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Small-multiples: BGPT seq (top) and BGPT sim (bottom) overlaid with each Beetle curriculum."""
    btw = per_model_dataset_btw(df)
    beetle_curs = [
        ("balanced-b1-fineweb-nld-eng",         "B1 balanced",  PAL["B1"]),
        ("simultaneous-b2-fineweb-nld-eng",     "B2 sim",       PAL["B2"]),
        ("sequential-33-67-b3-fineweb-nld-eng", "B3 seq 33/67", PAL["B3"]),
        ("balanced-b5-fineweb-nld-eng",         "B5",           PAL["B5"]),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.8), sharex=True, sharey="row")
    panels = [("schoonbaert", "Schoonbaert"), ("bernolet", "Bernolet")]
    for row, (ds, dtitle) in enumerate(panels):
        for col, (pat, lbl, col_) in enumerate(beetle_curs):
            ax = axes[row][col]
            sub = btw[(btw.dataset == ds) & btw.ckpt_n.notna()]
            # BGPT both — normalised to fraction of training
            for m, c, ll in [("nl_en_7525", PAL["bgpt_seq"], "BGPT seq"),
                             ("nl_en_5050", PAL["bgpt_sim"], "BGPT sim")]:
                d = sub[sub.model == m].sort_values("ckpt_n")
                if len(d):
                    x = d.ckpt_n / d.ckpt_n.max()
                    ax.plot(x, _smooth(d.priming_effect, 3), color=c, lw=1.2,
                            label=ll, zorder=3)
            # Beetle — normalised to fraction of training
            d = sub[sub.model.str.contains(pat)].sort_values("ckpt_n")
            if len(d):
                x = d.ckpt_n / d.ckpt_n.max()
                ax.plot(x, d.priming_effect, color=col_, lw=0.4, alpha=0.25, zorder=3)
                ax.plot(x, _smooth(d.priming_effect, 7), color=col_, lw=1.4,
                        label=f"Beetle {lbl}", zorder=4)
            ax.axhline(0, color="#bbb", lw=0.4, ls=":")
            ax.grid(alpha=0.2, lw=0.3)
            ax.set_xlim(0, 1.0)
            if row == 0:
                ax.set_title(lbl, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{dtitle}\nbtw-lang priming", fontsize=9)
            if row == 1:
                ax.set_xlabel("fraction of training")
    handles = [Line2D([0], [0], color=PAL["bgpt_seq"], lw=1.4, label="BGPT seq (75/25)"),
               Line2D([0], [0], color=PAL["bgpt_sim"], lw=1.4, label="BGPT sim (50/50)"),
               Line2D([0], [0], color="#444", lw=1.4, label="Beetle (per curriculum)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Beetle reproduces the developmental peak-and-decay across curricula", y=1.0, fontsize=10)
    fig.tight_layout()
    save(fig, out_dir, "fig2_trajectory_per_curriculum")
    return "fig2_trajectory_per_curriculum"


def fig3_finalckpt_bars(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Final-checkpoint headline bars, between-language only, FineWeb-24B + BGPT."""
    btw = per_model_dataset_btw(df)
    fin = final_ckpt(btw)

    # Selection: BGPT nl-en sim/seq + en-nl sim/seq, Beetle FineWeb-24B 5 curricula (B1, B2, B3, B4, B5)
    selectors = [
        # (model_match_lambda, label, group, color)
        (lambda m: m == "nl_en_7525", "BGPT seq nl→en (75/25)",  "BGPT", PAL["bgpt_seq"]),
        (lambda m: m == "nl_en_5050", "BGPT sim nl/en (50/50)",  "BGPT", PAL["bgpt_sim"]),
        (lambda m: m == "en_nl_7525", "BGPT seq en→nl (75/25)",  "BGPT", "#606060"),
        (lambda m: m == "en_nl_5050", "BGPT sim en/nl (50/50)",  "BGPT", "#bdbdbd"),
        (lambda m: "balanced-b1-fineweb-nld-eng" in m,         "Beetle B1 nld→eng",  "Beetle-FineWeb-24B", PAL["B1"]),
        (lambda m: "simultaneous-b2-fineweb-nld-eng" in m,     "Beetle B2 nld→eng",  "Beetle-FineWeb-24B", PAL["B2"]),
        (lambda m: "sequential-33-67-b3-fineweb-nld-eng" in m, "Beetle B3 nld→eng",  "Beetle-FineWeb-24B", PAL["B3"]),
        (lambda m: "balanced-b4-fineweb-nld-eng" in m,         "Beetle B4 nld→eng",  "Beetle-FineWeb-24B", PAL["B4"]),
        (lambda m: "balanced-b5-fineweb-nld-eng" in m,         "Beetle B5 nld→eng",  "Beetle-FineWeb-24B", PAL["B5"]),
    ]

    rows = []
    for sel_fn, label, grp, color in selectors:
        for ds in ["schoonbaert", "bernolet"]:
            d = fin[(fin.dataset == ds) & fin.model.map(sel_fn)]
            if not len(d): continue
            rows.append({"label": label, "group": grp, "color": color, "dataset": ds,
                         "priming": d.priming_effect.iloc[0]})
    bars = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, ds, title in zip(axes, ["schoonbaert", "bernolet"], ["Schoonbaert", "Bernolet"]):
        sub = bars[bars.dataset == ds]
        order = [r["label"] for r in [
            {"label": "BGPT seq nl→en (75/25)"},
            {"label": "BGPT sim nl/en (50/50)"},
            {"label": "BGPT seq en→nl (75/25)"},
            {"label": "BGPT sim en/nl (50/50)"},
            {"label": "Beetle B1 nld→eng"},
            {"label": "Beetle B2 nld→eng"},
            {"label": "Beetle B3 nld→eng"},
            {"label": "Beetle B4 nld→eng"},
            {"label": "Beetle B5 nld→eng"},
        ]]
        sub = sub.set_index("label").reindex(order).dropna().reset_index()
        ypos = np.arange(len(sub))
        ax.barh(ypos, sub.priming, color=sub.color, edgecolor="#222", linewidth=0.5)
        ax.set_yticks(ypos); ax.set_yticklabels(sub.label, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="#888", lw=0.6)
        ax.set_xlabel("between-language priming (final ckpt)")
        ax.set_title(title)
        ax.grid(alpha=0.25, lw=0.4, axis="x")
        # Annotate values
        for y, v in zip(ypos, sub.priming):
            ax.text(v + (0.005 if v >= 0 else -0.005), y, f"{v:+.3f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=7)
    legend_elems = [Patch(color=PAL["bgpt_seq"], label="BGPT (Arnett 2025)"),
                    Patch(color=PAL["B1"],       label="Beetle FineWeb-24B")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Final-checkpoint between-language priming: BGPT vs Beetle FineWeb-24B", y=1.0)
    fig.tight_layout()
    save(fig, out_dir, "fig3_finalckpt_bars")
    return "fig3_finalckpt_bars"


def fig4_curriculum_coverage(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Visualise how much of the curriculum space BGPT covers (2 schedules) vs Beetle (5 curricula)."""
    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    # X axis: training-time L2 ratio. Y axis: schedule type.
    # BGPT: only sim 50/50 and seq 75/25 (= 25% L2).
    # Beetle: B1 balanced (50/50), B2 sim (50/50), B3 seq (33/67 = 67% L2), B4 classroom-20 (20% L2), B5 late 80% (80% L2 late only).
    rows = [
        # (label, l2_pct, training_pos, color)
        ("BGPT sim (nl_en_5050)", 50, "throughout",         PAL["bgpt_sim"]),
        ("BGPT seq (nl_en_7525)", 25, "after 75% L1 phase", PAL["bgpt_seq"]),
        ("Beetle B1 balanced",    50, "throughout",         PAL["B1"]),
        ("Beetle B2 sim 50/50",   50, "throughout",         PAL["B2"]),
        ("Beetle B3 seq 33/67",   67, "after 33% L1 phase", PAL["B3"]),
        ("Beetle B4 classroom-20",20, "interleaved blocks", PAL["B4"]),
        ("Beetle B5 late 80%",    80, "final 20% only",     PAL["B5"]),
    ]
    df_ = pd.DataFrame(rows, columns=["label", "l2_pct", "schedule", "color"])
    ypos = np.arange(len(df_))[::-1]
    ax.barh(ypos, df_.l2_pct, color=df_.color, edgecolor="#222", linewidth=0.4, height=0.6)
    for y, (lbl, l2, sched, _) in zip(ypos, rows):
        ax.text(l2 + 1.5, y, f"{l2}% L2 | {sched}", va="center", fontsize=8, color="#222")
    ax.set_yticks(ypos); ax.set_yticklabels(df_.label, fontsize=8)
    ax.set_xlabel("L2 token share (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Curriculum coverage: BGPT samples 2 points, Beetle samples 5")
    ax.grid(alpha=0.25, lw=0.4, axis="x")
    fig.tight_layout()
    save(fig, out_dir, "fig4_curriculum_coverage")
    return "fig4_curriculum_coverage"


def fig5_episodic_memory(df: pd.DataFrame, out_dir: Path) -> str | None:
    """B1 baseline vs B1+episodic on Schoonbaert + Bernolet.
    All-CL bars are shown on a separate panel with a CAVEAT footnote about
    the bitwise-identical surprisals across EWC/LAMOL/LR variants.
    """
    btw = per_model_dataset_btw(df)
    fin = final_ckpt(btw)
    targets = {
        "B1 baseline":              "balanced-b1-humanscale-nld-eng",
        "B1 + episodic":            "balanced-b1-humanscale-nld-eng-episodic",
        "B2 sim baseline":          "simultaneous-b2-humanscale-nld-eng",
        "B2 + EWC (l20)":           "simultaneous-b2-humanscale-nld-eng-ewc-l20",
        "B2 + LAMOL (l20)":         "simultaneous-b2-humanscale-nld-eng-lamol-l20",
        "B2 + LR-shrink":           "simultaneous-b2-humanscale-nld-eng-lr-b2",
    }
    rows = []
    for label, pat in targets.items():
        for ds in ["schoonbaert", "bernolet"]:
            # match exactly (no '-episodic' eaten by '-')
            d = fin[(fin.dataset == ds) & fin.model.str.endswith(pat)]
            if not len(d): continue
            rows.append({"label": label, "dataset": ds, "priming": d.priming_effect.iloc[0]})
    bars = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    # Left: B1 vs B1+episodic — clean single contrast
    ax = axes[0]
    ep = bars[bars.label.isin(["B1 baseline", "B1 + episodic"])]
    pivot = ep.pivot(index="dataset", columns="label", values="priming").reindex(["schoonbaert", "bernolet"])
    x = np.arange(len(pivot))
    w = 0.35
    ax.bar(x - w/2, pivot["B1 baseline"],   width=w, color=PAL["B1"],       label="B1 baseline",   edgecolor="#222", lw=0.4)
    ax.bar(x + w/2, pivot["B1 + episodic"], width=w, color=PAL["episodic"], label="B1 + episodic", edgecolor="#222", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(["Schoonbaert", "Bernolet"])
    ax.set_ylabel("between-language priming (final)")
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_title("Episodic memory boosts cross-lingual priming")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    for i, (do, dp) in enumerate(zip(pivot["B1 baseline"], pivot["B1 + episodic"])):
        ax.text(x[i]-w/2, do + 0.005, f"{do:+.3f}", ha="center", fontsize=7)
        ax.text(x[i]+w/2, dp + 0.005, f"{dp:+.3f}", ha="center", fontsize=7)

    # Right: B2 + CL variants (with caveat banner)
    ax = axes[1]
    cl = bars[bars.label.isin(["B2 sim baseline", "B2 + EWC (l20)", "B2 + LAMOL (l20)", "B2 + LR-shrink"])]
    pivot = cl.pivot(index="dataset", columns="label", values="priming").reindex(["schoonbaert", "bernolet"])
    cols = ["B2 sim baseline", "B2 + EWC (l20)", "B2 + LAMOL (l20)", "B2 + LR-shrink"]
    colors = [PAL["B2"], PAL["ewc"], PAL["lamol"], PAL["lr"]]
    pivot = pivot.reindex(columns=cols)
    x = np.arange(len(pivot)); w = 0.18
    for i, (col, color) in enumerate(zip(cols, colors)):
        ax.bar(x + (i - 1.5) * w, pivot[col], width=w, color=color, label=col, edgecolor="#222", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(["Schoonbaert", "Bernolet"])
    ax.set_ylabel("between-language priming (final)")
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_title("CL regularizers: identical surprisals (see caveat)")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.25, lw=0.4, axis="y")

    fig.suptitle("Memory & continual-learning interventions on Schoonbaert + Bernolet", y=1.02)
    # Caveat footnote
    fig.text(0.5, -0.10,
             "Caveat: EWC, LAMOL, and LR-shrink TSVs share bitwise-identical surprisal columns "
             "(see PRIMING_RESULTS.md §Data integrity).\nThe three CL variants in the figure should "
             "be read as ONE result re-emitted under three model names until the eval is re-run.",
             ha="center", va="top", fontsize=7.5, color="#a00", style="italic")
    fig.tight_layout()
    save(fig, out_dir, "fig5_episodic_memory")
    return "fig5_episodic_memory"


def fig6_direction_asymmetry_bernolet(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Forest plot of D→E vs E→D priming on BERNOLET only — where the L1→L2 dominance pattern holds."""
    btw = between_lang(df).copy()
    btw["final_ck"] = btw.groupby(["model","dataset"])["ckpt_n"].transform("max")
    fin = btw[(btw.dataset == "bernolet") & ((btw.ckpt_n == btw.final_ck) | (btw.ckpt_n.isna()))]

    # Show the canonical models in a clean order
    keep = [
        ("nl_en_7525",                                                                           "BGPT seq Nl→En (75/25)", "BGPT"),
        ("nl_en_5050",                                                                           "BGPT sim Nl/En (50/50)", "BGPT"),
        ("en_nl_7525",                                                                           "BGPT seq En→Nl (75/25)", "BGPT"),
        ("en_nl_5050",                                                                           "BGPT sim En/Nl (50/50)", "BGPT"),
        ("beetle-fineweb/beetle-bilingual-balanced-b1-fineweb-nld-eng",                          "Beetle B1 nld-eng",       "FW-24B"),
        ("beetle-fineweb/beetle-bilingual-l2-50-simultaneous-b2-fineweb-nld-eng",                "Beetle B2 nld-eng",       "FW-24B"),
        ("beetle-fineweb/beetle-bilingual-balanced-b4-fineweb-eng-nld",                          "Beetle B4 eng-nld",       "FW-24B"),
        ("beetle-fineweb/beetle-bilingual-balanced-b5-fineweb-nld-eng",                          "Beetle B5 nld-eng",       "FW-24B"),
        ("beetle-humanscale/beetle-bilingual-balanced-b1-humanscale-nld-eng",                    "Beetle B1 (HS)",          "HS"),
        ("beetle-humanscale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng",          "Beetle B2 (HS)",          "HS"),
        ("beetle-humanscale/beetle-bilingual-l2-50-sequential-33-67-b3-humanscale-nld-eng",      "Beetle B3 (HS)",          "HS"),
        ("beetle-humanscale/beetle-bilingual-l2-80-late-b5-humanscale-nld-eng",                  "Beetle B5 (HS)",          "HS"),
        ("beetle-humanscale/beetle-bilingual-l2-50-classroom-20-b4-humanscale-nld-eng",          "Beetle B4-classroom (HS)","HS"),
    ]
    rows = []
    for model, label, group in keep:
        sub = fin[fin.model == model]
        if not len(sub): continue
        de = sub[sub.direction == "D→E"].priming_effect.mean()
        ed = sub[sub.direction == "E→D"].priming_effect.mean()
        rows.append({"label": label, "group": group, "D→E": de, "E→D": ed})
    if not rows: return None
    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y = np.arange(len(d))[::-1]
    ax.scatter(d["D→E"], y, color="#1f4f8a", marker="o", s=42, label="Dutch → English (L1→L2)", zorder=3)
    ax.scatter(d["E→D"], y, color="#d7191c", marker="s", s=42, label="English → Dutch (L2→L1)", zorder=3)
    for i, row in enumerate(d.itertuples()):
        yi = y[i]
        ax.plot([row._3, row._4], [yi, yi], color="#888", lw=0.6, zorder=2)  # _3=D→E, _4=E→D
    ax.set_yticks(y); ax.set_yticklabels(d.label, fontsize=8)
    ax.axvline(0, color="#888", lw=0.6)
    ax.set_xlabel("between-language priming (Bernolet, final ckpt)")
    ax.set_title("Bernolet: L1→L2 (D→E) priming dominates across nearly all conditions")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, lw=0.4, axis="x")
    fig.tight_layout()
    save(fig, out_dir, "fig6_direction_asymmetry_bernolet")
    return "fig6_direction_asymmetry_bernolet"


def fig7_direction_asymmetry_full(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Same as fig6 but 2 panels (Bernolet + Schoonbaert), to make the Schoonbaert ambiguity visible."""
    btw = between_lang(df).copy()
    btw["final_ck"] = btw.groupby(["model","dataset"])["ckpt_n"].transform("max")

    keep = [
        ("nl_en_7525",                                                                       "BGPT seq Nl→En"),
        ("nl_en_5050",                                                                       "BGPT sim Nl/En"),
        ("en_nl_7525",                                                                       "BGPT seq En→Nl"),
        ("en_nl_5050",                                                                       "BGPT sim En/Nl"),
        ("beetle-fineweb/beetle-bilingual-balanced-b1-fineweb-nld-eng",                      "Beetle B1 (FW)"),
        ("beetle-fineweb/beetle-bilingual-l2-50-simultaneous-b2-fineweb-nld-eng",            "Beetle B2 (FW)"),
        ("beetle-fineweb/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng",        "Beetle B3 (FW)"),
        ("beetle-fineweb/beetle-bilingual-balanced-b4-fineweb-nld-eng",                      "Beetle B4 (FW)"),
        ("beetle-fineweb/beetle-bilingual-balanced-b5-fineweb-nld-eng",                      "Beetle B5 (FW)"),
        ("beetle-humanscale/beetle-bilingual-balanced-b1-humanscale-nld-eng",                "Beetle B1 (HS)"),
        ("beetle-humanscale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng",      "Beetle B2 (HS)"),
        ("beetle-humanscale/beetle-bilingual-l2-50-sequential-33-67-b3-humanscale-nld-eng",  "Beetle B3 (HS)"),
        ("beetle-humanscale/beetle-bilingual-l2-50-classroom-20-b4-humanscale-nld-eng",      "Beetle B4-cls (HS)"),
        ("beetle-humanscale/beetle-bilingual-l2-80-late-b5-humanscale-nld-eng",              "Beetle B5 (HS)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True)
    for ax, ds, title in zip(axes, ["bernolet", "schoonbaert"], ["Bernolet (consistent L1→L2)", "Schoonbaert (mixed)"]):
        rows = []
        sub = btw[(btw.dataset == ds) & ((btw.ckpt_n == btw.final_ck) | btw.ckpt_n.isna())]
        for m, lbl in keep:
            d = sub[sub.model == m]
            if not len(d): continue
            de = d[d.direction == "D→E"].priming_effect.mean()
            ed = d[d.direction == "E→D"].priming_effect.mean()
            if pd.isna(de) and pd.isna(ed): continue
            rows.append({"label": lbl, "D→E": de, "E→D": ed})
        if not rows: continue
        d = pd.DataFrame(rows)
        y = np.arange(len(d))[::-1]
        for i, row in enumerate(d.itertuples()):
            yi = y[i]
            ax.plot([row._2, row._3], [yi, yi], color="#888", lw=0.6, zorder=2)
        ax.scatter(d["D→E"], y, color="#1f4f8a", marker="o", s=36, label="D→E", zorder=3)
        ax.scatter(d["E→D"], y, color="#d7191c", marker="s", s=36, label="E→D", zorder=3)
        ax.set_yticks(y); ax.set_yticklabels(d.label, fontsize=8)
        ax.axvline(0, color="#888", lw=0.6)
        ax.set_xlabel("between-language priming (final ckpt)")
        ax.set_title(title)
        ax.grid(alpha=0.25, lw=0.4, axis="x")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Directional asymmetry holds on Bernolet, dissolves on Schoonbaert", y=1.0)
    fig.tight_layout()
    save(fig, out_dir, "fig7_direction_asymmetry_full")
    return "fig7_direction_asymmetry_full"


def fig8_humanscale_consistency(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Side-by-side FineWeb-24B vs HumanScale final-ckpt priming — does scale change the curriculum ordering?"""
    btw = per_model_dataset_btw(df)
    fin = final_ckpt(btw)

    pairs = [
        ("B1", "balanced-b1-fineweb-nld-eng",         "balanced-b1-humanscale-nld-eng"),
        ("B2", "simultaneous-b2-fineweb-nld-eng",     "simultaneous-b2-humanscale-nld-eng"),
        ("B3", "sequential-33-67-b3-fineweb-nld-eng", "sequential-33-67-b3-humanscale-nld-eng"),
        ("B5", "balanced-b5-fineweb-nld-eng",         "late-b5-humanscale-nld-eng"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for ax, ds, title in zip(axes, ["schoonbaert", "bernolet"], ["Schoonbaert", "Bernolet"]):
        rows = []
        for cur, fw_pat, hs_pat in pairs:
            fw = fin[(fin.dataset == ds) & fin.model.str.endswith(fw_pat)]
            hs = fin[(fin.dataset == ds) & fin.model.str.endswith(hs_pat)]
            rows.append({"cur": cur,
                         "FineWeb-24B": fw.priming_effect.mean() if len(fw) else np.nan,
                         "HumanScale":  hs.priming_effect.mean() if len(hs) else np.nan})
        d = pd.DataFrame(rows)
        x = np.arange(len(d)); w = 0.38
        ax.bar(x - w/2, d["FineWeb-24B"], width=w, color=PAL["beetle"], label="FineWeb-24B", edgecolor="#222", lw=0.4)
        ax.bar(x + w/2, d["HumanScale"],  width=w, color="#fdae61",      label="HumanScale",  edgecolor="#222", lw=0.4)
        ax.set_xticks(x); ax.set_xticklabels(d.cur)
        ax.set_ylabel("between-language priming (final)")
        ax.axhline(0, color="#888", lw=0.6)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25, lw=0.4, axis="y")
    fig.suptitle("Scale-vs-curriculum: does FineWeb-24B preserve HumanScale ordering?", y=1.02)
    fig.tight_layout()
    save(fig, out_dir, "fig8_humanscale_consistency")
    return "fig8_humanscale_consistency"


def fig9_bgpt_peak_decay(df: pd.DataFrame, out_dir: Path) -> str | None:
    """For every BGPT model with a trajectory, show peak vs final priming — quantifies the 'sampling artifact' claim."""
    btw = per_model_dataset_btw(df)
    bg = btw[(btw.model_source == "bgpt") & btw.ckpt_n.notna()]
    rows = []
    for (m, ds), sub in bg.groupby(["model", "dataset"]):
        if len(sub) < 5: continue
        peak = sub.priming_effect.max()
        peak_step = sub.loc[sub.priming_effect.idxmax(), "ckpt_n"]
        final = sub.sort_values("ckpt_n").priming_effect.iloc[-1]
        rows.append({"model": m, "dataset": ds, "peak": peak, "final": final,
                     "peak_step": peak_step,
                     "decay_pct": (peak - final) / peak if peak > 0 else 0})
    d = pd.DataFrame(rows)
    if not len(d): return None

    # Filter to non-trivial peaks (> 0.05) so % decay labels are meaningful;
    # sort by peak magnitude so the largest effects appear at the top.
    d = d[d.peak >= 0.05].copy()
    if not len(d): return None

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    d = d.sort_values("peak", ascending=False).reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    ax.barh(y, d.peak, color="#fdae61", label="peak priming", height=0.4, edgecolor="#222", lw=0.4)
    ax.barh(y - 0.4, d.final, color="#1f4f8a", label="final-ckpt priming", height=0.4, edgecolor="#222", lw=0.4)
    labels = [f"{r.model}  ·  {r.dataset}" for r in d.itertuples()]
    ax.set_yticks(y - 0.2); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("between-language priming")
    ax.axvline(0, color="#888", lw=0.6)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    for i, r in enumerate(d.itertuples()):
        ax.text(max(r.peak, r.final) + 0.005, y[i] - 0.2,
                f"{r.decay_pct*100:+.0f}%", fontsize=7, va="center", color="#444")
    ax.set_title("BGPT trajectories: peak vs final-checkpoint priming\n(% labels = decay from peak)")
    ax.grid(alpha=0.25, lw=0.4, axis="x")
    fig.tight_layout()
    save(fig, out_dir, "fig9_bgpt_peak_decay")
    return "fig9_bgpt_peak_decay"


def fig10_cl_variant_caveat(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Diagnostic visualisation of the EWC/LAMOL/LR-shrink bitwise-identical surprisal issue."""
    # Pull raw TSV mean surprisals as a side panel
    raw_files = {
        "B2 + EWC":   REPO_ROOT / "results/evals/priming/tier1_final/schoonbaert__beetle-humanscale_beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng-ewc-l20__main.tsv",
        "B2 + LAMOL": REPO_ROOT / "results/evals/priming/tier1_final/schoonbaert__beetle-humanscale_beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng-lamol-l20__main.tsv",
        "B2 + LR":    REPO_ROOT / "results/evals/priming/tier1_final/schoonbaert__beetle-humanscale_beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng-lr-b2__main.tsv",
        "B1+epi":     REPO_ROOT / "results/evals/priming/tier1_final/schoonbaert__beetle-humanscale_beetle-bilingual-balanced-b1-humanscale-nld-eng-episodic__main.tsv",
        "B1 base":    REPO_ROOT / "results/evals/priming/tier1_final/schoonbaert__beetle-humanscale_beetle-bilingual-balanced-b1-humanscale-nld-eng__main.tsv",
        "B2 sim base":REPO_ROOT / "results/evals/priming/tier1_final/schoonbaert__beetle-humanscale_beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng__main.tsv",
    }
    means, stds = {}, {}
    for n, fp in raw_files.items():
        if not fp.exists(): continue
        d = pd.read_csv(fp, sep="\t")
        if "Surprisal" in d.columns:
            means[n] = d["Surprisal"].mean()
            stds[n]  = d["Surprisal"].std()
    if not means: return None
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    names = list(means.keys())
    y = np.arange(len(names))[::-1]
    vals = [means[n] for n in names]
    errs = [stds[n] for n in names]
    colors = ["#fb9a99" if n.startswith("B2 + EWC") else
              "#fdbf6f" if n.startswith("B2 + LAMOL") else
              "#cab2d6" if n.startswith("B2 + LR") else
              "#d7191c" if n.startswith("B1+epi") else
              "#1f4f8a" if n.startswith("B1 base") else
              "#4daf4a" for n in names]
    ax.barh(y, vals, xerr=errs, color=colors, edgecolor="#222", lw=0.4, error_kw=dict(lw=0.6, ecolor="#444"))
    for yi, n, v in zip(y, names, vals):
        ax.text(v + 0.5, yi, f"{v:.4f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("mean surprisal (Schoonbaert TSV)")
    ax.set_title("Schoonbaert raw-TSV mean surprisal: EWC ≡ LAMOL ≡ LR-shrink (bitwise)")
    ax.grid(alpha=0.25, lw=0.4, axis="x")
    fig.text(0.5, -0.05,
             "EWC, LAMOL, LR-shrink TSVs share identical surprisal columns despite distinct file metadata. "
             "Their mean ± SD bars are visually indistinguishable. B1+episodic and the two baselines differ.",
             ha="center", va="top", fontsize=7.5, color="#a00", style="italic")
    fig.tight_layout()
    save(fig, out_dir, "fig10_cl_variant_caveat")
    return "fig10_cl_variant_caveat"


# ── Bernolet-only main-paper variants ──────────────────────────────────────
# These are streamlined single-dataset versions of figs 1, 2, 3, 8, 9
# intended for the EMNLP main paper. The dual-panel originals stay as
# supplementary so the Schoonbaert results are still archived.

def fig_main_trajectory_bernolet(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Main paper: single-panel Bernolet trajectory, training-time normalised."""
    btw = per_model_dataset_btw(df)
    sub = btw[(btw.dataset == "bernolet") & btw.ckpt_n.notna()]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    for m, color, lbl in [
        ("nl_en_5050", PAL["bgpt_sim"], "BGPT sim (50/50)"),
        ("nl_en_7525", PAL["bgpt_seq"], "BGPT seq (75/25)"),
    ]:
        d = sub[sub.model == m].sort_values("ckpt_n")
        if not len(d): continue
        x = d.ckpt_n / d.ckpt_n.max()
        ax.plot(x, _smooth(d.priming_effect, 3), color=color, lw=1.8, label=lbl, zorder=4)
        ax.scatter([x.iloc[-1]], [d.priming_effect.iloc[-1]],
                   s=32, color=color, edgecolor="white", linewidth=0.7, zorder=5)

    beetle_trajs = [
        ("balanced-b1-fineweb-nld-eng",         PAL["B1"], "Beetle B1 (balanced)"),
        ("simultaneous-b2-fineweb-nld-eng",     PAL["B2"], "Beetle B2 (sim 50/50)"),
        ("sequential-33-67-b3-fineweb-nld-eng", PAL["B3"], "Beetle B3 (seq 33/67)"),
        ("balanced-b5-fineweb-nld-eng",         PAL["B5"], "Beetle B5 (balanced)"),
    ]
    for pat, color, lbl in beetle_trajs:
        d = sub[sub.model.str.contains(pat)].sort_values("ckpt_n")
        if not len(d): continue
        x = d.ckpt_n / d.ckpt_n.max()
        ax.plot(x, d.priming_effect, color=color, lw=0.4, alpha=0.25, zorder=2)
        ax.plot(x, _smooth(d.priming_effect, 7), color=color, lw=1.4, alpha=0.95,
                label=lbl, zorder=3)

    ax.axhline(0, color="#999", lw=0.5, ls=":", zorder=1)
    ax.set_xlabel("fraction of training")
    ax.set_xlim(0, 1.0)
    ax.set_ylabel("between-language priming (Bernolet)")
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)
    fig.suptitle("Cross-lingual priming: BGPT vs Beetle (Bernolet, training-time normalised)",
                 y=0.98, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out_dir, "fig_main_trajectory_bernolet")
    return "fig_main_trajectory_bernolet"


def fig_main_per_curriculum_bernolet(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Main paper: small-multiples (1×4) Bernolet, each Beetle curriculum vs BGPT."""
    btw = per_model_dataset_btw(df)
    sub = btw[(btw.dataset == "bernolet") & btw.ckpt_n.notna()]
    beetle_curs = [
        ("balanced-b1-fineweb-nld-eng",         "B1 balanced",  PAL["B1"]),
        ("simultaneous-b2-fineweb-nld-eng",     "B2 sim",       PAL["B2"]),
        ("sequential-33-67-b3-fineweb-nld-eng", "B3 seq 33/67", PAL["B3"]),
        ("balanced-b5-fineweb-nld-eng",         "B5",           PAL["B5"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), sharey=True, sharex=True)
    for ax, (pat, lbl, col_) in zip(axes, beetle_curs):
        for m, c, ll in [("nl_en_7525", PAL["bgpt_seq"], "BGPT seq"),
                         ("nl_en_5050", PAL["bgpt_sim"], "BGPT sim")]:
            d = sub[sub.model == m].sort_values("ckpt_n")
            if len(d):
                x = d.ckpt_n / d.ckpt_n.max()
                ax.plot(x, _smooth(d.priming_effect, 3), color=c, lw=1.2, label=ll, zorder=3)
        d = sub[sub.model.str.contains(pat)].sort_values("ckpt_n")
        if len(d):
            x = d.ckpt_n / d.ckpt_n.max()
            ax.plot(x, d.priming_effect, color=col_, lw=0.4, alpha=0.25, zorder=3)
            ax.plot(x, _smooth(d.priming_effect, 7), color=col_, lw=1.4,
                    label=f"Beetle {lbl}", zorder=4)
        ax.axhline(0, color="#bbb", lw=0.4, ls=":")
        ax.grid(alpha=0.2, lw=0.3)
        ax.set_xlim(0, 1.0)
        ax.set_title(lbl, fontsize=9)
        ax.set_xlabel("fraction of training")
    axes[0].set_ylabel("between-language priming\n(Bernolet)", fontsize=9)
    handles = [Line2D([0], [0], color=PAL["bgpt_seq"], lw=1.4, label="BGPT seq (75/25)"),
               Line2D([0], [0], color=PAL["bgpt_sim"], lw=1.4, label="BGPT sim (50/50)"),
               Line2D([0], [0], color="#444", lw=1.4, label="Beetle (per curriculum)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Beetle reproduces BGPT priming on Bernolet across curricula", y=1.0, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save(fig, out_dir, "fig_main_per_curriculum_bernolet")
    return "fig_main_per_curriculum_bernolet"


def fig_main_finalckpt_bernolet(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Main paper: single-panel Bernolet final-checkpoint bars."""
    btw = per_model_dataset_btw(df)
    fin = final_ckpt(btw)
    selectors = [
        (lambda m: m == "nl_en_7525", "BGPT seq nl→en (75/25)",  PAL["bgpt_seq"]),
        (lambda m: m == "nl_en_5050", "BGPT sim nl/en (50/50)",  PAL["bgpt_sim"]),
        (lambda m: m == "en_nl_7525", "BGPT seq en→nl (75/25)",  "#606060"),
        (lambda m: m == "en_nl_5050", "BGPT sim en/nl (50/50)",  "#bdbdbd"),
        (lambda m: "balanced-b1-fineweb-nld-eng" in m,         "Beetle B1 nld→eng",  PAL["B1"]),
        (lambda m: "simultaneous-b2-fineweb-nld-eng" in m,     "Beetle B2 nld→eng",  PAL["B2"]),
        (lambda m: "sequential-33-67-b3-fineweb-nld-eng" in m, "Beetle B3 nld→eng",  PAL["B3"]),
        (lambda m: "balanced-b4-fineweb-nld-eng" in m,         "Beetle B4 nld→eng",  PAL["B4"]),
        (lambda m: "balanced-b5-fineweb-nld-eng" in m,         "Beetle B5 nld→eng",  PAL["B5"]),
    ]
    rows = []
    for sel_fn, label, color in selectors:
        d = fin[(fin.dataset == "bernolet") & fin.model.map(sel_fn)]
        if not len(d): continue
        rows.append({"label": label, "color": color, "priming": d.priming_effect.iloc[0]})
    if not rows: return None
    sub = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ypos = np.arange(len(sub))
    ax.barh(ypos, sub.priming, color=sub.color, edgecolor="#222", linewidth=0.5)
    ax.set_yticks(ypos); ax.set_yticklabels(sub.label, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#888", lw=0.6)
    ax.set_xlabel("between-language priming, Bernolet (final ckpt)")
    ax.grid(alpha=0.25, lw=0.4, axis="x")
    for y, v in zip(ypos, sub.priming):
        ax.text(v + (0.004 if v >= 0 else -0.004), y, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=7)
    fig.suptitle("Final-checkpoint between-language priming on Bernolet",
                 y=0.98, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out_dir, "fig_main_finalckpt_bernolet")
    return "fig_main_finalckpt_bernolet"


def fig_main_humanscale_bernolet(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Main paper: scale comparison FineWeb-24B vs HumanScale, Bernolet only."""
    btw = per_model_dataset_btw(df)
    fin = final_ckpt(btw)
    pairs = [
        ("B1", "balanced-b1-fineweb-nld-eng",         "balanced-b1-humanscale-nld-eng"),
        ("B2", "simultaneous-b2-fineweb-nld-eng",     "simultaneous-b2-humanscale-nld-eng"),
        ("B3", "sequential-33-67-b3-fineweb-nld-eng", "sequential-33-67-b3-humanscale-nld-eng"),
        ("B5", "balanced-b5-fineweb-nld-eng",         "late-b5-humanscale-nld-eng"),
    ]
    rows = []
    for cur, fw_pat, hs_pat in pairs:
        fw = fin[(fin.dataset == "bernolet") & fin.model.str.endswith(fw_pat)]
        hs = fin[(fin.dataset == "bernolet") & fin.model.str.endswith(hs_pat)]
        rows.append({"cur": cur,
                     "FineWeb-24B": fw.priming_effect.mean() if len(fw) else np.nan,
                     "HumanScale":  hs.priming_effect.mean() if len(hs) else np.nan})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    x = np.arange(len(d)); w = 0.38
    ax.bar(x - w/2, d["FineWeb-24B"], width=w, color=PAL["beetle"], label="FineWeb-24B",
           edgecolor="#222", lw=0.4)
    ax.bar(x + w/2, d["HumanScale"],  width=w, color="#fdae61", label="HumanScale",
           edgecolor="#222", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(d.cur)
    ax.set_ylabel("between-language priming, Bernolet (final)")
    ax.axhline(0, color="#888", lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    fig.suptitle("Scale × curriculum on Bernolet", y=0.98, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out_dir, "fig_main_humanscale_bernolet")
    return "fig_main_humanscale_bernolet"


def fig_main_peak_decay_bernolet(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Main paper: BGPT peak vs final priming, Bernolet only, peak ≥0.05, sorted by peak."""
    btw = per_model_dataset_btw(df)
    bg = btw[(btw.model_source == "bgpt") & (btw.dataset == "bernolet") & btw.ckpt_n.notna()]
    rows = []
    for m, sub in bg.groupby("model"):
        if len(sub) < 5: continue
        peak = sub.priming_effect.max()
        final = sub.sort_values("ckpt_n").priming_effect.iloc[-1]
        rows.append({"model": m, "peak": peak, "final": final,
                     "decay_pct": (peak - final) / peak if peak > 0 else 0})
    d = pd.DataFrame(rows)
    if not len(d): return None
    d = d[d.peak >= 0.05].sort_values("peak", ascending=False).reset_index(drop=True)
    if not len(d): return None

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d.peak, color="#fdae61", label="peak priming", height=0.4, edgecolor="#222", lw=0.4)
    ax.barh(y - 0.4, d.final, color="#1f4f8a", label="final-ckpt priming", height=0.4, edgecolor="#222", lw=0.4)
    ax.set_yticks(y - 0.2); ax.set_yticklabels(d.model, fontsize=8)
    ax.set_xlabel("between-language priming (Bernolet)")
    ax.axvline(0, color="#888", lw=0.6)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    for i, r in enumerate(d.itertuples()):
        ax.text(max(r.peak, r.final) + 0.005, y[i] - 0.2,
                f"{r.decay_pct*100:+.0f}%", fontsize=7, va="center", color="#444")
    ax.set_title("BGPT trajectories on Bernolet: peak vs final-checkpoint")
    fig.tight_layout()
    save(fig, out_dir, "fig_main_peak_decay_bernolet")
    return "fig_main_peak_decay_bernolet"


# ── Driver ─────────────────────────────────────────────────────────────────

# ── Reviewer-proofing extensions (Arnett 2025 inheritance) ─────────────────

TIER1_DIR = REPO_ROOT / "results/evals/priming/tier1_final"
TIER2_DIR = REPO_ROOT / "results/evals/priming/tier2_headline_trajectory"
STATS_DIR = REPO_ROOT / "results/evals/priming/stats"

_BEETLE_FW_FINALS = [
    ("B1", "bernolet*balanced-b1-fineweb-nld-eng__main.tsv",         PAL["B1"]),
    ("B2", "bernolet*simultaneous-b2-fineweb-nld-eng__main.tsv",     PAL["B2"]),
    ("B3", "bernolet*sequential-33-67-b3-fineweb-nld-eng__main.tsv", PAL["B3"]),
    ("B4", "bernolet*balanced-b4-fineweb-nld-eng__main.tsv",         PAL["B4"]),
    ("B5", "bernolet*balanced-b5-fineweb-nld-eng__main.tsv",         PAL["B5"]),
]

_BEETLE_FW_TRAJ = [
    ("B1", "bernolet__beetle-fineweb_beetle-bilingual-balanced-b1-fineweb-nld-eng",         PAL["B1"]),
    ("B2", "bernolet__beetle-fineweb_beetle-bilingual-l2-50-simultaneous-b2-fineweb-nld-eng", PAL["B2"]),
    ("B3", "bernolet__beetle-fineweb_beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng", PAL["B3"]),
    ("B5", "bernolet__beetle-fineweb_beetle-bilingual-balanced-b5-fineweb-nld-eng",         PAL["B5"]),
]


def _import_priming_stats():
    """Lazy-import priming_stats helpers without polluting top of file."""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "priming_stats", REPO_ROOT / "scripts/priming_stats.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["priming_stats"] = mod
    spec.loader.exec_module(mod)
    return mod


def fig_pn_robustness_bernolet(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Arnett 2025 normalised relative probability (P_N) per Beetle FineWeb
    final-ckpt model on Bernolet between-lang items. Reviewer prompt:
    'why surprisal-diff and not P_N?' — this figure shows the qualitative
    pattern survives the metric switch."""
    ps = _import_priming_stats()
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    xs, ys, cs, lbls = [], [], [], []
    for i, (label, pat, color) in enumerate(_BEETLE_FW_FINALS):
        f = next(iter(TIER1_DIR.glob(pat)), None)
        if f is None: continue
        per = ps.per_item_priming(f, between_only=True)
        for j, (pl, tl, mark) in enumerate([("Dutch", "English", "D→E"),
                                            ("English", "Dutch", "E→D")]):
            sub = per[(per.PrimeLanguage == pl) & (per.TargetLanguage == tl)]
            if not len(sub): continue
            xs.append(i + (j - 0.5) * 0.4)
            ys.append(float(sub["p_n"].mean()))
            cs.append(color if j == 0 else "white")
            lbls.append(f"{label} {mark}")
    bars = ax.bar(xs, ys, width=0.38, color=cs,
                  edgecolor=[PAL["B1"] if c == "white" else c for c in cs],
                  linewidth=1.0)
    ax.axhline(0.5, color="#999", lw=0.5, ls="--", zorder=1)
    ax.set_xticks([i for i in range(len(_BEETLE_FW_FINALS))])
    ax.set_xticklabels([l for l, *_ in _BEETLE_FW_FINALS])
    ax.set_ylim(0.45, 0.75)
    ax.set_ylabel(r"normalised $P_N$ of matched alternative")
    ax.set_xlabel("Beetle FineWeb-24B curriculum")
    ax.set_title("Arnett (2025) $P_N$ — robustness check (Bernolet, between-lang)")
    legend = [Patch(facecolor="#888", edgecolor="#888", label="Dutch→English (D→E)"),
              Patch(facecolor="white", edgecolor="#888", label="English→Dutch (E→D)")]
    ax.legend(handles=legend, frameon=False, loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    save(fig, out_dir, "fig_pn_robustness_bernolet")
    return "fig_pn_robustness_bernolet"


def fig_pn_trajectory_bernolet(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Trajectory of mean P_N (Bernolet, between-lang) for 4 Beetle curricula.
    Mirrors fig_main_trajectory_bernolet but in Arnett's metric."""
    ps = _import_priming_stats()
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for label, prefix, color in _BEETLE_FW_TRAJ:
        tsvs = sorted(TIER2_DIR.glob(f"{prefix}__step-*.tsv"))
        if not tsvs: continue
        rows = []
        for t in tsvs:
            m = re.search(r"__step-(\d+)\.tsv$", t.name)
            if not m: continue
            ck = int(m.group(1))
            try:
                per = ps.per_item_priming(t, between_only=True)
            except Exception:
                continue
            if not len(per): continue
            rows.append((ck, float(per["p_n"].mean())))
        if not rows: continue
        rows.sort()
        ck_arr = np.array([r[0] for r in rows])
        pn_arr = np.array([r[1] for r in rows])
        x = ck_arr / ck_arr.max()
        y = pd.Series(pn_arr).rolling(7, min_periods=1, center=True).mean()
        ax.plot(x, pn_arr, color=color, lw=0.4, alpha=0.25, zorder=2)
        ax.plot(x, y, color=color, lw=1.4, label=f"Beetle {label}", zorder=3)
    ax.axhline(0.5, color="#999", lw=0.5, ls=":")
    ax.set_xlim(0, 1.0); ax.set_ylim(0.40, 0.80)
    ax.set_xlabel("fraction of training")
    ax.set_ylabel(r"mean $P_N$ (Bernolet, between-lang)")
    ax.set_title(r"Trajectory in Arnett's $P_N$ metric — same shape as surprisal-diff")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, lw=0.4)
    save(fig, out_dir, "fig_pn_trajectory_bernolet")
    return "fig_pn_trajectory_bernolet"


def fig_within_vs_cross_bernolet(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Per Beetle FineWeb final-ckpt curriculum, four bars: within-Eng,
    within-Nld, cross-DE, cross-ED priming. Sharpens the claim that
    bilingual cross-lingual priming is real but smaller than monolingual
    priming."""
    ps = _import_priming_stats()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    width = 0.18
    cells = [("within-Eng", "Eng", "Eng", "#7f7f7f"),
             ("within-Nld", "Nld", "Nld", "#bdbdbd"),
             ("cross D→E", "Dutch", "English", PAL["B1"]),
             ("cross E→D", "English", "Dutch", PAL["B5"])]
    centers = np.arange(len(_BEETLE_FW_FINALS))
    for j, (lbl, pl, tl, c) in enumerate(cells):
        ys = []
        for label, pat, _ in _BEETLE_FW_FINALS:
            f = next(iter(TIER1_DIR.glob(pat)), None)
            if f is None: ys.append(np.nan); continue
            both = ps.per_item_priming(f, between_only=False)
            # Map "Eng"/"Nld" → actual language strings
            pl_full = "English" if pl == "Eng" else ("Dutch" if pl == "Nld" else pl)
            tl_full = "English" if tl == "Eng" else ("Dutch" if tl == "Nld" else tl)
            sub = both[(both.PrimeLanguage == pl_full) & (both.TargetLanguage == tl_full)]
            ys.append(float(sub["priming"].mean()) if len(sub) else np.nan)
        ax.bar(centers + (j - 1.5) * width, ys, width=width,
               color=c, edgecolor="#222", linewidth=0.4, label=lbl)
    ax.axhline(0, color="#999", lw=0.5)
    ax.set_xticks(centers)
    ax.set_xticklabels([l for l, *_ in _BEETLE_FW_FINALS])
    ax.set_ylabel("priming effect (nats)")
    ax.set_xlabel("Beetle FineWeb-24B curriculum")
    ax.set_title("Within-language vs cross-language priming on Bernolet")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    save(fig, out_dir, "fig_within_vs_cross_bernolet")
    return "fig_within_vs_cross_bernolet"


def fig_emergence_timeline(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Developmental arc per Beetle curriculum on a fraction-of-training axis.

    For each curriculum we mark three points joined by a thin line:
      • first-emergence ckpt  (priming first robustly > 0)
      • peak ckpt             (max mean priming over trajectory)
      • final ckpt            (1.0)

    Marker size encodes mean priming at each point; colour encodes
    curriculum. Replaces the earlier L2-onset-arrow design — in these
    curricula L2 is mixed in from step 0, so an "L2 onset" reference
    line was the wrong frame for the data.
    """
    g_csv = STATS_DIR / "g_emergence.csv"
    c_csv = STATS_DIR / "c_peak_vs_final.csv"
    if not g_csv.exists() or not c_csv.exists():
        logger.warning("emergence/peak CSV missing — run priming_stats.py first")
        return None
    g = pd.read_csv(g_csv).set_index("curriculum")
    c = pd.read_csv(c_csv).set_index("curriculum")

    cur_colors = {"B1": PAL["B1"], "B2": PAL["B2"], "B3": PAL["B3"], "B5": PAL["B5"]}
    order = ["B1", "B2", "B3", "B5"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ys = np.arange(len(order))[::-1]

    # Marker-size scaling: priming nats → point area
    def msize(p):  # mean priming → marker area
        return 30 + 110 * max(p, 0.0)

    for y, cur in zip(ys, order):
        if cur not in g.index or cur not in c.index:
            continue
        col = cur_colors[cur]
        max_ck = float(g.loc[cur, "max_ckpt"])
        emerge_frac = float(g.loc[cur, "first_sig_frac_training"]) if pd.notna(
            g.loc[cur, "first_sig_frac_training"]) else None
        emerge_val = float(g.loc[cur, "first_sig_mean_priming"])
        peak_frac  = float(c.loc[cur, "peak_ckpt"])  / max_ck
        peak_val   = float(c.loc[cur, "peak_priming"])
        final_frac = float(c.loc[cur, "final_ckpt"]) / max_ck
        final_val  = float(c.loc[cur, "final_priming"])
        decay      = float(c.loc[cur, "decay_pct"])

        # Connecting line spanning emergence → peak → final
        xs = ([emerge_frac, peak_frac, final_frac]
              if emerge_frac is not None else [peak_frac, final_frac])
        ax.plot(xs, [y] * len(xs), color=col, lw=1.4, alpha=0.6, zorder=2)

        if emerge_frac is not None:
            ax.scatter(emerge_frac, y, s=msize(emerge_val), color="white",
                       edgecolor=col, linewidth=1.4, zorder=4)
            ax.text(emerge_frac, y + 0.30, f"{emerge_frac:.0%}\nΔ={emerge_val:+.2f}",
                    ha="center", va="bottom", fontsize=6.5, color=col)
        ax.scatter(peak_frac, y, s=msize(peak_val), color=col,
                   edgecolor="white", linewidth=1.0, zorder=5)
        ax.text(peak_frac, y + 0.30, f"peak {peak_frac:.0%}\nΔ={peak_val:+.2f}",
                ha="center", va="bottom", fontsize=6.5, color=col)
        ax.scatter(final_frac, y, s=msize(final_val), color=col, alpha=0.45,
                   edgecolor="white", linewidth=1.0, zorder=5)
        ax.text(final_frac + 0.005, y - 0.32,
                f"final Δ={final_val:+.2f}\n({decay:+.0f}% decay)",
                ha="left", va="top", fontsize=6.5, color=col)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"Beetle {cur}" for cur in order])
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(-0.02, 1.10)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("fraction of training")
    ax.set_title("Developmental arc of cross-lingual priming on Bernolet "
                 "(emergence → peak → final)")

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#444", markersize=8,
               label="first robust emergence (p<0.01, sustained ≥3 ckpts, Δ>0.10)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#444",
               markersize=10, label="peak ckpt (max mean priming)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#444",
               alpha=0.45, markersize=10, label="final ckpt"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7,
              loc="lower right", bbox_to_anchor=(1.0, -0.45))
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    save(fig, out_dir, "fig_emergence_timeline")
    return "fig_emergence_timeline"


def fig_blimp_priming_overlay(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Per-curriculum dual-axis: BLiMP-eng accuracy + between-lang Bernolet
    priming vs fraction of training. Tests whether priming decay tracks
    language-competence improvement (it doesn't — competence keeps rising
    while priming partially decays)."""
    blimp_csv = REPO_ROOT / "results/evals_checkpoints/blimp/blimp_results.csv"
    if not blimp_csv.exists():
        logger.warning("BLiMP per-ckpt CSV missing"); return None
    blimp = pd.read_csv(blimp_csv)
    blimp = blimp[(blimp["benchmark"] == "blimp_eng") & (blimp["branch"].notna())]
    blimp["step"] = pd.to_numeric(
        blimp["branch"].astype(str).str.replace("step-", "", regex=False),
        errors="coerce"
    )
    blimp = blimp.dropna(subset=["step"])

    # Match Beetle priming trajectory (between-lang Bernolet) per curriculum.
    btw = per_model_dataset_btw(df)
    btw = btw[(btw.dataset == "bernolet") & btw.ckpt_n.notna()]

    curricula = [
        ("B1", "balanced-b1-fineweb-nld-eng",         PAL["B1"]),
        ("B2", "simultaneous-b2-fineweb-nld-eng",     PAL["B2"]),
        ("B3", "sequential-33-67-b3-fineweb-nld-eng", PAL["B3"]),
        ("B5", "balanced-b5-fineweb-nld-eng",         PAL["B5"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.8), sharey=True)
    for ax, (label, pat, color) in zip(axes, curricula):
        # Priming
        p = btw[btw.model.str.contains(pat)].sort_values("ckpt_n")
        if len(p):
            xp = p.ckpt_n / p.ckpt_n.max()
            ax.plot(xp, _smooth(p.priming_effect, 7),
                    color=color, lw=1.4, label=f"priming Δ", zorder=3)
            ax.scatter([xp.iloc[-1]], [p.priming_effect.iloc[-1]],
                       s=22, color=color, edgecolor="white", linewidth=0.6)
        ax.axhline(0, color="#999", lw=0.4, ls=":")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("fraction of training")
        ax.set_title(f"Beetle {label}", fontsize=9)
        # BLiMP-eng on twin axis
        ax2 = ax.twinx()
        bsub = blimp[blimp.model.str.contains(pat, case=False)].sort_values("step")
        if len(bsub):
            xb = bsub.step / bsub.step.max()
            ax2.plot(xb, bsub.accuracy.rolling(5, min_periods=1, center=True).mean(),
                     color="#666", lw=1.0, ls="-.", label="BLiMP-Eng acc")
        ax2.set_ylim(0.45, 0.75)
        ax2.tick_params(axis="y", colors="#666", labelsize=7)
        if ax is axes[-1]:
            ax2.set_ylabel("BLiMP-Eng accuracy", color="#666", fontsize=8)
        ax.grid(alpha=0.2, lw=0.4)
    axes[0].set_ylabel("between-lang priming (Bernolet)")
    handles = [Line2D([0], [0], color="#444", lw=1.4, label="priming Δ (left axis)"),
               Line2D([0], [0], color="#666", lw=1.0, ls="-.",
                      label="BLiMP-Eng accuracy (right axis)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Priming decays even as syntactic competence (BLiMP-Eng) rises",
                 y=1.03, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save(fig, out_dir, "fig_blimp_priming_overlay")
    return "fig_blimp_priming_overlay"


def fig_pn_main_humanscale_bernolet(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Arnett-metric variant of fig_main_humanscale_bernolet: scale comparison
    FineWeb-24B vs HumanScale on Bernolet between-lang, but each bar is the
    mean normalised P_N (Arnett 2025) instead of priming-effect nats."""
    ps = _import_priming_stats()
    pairs = [
        ("B1", "bernolet__beetle-fineweb_beetle-bilingual-balanced-b1-fineweb-nld-eng__main.tsv",
               "bernolet__beetle-humanscale_beetle-bilingual-balanced-b1-humanscale-nld-eng__main.tsv"),
        ("B2", "bernolet__beetle-fineweb_beetle-bilingual-l2-50-simultaneous-b2-fineweb-nld-eng__main.tsv",
               "bernolet__beetle-humanscale_beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng__main.tsv"),
        ("B3", "bernolet__beetle-fineweb_beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng__main.tsv",
               "bernolet__beetle-humanscale_beetle-bilingual-l2-50-sequential-33-67-b3-humanscale-nld-eng__main.tsv"),
        ("B5", "bernolet__beetle-fineweb_beetle-bilingual-balanced-b5-fineweb-nld-eng__main.tsv",
               "bernolet__beetle-humanscale_beetle-bilingual-l2-80-late-b5-humanscale-nld-eng__main.tsv"),
    ]

    def _mean_pn(fname: str) -> float:
        f = TIER1_DIR / fname
        if not f.exists():
            return float("nan")
        try:
            per = ps.per_item_priming(f, between_only=True)
        except Exception:
            return float("nan")
        if not len(per) or "p_n" not in per.columns:
            return float("nan")
        return float(per["p_n"].mean())

    rows = [{"cur": cur, "FineWeb-24B": _mean_pn(fw), "HumanScale": _mean_pn(hs)}
            for cur, fw, hs in pairs]
    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(d)); w = 0.38
    ax.bar(x - w/2, d["FineWeb-24B"], width=w, color=PAL["beetle"], label="FineWeb-24B",
           edgecolor="#222", lw=0.4)
    ax.bar(x + w/2, d["HumanScale"],  width=w, color="#fdae61", label="HumanScale",
           edgecolor="#222", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(d.cur)
    ax.set_ylabel(r"$P_N$ (Bernolet, between-lang)")
    ax.axhline(0.5, color="#888", lw=0.6, ls="--")
    ax.set_ylim(0.45, 0.75)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    fig.suptitle("Syntactic Priming Effects in Dutch-English Bilingual Models across Scales",
                 y=0.98, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out_dir, "fig_pn_main_humanscale_bernolet")
    return "fig_pn_main_humanscale_bernolet"


def fig_pn_blimp_priming_overlay(df: pd.DataFrame, out_dir: Path) -> str | None:
    """Arnett-metric variant of fig_blimp_priming_overlay: per-curriculum
    dual-axis with mean P_N (Bernolet, between-lang) on the left and
    BLiMP-Eng accuracy on the right vs fraction of training. Same panel
    layout as the surprisal-diff overlay; tests whether the
    competence-vs-priming dissociation survives the metric switch."""
    ps = _import_priming_stats()
    blimp_csv = REPO_ROOT / "results/evals_checkpoints/blimp/blimp_results.csv"
    if not blimp_csv.exists():
        logger.warning("BLiMP per-ckpt CSV missing"); return None
    blimp = pd.read_csv(blimp_csv)
    blimp = blimp[(blimp["benchmark"] == "blimp_eng") & (blimp["branch"].notna())]
    blimp["step"] = pd.to_numeric(
        blimp["branch"].astype(str).str.replace("step-", "", regex=False),
        errors="coerce"
    )
    blimp = blimp.dropna(subset=["step"])

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.8), sharey=True)
    for ax, (label, prefix, color) in zip(axes, _BEETLE_FW_TRAJ):
        # Mean P_N per checkpoint from tier2 step TSVs
        tsvs = sorted(TIER2_DIR.glob(f"{prefix}__step-*.tsv"))
        rows = []
        for t in tsvs:
            m = re.search(r"__step-(\d+)\.tsv$", t.name)
            if not m: continue
            ck = int(m.group(1))
            try:
                per = ps.per_item_priming(t, between_only=True)
            except Exception:
                continue
            if not len(per) or "p_n" not in per.columns: continue
            rows.append((ck, float(per["p_n"].mean())))
        if rows:
            rows.sort()
            ck_arr = np.array([r[0] for r in rows])
            pn_arr = np.array([r[1] for r in rows])
            xp = ck_arr / ck_arr.max()
            ax.plot(xp, pn_arr, color=color, lw=0.4, alpha=0.25, zorder=2)
            ax.plot(xp, pd.Series(pn_arr).rolling(7, min_periods=1, center=True).mean(),
                    color=color, lw=1.4, label=r"mean $P_N$", zorder=3)
            ax.scatter([xp[-1]], [pn_arr[-1]], s=22, color=color,
                       edgecolor="white", linewidth=0.6, zorder=4)
        ax.axhline(0.5, color="#999", lw=0.4, ls=":")
        ax.set_xlim(0, 1.0); ax.set_ylim(0.40, 0.80)
        ax.set_xlabel("fraction of training")
        ax.set_title(f"Beetle {label}", fontsize=9)
        # BLiMP-eng on twin axis (matches fig_blimp_priming_overlay)
        ax2 = ax.twinx()
        # Match tier2 prefix back to a substring usable against blimp.model
        # (mirror of pat used in fig_blimp_priming_overlay).
        pat = prefix.split("__")[-1].split("beetle-bilingual-")[-1]
        bsub = blimp[blimp.model.str.contains(pat, case=False)].sort_values("step")
        if len(bsub):
            xb = bsub.step / bsub.step.max()
            ax2.plot(xb, bsub.accuracy.rolling(5, min_periods=1, center=True).mean(),
                     color="#666", lw=1.0, ls="-.", label="BLiMP-Eng acc")
        ax2.set_ylim(0.45, 0.75)
        ax2.tick_params(axis="y", colors="#666", labelsize=7)
        if ax is axes[-1]:
            ax2.set_ylabel("BLiMP-Eng accuracy", color="#666", fontsize=8)
        ax.grid(alpha=0.2, lw=0.4)
    axes[0].set_ylabel(r"mean $P_N$ (Bernolet, between-lang)")
    handles = [Line2D([0], [0], color="#444", lw=1.4, label=r"mean $P_N$ (left axis)"),
               Line2D([0], [0], color="#666", lw=1.0, ls="-.",
                      label="BLiMP-Eng accuracy (right axis)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Syntactic Priming Effects across Beetle Curricula",
                 y=1.03, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save(fig, out_dir, "fig_pn_blimp_priming_overlay")
    return "fig_pn_blimp_priming_overlay"


FIGS = [
    fig1_trajectory_headline,
    fig2_trajectory_per_curriculum,
    fig3_finalckpt_bars,
    fig4_curriculum_coverage,
    fig5_episodic_memory,
    fig6_direction_asymmetry_bernolet,
    fig7_direction_asymmetry_full,
    fig8_humanscale_consistency,
    fig9_bgpt_peak_decay,
    fig10_cl_variant_caveat,
    # Bernolet-only main-paper variants:
    fig_main_trajectory_bernolet,
    fig_main_per_curriculum_bernolet,
    fig_main_finalckpt_bernolet,
    fig_main_humanscale_bernolet,
    fig_main_peak_decay_bernolet,
    # Reviewer-proofing extensions (Arnett 2025 inheritance):
    fig_pn_robustness_bernolet,
    fig_pn_trajectory_bernolet,
    fig_within_vs_cross_bernolet,
    fig_emergence_timeline,
    fig_blimp_priming_overlay,
    fig_pn_main_humanscale_bernolet,
    fig_pn_blimp_priming_overlay,
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",
                    default=str(REPO_ROOT / "results/evals/priming/tier2_headline_trajectory/figures"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir).resolve()

    df = load_data()
    logger.info("loaded %d rows from %d tiers", len(df), df.tier.nunique())

    written = []
    for fn in FIGS:
        try:
            res = fn(df, out_dir)
            if res: written.append(res)
        except Exception as e:
            logger.exception("  %s failed: %s", fn.__name__, e)

    meta = {
        "n_models_total":    int(df["model"].nunique()),
        "n_beetle_models":   int(df[df.model_source=="beetle"]["model"].nunique()),
        "n_bgpt_models":     int(df[df.model_source=="bgpt"]["model"].nunique()),
        "datasets":          sorted(df.dataset.unique()),
        "figures_written":   written,
        "tiers_combined":    list(TIERS),
        "between_language_only": True,
    }
    (out_dir / "fig_summary_metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info("done: %d figure(s)", len(written))
    return 0 if written else 5


if __name__ == "__main__":
    raise SystemExit(main())
