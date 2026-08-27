"""blimp_scale_curriculum.py — EMNLP main-body figure: BLiMP × scale × curriculum.

Two subplots per BLiMP metric (left/right), encoded as the role of the
benchmark's evaluation language inside each evaluated bilingual model:

    Left  panel — "L1" : the test language is the model's first-acquired language.
                  ⇒ measures *native-language preservation* under L2 introduction.
    Right panel — "L2" : the test language is the model's second-acquired language.
                  ⇒ measures *L2 acquisition / transfer*.

X-axis is the per-org training token budget (log scale: 100M, 2B, 24B). Each
point is one model's final-checkpoint accuracy on a single BLiMP suite. Lines
group models with identical (curriculum, partner-language) over scale.

Inputs
------
``results/evals/blimp/blimp_results.csv`` with columns
    benchmark, model, branch, model_family, eval_language, iso_code,
    accuracy, n_correct, n_total

Beetle slug parsing for the new -100M / -2B / -24B / -Variants orgs is
handled here directly; the canonical ``analyze.model_naming.parse_repo``
regex requires ``dataset ∈ {humanscale, fineweb}`` and would silently
drop the new orgs.

Outputs
-------
``results/figures/main/fig_blimp_l1l2_<task>.{pdf,png}``      — main body (per task)
``results/figures/appendix/fig_blimp_l1l2_grid.{pdf,png}``    — 4×2 appendix grid
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze.plot_style import (
    PALETTE_ACL,
    apply_acl_style,
    fig_double,
    savefig_pub,
    style_axes,
)

log = logging.getLogger("blimp_scale_curriculum")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BLIMP_CSV = RESULTS / "evals" / "blimp" / "blimp_results.csv"
MAIN_DIR = RESULTS / "figures" / "main"
APP_DIR = RESULTS / "figures" / "appendix"

# Token budget per org. "fineweb" without suffix = the original 24B sweep.
ORG_TOKENS: dict[str, float] = {
    "Beetle-HumanScale": 1.0e8,
    "Beetle-FineWeb-100M": 1.0e8,
    "Beetle-FineWeb-2B": 2.0e9,
    "Beetle-nld-eng-Variants": 2.0e9,  # built atop the FineWeb-2B base
    "Beetle-FineWeb": 2.4e10,
    "Beetle-FineWeb2-24B": 2.4e10,
}

# Display name for the scale axis ticks.
ORG_SCALE_LABEL: dict[str, str] = {
    "Beetle-HumanScale": "100M·HS",
    "Beetle-FineWeb-100M": "100M·FW",
    "Beetle-FineWeb-2B": "2B",
    "Beetle-nld-eng-Variants": "2B·var",
    "Beetle-FineWeb": "24B",
    "Beetle-FineWeb2-24B": "24B·v2",
}

# Curriculum codes we plot. B1 = balanced-50:50 baseline; B2/B3 are the
# "simultaneous vs sequential" contrast that mirrors the F.13 framing.
CURRICULA: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5")
CURRICULUM_LABEL = {
    "B1": "balanced 50:50",
    "B2": "simultaneous",
    "B3": "sequential 33:67",
    "B4": "classroom 20%",
    "B5": "late 80%",
}

TASK_LABEL = {
    "blimp_eng": "BLiMP (English)",
    "blimp_nl":  "BLiMP-NL (Dutch)",
    "zhoblimp":  "Zho-BLiMP (Chinese)",
    "multiblimp":"MultiBLiMP (typological)",
}
TASK_ISO = {
    "blimp_eng": "eng",
    "blimp_nl":  "nld",
    "zhoblimp":  "zho",
    "multiblimp": None,  # iso varies per row; handled by eval_language column
}


# ────────────────────────────────────────────────────────────────────────────
# Slug parsing — lenient; covers the five new orgs the canonical regex misses
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Slug:
    repo: str
    org: str
    condition: str  # B1..B5 or "mono" or "unknown"
    l1: Optional[str]
    l2: Optional[str]
    variant: Optional[str]  # ablation tag, e.g. "ewc", "pre-pretrain", None

# All slug forms of interest end in "-{l1}-{l2}" or have an extra ablation
# suffix after the language pair. The "(?:-(?P<tag>...))?$" captures the
# ablation tail (e.g. "-ewc", "-pre-pretrain", "-arch-bgpt").
_PAIR = r"(?P<l1>[a-z]{3})-(?P<l2>[a-z]{3})"
_TAIL = r"(?:-(?P<tag>[a-z0-9-]+))?$"

_SLUG_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"^beetle-bilingual-balanced-b1-[a-z0-9-]*?-{_PAIR}{_TAIL}"),                             "B1"),
    (re.compile(rf"^beetle-bilingual-l2-50-simultaneous-b2-[a-z0-9-]*?-{_PAIR}{_TAIL}"),                   "B2"),
    (re.compile(rf"^beetle-bilingual-l2-50-sequential-(?:\d+-\d+-)?b3-[a-z0-9-]*?-{_PAIR}{_TAIL}"),        "B3"),
    (re.compile(rf"^beetle-bilingual-l2-50-classroom-20-b4-[a-z0-9-]*?-{_PAIR}{_TAIL}"),                   "B4"),
    (re.compile(rf"^beetle-bilingual-l2-80-late-b5-[a-z0-9-]*?-{_PAIR}{_TAIL}"),                           "B5"),
    (re.compile(rf"^beetle-bilingual-balanced-b(?P<n>[1-5])-[a-z0-9-]*?-{_PAIR}{_TAIL}"),                  "B?"),
    (re.compile(r"^beetle-monolingual-[a-z0-9-]+-(?P<l1>[a-z]{3})$"),                                       "mono"),
]


def parse_slug(repo: str) -> Optional[Slug]:
    if "/" not in repo:
        return None
    org, name = repo.split("/", 1)
    name = name.lower()
    for pat, cond in _SLUG_PATTERNS:
        m = pat.match(name)
        if not m:
            continue
        gd = m.groupdict()
        if cond == "mono":
            return Slug(repo, org, "mono", gd["l1"], None, None)
        c = cond
        if c == "B?":
            c = f"B{gd['n']}"
        return Slug(repo, org, c, gd.get("l1"), gd.get("l2"),
                    (gd.get("tag") or None))
    return None


# ────────────────────────────────────────────────────────────────────────────
# Data wrangling
# ────────────────────────────────────────────────────────────────────────────

def _read_blimp_csv(csv_path: Path) -> pd.DataFrame:
    """The repo's BLiMP CSV mixes a 7-column legacy schema (no ``iso_code``)
    with the current 8-column schema. Read both, normalise to the union."""
    rows_8: list[list[str]] = []
    rows_7: list[list[str]] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        next(fh, None)  # header
        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) == 8:
                rows_8.append(parts)
            elif len(parts) == 7:
                rows_7.append(parts)
    cols = ["benchmark", "model", "branch", "model_family",
            "eval_language", "iso_code", "accuracy", "n_correct"]
    # Some 8-col CSVs include n_total too — sniff the dtype of the last col.
    cols_8 = ["benchmark", "model", "model_family", "eval_language",
              "iso_code", "accuracy", "n_correct", "n_total"]
    df8 = pd.DataFrame(rows_8, columns=cols_8) if rows_8 else pd.DataFrame()
    cols_7 = ["benchmark", "model", "model_family", "eval_language",
              "accuracy", "n_correct", "n_total"]
    df7 = pd.DataFrame(rows_7, columns=cols_7) if rows_7 else pd.DataFrame()
    if not df7.empty:
        df7["iso_code"] = ""
    df = pd.concat([df8, df7], ignore_index=True, sort=False)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["n_correct"] = pd.to_numeric(df["n_correct"], errors="coerce")
    df["n_total"] = pd.to_numeric(df["n_total"], errors="coerce")
    df = df.dropna(subset=["accuracy"]).reset_index(drop=True)
    if "branch" not in df.columns:
        df["branch"] = ""
    return df


def load_blimp(csv_path: Path = BLIMP_CSV) -> pd.DataFrame:
    df = _read_blimp_csv(csv_path)
    df["org"] = df["model"].str.split("/").str[0]
    df["slug"] = df["model"].apply(parse_slug)
    df = df[df["slug"].notna()].copy()
    df["condition"] = df["slug"].apply(lambda s: s.condition)
    df["l1"] = df["slug"].apply(lambda s: s.l1)
    df["l2"] = df["slug"].apply(lambda s: s.l2)
    df["variant"] = df["slug"].apply(lambda s: s.variant)
    df["tokens"] = df["org"].map(ORG_TOKENS)
    df = df[df["tokens"].notna()].copy()
    # Final checkpoint only: drop rows whose branch is a step-N intermediate
    # if a "main" / no-branch row exists for the same (model, benchmark).
    if "branch" in df.columns:
        df["is_final"] = df["branch"].isna() | df["branch"].isin(["", "main"])
        finals = df[df["is_final"]]
        if len(finals) > 0:
            df = finals
    return df


def split_l1_l2(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (test_lang_is_L1, test_lang_is_L2) DataFrames for the given task.

    For multiblimp, the test language varies per row, so the iso comes from
    the existing ``iso_code`` column. For per-language tasks the iso is fixed.
    """
    sub = df[df["benchmark"] == task].copy()
    if task == "multiblimp":
        sub["iso"] = sub["iso_code"].str.lower()
    else:
        sub["iso"] = TASK_ISO[task]
    sub = sub[sub["iso"].notna() & (sub["iso"] != "")]
    is_l1 = sub["l1"] == sub["iso"]
    is_l2 = sub["l2"] == sub["iso"]
    return sub[is_l1].copy(), sub[is_l2].copy()


# ────────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────────

def _plot_panel(ax, panel_df: pd.DataFrame, *, title: str,
                show_ylabel: bool = True, show_legend: bool = False) -> None:
    if panel_df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="#888", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(8e7, 4e10)
        ax.set_ylim(0.45, 1.0)
        style_axes(ax, xlabel="training tokens",
                   ylabel="accuracy" if show_ylabel else None,
                   grid=True)
        ax.set_title(title, fontsize=8)
        return

    # Mean accuracy across (L1, condition, tokens) so the line connects
    # token-budget points within a curriculum/partner combination.
    g = (panel_df.groupby(["condition", "l1", "l2", "tokens"], dropna=False)
                  ["accuracy"].mean().reset_index())

    for cond in CURRICULA:
        cond_df = g[g["condition"] == cond]
        if cond_df.empty:
            continue
        color = PALETTE_ACL.get(cond, "#888")
        # One line per partner-language combo (so deu-eng vs nld-eng don't
        # collapse). All lines for a curriculum share the colour.
        for (l1, l2), trace in cond_df.groupby(["l1", "l2"]):
            trace = trace.sort_values("tokens")
            ax.plot(trace["tokens"], trace["accuracy"],
                    color=color, marker="o", markersize=3.2,
                    linewidth=1.2, alpha=0.85, zorder=3)
        # Scatter unconnected single-scale datapoints too so partner
        # languages with only one scale-cell still appear.
        single = cond_df.groupby(["l1", "l2"]).size()
        for (l1, l2), n in single.items():
            if n == 1:
                row = cond_df[(cond_df["l1"] == l1) & (cond_df["l2"] == l2)]
                ax.scatter(row["tokens"], row["accuracy"],
                           color=color, s=14, zorder=4, edgecolors="white",
                           linewidths=0.4)

    ax.set_xscale("log")
    ax.axhline(0.5, color="#bbb", linestyle=":", linewidth=0.7, zorder=1)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlim(8e7, 4e10)
    style_axes(ax, xlabel="training tokens (log)",
               ylabel="BLiMP accuracy" if show_ylabel else None,
               grid=True)
    ax.set_title(title, fontsize=8.5)


def _legend_handles() -> list:
    return [mlines.Line2D([], [], color=PALETTE_ACL[c], marker="o",
                          markersize=3.5, linewidth=1.2,
                          label=f"{c} — {CURRICULUM_LABEL[c]}")
            for c in CURRICULA if c in PALETTE_ACL]


def render_main_blimp_eng(df: pd.DataFrame, out_dir: Path = MAIN_DIR) -> Path:
    """Main-body figure: 1×2 BLiMP-eng — L1 (left) vs L2 (right)."""
    apply_acl_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = fig_double(height=2.6, ncols=2, sharey=True)
    l1_df, l2_df = split_l1_l2(df, "blimp_eng")
    _plot_panel(axes[0], l1_df,
                title="English as L1 (preservation)", show_ylabel=True)
    _plot_panel(axes[1], l2_df,
                title="English as L2 (acquisition)",  show_ylabel=False)
    fig.legend(handles=_legend_handles(), loc="lower center",
               ncol=len(CURRICULA), bbox_to_anchor=(0.5, -0.04),
               frameon=False, fontsize=7.5)
    fig.suptitle(TASK_LABEL["blimp_eng"], fontsize=9.5, y=1.01)
    fig.tight_layout()
    out = out_dir / "fig_blimp_l1l2_blimp_eng"
    savefig_pub(fig, out)
    plt.close(fig)
    return out.with_suffix(".pdf")


def render_appendix_grid(df: pd.DataFrame, out_dir: Path = APP_DIR) -> Path:
    """Appendix 4×2 grid: rows = BLiMP tasks, cols = {L1-test, L2-test}."""
    apply_acl_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = ["blimp_eng", "blimp_nl", "zhoblimp", "multiblimp"]
    fig, axes = fig_double(height=8.0, ncols=2, nrows=4, sharex=True)
    for i, task in enumerate(tasks):
        l1_df, l2_df = split_l1_l2(df, task)
        _plot_panel(axes[i, 0], l1_df,
                    title=f"{TASK_LABEL[task]} — test lang as L1",
                    show_ylabel=True)
        _plot_panel(axes[i, 1], l2_df,
                    title=f"{TASK_LABEL[task]} — test lang as L2",
                    show_ylabel=False)
    fig.legend(handles=_legend_handles(), loc="lower center",
               ncol=len(CURRICULA), bbox_to_anchor=(0.5, -0.015),
               frameon=False, fontsize=7.5)
    fig.tight_layout()
    out = out_dir / "fig_blimp_l1l2_grid"
    savefig_pub(fig, out)
    plt.close(fig)
    return out.with_suffix(".pdf")


def render_all(csv_path: Path = BLIMP_CSV) -> dict[str, Path]:
    df = load_blimp(csv_path)
    out: dict[str, Path] = {}
    if df.empty:
        log.warning("no parseable BLiMP rows; skipping figure")
        return out
    out["main_blimp_eng"] = render_main_blimp_eng(df)
    out["appendix_grid"] = render_appendix_grid(df)
    return out


def main() -> None:  # CLI
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=BLIMP_CSV,
                   help="Path to results/evals/blimp/blimp_results.csv")
    args = p.parse_args()
    paths = render_all(args.csv)
    for k, v in paths.items():
        print(f"[blimp_scale_curriculum] wrote {k}: {v}")


if __name__ == "__main__":
    main()
