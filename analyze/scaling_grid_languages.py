"""scaling_grid_languages.py — 4-eval × 3-language token-budget grid.

Companion to ``scaling_grid_token_budget.py``. Same four eval suites, but laid
out as a 3-row × 4-column grid where rows are language pairs and columns are
evaluation suites:

                BLiSS      FLORES     CEFR-grad   BLiMP
    nld-eng     [ a ]      [ b ]      [ c ]       [ d ]
    deu-eng     [ e ]      [ f ]      [ g ]       [ h ]
    zho-eng     [ i ]      [ j ]      [ k ]       [ l ]

Many cells are empty (FLORES + CEFR are nld-only at multiple budgets, and
zho-eng has only HumanScale BLiSS). Empty cells render an "n/a" annotation
so the figure honestly documents the language-coverage gap.

Inputs / outputs match ``scaling_grid_token_budget.py``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze.plot_style import (
    apply_acl_style,
    fig_double,
    savefig_pub,
    style_axes,
)

_log = logging.getLogger("scaling_grid_languages")
ROOT = Path(__file__).resolve().parents[1]

LANGS    = [("nld-eng", "Dutch"),
            ("deu-eng", "German"),
            ("zho-eng", "Chinese")]
CURRICULA     = ["B1", "B2", "B3", "B4", "B5"]
TOKEN_BUDGETS = [0.1, 2.0, 24.0]
BUDGET_COLOURS = {0.1: "#a6cee3", 2.0: "#1f78b4", 24.0: "#08306b"}
BUDGET_LABELS  = {0.1: "100M",   2.0: "2B",      24.0: "24B"}


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders parameterised by language pair.
# ─────────────────────────────────────────────────────────────────────────────

def _read_zhoblimp_rows() -> pd.DataFrame:
    """Robustly load zhoblimp rows from the heterogeneous combined CSV."""
    import csv
    rows = []
    with open(ROOT / "results/evals/blimp/blimp_results.csv") as f:
        for r in csv.reader(f):
            if len(r) >= 7 and r[0] == "zhoblimp":
                # Schema may be 7 cols (model_family present) or 8 cols
                # (model_family + iso_code). The accuracy is always the
                # 3rd-from-last column.
                acc = r[-3]
                rows.append({"benchmark": r[0], "model": r[1], "accuracy": acc})
    return pd.DataFrame(rows)


def _load_bliss(pair: str, target_lang: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "paper/emnlp/main_body/data/F7_bliss_curriculum_x_scale.csv")
    df = df[(df["pair"] == pair) & (df["target_lang"] == target_lang)].copy()
    scale_to_b = {"HumanScale": 0.1, "FineWeb": 24.0}
    df["tokens_b"] = df["scale"].map(scale_to_b)
    df = df.rename(columns={"condition": "curriculum", "rp0": "y"})
    return df[["curriculum", "tokens_b", "y"]].dropna()


def _load_flores(pair: str) -> pd.DataFrame:
    if pair != "nld-eng":
        return pd.DataFrame(columns=["curriculum", "tokens_b", "y"])
    df = pd.read_csv(ROOT / "paper/emnlp/main_body/data/F10_flores_scale_curve.csv")
    df = df.rename(columns={"scale_b": "tokens_b", "nld_bpt": "y"})
    return df[["curriculum", "tokens_b", "y"]]


def _load_cefr(pair: str) -> pd.DataFrame:
    if pair != "nld-eng":
        return pd.DataFrame(columns=["curriculum", "tokens_b", "y"])
    df = pd.read_csv(ROOT / "plots/headline_summary_table.csv")
    df = df[df["lang_pair"] == "nld-eng"].copy()
    family_to_b = {"HumanScale": 0.1, "FineWeb-2B": 2.0, "FineWeb-24B": 24.0}
    df["tokens_b"]  = df["family"].map(family_to_b)
    df["curriculum"] = df["curriculum"].str.upper()
    df = df.rename(columns={"cefr_gradient_bits": "y"})
    df = df.dropna(subset=["tokens_b", "y"])
    return df[df["curriculum"].isin(CURRICULA)][["curriculum", "tokens_b", "y"]]


def _load_blimp(pair: str) -> pd.DataFrame:
    """Picks the appropriate BLiMP suite per language."""
    bench = {"nld-eng": "blimp_nl", "deu-eng": "blimp_eng", "zho-eng": "zhoblimp"}[pair]
    if bench == "blimp_nl":
        df = pd.read_csv(ROOT / "results/blimp/blimp_nl_results.csv")
    elif bench == "blimp_eng":
        df = pd.read_csv(ROOT / "results/blimp/blimp_eng_results.csv")
    else:
        # zhoblimp lives only in the heterogeneous combined CSV; read robustly.
        df = _read_zhoblimp_rows()
    df = df[df["benchmark"] == bench].copy()
    df = df[df["model"].str.contains(pair, regex=False)].copy()
    df = df[~df["model"].str.contains(
        "postfix|alibi|ewc|lamol|safe|pre-pretrain|moe|sas|tilt|lr-boost|gni|align|lang-id|layer-growth|episodic"
    )]
    family_to_b = {
        "Beetle-FineWeb-100M": 0.1,
        "Beetle-FineWeb-2B":   2.0,
        "Beetle-FineWeb":      24.0,
    }
    def parse(model: str):
        org, repo = model.split("/", 1)
        tokens_b = family_to_b.get(org, np.nan)
        for tag, label in [("balanced-b1", "B1"), ("simultaneous-b2", "B2"),
                           ("sequential-33-67-b3", "B3"),
                           ("classroom-20-b4", "B4"), ("late-b5", "B5")]:
            if tag in repo:
                return label, tokens_b
        return None, tokens_b
    df[["curriculum", "tokens_b"]] = df["model"].apply(lambda m: pd.Series(parse(m)))
    df = df.dropna(subset=["curriculum", "tokens_b"])
    df = df.rename(columns={"accuracy": "y"})
    df = df.groupby(["curriculum", "tokens_b"], as_index=False)["y"].mean()
    return df[["curriculum", "tokens_b", "y"]]


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

def _draw(ax, df: pd.DataFrame, *, ylim: tuple[float, float] | None,
          baseline: float | None, higher_is_better: bool = True) -> bool:
    """Draw a single (lang × eval) panel. Returns True iff any data plotted."""
    bar_w = 0.26
    offsets = {0.1: -bar_w, 2.0: 0.0, 24.0: bar_w}
    if ylim is not None:
        ax.set_ylim(*ylim)
    base = baseline if baseline is not None else (ylim[0] if ylim else 0.0)

    x_centres = np.arange(len(CURRICULA))
    any_plotted = False
    for c_idx, curr in enumerate(CURRICULA):
        for tb in TOKEN_BUDGETS:
            row = df[(df["curriculum"] == curr) & (df["tokens_b"] == tb)]
            if row.empty:
                continue
            y = float(row["y"].iloc[0])
            ax.bar(x_centres[c_idx] + offsets[tb], y - base,
                   width=bar_w * 0.9, bottom=base,
                   color=BUDGET_COLOURS[tb], edgecolor="white",
                   linewidth=0.4, zorder=3)
            any_plotted = True

    ax.set_xticks(x_centres)
    ax.set_xticklabels(CURRICULA)
    ax.set_xlim(-0.6, len(CURRICULA) - 0.4)
    if not higher_is_better:
        ax.invert_yaxis()
    return any_plotted


def make_figure(out_main: Path, out_appendix: Path) -> None:
    apply_acl_style()

    eval_specs = [
        ("BLiSS RP@0",            _load_bliss, dict(ylim=(45, 70),
                                                    baseline=45,
                                                    higher_is_better=True)),
        ("FLORES BPT (↓)",        _load_flores, dict(ylim=(8.0, 5.5),
                                                     baseline=8.0,
                                                     higher_is_better=False)),
        ("CEFR gradient (bits)",  _load_cefr,  dict(ylim=(-0.6, 3.5),
                                                    baseline=0.0,
                                                    higher_is_better=True)),
        ("BLiMP accuracy",        _load_blimp, dict(ylim=(0.20, 0.85),
                                                    baseline=0.20,
                                                    higher_is_better=True)),
    ]

    fig, axes = fig_double(height=5.4, nrows=len(LANGS), ncols=len(eval_specs),
                            sharex=False)

    for r, (pair, target) in enumerate(LANGS):
        for c, (label, loader, kw) in enumerate(eval_specs):
            ax = axes[r][c]
            if loader is _load_bliss:
                df = loader(pair, target)
            else:
                df = loader(pair)
            plotted = _draw(ax, df, **kw)
            style_axes(
                ax,
                ylabel=label if c == 0 else None,
                title=f"{pair} · {label}" if r == 0 else None,
                grid=True,
            )
            if r == 0:
                ax.set_title(label, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{pair}\n\n{label}", fontsize=7)
            else:
                ax.set_ylabel("")
            if r == len(LANGS) - 1:
                ax.set_xlabel("Curriculum")
            if not plotted:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10,
                        color="#bbb", style="italic")
                ax.set_xticklabels([])

    handles = [
        mlines.Line2D([], [], color=BUDGET_COLOURS[tb], marker="s",
                      markersize=7, linestyle="None",
                      label=f"FineWeb-{BUDGET_LABELS[tb]} tokens")
        for tb in TOKEN_BUDGETS
    ]
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles), frameon=False, fontsize=7,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    for out in (out_main, out_appendix):
        out.parent.mkdir(parents=True, exist_ok=True)
        savefig_pub(fig, out)
    plt.close(fig)
    _log.info("wrote %s and %s", out_main, out_appendix)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_main = ROOT / "results/figures/main/fig_token_budget_scaling_languages"
    out_appendix = ROOT / "results/figures/appendix/fig_token_budget_scaling_languages"
    make_figure(out_main, out_appendix)


if __name__ == "__main__":
    main()
