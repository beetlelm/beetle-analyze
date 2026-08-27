"""blimp_checkpoint_trajectory.py — F.13 analogue redesigned per user request.

Single figure (the only main artefact): a curriculum × benchmark GRID with
*every L1-L2 pair overlaid* in each panel so cross-lingual transfer (or its
absence) is legible at a glance, and so the catastrophic-forgetting "drop"
under B3-sequential / B5-late shows up as a visible dip in the curve.

Layout
------
Rows: curricula B1, B2, B3, B4, B5
Cols: BLiMP-eng / BLiMP-NL / Zho-BLiMP / MultiBLiMP-NL / MultiBLiMP-DE
      (per-language MultiBLiMP slices, as requested)

Each panel
----------
* X-axis: training step on a symlog scale.
* Y-axis: BLiMP accuracy (0.45–0.95).
* One curve per (l1, l2, dataset) cell; colour encodes the L1–L2 pair,
  linestyle encodes the dataset scale (HumanScale 100 M = solid,
  FineWeb 24 B = dashed). B-GPT is intentionally excluded so the
  cross-Beetle-scale comparison stays clean.

Inputs
------
``results/evals_checkpoints/blimp/blimp_results.csv``

Outputs
-------
``results/figures/appendix/fig_blimp_traj_grid.{pdf,png}``  — the only artefact.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

from analyze.plot_style import apply_acl_style, savefig_pub

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "evals_checkpoints" / "blimp" / "blimp_results.csv"
APP_DIR = ROOT / "results" / "figures" / "appendix"
log = logging.getLogger("blimp_checkpoint_trajectory")

# Columns we plot (benchmark, iso_code) — order is left-to-right.
COLS: list[tuple[str, str, str]] = [
    ("blimp_eng",  "eng", "BLiMP (English L2)"),
    ("blimp_nl",   "nld", "BLiMP-NL (Dutch)"),
    ("zhoblimp",   "zho", "Zho-BLiMP (Chinese)"),
    ("multiblimp", "nld", "MultiBLiMP / Dutch"),
    ("multiblimp", "deu", "MultiBLiMP / German"),
]

ROWS: list[str] = ["B1", "B2", "B3", "B4", "B5"]
ROW_LABEL = {
    "B1": "B1 — balanced 50:50",
    "B2": "B2 — simultaneous",
    "B3": "B3 — sequential 33:67",
    "B4": "B4 — classroom 20%",
    "B5": "B5 — late onset 80%",
}

# Colour by L1-L2 pair so the same colour in different rows/cols is the
# same model family — the eye can chase a curriculum manipulation across
# the grid by following one colour.
PAIR_COLOR: dict[tuple[str, str], str] = {
    ("nld", "eng"): "#1b9e77",
    ("deu", "eng"): "#d95f02",
    ("zho", "eng"): "#b2182b",
    ("eng", "nld"): "#2166ac",
    ("eng", "deu"): "#a6761d",
}
PAIR_LABEL = {
    ("nld", "eng"): "nld-eng",
    ("deu", "eng"): "deu-eng",
    ("zho", "eng"): "zho-eng",
    ("eng", "nld"): "eng-nld",
    ("eng", "deu"): "eng-deu",
}

DATASET_LS = {
    "humanscale": "-",   # solid — 100 M
    "fineweb":    "--",  # dashed — 24 B
}
DATASET_LABEL = {"humanscale": "Beetle-HS (100 M)", "fineweb": "Beetle-FW (24 B)"}

_STEP_RE = re.compile(r"^step-(\d+)$")
_NEW_SLUG = re.compile(
    r"^beetle-bilingual-(?P<desc>[a-z0-9-]+?)-(?:b1|b2|b3|b4|b5)-"
    r"(?P<dataset>humanscale|fineweb)-(?P<l1>[a-z]{3})-(?P<l2>[a-z]{3})$"
)


def _parse(model: str) -> dict | None:
    if "/" not in model:
        return None
    name = model.split("/", 1)[1].lower()
    m = _NEW_SLUG.match(name)
    if not m:
        return None
    desc = m.group("desc")
    if "simultaneous" in desc:
        cond = "B2"
    elif "sequential" in desc:
        cond = "B3"
    elif "classroom-20" in desc:
        cond = "B4"
    elif "late" in desc:
        cond = "B5"
    elif "balanced" in desc:
        cond = "B1"
    else:
        return None
    return {
        "condition": cond,
        "dataset":   m.group("dataset"),
        "l1":        m.group("l1"),
        "l2":        m.group("l2"),
    }


def _step(branch: str) -> int | None:
    if not isinstance(branch, str):
        return None
    m = _STEP_RE.match(branch)
    return int(m.group(1)) if m else None


def load_traj(csv: Path = CSV) -> pd.DataFrame:
    df = pd.read_csv(csv, low_memory=False)
    parsed = df["model"].apply(_parse)
    df["meta"] = parsed
    df = df[df["meta"].notna()].copy()
    df["condition"] = df["meta"].apply(lambda d: d["condition"])
    df["dataset"]   = df["meta"].apply(lambda d: d["dataset"])
    df["l1"]        = df["meta"].apply(lambda d: d["l1"])
    df["l2"]        = df["meta"].apply(lambda d: d["l2"])
    df["step"]      = df["branch"].apply(_step)
    df = df[df["step"].notna()].copy()
    df["step"]     = df["step"].astype(int)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df = df.dropna(subset=["accuracy"])
    return df


def _plot_panel(ax, sub: pd.DataFrame, *, title: str, ylim=(0.45, 0.95)) -> None:
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="#aaa", fontsize=8)
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        return

    for (l1, l2, dataset), g in sub.groupby(["l1", "l2", "dataset"]):
        g = g.sort_values("step")
        if len(g) < 3:
            continue
        color = PAIR_COLOR.get((l1, l2))
        if color is None:
            continue
        ls = DATASET_LS[dataset]
        ax.plot(g["step"].clip(lower=1), g["accuracy"],
                color=color, linestyle=ls, linewidth=1.0, alpha=0.9,
                marker="o", markersize=2.4, zorder=3)

    ax.set_xscale("log")
    ax.set_xlim(1, 1.2e5)
    ax.set_ylim(*ylim)
    ax.axhline(0.5, color="#bbb", linestyle=":", linewidth=0.5, zorder=1)
    ax.grid(True, axis="y", linestyle=":", color="#dcdcdc", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(labelsize=7)


def _legend_handles() -> list:
    h = [mlines.Line2D([], [], color=c, linewidth=1.4, marker="o",
                       markersize=3, label=PAIR_LABEL[k])
         for k, c in PAIR_COLOR.items()]
    h.append(mlines.Line2D([], [], color="#444", linestyle="-", linewidth=1.4,
                           label=DATASET_LABEL["humanscale"]))
    h.append(mlines.Line2D([], [], color="#444", linestyle="--", linewidth=1.4,
                           label=DATASET_LABEL["fineweb"]))
    return h


def render_grid(df: pd.DataFrame, output_dir: Path = APP_DIR,
                fname: str = "fig_blimp_traj_grid",
                title_suffix: str = "") -> Path:
    """5 (curriculum) × 5 (benchmark) trajectory grid with overlaid L1-L2 pairs.

    Pass ``fname`` to disambiguate the HumanScale-only variant from the full
    HumanScale + FineWeb grid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    nrows, ncols = len(ROWS), len(COLS)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.45 * ncols + 0.6, 1.95 * nrows + 0.6),
                             sharex=True, sharey=True)
    for r, cond in enumerate(ROWS):
        for c, (bench, iso, title) in enumerate(COLS):
            ax = axes[r, c]
            sub = df[(df["condition"] == cond) &
                     (df["benchmark"] == bench) &
                     (df["iso_code"] == iso)]
            _plot_panel(ax, sub,
                        title=title if r == 0 else "")
            if c == 0:
                ax.set_ylabel(ROW_LABEL[cond], fontsize=8.5)
            if r == nrows - 1:
                ax.set_xlabel("training step (log)", fontsize=8)
    fig.suptitle(
        "BLiMP accuracy across training — curriculum (rows) × benchmark (cols)\n"
        "Solid = Beetle HumanScale (100 M);  dashed = Beetle FineWeb (24 B). "
        "Drops after L2 introduction are visible on B3 / B5 rows."
        + (f"\n{title_suffix}" if title_suffix else ""),
        fontsize=10, y=1.02,
    )
    fig.legend(handles=_legend_handles(), loc="lower center",
               bbox_to_anchor=(0.5, -0.025),
               ncol=min(len(_legend_handles()), 7),
               frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.025, 1, 0.99])
    out = output_dir / fname
    savefig_pub(fig, out)
    plt.close(fig)
    return out.with_suffix(".pdf")


def render_all() -> dict[str, Path]:
    apply_acl_style()
    df = load_traj()
    if df.empty:
        log.warning("no parseable checkpoint rows")
        return {}
    out = {"grid": render_grid(df)}
    # HumanScale-only variant (P2): same layout, FineWeb rows omitted.
    df_hs = df[df["dataset"] == "humanscale"]
    if not df_hs.empty:
        out["grid_humanscale"] = render_grid(
            df_hs,
            fname="fig_blimp_traj_grid_humanscale",
            title_suffix="HumanScale (100 M) only",
        )
    return out


def main() -> None:
    paths = render_all()
    for k, v in paths.items():
        print(f"[blimp_checkpoint_trajectory] {k} -> {v}")


if __name__ == "__main__":
    main()
