"""blimp_per_model_grids.py — EMNLP-appendix per-model trajectory grids.

Each panel is one Beetle model. Two lines per panel:
  L1 line (solid)  : BLiMP suite covering the model's L1 language.
  L2 line (dashed) : BLiMP suite covering the model's L2 language.
The two lines on the same axes make catastrophic-forgetting drops on the
L1 side legible against L2 acquisition on the same training-step clock.

Suite mapping (uses ``benchmark`` × ``iso_code`` as the unique suite key):
  eng → blimp_eng / eng
  nld → blimp_nl  / nld
  deu → multiblimp / deu
  zho → zhoblimp  / zho

We produce four grouped grids:

  A. Beetle-HumanScale curriculum × pair
  B. Beetle-FineWeb (24 B) curriculum × pair
  C. HumanScale B2 simultaneous typology — nld vs deu vs zho L1
  D. HumanScale B2 simultaneous CL ablations — baseline vs EWC vs LAMOL

Inputs
------
``results/evals_checkpoints/blimp/blimp_results.csv``

Outputs
-------
``results/figures/appendix/fig_blimp_traj_grid_humanscale.{pdf,png}``
``results/figures/appendix/fig_blimp_traj_grid_fineweb.{pdf,png}``
``results/figures/appendix/fig_blimp_traj_grid_typology.{pdf,png}``
``results/figures/appendix/fig_blimp_traj_grid_cl_ablations.{pdf,png}``
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
log = logging.getLogger("blimp_per_model_grids")

ISO_TO_SUITE: dict[str, tuple[str, str]] = {
    "eng": ("blimp_eng",  "eng"),
    "nld": ("blimp_nl",   "nld"),
    "deu": ("multiblimp", "deu"),
    "zho": ("zhoblimp",   "zho"),
}
ISO_LABEL = {"eng": "English", "nld": "Dutch",
             "deu": "German",  "zho": "Chinese"}

L1_COLOR = "#1b9e77"   # green — model's L1 language
L2_COLOR = "#d95f02"   # orange — model's L2 language

CURRICULA: list[str] = ["B1", "B2", "B3", "B4", "B5"]
CURRICULUM_LABEL = {
    "B1": "B1 — balanced 50:50",
    "B2": "B2 — simultaneous",
    "B3": "B3 — sequential 33:67",
    "B4": "B4 — classroom 20%",
    "B5": "B5 — late onset 80%",
}

_STEP_RE = re.compile(r"^step-(\d+)$")
_NEW_SLUG = re.compile(
    r"^beetle-bilingual-(?P<desc>[a-z0-9-]+?)-(?:b1|b2|b3|b4|b5)-"
    r"(?P<dataset>humanscale|fineweb)-(?P<l1>[a-z]{3})-(?P<l2>[a-z]{3})"
    r"(?P<tail>-[a-z0-9-]+)?$"
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
    tail = (m.group("tail") or "").lstrip("-") or None
    variant = None
    if tail:
        if "ewc" in tail:
            variant = "ewc"
        elif "lamol" in tail:
            variant = "lamol"
        elif "lr-b" in tail or tail.startswith("lr-"):
            variant = "lr"
        elif "episodic" in tail:
            variant = "episodic"
    return {
        "condition": cond, "dataset": m.group("dataset"),
        "l1": m.group("l1"), "l2": m.group("l2"),
        "variant": variant,
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
    df["variant"]   = df["meta"].apply(lambda d: d["variant"])
    df["step"]      = df["branch"].apply(_step)
    df = df[df["step"].notna()].copy()
    df["step"] = df["step"].astype(int)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df = df.dropna(subset=["accuracy"])
    return df


def _two_line_panel(ax, df: pd.DataFrame, *, model: str, l1: str, l2: str,
                    title: str, show_ylabel: bool, show_xlabel: bool) -> None:
    sub = df[df["model"] == model]
    plotted_any = False
    if sub.empty:
        pass
    else:
        l1_b, l1_iso = ISO_TO_SUITE.get(l1, (None, None))
        if l1_b is not None:
            l1_data = (sub[(sub["benchmark"] == l1_b) & (sub["iso_code"] == l1_iso)]
                          .sort_values("step"))
            if len(l1_data) >= 3:
                ax.plot(l1_data["step"].clip(lower=1), l1_data["accuracy"],
                        color=L1_COLOR, linestyle="-", linewidth=1.2,
                        marker="o", markersize=2.6, alpha=0.95, zorder=3)
                plotted_any = True
        l2_b, l2_iso = ISO_TO_SUITE.get(l2, (None, None))
        if l2_b is not None:
            l2_data = (sub[(sub["benchmark"] == l2_b) & (sub["iso_code"] == l2_iso)]
                          .sort_values("step"))
            if len(l2_data) >= 3:
                ax.plot(l2_data["step"].clip(lower=1), l2_data["accuracy"],
                        color=L2_COLOR, linestyle="--", linewidth=1.2,
                        marker="s", markersize=2.4, alpha=0.95, zorder=3)
                plotted_any = True

    if not plotted_any:
        ax.text(0.5, 0.5, "no checkpoint data", ha="center", va="center",
                transform=ax.transAxes, color="#aaa", fontsize=7)

    ax.set_xscale("log")
    ax.set_xlim(1, 1.2e5)
    ax.set_ylim(0.45, 0.95)
    ax.axhline(0.5, color="#bbb", linestyle=":", linewidth=0.5, zorder=1)
    ax.grid(True, axis="y", linestyle=":", color="#dcdcdc", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=8.5)
    if show_ylabel:
        ax.set_ylabel("accuracy", fontsize=8.5)
    if show_xlabel:
        ax.set_xlabel("training step (log)", fontsize=8)
    ax.tick_params(labelsize=7)


def _legend_handles(l1_iso: list[str] | None = None,
                    l2_iso: list[str] | None = None) -> list:
    h = [
        mlines.Line2D([], [], color=L1_COLOR, linestyle="-", linewidth=1.4,
                      marker="o", markersize=3,
                      label=f"L1 BLiMP ({'/'.join(ISO_LABEL[i] for i in l1_iso) if l1_iso else 'native'})"),
        mlines.Line2D([], [], color=L2_COLOR, linestyle="--", linewidth=1.4,
                      marker="s", markersize=3,
                      label=f"L2 BLiMP ({'/'.join(ISO_LABEL[i] for i in l2_iso) if l2_iso else 'second'})"),
    ]
    return h


def render_grid(df: pd.DataFrame, *,
                cells: list[list[dict | None]],
                row_labels: list[str], col_labels: list[str],
                output: Path, suptitle: str,
                figsize: tuple[float, float]) -> Path:
    """Generic grid renderer.

    `cells[r][c]` is either None (empty cell) or a dict with keys
    {model, l1, l2, title?}. Each cell becomes a per-model two-line panel.
    """
    nrows, ncols = len(cells), len(cells[0]) if cells else 0
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            cell = cells[r][c]
            if cell is None:
                ax.set_visible(False)
                continue
            t = cell.get("title", f"{cell['l1']}-{cell['l2']}")
            _two_line_panel(ax, df, model=cell["model"],
                            l1=cell["l1"], l2=cell["l2"],
                            title=t,
                            show_ylabel=(c == 0),
                            show_xlabel=(r == nrows - 1))
            if c == 0 and row_labels:
                ax.text(-0.42, 0.5, row_labels[r], rotation=90,
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9)
            if r == 0 and col_labels:
                # Column super-title above the topmost row (set above title)
                ax.set_title(col_labels[c] + ("\n" + cell.get("title", "") if cell.get("title") else ""),
                             fontsize=8.5)
    fig.suptitle(suptitle, fontsize=10.5, y=1.005)
    fig.legend(handles=_legend_handles(), loc="lower center",
               bbox_to_anchor=(0.5, -0.025), ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.99])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Type-42 / ACL-compliant via savefig_pub (PDF + PNG, embedded fonts).
    savefig_pub(fig, output)
    plt.close(fig)
    return output.with_suffix(".pdf")


# ── Cell-builder helpers ────────────────────────────────────────────────────

def _model_id(dataset: str, condition: str, l1: str, l2: str,
              variant: str | None = None) -> str:
    """Reverse a (dataset, condition, l1, l2) tuple to the canonical Beetle slug."""
    desc = {"B1": "balanced-b1", "B2": "l2-50-simultaneous-b2",
            "B3": "l2-50-sequential-33-67-b3",
            "B4": "l2-50-classroom-20-b4",
            "B5": "l2-80-late-b5"}[condition]
    org = "beetle-humanscale" if dataset == "humanscale" else "beetle-fineweb"
    base = f"{org}/beetle-bilingual-{desc}-{dataset}-{l1}-{l2}"
    if variant:
        base += f"-{variant}"
    return base


def cells_curriculum_x_pair(df: pd.DataFrame, dataset: str,
                            pairs: list[tuple[str, str]]) -> list[list]:
    out: list[list] = []
    for cond in CURRICULA:
        row = []
        for (l1, l2) in pairs:
            mid = _model_id(dataset, cond, l1, l2)
            if (df["model"] == mid).any():
                row.append({"model": mid, "l1": l1, "l2": l2,
                            "title": f"{l1}-{l2}"})
            else:
                row.append(None)
        out.append(row)
    return out


def render_humanscale_grid(df: pd.DataFrame) -> Path:
    pairs = [("nld", "eng"), ("deu", "eng"), ("zho", "eng")]
    cells = cells_curriculum_x_pair(df, "humanscale", pairs)
    return render_grid(
        df, cells=cells,
        row_labels=[CURRICULUM_LABEL[c] for c in CURRICULA],
        col_labels=["", "", ""],
        output=APP_DIR / "fig_blimp_traj_grid_humanscale",
        suptitle="Beetle-HumanScale (100 M tokens) — per-model trajectories: L1 vs L2 BLiMP",
        figsize=(9.5, 11.0),
    )


def render_fineweb_grid(df: pd.DataFrame) -> Path:
    pairs = [("nld", "eng"), ("deu", "eng"), ("eng", "nld"), ("eng", "deu")]
    sub_curr = ["B1", "B2", "B3"]
    cells: list[list] = []
    for cond in sub_curr:
        row = []
        for (l1, l2) in pairs:
            mid = _model_id("fineweb", cond, l1, l2)
            if (df["model"] == mid).any():
                row.append({"model": mid, "l1": l1, "l2": l2,
                            "title": f"{l1}-{l2}"})
            else:
                row.append(None)
        cells.append(row)
    return render_grid(
        df, cells=cells,
        row_labels=[CURRICULUM_LABEL[c] for c in sub_curr],
        col_labels=["", "", "", ""],
        output=APP_DIR / "fig_blimp_traj_grid_fineweb",
        suptitle="Beetle-FineWeb (24 B tokens) — per-model trajectories: L1 vs L2 BLiMP",
        figsize=(11.5, 7.0),
    )


def render_typology_grid(df: pd.DataFrame) -> Path:
    """HumanScale B2 simultaneous: how does typological distance affect transfer?

    Cols: nld-eng (Indo-Eu, close), deu-eng (Indo-Eu, close), zho-eng (Sino-Tib, far).
    """
    pairs = [("nld", "eng"), ("deu", "eng"), ("zho", "eng")]
    cells = []
    for cond in CURRICULA:
        row = []
        for (l1, l2) in pairs:
            mid = _model_id("humanscale", cond, l1, l2)
            if (df["model"] == mid).any():
                row.append({"model": mid, "l1": l1, "l2": l2,
                            "title": f"{l1}-{l2}"})
            else:
                row.append(None)
        # Only include rows that have at least one cell
        if any(c is not None for c in row):
            cells.append(row)
    return render_grid(
        df, cells=cells,
        row_labels=[CURRICULUM_LABEL[c] for c, row in zip(CURRICULA, cells) if any(row)],
        col_labels=["nld-eng (close IE)", "deu-eng (close IE)", "zho-eng (distant)"],
        output=APP_DIR / "fig_blimp_traj_grid_typology",
        suptitle="HumanScale typology — does L1 distance modulate L2 acquisition?",
        figsize=(9.5, 2.4 * len(cells) + 1.0),
    )


def render_cl_ablations_grid(df: pd.DataFrame) -> Path:
    """HumanScale B2 nld-eng baseline vs EWC vs LAMOL — single-row grid."""
    base = "beetle-humanscale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng"
    cells = [[
        {"model": base, "l1": "nld", "l2": "eng", "title": "B2 baseline"},
        {"model": base + "-ewc-l20", "l1": "nld", "l2": "eng",
         "title": "B2 + EWC (λ=20)"},
        {"model": base + "-lamol-l20", "l1": "nld", "l2": "eng",
         "title": "B2 + LAMOL (λ=20)"},
    ]]
    return render_grid(
        df, cells=cells,
        row_labels=["B2 simultaneous nld-eng"],
        col_labels=["", "", ""],
        output=APP_DIR / "fig_blimp_traj_grid_cl_ablations",
        suptitle="Continual-learning ablations on HumanScale B2 nld-eng",
        figsize=(9.5, 3.2),
    )


def render_all() -> dict[str, Path]:
    apply_acl_style()
    df = load_traj()
    if df.empty:
        log.warning("no parseable checkpoint rows")
        return {}
    out: dict[str, Path] = {}
    out["A_humanscale"]    = render_humanscale_grid(df)
    out["B_fineweb"]       = render_fineweb_grid(df)
    out["C_typology"]      = render_typology_grid(df)
    out["D_cl_ablations"]  = render_cl_ablations_grid(df)
    return out


def main() -> None:
    paths = render_all()
    for k, v in paths.items():
        print(f"[blimp_per_model_grids] {k} -> {v}")


if __name__ == "__main__":
    main()
