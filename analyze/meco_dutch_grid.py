"""meco_dutch_grid.py — Dutch-MECO training-trajectory grid for nld-eng B1–B5.

Builds a single multi-panel figure of training trajectories on the MECO L2
reading-time corpus *restricted to Dutch L1 readers* (``lang == "du"``) for
every nld-eng bilingual model that has per-checkpoint surprisal data.

Two metrics are tracked over training:
    1. **Mean surprisal** on the (item, word) rows that Dutch L1 readers fixated.
       Lower = better English LM relative to those stimuli. (Cheap; available for
       every checkpoint with surprisals.)
    2. **Spearman ρ(surprisal, RT)** on the same (item, word) rows. This is the
       psycholinguistic alignment metric — high ρ means surprisal predicts
       reading times. (Requires the joined RT data; computed on the fly.)

Layout: rows = scale/org, cols = curriculum (B1 / B2 / B3 / B4 / B5 / variants).

Inputs
------
``results/evals_checkpoints/meco/surprisals_*nld-eng*__step-*.csv`` and
``results/evals_b3_checkpoints/meco/surprisals_*nld-eng*__step-*.csv``.

The reading-time / language join is loaded once from
``data/meco_l2/joint_data_l2_trimmed_version2.2.rda`` via
``analyze.meco_canonical.load_joined_meco_l2``.

Outputs
-------
``results/figures/main/fig_meco_dutch_grid_nld_eng.{pdf,png}``
``results/figures/appendix/fig_meco_dutch_grid_nld_eng_rho.{pdf,png}``
``results/figures/appendix/fig_meco_dutch_overlay_nld_eng.{pdf,png}``
``results/meco/meco_dutch_trajectory_summary.csv``
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyze.plot_style import (
    PALETTE_ACL,
    apply_acl_style,
    fig_double,
    savefig_pub,
    style_axes,
)

log = logging.getLogger("meco_dutch_grid")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SURPR_DIRS = [
    RESULTS / "evals_checkpoints" / "meco",
    RESULTS / "evals_b3_checkpoints" / "meco",
]
MAIN_DIR = RESULTS / "figures" / "main"
APP_DIR = RESULTS / "figures" / "appendix"
SUMMARY_CSV = RESULTS / "meco" / "meco_dutch_trajectory_summary.csv"
# ΔlogLik trajectory cache produced by scripts/p1_meco_deltalogl_nld_eng_traj.py.
DELTALOGL_CACHE = RESULTS / "meco" / "canonical_deltalogl_nld_eng_traj.csv"

ORG_TOKENS: dict[str, float] = {
    "beetle-humanscale": 1.0e8,
    "beetle-fineweb": 2.4e10,
    "beetle-nld-eng-variants": 2.0e9,
}
ORG_LABEL: dict[str, str] = {
    "beetle-humanscale": "HumanScale 100M",
    "beetle-fineweb": "FineWeb 24B",
    "beetle-nld-eng-variants": "nld-eng-Variants 2B",
}
ORG_ROW_ORDER = ["beetle-humanscale", "beetle-fineweb", "beetle-nld-eng-variants"]

# Display order for curriculum / variant tags (one per panel column).
CONDITION_ORDER = ["B1", "B2", "B3", "B4", "B5"]
CONDITION_LABEL = {
    "B1": "B1 balanced",
    "B2": "B2 simultaneous",
    "B3": "B3 sequential",
    "B4": "B4 classroom",
    "B5": "B5 late",
}
# Variants on the nld-eng-Variants org get their own column tags.
VARIANT_ORDER = ["alibi", "layer-growth", "pre-pretrain", "sas", "tilt"]


# ────────────────────────────────────────────────────────────────────────────
# Filename + slug parsing
# ────────────────────────────────────────────────────────────────────────────

_FNAME = re.compile(
    r"^surprisals_(?P<org>[a-z0-9-]+)_(?P<name>[a-z0-9-]+?)__step-(?P<step>\d+)\.csv$"
)

_COND = [
    (re.compile(r"-balanced-b1-"),                 "B1"),
    (re.compile(r"-l2-50-simultaneous-b2-"),       "B2"),
    (re.compile(r"-l2-50-sequential-(?:\d+-\d+-)?b3-"), "B3"),
    (re.compile(r"-l2-50-classroom-20-b4-"),       "B4"),
    (re.compile(r"-l2-80-late-b5-"),               "B5"),
    (re.compile(r"-balanced-b4-"),                 "B4"),  # FineWeb b4 uses 'balanced'
    (re.compile(r"-balanced-b5-"),                 "B5"),  # FineWeb b5 uses 'balanced'
]


@dataclass
class FileMeta:
    path: Path
    org: str
    model_name: str
    step: int
    condition: str
    variant: Optional[str]


def _parse_filename(p: Path) -> Optional[FileMeta]:
    m = _FNAME.match(p.name)
    if not m:
        return None
    org = m.group("org").lower()
    name = m.group("name")
    step = int(m.group("step"))
    cond = "unknown"
    for pat, c in _COND:
        if pat.search(name):
            cond = c
            break
    # Capture the trailing variant tag (after the language pair, e.g. "alibi").
    # Slugs end in -nld-eng or -nld-eng-<variant>.
    variant: Optional[str] = None
    m2 = re.search(r"-nld-eng-(?P<v>[a-z0-9-]+?)(?:-postfix)?$", name)
    if m2:
        variant = m2.group("v")
    return FileMeta(p, org, name, step, cond, variant)


# ────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────

def _gather_files() -> list[FileMeta]:
    files: list[FileMeta] = []
    for d in SURPR_DIRS:
        if not d.exists():
            continue
        for p in d.glob("surprisals_*nld-eng*__step-*.csv"):
            meta = _parse_filename(p)
            if meta is None:
                continue
            # Restrict to nld-eng (i.e. Dutch L1, English L2). Skip eng-nld.
            if "-nld-eng" not in meta.model_name:
                continue
            files.append(meta)
    return files


def _load_dutch_rt() -> pd.DataFrame:
    """Return the Dutch-L1 (lang='du') subset of MECO L2 RT data.

    Joined wave-1 + wave-2 via analyze.meco_canonical.load_joined_meco_l2.
    Drops rows with missing wordnum and zero/NA RT, but keeps all 12 items.
    """
    from analyze.meco_canonical import load_joined_rt
    rt = load_joined_rt()
    rt = rt[rt["lang"].astype(str) == "du"].copy()
    rt = rt.dropna(subset=["firstrun_dur"])
    rt = rt[rt["firstrun_dur"] > 0]
    # Aggregate across subjects to one RT per (item, word).
    rt["itemid"] = rt["itemid"].astype(str)
    rt = (rt.groupby(["itemid", "wordnum"], dropna=False)["firstrun_dur"]
            .mean().reset_index().rename(columns={"firstrun_dur": "rt_du"}))
    return rt


def _aggregate_one_file(meta: FileMeta, rt_du: pd.DataFrame) -> Optional[dict]:
    """Compute Dutch-MECO trajectory metrics for one (model, step) CSV.

    Restricts surprisal rows to the (itemid, wordnum) keys that have Dutch RT
    data (which spans every L2 stim item) and returns:
        mean_surprisal, spearman_rho, n_obs.
    """
    try:
        df = pd.read_csv(meta.path, usecols=["itemid", "wordnum", "surprisal"])
    except Exception as e:
        log.warning("[%s] read failed: %s", meta.path.name, e)
        return None
    df["itemid"] = df["itemid"].astype(str)
    df["wordnum"] = pd.to_numeric(df["wordnum"], errors="coerce")
    df["surprisal"] = pd.to_numeric(df["surprisal"], errors="coerce")
    df = df.dropna(subset=["wordnum", "surprisal"])
    df["wordnum"] = df["wordnum"].astype(int)
    j = df.merge(rt_du, on=["itemid", "wordnum"], how="inner")
    if j.empty:
        return None
    rho_val: float
    if j["surprisal"].nunique() > 1 and j["rt_du"].nunique() > 1:
        rho, _ = spearmanr(j["surprisal"], j["rt_du"])
        rho_val = float(rho) if rho is not None and not np.isnan(rho) else float("nan")
    else:
        rho_val = float("nan")
    return {
        "org": meta.org,
        "model": meta.model_name,
        "condition": meta.condition,
        "variant": meta.variant,
        "step": meta.step,
        "mean_surprisal": float(j["surprisal"].mean()),
        "spearman_rho": rho_val,
        "n_obs": int(len(j)),
    }


def build_trajectory_summary() -> pd.DataFrame:
    files = _gather_files()
    if not files:
        log.warning("no nld-eng MECO surprisal step-files found")
        return pd.DataFrame()
    rt_du = _load_dutch_rt()
    log.info("loaded Dutch-L1 RT join: %d (item, word) rows", len(rt_du))
    rows: list[dict] = []
    for i, meta in enumerate(files, 1):
        rec = _aggregate_one_file(meta, rt_du)
        if rec is not None:
            rows.append(rec)
        if i % 50 == 0:
            log.info("aggregated %d / %d files", i, len(files))
    summary = pd.DataFrame(rows).sort_values(["org", "condition", "model", "step"])
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    log.info("wrote %s (%d rows, %d models)",
             SUMMARY_CSV, len(summary), summary["model"].nunique())
    return summary


# ────────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────────

def _short_model_label(name: str) -> str:
    """Compress the slug for in-panel labels."""
    s = name
    s = re.sub(r"^beetle-bilingual-", "", s)
    s = re.sub(r"^l2-50-", "", s)
    s = re.sub(r"^l2-80-", "", s)
    s = s.replace("humanscale-nld-eng", "HS").replace("fineweb-nld-eng", "FW")
    s = s.replace("fineweb-2b-nld-eng", "FW2B")
    return s


def _col_keys(org: str, summary: pd.DataFrame) -> list[str]:
    """Per-row column ordering: B1..B5 for the canonical orgs, variant tags
    for nld-eng-Variants."""
    if org == "beetle-nld-eng-variants":
        return [v for v in VARIANT_ORDER
                if not summary[(summary["org"] == org) &
                               (summary["variant"] == v)].empty]
    return [c for c in CONDITION_ORDER
            if not summary[(summary["org"] == org) &
                           (summary["condition"] == c)].empty]


def _panel(ax, df: pd.DataFrame, *, metric: str, title: str,
           ylabel: Optional[str], invert_y: bool = False) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="#888", fontsize=7.5)
        style_axes(ax, xlabel=None, ylabel=ylabel, grid=True)
        ax.set_title(title, fontsize=8)
        return
    for model_name, trace in df.groupby("model"):
        trace = trace.sort_values("step")
        cond = trace["condition"].iloc[0]
        color = PALETTE_ACL.get(cond, "#888")
        # If an ablation variant exists, lighten the line so the variant grid
        # cell stays distinguishable from the base condition cell.
        is_variant = trace["variant"].notna().any()
        ax.plot(trace["step"].clip(lower=1), trace[metric],
                color=color, marker="o", markersize=2.6,
                linewidth=1.0 if is_variant else 1.4,
                alpha=0.9, zorder=3,
                label=_short_model_label(model_name))
    ax.set_xscale("log")
    if invert_y:
        ax.invert_yaxis()
    style_axes(ax, xlabel="training step (log)",
               ylabel=ylabel, grid=True)
    ax.set_title(title, fontsize=8.5)
    # Tight per-panel y-limits so trajectories fill each cell.
    ys = df[metric].dropna()
    if not ys.empty:
        lo, hi = float(ys.min()), float(ys.max())
        pad = max(0.05 * (hi - lo), 0.01)
        ax.set_ylim(lo - pad, hi + pad)


def render_grid(summary: pd.DataFrame, *, metric: str = "mean_surprisal",
                out_dir: Path = MAIN_DIR,
                fname: str = "fig_meco_dutch_grid_nld_eng",
                invert_y: bool = False) -> Path:
    apply_acl_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        raise SystemExit("no MECO Dutch trajectory data — run build_trajectory_summary first")

    rows = [o for o in ORG_ROW_ORDER if not summary[summary["org"] == o].empty]
    n_rows = len(rows)
    n_cols = max(len(_col_keys(o, summary)) for o in rows)
    if n_cols == 0:
        raise SystemExit("no curriculum/variant cells with data")

    fig, axes = fig_double(height=2.2 * n_rows, ncols=n_cols, nrows=n_rows,
                            sharex=True)
    axes = np.atleast_2d(axes)
    if metric == "mean_surprisal":
        ylabel = "mean surprisal"
    elif metric == "delta_loglik":
        ylabel = "ΔlogLik (Dutch L1)"
    else:
        ylabel = "Spearman ρ(surp, RT)"
    for r, org in enumerate(rows):
        cols = _col_keys(org, summary)
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(cols):
                ax.set_visible(False)
                continue
            col_key = cols[c]
            if org == "beetle-nld-eng-variants":
                sub = summary[(summary["org"] == org) &
                              (summary["variant"] == col_key)]
                col_title = col_key
            else:
                sub = summary[(summary["org"] == org) &
                              (summary["condition"] == col_key) &
                              (summary["variant"].isna())]
                col_title = CONDITION_LABEL.get(col_key, col_key)
            row_label = ORG_LABEL[org] if c == 0 else None
            title = f"{ORG_LABEL[org].split()[0]} · {col_title}" if c == 0 else col_title
            _panel(ax, sub, metric=metric, title=title,
                   ylabel=row_label, invert_y=invert_y)

    fig.suptitle("Dutch-MECO training trajectories — nld-eng (L1=Dutch, L2=English)",
                 fontsize=9.5, y=1.005)
    fig.tight_layout()
    out = out_dir / fname
    savefig_pub(fig, out)
    plt.close(fig)
    return out.with_suffix(".pdf")


def render_overlay(summary: pd.DataFrame, *, metric: str = "mean_surprisal",
                   out_dir: Path = APP_DIR,
                   fname: str = "fig_meco_dutch_overlay_nld_eng") -> Path:
    """Single-panel overlay per scale-org, color = condition. Companion to the
    grid for at-a-glance comparison."""
    apply_acl_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [o for o in ORG_ROW_ORDER if not summary[summary["org"] == o].empty]
    fig, axes = fig_double(height=2.6, ncols=len(rows), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, org in zip(axes, rows):
        sub = summary[summary["org"] == org]
        for cond in CONDITION_ORDER:
            slice_ = sub[(sub["condition"] == cond) & (sub["variant"].isna())]
            if slice_.empty:
                continue
            for model_name, trace in slice_.groupby("model"):
                trace = trace.sort_values("step")
                ax.plot(trace["step"].clip(lower=1), trace[metric],
                        color=PALETTE_ACL.get(cond, "#888"),
                        marker="o", markersize=2.4, linewidth=1.2, alpha=0.85)
        ax.set_xscale("log")
        style_axes(ax, xlabel="training step (log)",
                   ylabel="mean surprisal" if ax is axes[0] else None, grid=True)
        ax.set_title(ORG_LABEL[org], fontsize=8.5)
    handles = [mlines.Line2D([], [], color=PALETTE_ACL[c], marker="o",
                             markersize=3, linewidth=1.2,
                             label=CONDITION_LABEL[c])
               for c in CONDITION_ORDER if c in PALETTE_ACL]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=7.5)
    fig.suptitle("Dutch-MECO trajectories — overlay by curriculum",
                 fontsize=9.5, y=1.01)
    fig.tight_layout()
    out = out_dir / fname
    savefig_pub(fig, out)
    plt.close(fig)
    return out.with_suffix(".pdf")


_DELTALOGL_MODEL_RE = re.compile(
    r"^(?P<org>[a-z0-9-]+)/(?P<name>[a-z0-9-]+)@step-(?P<step>\d+)$"
)


def _join_deltalogl(summary: pd.DataFrame, lang: str = "du") -> pd.DataFrame:
    """Attach ΔlogLik values from canonical_deltalogl_nld_eng_traj.csv to the
    summary frame on (org, model, step). Returns a new frame with a
    ``delta_loglik`` column; rows without a fit get NaN."""
    if not DELTALOGL_CACHE.exists():
        log.warning(
            "ΔlogLik cache missing (%s); run scripts/p1_meco_deltalogl_nld_eng_traj.py first",
            DELTALOGL_CACHE,
        )
        out = summary.copy()
        out["delta_loglik"] = float("nan")
        return out

    cache = pd.read_csv(DELTALOGL_CACHE)
    cache = cache[cache["lang"].astype(str) == lang].copy()
    if cache.empty:
        log.warning("ΔlogLik cache has no rows for lang=%s", lang)
        out = summary.copy()
        out["delta_loglik"] = float("nan")
        return out

    parsed = cache["model"].astype(str).str.extract(_DELTALOGL_MODEL_RE)
    cache["org"] = parsed["org"].str.lower()
    cache["model_name"] = parsed["name"]
    cache["step"] = pd.to_numeric(parsed["step"], errors="coerce").astype("Int64")
    cache = cache.dropna(subset=["org", "model_name", "step"])
    cache["step"] = cache["step"].astype(int)

    keep = cache[["org", "model_name", "step", "delta_loglik"]].rename(
        columns={"model_name": "model"}
    )
    return summary.merge(keep, on=["org", "model", "step"], how="left")


def render_all() -> dict[str, Path]:
    summary = (pd.read_csv(SUMMARY_CSV)
               if SUMMARY_CSV.exists() else build_trajectory_summary())
    if summary.empty:
        return {}
    out: dict[str, Path] = {}
    out["main_grid"] = render_grid(summary, metric="mean_surprisal",
                                   out_dir=MAIN_DIR,
                                   fname="fig_meco_dutch_grid_nld_eng")
    out["app_grid_rho"] = render_grid(summary, metric="spearman_rho",
                                      out_dir=APP_DIR,
                                      fname="fig_meco_dutch_grid_nld_eng_rho",
                                      invert_y=False)
    out["app_overlay"] = render_overlay(summary, metric="mean_surprisal")

    # Appendix ΔlogLik panel: per-checkpoint Dutch-L1 mixed-model ΔlogLik
    # joined from canonical_deltalogl_nld_eng_traj.csv. Replaces the rho figure
    # as the headline appendix metric (rho is retained alongside).
    summary_dll = _join_deltalogl(summary, lang="du")
    if summary_dll["delta_loglik"].notna().any():
        out["app_grid_deltalogl"] = render_grid(
            summary_dll, metric="delta_loglik",
            out_dir=APP_DIR,
            fname="fig_meco_dutch_grid_nld_eng_deltalogl",
            invert_y=False,
        )
    else:
        log.warning(
            "no ΔlogLik fits available — skipping fig_meco_dutch_grid_nld_eng_deltalogl"
        )
    return out


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true",
                   help="Re-aggregate the per-checkpoint summary CSV.")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    if args.rebuild or not SUMMARY_CSV.exists():
        build_trajectory_summary()
    paths = render_all()
    for k, v in paths.items():
        print(f"[meco_dutch_grid] wrote {k}: {v}")


if __name__ == "__main__":
    main()
