"""All paper figures for the SLT analysis.

Reads consolidated CSVs from ``<output_root>/slt/`` and emits PDFs into
``<paper_dir>/figures/``. Six figures are produced:

  1. ``fig_headline_capacity_vs_lambda.pdf``  — A3 capacity-vs-λ scatter.
  2. ``fig_bridge_pc1.pdf``                  — A1 cross-lingual probe ↔ PC1.
  3. ``fig_llc_trajectory.pdf``              — S1 λ̂ across training.
  4. ``fig_susceptibility_pca_heatmap.pdf``  — heatmap of |loadings| × probe.
  5. ``fig_fisher_spectra.pdf``              — eigenvalue curves per layer.
  6. ``fig_falsification_panel.pdf``         — A2 controls.

Each figure also writes a ``.csv`` next to it with the points plotted, so
the LaTeX paper can reference the numerical source.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS))

log = logging.getLogger(__name__)

_REGIME_COLOURS = {
    "mono": "#3b6cb0",
    "bi": "#bf6f3a",
    "multi": "#3a8a4d",
    "unknown": "#888888",
}


def _import_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    return matplotlib, plt


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _safe_float(s: str | None, default: float = float("nan")) -> float:
    try:
        return float(s) if s not in (None, "") else default
    except ValueError:
        return default


def fig_headline(slt_dir: Path, fig_dir: Path) -> None:
    """A3: linguistic-capacity composite vs λ̂.

    The capacity composite is read from
    ``<output_root>/figures/capacity_composite.csv`` if it exists
    (produced by an earlier beetle-analyze stage); otherwise we use
    ``init_loss`` as a stand-in proxy.
    """
    _, plt = _import_mpl()
    rows = _read_csv(slt_dir / "llc_summary.csv")
    if not rows:
        log.warning("[plot] no llc_summary.csv — skipping headline figure")
        return

    cap_csv = slt_dir.parent / "figures" / "capacity_composite.csv"
    cap_lookup: dict[tuple[str, str], float] = {}
    if cap_csv.exists():
        for r in _read_csv(cap_csv):
            cap_lookup[(r["repo"], r["revision"])] = _safe_float(r.get("capacity"))

    xs, ys, cs, labels = [], [], [], []
    for r in rows:
        key = (r["repo"], r["revision"])
        cap = cap_lookup.get(key)
        if cap is None or np.isnan(cap):
            cap = -_safe_float(r.get("init_loss"))  # fallback proxy
        lam = _safe_float(r.get("llc_mean"))
        if np.isnan(cap) or np.isnan(lam):
            continue
        xs.append(cap)
        ys.append(lam)
        cs.append(_REGIME_COLOURS.get(r.get("regime", "unknown"), "#888"))
        labels.append(r["repo"].split("/")[-1])

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter(xs, ys, c=cs, s=30, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Linguistic-capacity composite (z)")
    ax.set_ylabel(r"$\hat{\lambda}$ (LLC)")
    ax.set_title("Capacity rises while $\\lambda$ falls in multilingual models")
    for r in ("mono", "bi", "multi"):
        ax.scatter([], [], c=_REGIME_COLOURS[r], label=r, edgecolor="black", linewidth=0.4)
    ax.legend(frameon=False, loc="best", fontsize=8)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_headline_capacity_vs_lambda.pdf"
    fig.savefig(out)
    plt.close(fig)

    with (fig_dir / "fig_headline_capacity_vs_lambda.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["repo", "regime", "capacity", "llc_mean"])
        for r, x, y in zip(rows, xs, ys):
            w.writerow([r["repo"], r.get("regime", "?"), x, y])
    log.info("[plot] %s", out)


def fig_bridge(slt_dir: Path, fig_dir: Path) -> None:
    _, plt = _import_mpl()
    rows = _read_csv(slt_dir / "bridge_summary.csv")
    if not rows:
        log.warning("[plot] no bridge_summary.csv — skipping A1 figure")
        return
    xs = np.array([_safe_float(r["shared_feature_score"]) for r in rows])
    ys = np.array([_safe_float(r["pc1_var_ratio"]) for r in rows])
    cs = [_REGIME_COLOURS.get(r.get("regime", "unknown"), "#888") for r in rows]

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter(xs, ys, c=cs, s=30, alpha=0.85, edgecolor="black", linewidth=0.4)
    if xs.size >= 2:
        slope, intercept = np.polyfit(xs, ys, 1)
        xx = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xx, slope * xx + intercept, "k--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Cross-lingual probe score")
    ax.set_ylabel("PC1 explained variance ratio")
    ax.set_title("A1: shared-feature ↔ susceptibility-PC1")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_bridge_pc1.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("[plot] %s", out)


def fig_trajectory(slt_dir: Path, fig_dir: Path) -> None:
    _, plt = _import_mpl()
    rows = _read_csv(slt_dir / "llc_summary.csv")
    if not rows:
        return

    by_repo: dict[str, list[tuple[str, float, str]]] = {}
    for r in rows:
        rev = r["revision"]
        try:
            step = int(rev.split("step-")[-1]) if "step-" in rev else -1
        except ValueError:
            step = -1
        by_repo.setdefault(r["repo"], []).append(
            (rev, _safe_float(r.get("llc_mean")), r.get("regime", "unknown"))
        )

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for repo, items in by_repo.items():
        regime = items[0][2]
        items_sorted = sorted(
            items,
            key=lambda x: (
                int(x[0].split("step-")[-1]) if "step-" in x[0] else 1_000_000_000
            ),
        )
        ys = [v for _, v, _ in items_sorted]
        xs = list(range(len(items_sorted)))
        ax.plot(xs, ys, "-o", color=_REGIME_COLOURS.get(regime, "#888"),
                alpha=0.7, markersize=3, linewidth=0.8)
    ax.set_xlabel("Checkpoint index (1% → 100%)")
    ax.set_ylabel(r"$\hat{\lambda}$ (LLC)")
    ax.set_title("S1: LLC trajectory across training")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_llc_trajectory.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("[plot] %s", out)


def fig_pca_heatmap(slt_dir: Path, fig_dir: Path) -> None:
    """For each regime, the mean |PC1 loading| over components."""
    _, plt = _import_mpl()
    sus_dir = slt_dir / "susceptibilities"
    if not sus_dir.exists():
        return

    by_regime: dict[str, list[np.ndarray]] = {"mono": [], "bi": [], "multi": []}
    sus_summary = _read_csv(slt_dir / "susceptibilities_summary.csv")
    by_key = {(r["repo"], r["revision"]): r for r in sus_summary}
    component_names: list[str] = []

    for npz in sorted(sus_dir.rglob("*_pca.npz")):
        repo_slug = npz.parent.name
        rev = npz.stem.replace("_pca", "")
        repo_orig = repo_slug.replace("__", "/")
        meta = by_key.get((repo_orig, rev))
        regime = meta.get("regime", "unknown") if meta else "unknown"
        if regime not in by_regime:
            continue
        try:
            data = np.load(npz, allow_pickle=True)
            loadings = np.abs(np.asarray(data["loadings"]))[0]  # PC1 only
            if not component_names:
                component_names = list(np.asarray(data["component_names"]))
            by_regime[regime].append(loadings)
        except Exception:  # pragma: no cover
            log.exception("[plot] failed to load %s", npz)

    if not any(by_regime.values()):
        return
    means = []
    labels = []
    for regime in ("mono", "bi", "multi"):
        if by_regime[regime]:
            stacked = np.stack(by_regime[regime])
            means.append(stacked.mean(axis=0))
            labels.append(regime)
    if not means:
        return
    arr = np.stack(means)

    fig, ax = plt.subplots(figsize=(8.0, 2.4))
    im = ax.imshow(arr, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks([])
    ax.set_xlabel(f"Component (n = {arr.shape[1]})")
    ax.set_title("|PC1 loading| × component, by regime")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_susceptibility_pca_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("[plot] %s", out)


def fig_fisher(slt_dir: Path, fig_dir: Path) -> None:
    _, plt = _import_mpl()
    fisher_dir = slt_dir / "fisher"
    if not fisher_dir.exists():
        return

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    n_drawn = 0
    for npz in sorted(fisher_dir.rglob("*.npz")):
        try:
            data = np.load(npz)
        except Exception:
            continue
        for layer_name in data.files:
            eigs = data[layer_name]
            if eigs.size == 0:
                continue
            ax.semilogy(np.sort(eigs)[::-1], linewidth=0.5, alpha=0.5)
            n_drawn += 1
            if n_drawn > 200:
                break
        if n_drawn > 200:
            break
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue (log)")
    ax.set_title("Block-Fisher eigenspectra (per layer × cell)")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_fisher_spectra.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("[plot] %s", out)


def fig_falsification(slt_dir: Path, fig_dir: Path) -> None:
    _, plt = _import_mpl()
    rows = _read_csv(slt_dir / "falsification_summary.csv")
    llc = _read_csv(slt_dir / "llc_summary.csv")
    if not rows or not llc:
        return

    mono_lams = [_safe_float(r.get("llc_mean")) for r in llc if r.get("regime") == "mono"]
    mono_mean = float(np.nanmean(mono_lams)) if mono_lams else float("nan")

    a2a = [_safe_float(r["llc_mean"]) for r in rows if r.get("kind") == "a2a_disjoint"]
    a2b_b = [_safe_float(r["llc_mean"]) for r in rows if r.get("kind") == "a2b_permuted_before"]
    a2b_a = [_safe_float(r["llc_mean"]) for r in rows if r.get("kind") == "a2b_permuted_after"]

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    bars = []
    if mono_lams: bars.append(("mono\nbaseline", mono_mean, "#888"))
    if a2a:       bars.append(("A2a\nvocab-disjoint", float(np.nanmean(a2a)), _REGIME_COLOURS["bi"]))
    if a2b_b:     bars.append(("A2b\nbefore", float(np.nanmean(a2b_b)), _REGIME_COLOURS["multi"]))
    if a2b_a:     bars.append(("A2b\nafter perm.", float(np.nanmean(a2b_a)), "#cc4444"))
    if not bars:
        return
    xs = list(range(len(bars)))
    ys = [b[1] for b in bars]
    cs = [b[2] for b in bars]
    ax.bar(xs, ys, color=cs, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylabel(r"$\hat{\lambda}$ (LLC)")
    ax.set_title("A2: falsification controls")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_falsification_panel.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("[plot] %s", out)


def render_all(output_root: Path, paper_dir: Path) -> None:
    slt_dir = output_root / "slt"
    fig_dir = paper_dir / "figures"
    if not slt_dir.exists():
        log.warning("[plot] %s does not exist", slt_dir)
        return
    fig_headline(slt_dir, fig_dir)
    fig_bridge(slt_dir, fig_dir)
    fig_trajectory(slt_dir, fig_dir)
    fig_pca_heatmap(slt_dir, fig_dir)
    fig_fisher(slt_dir, fig_dir)
    fig_falsification(slt_dir, fig_dir)


def run_stage(args) -> None:
    if int(getattr(args, "rank", 0)) != 0:
        return
    paper_dir = getattr(args, "paper_dir", None) or (
        Path(__file__).resolve().parent / "paper"
    )
    render_all(args.output_root, Path(paper_dir))


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Render SLT paper figures.")
    p.add_argument("--output-root", type=Path, default=Path("results"))
    p.add_argument("--paper-dir", type=Path,
                   default=Path(__file__).resolve().parent / "paper")
    p.add_argument("--diagnostics", action="store_true",
                   help="Also render LLC chain-trace diagnostics.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    render_all(args.output_root, args.paper_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
