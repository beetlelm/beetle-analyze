"""Per-component susceptibilities + structural-inference PCA.

Implements the Baker et al. 2026 framework via ``devinterp``:
  - Build the unperturbed sampling task (multilingual mixture).
  - Build four probe distributions: eng, nld, mix_code, multi.
  - Enumerate components: full, embed, unembed, per-MLP, per-head.
  - Call ``devinterp.slt.susceptibilities.susceptibilities`` to estimate
    χ_C for every (component, probe) pair.
  - PCA on the standardised response matrix X. Persist top 3 PCs.

Outputs:
  ``<output_root>/slt/susceptibilities/<repo>/<rev>.zarr``  raw devinterp DataTree
  ``<output_root>/slt/susceptibilities/<repo>/<rev>_pca.npz``  PCs + variance ratios
  ``<output_root>/slt/susceptibilities_summary.rank{R}.csv``  flat per-cell row
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS))

from analyze.ppl_utils import load_model_and_tokenizer  # noqa: E402

from slt.components import enumerate_components, build_devinterp_masks  # noqa: E402
from slt.probe_data import build_probes, build_sampling_task, PROBE_NAMES  # noqa: E402
from slt.run_llc import _detect_regime, _scaled_hyperparams  # noqa: E402

log = logging.getLogger(__name__)


@dataclass
class SusceptibilityCellSummary:
    repo: str
    revision: str
    regime: str
    languages: str
    n_components: int
    n_probes: int
    pc1_var_ratio: float
    pc2_var_ratio: float
    pc3_var_ratio: float
    matrix_rank_99pct: int     # # PCs to explain 99% of variance
    pc1_loading_uniformity: float  # 1 - normalized std of |PC1 loadings|


def _standardize_columns(X: np.ndarray) -> np.ndarray:
    """Subtract row means, then standardise columns (zero-mean, unit-var)."""
    X = X - X.mean(axis=1, keepdims=True)  # cancels C-only term per Baker §3.2
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd


def _pca(X: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD-based PCA. Returns ``(components, explained_variance, loadings)``.

    Conventions:
      X has shape (n_probes, n_components).
      ``components``: (k, n_probes)        — left singular vectors × singular values
      ``loadings``: (k, n_components)      — right singular vectors
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = (S ** 2) / max(X.shape[0] - 1, 1)
    var_ratio = var / var.sum() if var.sum() > 0 else var
    components = (U[:, :k] * S[:k])
    loadings = Vt[:k]
    return components.T, var_ratio[:k], loadings


def _matrix_rank_99(X: np.ndarray) -> int:
    """Number of PCs needed to capture 99% of variance."""
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    var = (S ** 2) / max(X.shape[0] - 1, 1)
    if var.sum() == 0:
        return 0
    cum = np.cumsum(var) / var.sum()
    return int(np.searchsorted(cum, 0.99) + 1)


def _aggregate_per_token_susceptibilities(
    sus_data: "xr.Dataset",
    probe_names: list[str],
    component_names: list[str],
) -> np.ndarray:
    """Reduce devinterp's (sus_flat, wr) per-token output to a (probes, components) matrix.

    devinterp returns ``sus`` with dims ``(sus_flat, wr)`` where
    ``sus_flat`` indexes (dataset_id, batch, target_position) tuples.
    We average over batch × target_position within each dataset_id to
    get the per-probe susceptibility, and use the order in ``component_names``
    to align the ``wr`` axis.

    Returns array of shape ``(len(probe_names), len(component_names))``.
    """
    import xarray as xr  # type: ignore

    sus = sus_data["sus"]                         # (sus_flat, wr)
    if "dataset_id" not in sus_data.coords:
        # Fallback: assume sus_flat is already (probe, ...) ordered.
        flat = sus.values
        per_probe = flat.reshape(len(probe_names), -1, flat.shape[-1]).mean(axis=1)
        return per_probe[:, : len(component_names)]

    ds_ids = sus_data.coords["dataset_id"].values
    rows = []
    for pname in probe_names:
        # dataset_id encodes the probe; we map by index.
        mask = ds_ids == pname
        if mask.sum() == 0:
            # Fallback: numeric id.
            try:
                idx = probe_names.index(pname)
                mask = ds_ids == idx
            except ValueError:
                rows.append(np.zeros(len(component_names)))
                continue
        rows.append(np.asarray(sus.values[mask]).mean(axis=0))
    arr = np.asarray(rows)
    if arr.shape[1] != len(component_names):
        # Truncate/pad — devinterp orders 'wr' by the dict iteration order.
        arr = arr[:, : len(component_names)]
    return arr


def estimate_susceptibilities(
    repo: str,
    revision: str,
    *,
    output_root: Path,
    n_chains: int = 4,
    n_draws: int = 200,
    n_burnin: int = 100,
    batch_size: int = 8,
    n_per_probe: int = 256,
    smoke: bool = False,
) -> SusceptibilityCellSummary:
    """Run the susceptibility analysis for one (repo, revision) cell."""
    from devinterp.slt.susceptibilities import susceptibilities  # type: ignore

    log.info("[sus] %s @ %s", repo, revision)
    if smoke:
        n_chains, n_draws, n_burnin, n_per_probe = 1, 20, 10, 32

    model, tokenizer = load_model_and_tokenizer(repo, revision=revision)
    model = model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    hp = _scaled_hyperparams(n_params, batch_size)

    # Build observables and weight restrictions.
    sampling_task = build_sampling_task(tokenizer, n=1024, max_length=256)
    probes = build_probes(tokenizer, n_per_probe=n_per_probe, max_length=256)

    observables = {"sampling": (sampling_task, 2)}
    for pname in PROBE_NAMES:
        observables[pname] = (probes[pname].dataset, 2)

    components = enumerate_components(model)
    weight_restrictions = build_devinterp_masks(model, components)

    out_dir = output_root / "slt" / "susceptibilities" / repo.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = out_dir / f"{revision}.zarr"

    sus_tree = susceptibilities(
        model=model,
        dataset=sampling_task,
        observables=observables,
        weight_restrictions=weight_restrictions,
        sampling_task="sampling",
        lr=hp["lr"],
        n_beta=hp["n_beta"],
        num_chains=n_chains,
        num_draws=n_draws,
        num_burnin_steps=n_burnin,
        batch_size=batch_size,
        num_init_loss_batches=4,
        localization=hp["localization"],
        output_path=str(zarr_path),
    )

    # Pull out the response matrix and PCA it.
    sus_data = sus_tree["susceptibilities"].dataset

    component_names = [c.name for c in components]
    X = _aggregate_per_token_susceptibilities(
        sus_data, list(PROBE_NAMES), component_names
    )

    Xs = _standardize_columns(X)
    pcs, var_ratio, loadings = _pca(Xs, k=3)

    pc1_load_abs = np.abs(loadings[0])
    uniformity = float(1.0 - pc1_load_abs.std() / (pc1_load_abs.mean() + 1e-12))

    rank99 = _matrix_rank_99(Xs)

    np.savez(
        out_dir / f"{revision}_pca.npz",
        X=X, X_std=Xs,
        pcs=pcs, loadings=loadings,
        var_ratio=var_ratio,
        component_names=np.array(component_names),
        component_kinds=np.array([c.kind for c in components]),
        component_layers=np.array([c.layer if c.layer is not None else -1 for c in components]),
        component_heads=np.array([c.head if c.head is not None else -1 for c in components]),
        probe_names=np.array(list(PROBE_NAMES)),
    )

    regime, languages = _detect_regime(repo)

    return SusceptibilityCellSummary(
        repo=repo,
        revision=revision,
        regime=regime,
        languages=languages,
        n_components=len(components),
        n_probes=len(PROBE_NAMES),
        pc1_var_ratio=float(var_ratio[0]) if len(var_ratio) > 0 else 0.0,
        pc2_var_ratio=float(var_ratio[1]) if len(var_ratio) > 1 else 0.0,
        pc3_var_ratio=float(var_ratio[2]) if len(var_ratio) > 2 else 0.0,
        matrix_rank_99pct=rank99,
        pc1_loading_uniformity=uniformity,
    )


def append_summary(s: SusceptibilityCellSummary, output_root: Path) -> None:
    rank = int(os.environ.get("RANK", "0"))
    summary_dir = output_root / "slt"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"susceptibilities_summary.rank{rank}.csv"
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(s).keys()))
        if new:
            w.writeheader()
        w.writerow(asdict(s))


def run_stage(args) -> None:
    from slt.run_llc import _resolve_pool

    cells = _resolve_pool(args)
    rank = getattr(args, "rank", 0)
    world = max(getattr(args, "world_size", 1), 1)
    my_cells = cells[rank::world]
    log.info("[sus] rank %d/%d owns %d/%d cells", rank, world, len(my_cells), len(cells))

    for i, (repo, rev) in enumerate(my_cells, 1):
        try:
            s = estimate_susceptibilities(
                repo, rev, output_root=args.output_root,
                n_chains=getattr(args, "sus_num_chains", 4),
                n_draws=getattr(args, "sus_num_draws", 200),
                n_burnin=getattr(args, "sus_num_burnin", 100),
                batch_size=getattr(args, "sus_batch_size", 8),
                n_per_probe=getattr(args, "sus_n_per_probe", 256),
            )
            append_summary(s, args.output_root)
            log.info("[sus %d/%d] %s@%s  PC1=%.1f%%", i, len(my_cells), repo, rev,
                     s.pc1_var_ratio * 100)
        except Exception:  # pragma: no cover
            log.exception("[sus %d/%d] failed: %s @ %s", i, len(my_cells), repo, rev)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Component-localised susceptibilities.")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--output-root", type=Path, default=Path("results"))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    s = estimate_susceptibilities(args.model, args.revision,
                                  output_root=args.output_root, smoke=args.smoke)
    append_summary(s, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
