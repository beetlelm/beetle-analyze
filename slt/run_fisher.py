"""Block-diagonal Fisher spectroscopy with Lanczos top-k eigenvalues.

The full empirical FIM is intractable at 100M+ params. We compute the
per-layer block diagonal:

    F_l ≈ (1/N) Σ_i  ∇_{w_l} log p(y_i | x_i, w)  ∇_{w_l} log p(...)^T

then extract the top-k eigenvalues via Lanczos on a matrix-vector product
that uses Fisher-vector products. Reports:

  - top-k eigenspectrum per layer
  - effective rank r_eff = (Σ λ)^2 / Σ λ^2
  - null-space dimension (eigenvalues < 1e-6 * λ_max)

Memory cost: O(N · |w_l|) for the gradient buffer per layer; we stream
N=8k tokens through the model and accumulate the layer-block Fisher
implicitly (using gradient outer products on the fly).
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
from torch.utils.data import DataLoader

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS))

from analyze.ppl_utils import load_model_and_tokenizer  # noqa: E402

from slt.probe_data import build_sampling_task  # noqa: E402

log = logging.getLogger(__name__)


@dataclass
class FisherLayerStats:
    repo: str
    revision: str
    layer_name: str
    layer_idx: int
    n_params: int
    top_k: int
    eigval_max: float
    eigval_min_topk: float
    effective_rank: float
    null_space_dim: int


def _layer_param_groups(model: torch.nn.Module) -> list[tuple[str, list[torch.nn.Parameter]]]:
    """Group parameters by transformer layer.

    Returns ``[(label, [params...]), ...]`` covering every layer plus
    embed / unembed / final-norm.
    """
    groups: dict[str, list[torch.nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Layer index extraction
        if ".layers." in name:
            label = name.split(".layers.")[1].split(".", 1)[0]
            label = f"layer_{int(label):02d}"
        elif "embed_tokens" in name or "wte" in name:
            label = "embed"
        elif "lm_head" in name:
            label = "unembed"
        elif "norm" in name and "layers" not in name:
            label = "final_norm"
        else:
            label = "other"
        groups.setdefault(label, []).append(p)
    return [(k, v) for k, v in sorted(groups.items())]


def _flatten_grads(params: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.grad.detach().reshape(-1) for p in params if p.grad is not None])


def _accumulate_fisher_block(
    model: torch.nn.Module,
    params: list[torch.nn.Parameter],
    loader: DataLoader,
    device: torch.device,
    *,
    max_samples: int,
) -> torch.Tensor:
    """Accumulate a block-diagonal Fisher matrix for a single layer.

    Computes Σ g g^T / N where g is the per-sample gradient over only the
    parameters in ``params``. For large blocks (>8k) we do not store F
    densely; instead we expose Lanczos via a closure (see below).

    For now: dense for blocks ≤ 8k params; raise for larger blocks.
    """
    n = sum(p.numel() for p in params)
    if n > 8192:
        raise ValueError(
            f"Block too large for dense accumulation ({n} params); "
            "use Lanczos closure path instead."
        )
    F = torch.zeros((n, n), device=device, dtype=torch.float32)
    seen = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        for s in range(ids.shape[0]):
            model.zero_grad(set_to_none=True)
            out = model(ids[s : s + 1], labels=ids[s : s + 1])
            out.loss.backward()
            g = _flatten_grads(params).to(torch.float32)
            F.addr_(g, g)
            seen += 1
            if seen >= max_samples:
                break
        if seen >= max_samples:
            break
    return F / max(seen, 1)


def _lanczos_topk(
    fvp: callable,
    dim: int,
    k: int,
    *,
    n_iter: int | None = None,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> np.ndarray:
    """Lanczos top-k eigenvalues of a symmetric PSD operator ``fvp``.

    Parameters
    ----------
    fvp : callable
        A function taking a torch tensor of shape ``(dim,)`` and returning
        ``F @ v`` of the same shape.
    dim : int
        Ambient dimension.
    k : int
        Number of eigenvalues to return.
    n_iter : int
        Number of Lanczos iterations (defaults to ``2*k + 20``).
    """
    n_iter = n_iter or (2 * k + 20)
    n_iter = min(n_iter, dim)
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(dim, generator=g, device=device, dtype=dtype)
    q /= q.norm()
    Q = [q]
    alphas: list[float] = []
    betas: list[float] = []
    prev_q = torch.zeros_like(q)
    prev_beta = 0.0
    for _ in range(n_iter):
        v = fvp(Q[-1])
        a = float(v @ Q[-1])
        v = v - a * Q[-1] - prev_beta * prev_q
        # Reorthogonalise against all previous q's (numerical stability).
        for qi in Q:
            v = v - (v @ qi) * qi
        beta = float(v.norm())
        alphas.append(a)
        if beta < 1e-12:
            break
        betas.append(beta)
        prev_q = Q[-1]
        prev_beta = beta
        Q.append(v / beta)
    T = np.diag(alphas)
    for i, b in enumerate(betas):
        T[i, i + 1] = T[i + 1, i] = b
    eigs = np.linalg.eigvalsh(T)
    return np.sort(eigs)[::-1][:k]


def _fisher_lanczos_block(
    model: torch.nn.Module,
    params: list[torch.nn.Parameter],
    loader: DataLoader,
    device: torch.device,
    *,
    max_samples: int,
    k: int,
) -> np.ndarray:
    """Lanczos top-k eigenvalues of the empirical block Fisher.

    The Fisher-vector product is computed implicitly:
        F v ≈ (1/N) Σ_i (g_i^T v) g_i
    so we only need O(|w_l|) memory for ``v``, not O(|w_l|^2) for ``F``.
    """
    grads_buffer: list[torch.Tensor] = []
    seen = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        for s in range(ids.shape[0]):
            model.zero_grad(set_to_none=True)
            out = model(ids[s : s + 1], labels=ids[s : s + 1])
            out.loss.backward()
            g = _flatten_grads(params).to(torch.float32).cpu()
            grads_buffer.append(g)
            seen += 1
            if seen >= max_samples:
                break
        if seen >= max_samples:
            break
    G = torch.stack(grads_buffer)  # (N, dim)
    G = G.to(device)
    N, dim = G.shape

    def fvp(v: torch.Tensor) -> torch.Tensor:
        return (G.T @ (G @ v)) / N

    return _lanczos_topk(fvp, dim, k, device=device)


def estimate_fisher(
    repo: str,
    revision: str,
    *,
    output_root: Path,
    n_samples: int = 8192,
    k: int = 128,
    batch_size: int = 4,
    smoke: bool = False,
) -> list[FisherLayerStats]:
    """Block-diagonal Fisher spectroscopy for one (repo, revision) cell."""
    log.info("[fisher] %s @ %s", repo, revision)
    if smoke:
        n_samples = 64
        k = 16

    model, tokenizer = load_model_and_tokenizer(repo, revision=revision)
    model = model.eval()
    device = next(model.parameters()).device

    ds = build_sampling_task(tokenizer, n=n_samples, max_length=128)
    loader = DataLoader(ds.with_format("torch"), batch_size=batch_size, shuffle=False)

    out_dir = output_root / "slt" / "fisher" / repo.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: list[FisherLayerStats] = []
    spectra: dict[str, np.ndarray] = {}

    for label, params in _layer_param_groups(model):
        n = sum(p.numel() for p in params)
        if n == 0:
            continue
        log.info("  layer=%s n_params=%d", label, n)
        try:
            eigs = _fisher_lanczos_block(
                model, params, loader, device,
                max_samples=n_samples, k=k,
            )
        except Exception:  # pragma: no cover
            log.exception("    layer %s failed; skipping", label)
            continue
        eigs = np.clip(eigs, 0.0, None)
        spectra[label] = eigs

        s1 = float(eigs.sum())
        s2 = float((eigs ** 2).sum())
        r_eff = (s1 * s1) / s2 if s2 > 0 else 0.0
        eigval_max = float(eigs[0]) if len(eigs) else 0.0
        null_dim = int((eigs < 1e-6 * max(eigval_max, 1e-12)).sum())
        layer_idx = int(label.split("_")[-1]) if label.startswith("layer_") else -1

        stats.append(FisherLayerStats(
            repo=repo,
            revision=revision,
            layer_name=label,
            layer_idx=layer_idx,
            n_params=n,
            top_k=k,
            eigval_max=eigval_max,
            eigval_min_topk=float(eigs[-1]) if len(eigs) else 0.0,
            effective_rank=r_eff,
            null_space_dim=null_dim,
        ))

    np.savez(out_dir / f"{revision}.npz", **spectra)
    return stats


def append_summary(stats: list[FisherLayerStats], output_root: Path) -> None:
    rank = int(os.environ.get("RANK", "0"))
    summary_dir = output_root / "slt"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"fisher_summary.rank{rank}.csv"
    new = not path.exists()
    if not stats:
        return
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(stats[0]).keys()))
        if new:
            w.writeheader()
        for s in stats:
            w.writerow(asdict(s))


def run_stage(args) -> None:
    """Stage entry point invoked from ``analyze.run.main``."""
    from slt.run_llc import _resolve_pool

    cells = _resolve_pool(args)
    rank = getattr(args, "rank", 0)
    world = max(getattr(args, "world_size", 1), 1)
    my_cells = cells[rank::world]
    log.info("[fisher] rank %d/%d owns %d/%d cells", rank, world, len(my_cells), len(cells))

    for i, (repo, rev) in enumerate(my_cells, 1):
        try:
            stats = estimate_fisher(repo, rev, output_root=args.output_root,
                                    n_samples=getattr(args, "fisher_samples", 8192),
                                    k=getattr(args, "fisher_topk", 128))
            append_summary(stats, args.output_root)
            log.info("[fisher %d/%d] %s@%s  layers=%d", i, len(my_cells), repo, rev, len(stats))
        except Exception:  # pragma: no cover
            log.exception("[fisher %d/%d] failed: %s @ %s", i, len(my_cells), repo, rev)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Block-diagonal Fisher spectroscopy.")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--output-root", type=Path, default=Path("results"))
    p.add_argument("--n-samples", type=int, default=8192)
    p.add_argument("--top-k", type=int, default=128)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    stats = estimate_fisher(args.model, args.revision,
                            output_root=args.output_root,
                            n_samples=args.n_samples, k=args.top_k,
                            smoke=args.smoke)
    append_summary(stats, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
