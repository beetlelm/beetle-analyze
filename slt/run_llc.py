"""Local Learning Coefficient (LLC) estimation for the SLT sweep.

For each (model, checkpoint) cell, this module:
  1. Loads the model + tokenizer at the requested revision.
  2. Builds a multilingual sampling-task dataset.
  3. Runs ``devinterp.slt.llc.llc`` with SGLD.
  4. Persists the result to ``<output_root>/slt/llc/<repo>/<rev>.zarr``
     and a flat row to ``<output_root>/slt/llc_summary.csv``.

Hyperparameters are taken from the plan (see refactored-puzzling-matsumoto.md):
  ``batch_size=8``, ``lr=1e-4``, ``localization=100``, ``num_chains=4``,
  ``num_draws=200``, ``num_burnin_steps=100``.

Larger Beetle-FineWeb-2B cells override to ``lr=5e-5``,
``localization=200``.
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

import torch

# Path bootstrapping for analyze.* imports.
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS))

from analyze.ppl_utils import load_model_and_tokenizer  # noqa: E402
from analyze.model_naming import parse_repo  # noqa: E402

from slt.probe_data import build_sampling_task  # noqa: E402

log = logging.getLogger(__name__)


@dataclass
class LLCResult:
    repo: str
    revision: str
    family: str
    regime: str          # "mono" | "bi" | "multi"
    languages: str       # "nld,eng" etc.
    n_params: int
    init_loss: float
    llc_mean: float
    llc_std: float
    n_chains: int
    n_draws: int
    n_beta: float
    lr: float
    localization: float


def _detect_regime(repo: str) -> tuple[str, str]:
    """Map a repo name to ``(regime, languages)``.

    Heuristic that consults ``parse_repo`` first; falls back to
    substring matching.
    """
    try:
        meta = parse_repo(repo)
        l1 = getattr(meta, "l1", None)
        l2 = getattr(meta, "l2", None)
        if l1 and l2 and l1 != l2:
            return "bi", f"{l1},{l2}"
        if l1 and not l2:
            return "mono", l1
    except Exception:  # pragma: no cover
        pass
    rl = repo.lower()
    if "fineweb" in rl:
        return "multi", "fineweb_mix"
    if "_mono" in rl:
        return "mono", rl.split("_mono")[0].split("/")[-1].split("_")[-1]
    if "-eng" in rl or "-nld" in rl or "-deu" in rl:
        return "bi", "unknown_pair"
    if "pico" in rl:
        return "mono", "eng"
    return "unknown", "unknown"


def _scaled_hyperparams(n_params: int, batch_size: int) -> dict[str, float]:
    """Return ``(lr, localization)`` scaled by parameter count.

    Defaults from the plan:
      ``< 500M``  : lr=1e-4, gamma=100
      ``500M-2B`` : lr=5e-5, gamma=200
      ``> 2B``    : lr=2e-5, gamma=400
    """
    from devinterp.utils import default_nbeta  # type: ignore

    n_beta = float(default_nbeta(batch_size))
    if n_params < 500_000_000:
        return {"lr": 1e-4, "localization": 100.0, "n_beta": n_beta}
    if n_params < 2_000_000_000:
        return {"lr": 5e-5, "localization": 200.0, "n_beta": n_beta}
    return {"lr": 2e-5, "localization": 400.0, "n_beta": n_beta}


def estimate_llc(
    repo: str,
    revision: str,
    *,
    output_root: Path,
    batch_size: int = 8,
    num_chains: int = 4,
    num_draws: int = 200,
    num_burnin_steps: int = 100,
    smoke: bool = False,
) -> LLCResult:
    """Estimate the LLC for one (repo, revision) cell."""
    from devinterp.slt.llc import llc  # type: ignore

    log.info("[llc] %s @ %s", repo, revision)
    model, tokenizer = load_model_and_tokenizer(repo, revision=revision)
    model = model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    hp = _scaled_hyperparams(n_params, batch_size)

    if smoke:
        num_chains, num_draws, num_burnin_steps = 1, 20, 10

    ds = build_sampling_task(tokenizer, n=1024, max_length=256)

    out_dir = output_root / "slt" / "llc" / repo.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = out_dir / f"{revision}.zarr"

    result = llc(
        model=model,
        dataset=ds,
        observables={"train": ds},
        lr=hp["lr"],
        n_beta=hp["n_beta"],
        num_chains=num_chains,
        num_draws=num_draws,
        num_burnin_steps=num_burnin_steps,
        batch_size=batch_size,
        num_init_loss_batches=4,
        localization=hp["localization"],
        output_path=str(zarr_path),
        save_metrics=False,
    )

    regime, languages = _detect_regime(repo)
    family = repo.split("/")[0]

    return LLCResult(
        repo=repo,
        revision=revision,
        family=family,
        regime=regime,
        languages=languages,
        n_params=n_params,
        init_loss=float(result["init_loss"].item()),
        llc_mean=float(result["llc_mean"].item()),
        llc_std=float(result["llc_std"].item()),
        n_chains=num_chains,
        n_draws=num_draws,
        n_beta=hp["n_beta"],
        lr=hp["lr"],
        localization=hp["localization"],
    )


def append_summary(result: LLCResult, output_root: Path) -> None:
    """Append a single LLC row to ``slt/llc_summary.csv``.

    Writes the header on first creation. Safe to call concurrently from
    multiple ranks: each rank writes a per-rank shard, merged downstream
    by ``slt.aggregate``.
    """
    rank = int(os.environ.get("RANK", "0"))
    summary_dir = output_root / "slt"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"llc_summary.rank{rank}.csv"
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(result).keys()))
        if new:
            w.writeheader()
        w.writerow(asdict(result))


def _resolve_pool(args) -> list[tuple[str, str]]:
    """Reuse beetle-analyze's repo resolver to get a list of cells.

    Returns ``[(repo, revision), ...]``. Driven by ``--family`` /
    ``--orgs`` / ``--ckpt-scope`` from the parent ``analyze.run`` flags.
    """
    from analyze.run import _resolve_repos  # type: ignore
    from analyze.convergence import list_checkpoints  # type: ignore

    repos = _resolve_repos(args)
    cells: list[tuple[str, str]] = []
    if getattr(args, "all_checkpoints", False):
        for r in repos:
            try:
                revs = list_checkpoints(r)
            except Exception:
                revs = ["main"]
            cells.extend((r, rv) for rv in revs)
    else:
        cells.extend((r, "main") for r in repos)
    return cells


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="LLC estimation per (model, checkpoint).")
    p.add_argument("--model", required=False, help="Single repo (overrides --family).")
    p.add_argument("--revision", default="main")
    p.add_argument("--output-root", type=Path, default=Path("results"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-chains", type=int, default=4)
    p.add_argument("--num-draws", type=int, default=200)
    p.add_argument("--num-burnin-steps", type=int, default=100)
    p.add_argument("--smoke", action="store_true",
                   help="1 chain, 20 draws — for sanity checking the schema only.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.model:
        log.error("No --model provided; use --model or invoke from analyze.run.")
        return 1

    res = estimate_llc(
        args.model,
        args.revision,
        output_root=args.output_root,
        batch_size=args.batch_size,
        num_chains=args.num_chains,
        num_draws=args.num_draws,
        num_burnin_steps=args.num_burnin_steps,
        smoke=args.smoke,
    )
    append_summary(res, args.output_root)
    log.info("[llc] %s@%s  λ̂=%.3f ± %.3f", res.repo, res.revision, res.llc_mean, res.llc_std)
    return 0


def run_stage(args) -> None:
    """Stage entry point invoked from ``analyze.run.main``.

    Iterates over the rank's share of (repo, revision) cells.
    """
    cells = _resolve_pool(args)
    rank = getattr(args, "rank", 0)
    world = max(getattr(args, "world_size", 1), 1)
    my_cells = cells[rank::world]
    log.info("[llc] rank %d/%d owns %d/%d cells", rank, world, len(my_cells), len(cells))

    for i, (repo, rev) in enumerate(my_cells, 1):
        try:
            res = estimate_llc(repo, rev, output_root=args.output_root,
                               batch_size=getattr(args, "llc_batch_size", 8),
                               num_chains=getattr(args, "llc_num_chains", 4),
                               num_draws=getattr(args, "llc_num_draws", 200),
                               num_burnin_steps=getattr(args, "llc_num_burnin", 100))
            append_summary(res, args.output_root)
            log.info("[llc %d/%d] %s@%s  λ̂=%.3f", i, len(my_cells), repo, rev, res.llc_mean)
        except Exception:  # pragma: no cover
            log.exception("[llc %d/%d] failed: %s @ %s", i, len(my_cells), repo, rev)


if __name__ == "__main__":
    raise SystemExit(main())
