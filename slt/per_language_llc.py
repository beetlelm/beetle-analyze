"""S3 — Per-language LLC decomposition.

For each multilingual checkpoint, estimate λ̂ separately on per-language
FLORES batches. A "shared geometry" model has tightly clustered
λ_per_lang; a "language-specialised" multilingual model has high
variance.

Outputs
-------
  ``<output_root>/slt/per_language_llc/<repo>/<rev>.csv``  one row per language
  ``<output_root>/slt/per_language_llc_summary.rank{R}.csv``  flat ledger
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS))

from analyze.ppl_utils import load_model_and_tokenizer  # noqa: E402

from slt.probe_data import per_language_dataset, _supported_flores_langs  # noqa: E402
from slt.run_llc import _scaled_hyperparams, _detect_regime  # noqa: E402

log = logging.getLogger(__name__)


def estimate_per_language(
    repo: str,
    revision: str,
    *,
    output_root: Path,
    n_per_lang: int = 512,
    max_length: int = 256,
    batch_size: int = 8,
    n_chains: int = 4,
    n_draws: int = 200,
    n_burnin: int = 100,
    smoke: bool = False,
) -> list[dict]:
    """Estimate λ̂ per language for one (repo, revision) cell."""
    from devinterp.slt.llc import llc  # type: ignore

    if smoke:
        n_chains, n_draws, n_burnin, n_per_lang = 1, 20, 10, 64

    model, tokenizer = load_model_and_tokenizer(repo, revision=revision)
    model = model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    hp = _scaled_hyperparams(n_params, batch_size)

    langs = _supported_flores_langs(tokenizer)
    log.info("[per-lang LLC] %s @ %s — langs=%s", repo, revision, langs)

    out_dir = output_root / "slt" / "per_language_llc" / repo.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    regime, languages = _detect_regime(repo)
    for iso in langs:
        try:
            ds = per_language_dataset(tokenizer, iso, n=n_per_lang, max_length=max_length)
            res = llc(
                model=model,
                dataset=ds,
                observables={"train": ds},
                lr=hp["lr"],
                n_beta=hp["n_beta"],
                num_chains=n_chains,
                num_draws=n_draws,
                num_burnin_steps=n_burnin,
                batch_size=batch_size,
                num_init_loss_batches=4,
                localization=hp["localization"],
                save_metrics=False,
            )
            rows.append({
                "repo": repo,
                "revision": revision,
                "regime": regime,
                "languages": languages,
                "language": iso,
                "n_per_lang": n_per_lang,
                "llc_mean": float(res["llc_mean"].item()),
                "llc_std": float(res["llc_std"].item()),
            })
        except Exception:  # pragma: no cover
            log.exception("[per-lang LLC] failed for %s/%s", repo, iso)

    cell_csv = out_dir / f"{revision}.csv"
    if rows:
        with cell_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    return rows


def append_summary(rows: list[dict], output_root: Path) -> None:
    rank = int(os.environ.get("RANK", "0"))
    summary_dir = output_root / "slt"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"per_language_llc_summary.rank{rank}.csv"
    if not rows:
        return
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def run_stage(args) -> None:
    from slt.run_llc import _resolve_pool

    cells = _resolve_pool(args)
    rank = getattr(args, "rank", 0)
    world = max(getattr(args, "world_size", 1), 1)
    my_cells = cells[rank::world]
    log.info("[per-lang LLC] rank %d/%d owns %d/%d cells",
             rank, world, len(my_cells), len(cells))

    for i, (repo, rev) in enumerate(my_cells, 1):
        try:
            rows = estimate_per_language(
                repo, rev, output_root=args.output_root,
                n_chains=getattr(args, "llc_num_chains", 4),
                n_draws=getattr(args, "llc_num_draws", 200),
                n_burnin=getattr(args, "llc_num_burnin", 100),
            )
            append_summary(rows, args.output_root)
            log.info("[per-lang LLC %d/%d] %s@%s  langs=%d",
                     i, len(my_cells), repo, rev, len(rows))
        except Exception:  # pragma: no cover
            log.exception("[per-lang LLC %d/%d] failed: %s @ %s",
                          i, len(my_cells), repo, rev)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Per-language LLC decomposition (S3).")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--output-root", type=Path, default=Path("results"))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rows = estimate_per_language(args.model, args.revision,
                                 output_root=args.output_root, smoke=args.smoke)
    append_summary(rows, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
