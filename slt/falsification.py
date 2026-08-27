"""A2 — Falsification axis.

Two controls that distinguish "λ tracks shared structure" from "λ tracks
any artefact":

  A2a  Vocabulary-disjoint bilingual: estimate λ on a bilingual checkpoint
       whose vocabulary is *partitioned* across languages (no shared
       token IDs). If λ_bi-disjoint < λ_mono, then sharing must live in
       the body weights — the geometry signal cannot be reduced to vocab
       overlap.

  A2b  Token-permutation control: take a multilingual checkpoint and
       apply a fixed token-ID permutation to one language at evaluation.
       The model now treats permuted-Dutch as a different surface form.
       Predict λ rises (geometry was *real*, not a labelling artefact).

Outputs
-------
  ``<output_root>/slt/falsification/<repo>/<rev>__a2a.csv``   λ_bi-disjoint
  ``<output_root>/slt/falsification/<repo>/<rev>__a2b.csv``   λ before/after permutation
  ``<output_root>/slt/falsification_summary.csv``             flat results
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
import torch
from datasets import Dataset

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS))

from analyze.ppl_utils import load_model_and_tokenizer  # noqa: E402

from slt.probe_data import build_sampling_task, per_language_dataset  # noqa: E402
from slt.run_llc import _scaled_hyperparams, _detect_regime  # noqa: E402

log = logging.getLogger(__name__)


def _llc_on(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    n_chains: int,
    n_draws: int,
    n_burnin: int,
    hp: dict,
) -> tuple[float, float]:
    """One-shot LLC estimate on an in-memory dataset."""
    from devinterp.slt.llc import llc  # type: ignore

    res = llc(
        model=model,
        dataset=dataset,
        observables={"train": dataset},
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
    return float(res["llc_mean"].item()), float(res["llc_std"].item())


def _permute_token_ids(ds: Dataset, perm: np.ndarray) -> Dataset:
    """Apply a fixed token permutation to ``input_ids``.

    Tokens whose ID is out of bounds are left as-is (they may be special
    tokens like ``pad``).
    """
    n = perm.shape[0]
    arr = np.asarray(ds["input_ids"])
    masked = np.where(arr < n, perm[np.clip(arr, 0, n - 1)], arr)
    return Dataset.from_dict({"input_ids": masked.tolist()})


def run_a2a_disjoint(
    repo: str,
    revision: str,
    *,
    output_root: Path,
    batch_size: int = 8,
    n_chains: int = 4,
    n_draws: int = 200,
    n_burnin: int = 100,
    smoke: bool = False,
) -> dict:
    """A2a: λ on a vocab-disjoint bilingual checkpoint.

    Caller provides a repo whose tokenizer partitions IDs by language.
    We estimate λ on the multilingual sampling task as for the regular
    LLC stage; the falsification reading is downstream
    (compare to the matched mono baseline).
    """
    if smoke:
        n_chains, n_draws, n_burnin = 1, 20, 10

    model, tokenizer = load_model_and_tokenizer(repo, revision=revision)
    model = model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    hp = _scaled_hyperparams(n_params, batch_size)

    ds = build_sampling_task(tokenizer, n=512 if smoke else 1024, max_length=256)
    mean, std = _llc_on(
        model, ds, batch_size=batch_size,
        n_chains=n_chains, n_draws=n_draws, n_burnin=n_burnin, hp=hp,
    )
    regime, languages = _detect_regime(repo)
    return {
        "kind": "a2a_disjoint",
        "repo": repo,
        "revision": revision,
        "regime": regime,
        "languages": languages,
        "llc_mean": mean,
        "llc_std": std,
        "perm_seed": "",
        "permuted_lang": "",
    }


def run_a2b_permuted(
    repo: str,
    revision: str,
    *,
    output_root: Path,
    permuted_lang: str = "nld",
    perm_seed: int = 1337,
    batch_size: int = 8,
    n_chains: int = 4,
    n_draws: int = 200,
    n_burnin: int = 100,
    smoke: bool = False,
) -> list[dict]:
    """A2b: estimate λ on (a) the natural mixture and (b) the same mixture
    with a fixed token permutation applied to ``permuted_lang``.

    Returns two rows: ``before`` and ``after``.
    """
    if smoke:
        n_chains, n_draws, n_burnin = 1, 20, 10

    model, tokenizer = load_model_and_tokenizer(repo, revision=revision)
    model = model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    hp = _scaled_hyperparams(n_params, batch_size)

    natural = build_sampling_task(tokenizer, n=512 if smoke else 1024, max_length=256)

    # Build a permuted variant by replacing the "permuted_lang" subset.
    permuted_subset = per_language_dataset(
        tokenizer, permuted_lang, n=128 if smoke else 256, max_length=256
    )
    rng = np.random.default_rng(perm_seed)
    vocab_size = int(getattr(tokenizer, "vocab_size", 32000))
    perm = rng.permutation(vocab_size)
    permuted_subset = _permute_token_ids(permuted_subset, perm)

    # Replace the first |permuted_subset| rows of the natural mixture.
    n_swap = len(permuted_subset)
    ids_before = list(natural["input_ids"])
    ids_after = list(natural["input_ids"])
    ids_after[:n_swap] = list(permuted_subset["input_ids"])
    after = Dataset.from_dict({"input_ids": ids_after})
    before = Dataset.from_dict({"input_ids": ids_before})

    rows: list[dict] = []
    for label, ds in (("before", before), ("after", after)):
        mean, std = _llc_on(
            model, ds, batch_size=batch_size,
            n_chains=n_chains, n_draws=n_draws, n_burnin=n_burnin, hp=hp,
        )
        regime, languages = _detect_regime(repo)
        rows.append({
            "kind": f"a2b_permuted_{label}",
            "repo": repo,
            "revision": revision,
            "regime": regime,
            "languages": languages,
            "llc_mean": mean,
            "llc_std": std,
            "perm_seed": perm_seed,
            "permuted_lang": permuted_lang,
        })
    return rows


def append_summary(rows: list[dict], output_root: Path) -> None:
    rank = int(os.environ.get("RANK", "0"))
    summary_dir = output_root / "slt"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"falsification_summary.rank{rank}.csv"
    new = not path.exists()
    if not rows:
        return
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def run_stage(args) -> None:
    """Stage entry point.

    Reads ``args.falsification_a2a_repos`` and ``args.falsification_a2b_repos``
    (lists of repos) — both default to empty.
    """
    rank = int(getattr(args, "rank", 0))
    world = max(int(getattr(args, "world_size", 1)), 1)

    a2a = list(getattr(args, "falsification_a2a_repos", []) or [])
    a2b = list(getattr(args, "falsification_a2b_repos", []) or [])
    cells = (
        [("a2a", r, getattr(args, "revision", "main")) for r in a2a]
        + [("a2b", r, getattr(args, "revision", "main")) for r in a2b]
    )
    my_cells = cells[rank::world]
    log.info("[falsification] rank %d/%d owns %d/%d cells",
             rank, world, len(my_cells), len(cells))

    for kind, repo, rev in my_cells:
        try:
            if kind == "a2a":
                row = run_a2a_disjoint(repo, rev, output_root=args.output_root)
                append_summary([row], args.output_root)
            else:
                rows = run_a2b_permuted(repo, rev, output_root=args.output_root)
                append_summary(rows, args.output_root)
        except Exception:  # pragma: no cover
            log.exception("[falsification] failed: %s @ %s", repo, rev)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="A2 falsification axis.")
    p.add_argument("--mode", choices=["a2a", "a2b"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--output-root", type=Path, default=Path("results"))
    p.add_argument("--permuted-lang", default="nld")
    p.add_argument("--perm-seed", type=int, default=1337)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.mode == "a2a":
        row = run_a2a_disjoint(args.model, args.revision,
                               output_root=args.output_root, smoke=args.smoke)
        append_summary([row], args.output_root)
    else:
        rows = run_a2b_permuted(args.model, args.revision,
                                output_root=args.output_root,
                                permuted_lang=args.permuted_lang,
                                perm_seed=args.perm_seed,
                                smoke=args.smoke)
        append_summary(rows, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
