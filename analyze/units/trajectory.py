"""
analyze/units/trajectory.py — Developmental interpretability of syntactic units.

This is what turns the (static) localize→ablate→overlap replication into a
*developmental* interpretability tool, and the reason the package lives inside
``beetle-analyze``: it re-runs functional localization at every training
checkpoint of a Beetle model and traces how the units emerge, consolidate, and
specialize over the course of learning.

For each checkpoint returned by ``analyze.convergence.list_checkpoints`` it can
record, per phenomenon:

  * ``n_units``            — size of the localized mask;
  * ``cv_stability``       — cross-validation reliability of the mask
    (``overlap.cross_validation_stability``);
  * ``mean_cross_overlap`` — mean pairwise overlap with the *other* phenomena's
    masks, i.e. the developmental trajectory of the paper's "same units" effect;
  * ``persistence``        — Jaccard overlap of a phenomenon's mask with its own
    mask at the previous checkpoint (how stable the localizer is over training);
  * optionally ``ablation_delta`` — the causal accuracy drop from knocking the
    mask out, per checkpoint.

Everything is emitted as one tidy long-format ``DataFrame`` (``step``,
``phenomenon``, ``category``, metric columns) written via
``write_csv_atomic``.

CLI
---
    python -m analyze.units.trajectory \\
        --repo Beetle-FineWeb-2B/beetle-bilingual-balanced-b1-fineweb-2b-nld-eng \\
        --ckpt-scope all --log-spaced \\
        --categories subject_verb_agreement anaphor_agreement determiner_noun_agreement \\
        --with-ablation --out results/units/trajectory.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .stimuli import MinimalPairSet
from .localize import localize, SELECTIVITY_METHODS, SITES
from .overlap import jaccard, cross_validation_stability
from .ablate import ablation_effect


def _step_of(revision: str) -> int:
    from analyze.convergence import step_label
    return step_label(revision)


def localize_trajectory(
    repo: str,
    pair_sets: Sequence[MinimalPairSet],
    *,
    checkpoints: Optional[Sequence[str]] = None,
    all_checkpoints: bool = False,
    log_spaced: bool = False,
    top_frac: float = 0.01,
    method: str = "ttest",
    site: str = "resid",
    with_cv: bool = True,
    with_cross_overlap: bool = True,
    with_ablation: bool = False,
    ablation_mode: str = "zero",
    hf_token: Optional[str] = None,
    device: Optional[str] = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Trace localized syntactic units across ``repo``'s training checkpoints.

    Returns a long-format DataFrame with one row per (checkpoint, phenomenon).
    """
    from analyze.convergence import list_checkpoints, _load_checkpoint, _free
    from analyze._interp_helpers import device_for

    dev = device or device_for()
    revs = list(checkpoints) if checkpoints is not None else list_checkpoints(
        repo, hf_token, all_checkpoints=all_checkpoints, log_spaced=log_spaced)

    prev_masks: Dict[str, object] = {}
    rows: List[dict] = []

    iterator = revs
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(revs, desc=f"units-traj[{repo.split('/')[-1]}]", unit="ckpt")
        except Exception:
            pass

    for rev in iterator:
        loaded = _load_checkpoint(repo, rev, hf_token, device=dev)
        if loaded is None:
            continue
        model, tok = loaded
        try:
            masks = {s.phenomenon: localize(model, tok, s, top_frac=top_frac,
                                            method=method, site=site)
                     for s in pair_sets}
            for s in pair_sets:
                m = masks[s.phenomenon]
                others = [mm for name, mm in masks.items() if name != s.phenomenon]
                row = {
                    "repo": repo,
                    "revision": rev,
                    "step": _step_of(rev),
                    "phenomenon": s.phenomenon,
                    "category": s.category,
                    "n_units": m.count(),
                    "total_units": m.total_units(),
                }
                if with_cross_overlap and others:
                    # Mean overlap of THIS phenomenon's mask with every other
                    # phenomenon's mask — the developmental "same units" signal.
                    row["mean_cross_overlap"] = float(
                        np.mean([jaccard(m, o) for o in others]))
                if s.phenomenon in prev_masks:
                    row["persistence"] = jaccard(m, prev_masks[s.phenomenon])
                if with_cv and len(s) >= 2 * 2:
                    cv = cross_validation_stability(model, tok, s, top_frac=top_frac,
                                                    method=method, site=site,
                                                    n_folds=min(5, len(s)))
                    row["cv_stability"] = cv["mean"]
                if with_ablation:
                    eff = ablation_effect(model, tok, s, m, mode=ablation_mode)
                    row["ablation_delta"] = eff["delta"]
                    row["baseline_acc"] = eff["baseline"]
                    row["ablated_acc"] = eff["ablated"]
                rows.append(row)
            prev_masks = masks
        finally:
            _free(model, tok)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Iterable[str]] = None) -> None:
    from analyze._interp_helpers import write_csv_atomic
    from . import stimuli as _stim

    p = argparse.ArgumentParser(
        description="Developmental trajectory of localized syntactic units.")
    p.add_argument("--repo", required=True)
    p.add_argument("--ckpt-scope", choices=["final", "all"], default="all")
    p.add_argument("--log-spaced", action="store_true")
    p.add_argument("--revisions", nargs="*", default=None,
                   help="Explicit revision list (overrides --ckpt-scope).")
    p.add_argument("--categories", nargs="*", default=list(_stim.AGREEMENT_CATEGORIES))
    p.add_argument("--max-pairs", type=int, default=150)
    p.add_argument("--top-frac", type=float, default=0.01)
    p.add_argument("--method", default="ttest", choices=SELECTIVITY_METHODS)
    p.add_argument("--site", default="resid", choices=SITES)
    p.add_argument("--with-ablation", action="store_true")
    p.add_argument("--ablation-mode", default="zero", choices=["zero", "mean"])
    p.add_argument("--no-cv", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--out", default="results/units/trajectory.csv")
    args = p.parse_args(list(argv) if argv is not None else None)

    sets = (_stim.demo_pairs() if args.demo else
            _stim.load_blimp(categories=args.categories, max_pairs=args.max_pairs))

    df = localize_trajectory(
        args.repo, sets,
        checkpoints=args.revisions,
        all_checkpoints=(args.ckpt_scope == "all"),
        log_spaced=args.log_spaced,
        top_frac=args.top_frac, method=args.method, site=args.site,
        with_cv=not args.no_cv, with_ablation=args.with_ablation,
        ablation_mode=args.ablation_mode,
    )
    write_csv_atomic(df, args.out)
    print(df.to_string(index=False))
    print(f"[units.trajectory] wrote {len(df)} rows → {args.out}")


if __name__ == "__main__":
    main()
