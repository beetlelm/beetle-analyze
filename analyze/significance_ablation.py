"""
significance_ablation.py — MECO-primary within-family significance selection.

For a multi-seed ablation sweep over B1 / B2 / B3 families produced by
``beetlelm/script/ablation_minimal_nld_eng.sh``, this driver:

1. Loads a manifest CSV of (variant, seed, config_path, log_path,
   hf_push_suffix) rows.
2. For each (variant, seed) pair, obtains a scalar MECO alignment score
   (Spearman rho averaged across languages) using the pipeline upgraded in
   ``analyze/meco_extended.py``.
3. Within each base-condition family (B1 = {B1, B1b}, B2 = {B2, B2_ewc,
   B2_lamol, B2_lr}, B3 = {B3, B3_full}):
     - runs pairwise paired permutation tests (exact at n=3),
     - computes paired Cohen's d_z,
     - applies Holm-Bonferroni multiple-comparison correction *within* the
       family only (not across families).
4. Declares a winner iff it dominates every family member with Holm-corrected
   p < 0.05. Otherwise it reports "no significant winner" together with the
   largest d_z — it does **not** fall through to PPL / CKA / AOA.

Secondary metrics (PPL, CKA, AOA) are still recorded in the output JSON so
reviewers can inspect them, but they are explicitly flagged as exploratory
and do not drive selection.

Usage
-----
    python -m analyze.significance_ablation \\
        --variants-file logs_ablation_nld_eng/variants_manifest.csv \\
        --family B2 \\
        --output results/ablation_significance_B2.json

Or run all three families at once:

    python -m analyze.significance_ablation \\
        --variants-file logs_ablation_nld_eng/variants_manifest.csv \\
        --all-families \\
        --output-dir results/

Implementation notes
--------------------
Paired permutation test: for n = 3 paired samples we have exactly
2^3 = 8 sign permutations. ``scipy.stats.permutation_test`` with
``permutation_type="samples"`` enumerates these exactly (no sampling error).

Cohen's d_z (paired):   d_z = mean(diffs) / std(diffs, ddof=1)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError as e:
    raise SystemExit(
        "scipy is required for significance_ablation.py — "
        "install via `pip install scipy`."
    ) from e

try:
    from statsmodels.stats.multitest import multipletests
except ImportError as e:
    raise SystemExit(
        "statsmodels is required for Holm correction — "
        "install via `pip install statsmodels`."
    ) from e


logger = logging.getLogger("significance_ablation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(_h)


# ─────────────────────────────────────────────────────────────────────────────
# Family definitions
# ─────────────────────────────────────────────────────────────────────────────

FAMILY_MEMBERS: Dict[str, List[str]] = {
    "B1": ["B1", "B1b"],
    "B2": ["B2", "B2_ewc", "B2_lamol", "B2_lr"],
    "B3": ["B3", "B3_full"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Manifest + metric ingestion
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunRecord:
    variant: str
    seed: int
    config_path: str
    log_path: str
    hf_push_suffix: str
    # Populated by :func:`compute_metrics_for_run`.
    meco_rho: Optional[float] = None          # primary
    ppl_l2: Optional[float] = None            # secondary (exploratory)
    cka_sim: Optional[float] = None           # secondary (exploratory)
    aoa_spearman: Optional[float] = None      # tertiary (exploratory)
    metadata: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict:
        d = asdict(self)
        return d


def load_manifest(path: str) -> List[RunRecord]:
    """Parse the CSV written by ablation_minimal_nld_eng.sh."""
    records: List[RunRecord] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"variant", "seed", "config_path", "log_path", "hf_push_suffix"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")
        for row in reader:
            records.append(
                RunRecord(
                    variant=row["variant"],
                    seed=int(row["seed"]),
                    config_path=row["config_path"],
                    log_path=row["log_path"],
                    hf_push_suffix=row["hf_push_suffix"],
                )
            )
    return records


def compute_metrics_for_run(
    record: RunRecord,
    *,
    meco_languages: Sequence[str] = ("en", "nl"),
    meco_backend: Optional[callable] = None,
    secondary_backend: Optional[callable] = None,
) -> RunRecord:
    """
    Populate the ``meco_rho`` (primary) and optionally secondary metrics on
    ``record``.

    The caller supplies pluggable backends so this module is test-friendly
    and so that heavy imports (transformers, statsmodels mixedlm) stay in
    meco_extended.py rather than leaking here:

        meco_backend(record, languages) -> float           # mean Spearman rho
        secondary_backend(record) -> dict with keys {ppl_l2, cka_sim, aoa_spearman}

    When backends are not provided, the environment variable
    ``BEETLE_MOCK_METRICS=1`` triggers a deterministic mock based on a hash
    of (variant, seed) — useful for dry-runs and unit tests.
    """
    if meco_backend is None and os.environ.get("BEETLE_MOCK_METRICS") == "1":
        rng = np.random.default_rng(hash((record.variant, record.seed)) & 0xFFFFFFFF)
        record.meco_rho = float(np.clip(0.25 + 0.1 * rng.standard_normal(), -1, 1))
        record.ppl_l2 = float(10 + 5 * rng.standard_normal())
        record.cka_sim = float(np.clip(0.5 + 0.2 * rng.standard_normal(), 0, 1))
        record.aoa_spearman = float(np.clip(0.3 + 0.1 * rng.standard_normal(), -1, 1))
        return record

    if meco_backend is None:
        # Late-import to avoid a hard dependency at import time.
        from analyze.meco_extended import compute_alignment_for_run  # type: ignore

        meco_backend = compute_alignment_for_run

    record.meco_rho = float(meco_backend(record, meco_languages))

    if secondary_backend is not None:
        sec = secondary_backend(record)
        record.ppl_l2 = sec.get("ppl_l2")
        record.cka_sim = sec.get("cka_sim")
        record.aoa_spearman = sec.get("aoa_spearman")

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Statistical core
# ─────────────────────────────────────────────────────────────────────────────

def cohens_dz(diffs: np.ndarray) -> float:
    """
    Paired Cohen's d_z = mean(diffs) / std(diffs, ddof=1).

    Returns 0.0 when std(diffs) == 0 (all paired differences identical) to
    avoid propagating nan into downstream summaries.
    """
    diffs = np.asarray(diffs, dtype=float)
    sd = diffs.std(ddof=1) if diffs.size > 1 else 0.0
    if not np.isfinite(sd) or sd == 0.0:
        return 0.0
    return float(diffs.mean() / sd)


def paired_permutation_p(
    a: Sequence[float],
    b: Sequence[float],
    *,
    alternative: str = "two-sided",
) -> float:
    """
    Exact paired permutation test via sign-flipping of pairwise differences.

    At n = 3 there are 2^3 = 8 permutations, so this is exact with zero
    sampling error. We roll our own so the result is reproducible and has
    no scipy-version-dependent behaviour.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_permutation_p requires equal-shape arrays")
    diffs = a - b
    n = diffs.size
    observed = diffs.mean()

    # Enumerate all sign patterns.
    extreme = 0
    total = 2 ** n
    for mask in range(total):
        signs = np.array([1 if (mask >> k) & 1 else -1 for k in range(n)])
        m = (signs * diffs).mean()
        if alternative == "two-sided":
            if abs(m) >= abs(observed):
                extreme += 1
        elif alternative == "greater":
            if m >= observed:
                extreme += 1
        elif alternative == "less":
            if m <= observed:
                extreme += 1
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
    return extreme / total


@dataclass
class PairwiseResult:
    variant_a: str
    variant_b: str
    mean_a: float
    mean_b: float
    mean_diff: float
    cohens_dz: float
    p_two_sided: float
    p_one_sided_greater: float  # exploratory: H1 that A > B
    p_holm: Optional[float] = None


def within_family_pairwise(
    metric_by_variant_seed: Dict[str, Dict[int, float]],
    family_variants: Sequence[str],
) -> List[PairwiseResult]:
    """
    Pairwise paired permutation tests within a family.

    ``metric_by_variant_seed`` maps variant → {seed → metric_value}. Seeds
    must match across variants (the sweep pairs them). Variants with fewer
    than 2 seeds are skipped with a warning.
    """
    out: List[PairwiseResult] = []

    # Find the intersection of seed sets so pairing is well defined.
    seed_sets = {
        v: set(metric_by_variant_seed[v].keys()) for v in family_variants
        if v in metric_by_variant_seed
    }
    if len(seed_sets) < 2:
        logger.warning(
            "within_family_pairwise: fewer than 2 variants have metrics; skipping."
        )
        return out
    shared_seeds = sorted(set.intersection(*seed_sets.values()))
    if len(shared_seeds) < 2:
        logger.warning(
            "within_family_pairwise: fewer than 2 shared seeds; paired tests "
            "require seed pairing. Shared seeds=%s", shared_seeds,
        )
        return out

    for i, va in enumerate(family_variants):
        for vb in family_variants[i + 1 :]:
            if va not in metric_by_variant_seed or vb not in metric_by_variant_seed:
                continue
            a = np.array([metric_by_variant_seed[va][s] for s in shared_seeds])
            b = np.array([metric_by_variant_seed[vb][s] for s in shared_seeds])
            diffs = a - b
            out.append(
                PairwiseResult(
                    variant_a=va,
                    variant_b=vb,
                    mean_a=float(a.mean()),
                    mean_b=float(b.mean()),
                    mean_diff=float(diffs.mean()),
                    cohens_dz=cohens_dz(diffs),
                    p_two_sided=paired_permutation_p(a, b, alternative="two-sided"),
                    p_one_sided_greater=paired_permutation_p(
                        a, b, alternative="greater"
                    ),
                )
            )

    # Holm-Bonferroni within family only.
    if out:
        raw = [r.p_two_sided for r in out]
        _, p_adj, _, _ = multipletests(raw, alpha=0.05, method="holm")
        for r, p in zip(out, p_adj):
            r.p_holm = float(p)

    return out


def pick_winner(
    pairwise: Sequence[PairwiseResult],
    family_variants: Sequence[str],
    alpha: float = 0.05,
) -> Dict:
    """
    A variant wins iff it Pareto-dominates every other family member:
    for every pair (winner, other), Holm-corrected two-sided p < alpha AND
    winner's mean > other's mean.

    Primary metric assumption: *larger is better* (MECO rho). If you ever
    switch the primary metric to e.g. L2 PPL, invert the sign in
    compute_metrics_for_run before calling this function.
    """
    members = [v for v in family_variants
               if any(p.variant_a == v or p.variant_b == v for p in pairwise)]
    wins: Dict[str, List[str]] = {m: [] for m in members}
    losses: Dict[str, List[str]] = {m: [] for m in members}

    for r in pairwise:
        significant = (r.p_holm is not None and r.p_holm < alpha)
        if not significant:
            continue
        # Higher mean wins under the 'larger-is-better' convention.
        if r.mean_a > r.mean_b:
            wins[r.variant_a].append(r.variant_b)
            losses[r.variant_b].append(r.variant_a)
        elif r.mean_b > r.mean_a:
            wins[r.variant_b].append(r.variant_a)
            losses[r.variant_a].append(r.variant_b)

    needed_opponents = len(members) - 1
    winners = [m for m in members if len(wins[m]) == needed_opponents]
    max_dz = max((abs(r.cohens_dz) for r in pairwise), default=0.0)

    return {
        "winner": winners[0] if len(winners) == 1 else None,
        "winner_certain": len(winners) == 1,
        "winners_candidate": winners,
        "largest_abs_cohens_dz": max_dz,
        "alpha": alpha,
        "note": (
            "winner selected by within-family Holm-corrected paired permutation "
            "test on MECO rho (primary). Secondary metrics (PPL, CKA, AOA) are "
            "reported below for transparency but are NOT decisional."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def run_family(
    records: Sequence[RunRecord],
    family: str,
    *,
    alpha: float = 0.05,
) -> Dict:
    """Run the full within-family analysis and return a serialisable dict."""
    if family not in FAMILY_MEMBERS:
        raise ValueError(
            f"Unknown family {family}. Known: {sorted(FAMILY_MEMBERS.keys())}"
        )
    members = FAMILY_MEMBERS[family]
    in_family = [r for r in records if r.variant in members]
    if not in_family:
        return {"family": family, "error": "no records for family", "members": members}

    # Pivot: variant -> {seed: primary_metric}.
    primary: Dict[str, Dict[int, float]] = {}
    secondary_rows: List[Dict] = []
    for r in in_family:
        if r.meco_rho is None:
            logger.warning(
                "Skipping %s/seed=%s: meco_rho missing. Did you call "
                "compute_metrics_for_run?", r.variant, r.seed,
            )
            continue
        primary.setdefault(r.variant, {})[r.seed] = float(r.meco_rho)
        secondary_rows.append(
            {
                "variant": r.variant,
                "seed": r.seed,
                "meco_rho": r.meco_rho,
                "ppl_l2": r.ppl_l2,
                "cka_sim": r.cka_sim,
                "aoa_spearman": r.aoa_spearman,
            }
        )

    pairwise = within_family_pairwise(primary, members)
    decision = pick_winner(pairwise, members, alpha=alpha)

    return {
        "family": family,
        "members": members,
        "n_seeds_by_variant": {v: len(primary.get(v, {})) for v in members},
        "primary_metric": "meco_rho",
        "primary_metric_higher_is_better": True,
        "per_run_metrics": secondary_rows,
        "pairwise": [asdict(p) for p in pairwise],
        "decision": decision,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0] if __doc__ else "",
    )
    parser.add_argument(
        "--variants-file", required=True,
        help="CSV manifest produced by ablation_minimal_nld_eng.sh.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--family", choices=sorted(FAMILY_MEMBERS.keys()),
        help="Run a single family.",
    )
    group.add_argument(
        "--all-families", action="store_true",
        help="Run B1, B2, B3 and write one JSON each to --output-dir.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (single family mode).",
    )
    parser.add_argument(
        "--output-dir", default="results/",
        help="Directory for per-family JSON outputs (multi-family mode).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Family-wise error rate for Holm-Bonferroni.",
    )
    parser.add_argument(
        "--mock-metrics", action="store_true",
        help="Use deterministic mock MECO scores (sets BEETLE_MOCK_METRICS=1).",
    )
    args = parser.parse_args(argv)

    if args.mock_metrics:
        os.environ["BEETLE_MOCK_METRICS"] = "1"

    records = load_manifest(args.variants_file)
    logger.info("Loaded %d run records from %s", len(records), args.variants_file)

    # Populate metrics.
    for r in records:
        compute_metrics_for_run(r)

    families = [args.family] if args.family else sorted(FAMILY_MEMBERS.keys())
    results: Dict[str, Dict] = {}
    for fam in families:
        results[fam] = run_family(records, fam, alpha=args.alpha)

    if args.family:
        out_path = Path(
            args.output or f"results/ablation_significance_{args.family}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results[args.family], indent=2))
        logger.info("Wrote %s", out_path)
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for fam, payload in results.items():
            p = out_dir / f"ablation_significance_{fam}.json"
            p.write_text(json.dumps(payload, indent=2))
            logger.info("Wrote %s", p)

    # Console summary.
    for fam, payload in results.items():
        decision = payload.get("decision", {})
        winner = decision.get("winner")
        if winner:
            print(f"[{fam}] WINNER = {winner} (Holm alpha={args.alpha}).")
        else:
            print(
                f"[{fam}] no significant winner "
                f"(largest |d_z| = {decision.get('largest_abs_cohens_dz', 0):.3f})."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
