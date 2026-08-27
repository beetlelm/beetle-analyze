"""
analyze.meco_detailed._runner — in-process module execution for the orchestrator.

Each D-module's CLI is a thin wrapper around its ``compute()`` function. In
the orchestrated sweep we don't want N×M subprocess cold-imports (≈5 s each
for the pandas + statsmodels stack); instead we dispatch ``compute()`` calls
in-process inside ``ProcessPoolExecutor`` workers that survive across many
tasks.

Public surface:

  init_worker()
      Process-pool initialiser. Pre-loads the joined MECO RT cache so the
      first task in each worker doesn't pay the ~10 s rda + wordfreq cost.

  run_one(slug, model_id, *, fast=False, langs=None)
      Run module ``slug`` on a single model's surprisals, upsert the
      resulting rows into the canonical CSV, and return a small status
      record. Designed to be picklable (no closures over local state) so
      it can ride a ``ProcessPoolExecutor.submit``.

  regenerate_plots(slugs=None)
      Replot every module's PDF from its current CSV. Cheap (~1 s/module),
      called once at the end of the sweep to avoid mid-sweep PDF contention.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

# Tell numpy / OpenBLAS to play nice with multiprocessing — without this each
# worker spawns OS_CPU_COUNT BLAS threads that fight each other.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from analyze import _path  # noqa: F401 — ensure analyze.* resolves in workers


# Modules in canonical order. Each entry maps short slug ``d1`` → full module
# name. The orchestrator always feeds the full name to ``run_one``.
ALL_MODULE_SLUGS: tuple[str, ...] = (
    "d1_measure_robustness",
    "d2_spillover_decay",
    "d3_bilingual_decomp",
    "d4_reader_caterpillar",
    "d5_item_granularity",
    "d6_proficiency_mod",
    "d7_l1_control",
    "d8_comprehension_link",
)

# Composite keys used for CSV upsert per module. ``model`` is the focal
# beetle repo; ``lang`` is the MECO L1 code (single string).
_MODULE_KEYS: dict[str, tuple[str, ...]] = {
    "d1_measure_robustness": ("model", "lang", "measure"),
    "d2_spillover_decay":    ("model", "lang", "lag"),
    "d3_bilingual_decomp":   ("lang", "beetle"),
    "d4_reader_caterpillar": ("model", "lang", "subid"),
    "d5_item_granularity":   ("model_beetle", "model_baseline", "lang",
                              "bin_kind", "bin_key"),
    "d6_proficiency_mod":    ("model", "lang", "prof_quartile"),
    "d7_l1_control":         ("model", "reader_group", "lang"),
    "d8_comprehension_link": ("model", "lang"),
}


@dataclass
class _RunResult:
    slug: str
    model: str
    success: bool
    seconds: float
    n_rows: int
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Worker initialiser
# ---------------------------------------------------------------------------

def init_worker(*, fast: bool = False) -> None:
    """Process-pool initialiser. Pre-warms the RT cache + sets fast-fit env.

    Called once per worker before the first task is dispatched. After this
    returns, ``get_rt_with_features()`` and ``get_rt_full_with_features()``
    are O(1) for every subsequent call in this process.
    """
    if fast:
        os.environ["BEETLE_FAST_FIT"] = "1"
    from analyze.meco_detailed import _common
    # Touch both cached loaders to amortise the rdata + wordfreq cost
    # before the first compute() call charges it to a real task.
    _common.get_rt_with_features()
    try:
        _common.get_rt_full_with_features()
    except FileNotFoundError:
        # Multi-measure RT only required by D1; skip silently if absent.
        pass


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------

def _load_focal_surprisal(model_id: str) -> pd.DataFrame:
    """Load the single beetle model's MECO surprisal CSV(s).

    Filters the on-disk ``results/meco/`` surprisals to rows whose ``model``
    column matches ``model_id``, then collapses to the **final-checkpoint
    only** subset (matching the canonical pipeline convention).
    """
    from plots.checkpoint_loaders import load_meco_checkpoint_files  # type: ignore
    df = load_meco_checkpoint_files()
    if df.empty:
        return df
    df = df[df["model"] == model_id]
    if df.empty:
        return df
    last = df["step"].max()
    return df[df["step"] == last].copy()


def _load_surprisal_with_aux(model_id: str, slug: str) -> pd.DataFrame:
    """For D3 we also need L1-mono + L2-mono surprisals; everything else
    only needs the focal model.

    Returns a surprisal frame restricted to: the focal model + (for D3)
    every model referenced in ``d3_bilingual_decomp.DEFAULT_TRIPLETS``.
    """
    from plots.checkpoint_loaders import load_meco_checkpoint_files  # type: ignore
    df = load_meco_checkpoint_files()
    if df.empty:
        return df

    keep = {model_id}
    if slug == "d3_bilingual_decomp":
        from analyze.meco_detailed.d3_bilingual_decomp import DEFAULT_TRIPLETS
        for trip in DEFAULT_TRIPLETS.values():
            keep.update(trip.values())

    sub = df[df["model"].isin(keep)].copy()
    if sub.empty:
        return sub
    # Final-checkpoint per model only.
    last_step = sub.groupby("model")["step"].max().rename("_last")
    sub = sub.join(last_step, on="model")
    sub = sub[sub["step"] == sub["_last"]].drop(columns=["_last"])
    return sub


def run_one(
    slug: str,
    model_id: str,
    *,
    fast: bool = False,
    langs: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> dict:
    """Run module ``slug`` for a single beetle model in-process.

    Steps:
      1. Load (filtered) surprisal frame for the focal model (+ aux for D3).
      2. Import the module, call ``module.compute(surp_df, …)``.
      3. Upsert the returned rows into ``results/meco_detailed/<slug>.csv``.
      4. Return a small status dict — designed to be picklable across the
         ``ProcessPoolExecutor`` boundary.

    Plotting is *not* done here; call ``regenerate_plots()`` once after the
    sweep finishes to avoid mid-sweep contention on the PDF files.
    """
    if fast:
        os.environ["BEETLE_FAST_FIT"] = "1"

    from analyze.meco_detailed import _common

    t0 = time.time()
    try:
        surp = _load_surprisal_with_aux(model_id, slug)
        if surp.empty:
            return _RunResult(
                slug=slug, model=model_id, success=False, seconds=0.0,
                n_rows=0, error="no surprisal CSV on disk for this model",
            ).to_dict()

        module = importlib.import_module(f"analyze.meco_detailed.{slug}")

        # Build kwargs for the heterogeneous ``compute`` signatures.
        kwargs: dict = {}
        if langs is not None:
            kwargs["langs"] = list(langs)
        # D5 takes ``--beetle / --baseline`` and is per-baseline; skip in
        # the sweep unless the caller explicitly provides them. The CLI
        # entry-point still works for ad-hoc D5 runs.
        if slug == "d5_item_granularity":
            kwargs.setdefault("beetle", model_id)
            kwargs.setdefault("baseline", "bigscience/bloom-7b1")
            kwargs.setdefault("n_boot", 0)

        if verbose:
            print(f"[runner] {slug} :: {model_id}", file=sys.stderr, flush=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_rows = module.compute(surp, verbose=False, **kwargs)

        if new_rows is None or new_rows.empty:
            return _RunResult(
                slug=slug, model=model_id, success=True,
                seconds=time.time() - t0, n_rows=0,
                error="compute returned 0 rows (likely n<200)",
            ).to_dict()

        csv_path = _common.result_csv(slug)
        keys = _MODULE_KEYS.get(slug, ("model", "lang"))
        # ``key_cols`` must be present in new_rows; D3 doesn't have a ``model``
        # column so its keys list reflects that.
        existing = [k for k in keys if k in new_rows.columns]
        if not existing:
            existing = list(new_rows.columns[:1])  # last-resort fallback
        n_total = _common.upsert_csv(csv_path, new_rows, key_cols=existing)
        return _RunResult(
            slug=slug, model=model_id, success=True,
            seconds=time.time() - t0, n_rows=int(len(new_rows)),
            error="" if n_total > 0 else "upsert wrote 0 rows",
        ).to_dict()

    except Exception as e:
        return _RunResult(
            slug=slug, model=model_id, success=False,
            seconds=time.time() - t0, n_rows=0,
            error=f"{e.__class__.__name__}: {e}",
        ).to_dict()


# ---------------------------------------------------------------------------
# End-of-sweep plot regeneration
# ---------------------------------------------------------------------------

def regenerate_plots(slugs: Optional[Iterable[str]] = None) -> dict[str, str]:
    """Replot each module's PDF from its current CSV.

    Returns ``{slug: "ok" or error message}``. Designed to be called once
    from the orchestrator after every CPU task has finished so the PDFs
    reflect the full sweep, not a partial state.
    """
    from analyze.meco_detailed import _common
    slugs = list(slugs or ALL_MODULE_SLUGS)
    out: dict[str, str] = {}
    for slug in slugs:
        csv_path = _common.result_csv(slug)
        if not csv_path.exists():
            out[slug] = "skip: no csv"
            continue
        try:
            module = importlib.import_module(f"analyze.meco_detailed.{slug}")
            df = pd.read_csv(csv_path)
            module.plot(df, _common.figure_pdf(slug))
            out[slug] = "ok"
        except Exception as e:
            out[slug] = f"{e.__class__.__name__}: {e}"
    return out
