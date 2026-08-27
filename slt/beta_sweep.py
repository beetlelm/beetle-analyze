"""S2 — β-sweep validation on the four anchor cells.

Goal: confirm that the single-temperature LLC matches the slope-fit RLCT
from a four-temperature β-sweep within ~10%, on one anchor cell per
regime (mono / bi / multi / pico).

For each anchor cell, we call ``devinterp.slt.llc.llc`` four times with
``n_beta`` scaled by ∈ {0.3, 0.5, 0.7, 1.0} × default. We then regress
``E[L_beta] vs 1/beta`` to recover the asymptotic free-energy slope, which
equals λ in the SLT thermodynamic-integration sense (Watanabe 2009 §1.4).

Outputs
-------
  ``<output_root>/slt/beta_sweep/<repo>/<rev>.csv``    raw β samples
  ``<output_root>/slt/beta_sweep_summary.rank{R}.csv`` per-cell λ̂_TI
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

from analyze.ppl_utils import load_model_and_tokenizer  # noqa: E402

from slt.probe_data import build_sampling_task  # noqa: E402
from slt.run_llc import _scaled_hyperparams, _detect_regime  # noqa: E402

log = logging.getLogger(__name__)

BETA_FRACTIONS = (0.3, 0.5, 0.7, 1.0)


def beta_sweep(
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
    """Run β-sweep on a single anchor cell."""
    from devinterp.slt.llc import llc  # type: ignore

    if smoke:
        n_chains, n_draws, n_burnin = 1, 20, 10

    model, tokenizer = load_model_and_tokenizer(repo, revision=revision)
    model = model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    hp = _scaled_hyperparams(n_params, batch_size)
    base_n_beta = float(hp["n_beta"])

    ds = build_sampling_task(tokenizer, n=512 if smoke else 1024, max_length=256)
    out_dir = output_root / "slt" / "beta_sweep" / repo.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for frac in BETA_FRACTIONS:
        n_beta = base_n_beta * frac
        try:
            res = llc(
                model=model,
                dataset=ds,
                observables={"train": ds},
                lr=hp["lr"],
                n_beta=n_beta,
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
                "frac": frac,
                "n_beta": n_beta,
                "llc_mean": float(res["llc_mean"].item()),
                "llc_std": float(res["llc_std"].item()),
                "init_loss": float(res["init_loss"].item()),
            })
        except Exception:  # pragma: no cover
            log.exception("β-sweep failed at frac=%.2f for %s@%s", frac, repo, revision)

    raw_csv = out_dir / f"{revision}.csv"
    if rows:
        with raw_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Slope-fit λ_TI: regress llc_mean on log(1/n_beta).
    # If the LLC estimator is consistent across temperatures, the slope
    # should be ≈ 0 (it's already the local coefficient at every β),
    # whereas the *RLCT* under TI is the constant-term reading. We report
    # both: the mean, the slope, and the residual to flag drift.
    if len(rows) >= 2:
        x = np.log(np.array([1.0 / r["n_beta"] for r in rows]))
        y = np.array([r["llc_mean"] for r in rows])
        slope, intercept = np.polyfit(x, y, 1)
        lam_ti = float(intercept)
    else:
        slope, intercept, lam_ti = float("nan"), float("nan"), float("nan")

    if rows:
        lam_default = next((r["llc_mean"] for r in rows if r["frac"] == 1.0), float("nan"))
    else:
        lam_default = float("nan")
    rel_err = (
        abs(lam_ti - lam_default) / max(abs(lam_default), 1e-12)
        if rows else float("nan")
    )

    regime, languages = _detect_regime(repo)
    summary = {
        "repo": repo,
        "revision": revision,
        "regime": regime,
        "languages": languages,
        "lam_default_beta": float(lam_default),
        "lam_ti_intercept": lam_ti,
        "ti_slope": float(slope),
        "rel_err_default_vs_ti": float(rel_err),
        "passes_10pct": bool(rel_err < 0.10),
    }
    return summary


def append_summary(s: dict, output_root: Path) -> None:
    rank = int(os.environ.get("RANK", "0"))
    summary_dir = output_root / "slt"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"beta_sweep_summary.rank{rank}.csv"
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(s.keys()))
        if new:
            w.writeheader()
        w.writerow(s)


def run_stage(args) -> None:
    """Run β-sweep on a small set of anchor cells."""
    rank = int(getattr(args, "rank", 0))
    world = max(int(getattr(args, "world_size", 1)), 1)

    anchors = list(getattr(args, "beta_sweep_anchors", []) or [])
    if not anchors:
        log.warning("[β-sweep] no anchor cells configured (--beta-sweep-anchors)")
        return
    cells = [(a, getattr(args, "revision", "main")) for a in anchors]
    my_cells = cells[rank::world]
    log.info("[β-sweep] rank %d/%d owns %d/%d cells",
             rank, world, len(my_cells), len(cells))

    for repo, rev in my_cells:
        try:
            s = beta_sweep(repo, rev, output_root=args.output_root)
            append_summary(s, args.output_root)
            log.info("[β-sweep] %s@%s  λ_default=%.3f λ_ti=%.3f  pass=%s",
                     repo, rev, s["lam_default_beta"], s["lam_ti_intercept"],
                     s["passes_10pct"])
        except Exception:  # pragma: no cover
            log.exception("[β-sweep] failed: %s @ %s", repo, rev)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="β-sweep validation (S2).")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--output-root", type=Path, default=Path("results"))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    s = beta_sweep(args.model, args.revision,
                   output_root=args.output_root, smoke=args.smoke)
    append_summary(s, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
