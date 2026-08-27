"""A1 — Bridge regression: cross-lingual probing score vs susceptibility PC1.

The headline novelty claim of the paper. For each (model, checkpoint) cell,
we compute:

  * x = ``shared_feature_score``: average cross-lingual transfer accuracy of
    POS / dep / NER probes from ``analyze.xling.probes`` (already produced
    by the standard Beetle pipeline; we just read its CSVs).
  * y = ``pc1_var_ratio``: PC1 explained variance of the standardised
    susceptibility response matrix (from ``run_susceptibilities.py``).

We regress y on x with OLS, report Pearson r, plot the scatter, and write
a row to ``slt/bridge_summary.csv``. **Pass criterion: r > 0.7**.

Inputs
------
  ``<output_root>/slt/susceptibilities_summary.rank{R}.csv``  (PC1 ratios)
  ``<output_root>/xling/probes/<repo>__<rev>.csv``            (probe accuracies)

Outputs
-------
  ``<output_root>/slt/bridge_summary.csv``        — joined per-cell rows
  ``<output_root>/slt/bridge_regression.json``   — Pearson r, slope, p-value
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS))

log = logging.getLogger(__name__)


def _load_susceptibility_summary(output_root: Path) -> dict[tuple[str, str], dict]:
    """Read all per-rank susceptibility CSVs and key by (repo, revision)."""
    rows: dict[tuple[str, str], dict] = {}
    base = output_root / "slt"
    if not base.exists():
        return rows
    for csv_path in sorted(base.glob("susceptibilities_summary.rank*.csv")):
        with csv_path.open() as fh:
            for row in csv.DictReader(fh):
                rows[(row["repo"], row["revision"])] = row
    return rows


def _load_probe_scores(output_root: Path) -> dict[tuple[str, str], float]:
    """Cross-lingual probe transfer accuracy per (repo, revision).

    Reads ``output_root/xling/probes/<repo_slug>__<rev>.csv`` produced by
    ``analyze.xling.probes``. The "shared-feature score" is the mean
    cross-lingual transfer accuracy across all probe types and language
    pairs where train_lang ≠ eval_lang.
    """
    out: dict[tuple[str, str], float] = {}
    probes_dir = output_root / "xling" / "probes"
    if not probes_dir.exists():
        log.warning("[bridge] no probes dir at %s — bridge will be empty", probes_dir)
        return out
    for csv_path in probes_dir.glob("*.csv"):
        stem = csv_path.stem
        try:
            repo_slug, rev = stem.rsplit("__", 1)
        except ValueError:
            continue
        repo = repo_slug.replace("__", "/")
        accs: list[float] = []
        with csv_path.open() as fh:
            for row in csv.DictReader(fh):
                tl = row.get("train_lang") or row.get("source_lang")
                el = row.get("eval_lang") or row.get("target_lang")
                if not tl or not el or tl == el:
                    continue
                try:
                    accs.append(float(row.get("acc") or row.get("accuracy") or 0.0))
                except ValueError:
                    continue
        if accs:
            out[(repo, rev)] = float(np.mean(accs))
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return ``(r, p_two_sided)``. Uses scipy if available, else manual."""
    if x.size < 3:
        return 0.0, 1.0
    try:
        from scipy.stats import pearsonr  # type: ignore

        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        x_c = x - x.mean()
        y_c = y - y.mean()
        denom = float(np.sqrt((x_c ** 2).sum() * (y_c ** 2).sum()))
        if denom == 0.0:
            return 0.0, 1.0
        return float((x_c * y_c).sum() / denom), float("nan")


def run_bridge(output_root: Path) -> dict:
    """Join probe accuracies with PC1 ratios; run regression; persist results."""
    sus = _load_susceptibility_summary(output_root)
    probes = _load_probe_scores(output_root)

    keys = sorted(set(sus) & set(probes))
    if not keys:
        log.warning("[bridge] no overlapping (repo, revision) keys")
        result = {"n": 0, "pearson_r": 0.0, "p_value": float("nan")}
        (output_root / "slt").mkdir(parents=True, exist_ok=True)
        with (output_root / "slt" / "bridge_regression.json").open("w") as fh:
            json.dump(result, fh, indent=2)
        return result

    xs, ys, regimes = [], [], []
    out_rows = []
    for repo, rev in keys:
        x = probes[(repo, rev)]
        s = sus[(repo, rev)]
        y = float(s["pc1_var_ratio"])
        xs.append(x)
        ys.append(y)
        regimes.append(s.get("regime", "unknown"))
        out_rows.append({
            "repo": repo,
            "revision": rev,
            "regime": s.get("regime", "unknown"),
            "languages": s.get("languages", "unknown"),
            "shared_feature_score": x,
            "pc1_var_ratio": y,
            "matrix_rank_99pct": s.get("matrix_rank_99pct", ""),
            "pc1_loading_uniformity": s.get("pc1_loading_uniformity", ""),
        })

    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    r, p = _pearson(x_arr, y_arr)
    slope, intercept = np.polyfit(x_arr, y_arr, deg=1) if x_arr.size >= 2 else (0.0, 0.0)

    summary_dir = output_root / "slt"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "bridge_summary.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    result = {
        "n": int(x_arr.size),
        "pearson_r": float(r),
        "p_value": float(p),
        "slope": float(slope),
        "intercept": float(intercept),
        "pass_criterion_r_gt_0_7": bool(r > 0.7),
    }
    with (summary_dir / "bridge_regression.json").open("w") as fh:
        json.dump(result, fh, indent=2)
    log.info("[bridge] n=%d Pearson r=%.3f p=%.3g  pass=%s",
             result["n"], r, p, result["pass_criterion_r_gt_0_7"])
    return result


def run_stage(args) -> None:
    if int(getattr(args, "rank", 0)) != 0:
        log.info("[bridge] skipping on rank %d (rank-0 only)", args.rank)
        return
    run_bridge(args.output_root)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="A1 bridge regression.")
    p.add_argument("--output-root", type=Path, default=Path("results"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_bridge(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
