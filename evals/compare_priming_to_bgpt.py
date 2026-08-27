#!/usr/bin/env python3
"""
evals/compare_priming_to_bgpt.py — Side-by-side comparison of Beetle
priming results (this repo) and Arnett (2025)'s published BGPT numbers
(arnett-2025-priming-raw/).

Inputs
------
  Beetle priming surprisals
      results/evals/priming/priming_all.tsv   (produced by evals/eval_priming.py)
  Beetle loss curves
      results/evals/priming/loss_curves.csv   (produced by evals/eval_priming.py)
  Arnett BGPT priming results (per-dataset, already aggregated by their R script)
      arnett-2025-priming-raw/raw_structural_priming_results/
          {schoonbaert,bernolet,hartsuiker,kotzochampou,fleischer}_dataset_final.tsv
  Arnett BGPT loss curves
      arnett-2025-priming-raw/l2_loss.csv

Outputs
-------
  results/evals/priming/comparison/
      loss_curves_combined.csv      — Beetle loss + BGPT loss in one long table
      priming_combined.tsv          — per-stimulus surprisals from both sources
      priming_effects_per_model.csv — collapsed priming effect (match - mismatch
                                      mean relative-probability) per
                                      (model_source, model, dataset, checkpoint,
                                      PrimeLanguage, TargetLanguage)
      summary.md                    — short markdown report of overlapping
                                      conditions and effect-size deltas

Schema notes
------------

* Arnett's preprocess pipeline pivots wider on TargetType to compute, per
  (Set, PrimeType) row, a relative probability:

      DO/PO:    PO_Relative_Probability =
                    exp(-PO_Target) / (exp(-PO_Target) + exp(-DO_Target))
      s/of-gen: s-gen_Probability =
                    exp(-s-genitive_Target) /
                    (exp(-s-genitive_Target) + exp(-of-genitive_Target))
      Active/Pass: Passive_Relative_Probability =
                    exp(-Passive_Target) /
                    (exp(-Passive_Target) + exp(-Active_Target))

  We replicate the same pivot here so Beetle and BGPT rows are directly
  comparable.

* The structural priming effect for a given model/dataset/language pair is
  defined as

      effect = mean(P_match) - mean(P_mismatch)

  where P_match is the relative probability of the congruent target type
  given a same-type prime, P_mismatch the same target after the opposite
  prime. This collapses per-stimulus rows into a single scalar per condition,
  matching the y-axis of Arnett's Figures 4 and 5.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_BEETLE_DIR = _REPO_ROOT / "results" / "evals" / "priming"
DEFAULT_ARNETT_DIR = _REPO_ROOT / "arnett-2025-priming-raw"
DEFAULT_OUTPUT_DIR = DEFAULT_BEETLE_DIR / "comparison"

# Per-dataset metadata: which TargetType pair to pivot, and which type is
# treated as the "match" relative-probability numerator (by convention we
# pick the first alternative listed in Arnett's paper).
DATASET_PIVOT: dict[str, tuple[str, str, str]] = {
    # dataset → (target_type_a, target_type_b, prob_label_a)
    # prob_label_a is the column name we emit, mirroring Arnett's R script.
    "schoonbaert":  ("PO_Target",         "DO_Target",         "PO_Relative_Probability"),
    "bernolet":     ("s-genitive_Target", "of-genitive_Target","s-gen_Probability"),
    "hartsuiker":   ("Passive_Target",    "Active_Target",     "Passive_Relative_Probability"),
    "kotzochampou": ("Passive_Target",    "Active_Target",     "Passive_Relative_Probability"),
    "fleischer":    ("Passive_Target",    "Active_Target",     "Passive_Relative_Probability"),
}

# Map BGPT short pair codes (en_nl, nl_en, ...) → Beetle ISO-639-3 (eng, nld, ...)
BGPT_PAIR_TO_ISO3: dict[str, tuple[str, str]] = {
    "en_nl": ("eng", "nld"), "nl_en": ("nld", "eng"),
    "en_es": ("eng", "spa"), "es_en": ("spa", "eng"),
    "en_el": ("eng", "ell"), "el_en": ("ell", "eng"),
    "en_pl": ("eng", "pol"), "pl_en": ("pol", "eng"),
    "en_de": ("eng", "deu"), "de_en": ("deu", "eng"),
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_beetle_loss(beetle_dir: Path) -> pd.DataFrame:
    """Beetle stage-A output: results/evals/priming/loss_curves.csv."""
    p = beetle_dir / "loss_curves.csv"
    if not p.exists():
        logger.warning(f"No Beetle loss curves at {p}")
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["model_source"] = "beetle"
    return df


def load_arnett_loss(arnett_dir: Path) -> pd.DataFrame:
    """Arnett's BGPT loss CSV: arnett-2025-priming-raw/l2_loss.csv.

    Columns in the raw file: model, eval_file, checkpoint, mean_surprisal.
    We extract the BGPT pair code from `model` (e.g. ``en_nl_5050/model``)
    and the eval language from ``eval_file`` (e.g. ``en_tokenized_eval_*``
    → eng).
    """
    p = arnett_dir / "l2_loss.csv"
    if not p.exists():
        logger.warning(f"No Arnett loss CSV at {p}")
        return pd.DataFrame()
    # The file's header line has a leading empty column (",model,eval_file,...")
    # but data rows have only 4 fields. Skip the header and supply names so
    # the 4-field data lines align correctly.
    raw = pd.read_csv(
        p, skiprows=1, header=None,
        names=["model_str", "eval_file", "checkpoint", "mean_surprisal"],
    )
    raw["checkpoint"] = pd.to_numeric(raw["checkpoint"], errors="coerce")
    raw["mean_surprisal"] = pd.to_numeric(raw["mean_surprisal"], errors="coerce")

    def _extract_pair(s: str) -> Optional[str]:
        # e.g. "en_nl_5050/model" → "en_nl"
        s = str(s)
        head = s.split("/", 1)[0]
        # Strip ratio/suffix codes like _5050 / _7525.
        parts = head.split("_")
        if len(parts) >= 2 and len(parts[0]) == 2 and len(parts[1]) == 2:
            return f"{parts[0]}_{parts[1]}"
        return None

    def _extract_lang(s: str) -> Optional[str]:
        s = str(s)
        # e.g. "en_nl_7525/en_tokenized_eval_shuffled.txt" → "en"
        leaf = s.rsplit("/", 1)[-1]
        if leaf.startswith("en_"):
            return "eng"
        if leaf.startswith("nl_"):
            return "nld"
        if leaf.startswith("es_"):
            return "spa"
        if leaf.startswith("el_"):
            return "ell"
        if leaf.startswith("pl_"):
            return "pol"
        if leaf.startswith("de_"):
            return "deu"
        return None

    raw["bgpt_pair"] = raw["model_str"].map(_extract_pair)
    raw["language"] = raw["eval_file"].map(_extract_lang)
    raw["model_source"] = "bgpt"
    raw = raw.rename(columns={"model_str": "repo"})
    return raw[["model_source", "repo", "bgpt_pair", "language", "checkpoint", "mean_surprisal"]]


def load_arnett_priming(arnett_dir: Path) -> pd.DataFrame:
    """Concatenate every {dataset}_dataset_final.tsv with a `dataset` tag."""
    raw_dir = arnett_dir / "raw_structural_priming_results"
    parts: list[pd.DataFrame] = []
    for path in sorted(raw_dir.glob("*_dataset_final.tsv")):
        ds = path.name.replace("_dataset_final.tsv", "")
        try:
            df = pd.read_csv(path, sep="\t", low_memory=False)
        except Exception as exc:
            logger.warning(f"  could not read {path}: {exc}")
            continue
        df["dataset"] = ds
        df["model_source"] = "bgpt"
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_beetle_priming(beetle_dir: Path) -> pd.DataFrame:
    """Beetle stage-B summary: results/evals/priming/priming_all.tsv.

    The TSV has per-stimulus rows in long form (one row per TargetType).
    We pivot wider here to produce the same {dataset, *_Target columns,
    *_Relative_Probability} layout as Arnett's preprocessed BGPT TSVs.
    """
    p = beetle_dir / "priming_all.tsv"
    if not p.exists():
        logger.warning(f"No Beetle priming summary at {p} — run evals/eval_priming.py first.")
        return pd.DataFrame()
    long = pd.read_csv(p, sep="\t", low_memory=False)
    # eval_priming.py with --revisions main leaves `checkpoint` as NaN
    # (only `revision` is populated). pivot_table drops index rows with NaN,
    # which silently zeroes out the entire beetle side of the comparison.
    # Backfill checkpoint from revision so the pivot keeps the rows.
    if "checkpoint" in long.columns and long["checkpoint"].isna().all():
        if "revision" in long.columns:
            long["checkpoint"] = long["revision"].astype(str)
        else:
            long["checkpoint"] = "main"

    pivoted_parts: list[pd.DataFrame] = []
    for ds, sub in long.groupby("dataset"):
        if ds not in DATASET_PIVOT:
            logger.warning(f"  unknown dataset {ds!r}; skipping pivot.")
            continue
        a, b, prob_col = DATASET_PIVOT[ds]
        # Pivot wider on TargetType, value = Surprisal. dropna=False is needed
        # because some columns (condition/l1/l2) can legitimately be NaN for
        # repos that didn't parse cleanly — but we still want their rows.
        idx_cols = [c for c in (
            "dataset", "model", "revision", "checkpoint", "family",
            "condition", "l1", "l2",
            "Set", "ExtraCondition",
            "PrimeLanguage", "PrimeType", "TargetLanguage",
        ) if c in sub.columns and not sub[c].isna().all()]
        wide = sub.pivot_table(
            index=idx_cols, columns="TargetType", values="Surprisal",
            aggfunc="first", dropna=False,
        ).reset_index()
        wide.columns.name = None
        if a in wide.columns and b in wide.columns:
            denom = np.exp(-wide[a]) + np.exp(-wide[b])
            wide[prob_col] = np.exp(-wide[a]) / denom.replace(0, np.nan)
        wide["model_source"] = "beetle"
        pivoted_parts.append(wide)
    if not pivoted_parts:
        return pd.DataFrame()
    return pd.concat(pivoted_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Effect-size collapse
# ---------------------------------------------------------------------------

def collapse_priming_effects(wide: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-stimulus rows to one priming-effect scalar per
    (model_source, model_or_repo, dataset, checkpoint, PrimeLanguage,
    TargetLanguage).

    For each dataset we know the pivoted relative-probability column from
    DATASET_PIVOT. The priming effect is

        effect = mean(P | match-type prime) - mean(P | mismatch-type prime)

    where "match" means PrimeType encodes the same construction as the
    numerator TargetType. We detect a match by checking that the prefix of
    PrimeType (before "_Prime") matches the prefix of the relative-prob
    target type (before "_Target").
    """
    # Vectorized: a Python-level groupby loop over ~40k+ groups was the
    # bottleneck on trajectory tiers (33+ min and a SIGILL). Use single-pass
    # agg on (group_cols × is_match) then reshape — pure pandas, ~seconds.
    out_blocks: list[pd.DataFrame] = []
    for ds, sub in wide.groupby("dataset"):
        if ds not in DATASET_PIVOT:
            continue
        a_target, _b_target, prob_col = DATASET_PIVOT[ds]
        if prob_col not in sub.columns:
            continue
        target_prefix = a_target.replace("_Target", "")
        model_col = "model" if "model" in sub.columns else "repo"
        group_cols = [c for c in (
            "model_source", model_col, "dataset", "checkpoint",
            "PrimeLanguage", "TargetLanguage",
        ) if c in sub.columns]
        sub2 = sub.dropna(subset=[prob_col]).copy()
        if sub2.empty:
            continue
        sub2["_is_match"] = sub2["PrimeType"].astype(str).str.startswith(target_prefix)
        agg = (sub2.groupby(group_cols + ["_is_match"], dropna=False)[prob_col]
                    .agg(["mean", "count"])
                    .unstack("_is_match"))
        if (True not in agg["mean"].columns) or (False not in agg["mean"].columns):
            continue
        keep = agg["mean"][True].notna() & agg["mean"][False].notna()
        agg = agg.loc[keep]
        if agg.empty:
            continue
        block = pd.DataFrame({
            "mean_match":    agg["mean"][True].astype(float),
            "mean_mismatch": agg["mean"][False].astype(float),
            "n_match":       agg["count"][True].fillna(0).astype(int),
            "n_mismatch":    agg["count"][False].fillna(0).astype(int),
        })
        block["priming_effect"] = block["mean_match"] - block["mean_mismatch"]
        block = block.reset_index()
        out_blocks.append(block)
    return pd.concat(out_blocks, ignore_index=True) if out_blocks else pd.DataFrame()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    beetle_dir = Path(args.beetle_dir)
    arnett_dir = Path(args.arnett_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loss curves.
    beetle_loss = load_beetle_loss(beetle_dir)
    arnett_loss = load_arnett_loss(arnett_dir)
    loss = pd.concat([beetle_loss, arnett_loss], ignore_index=True, sort=False)
    if not loss.empty:
        loss.to_csv(out_dir / "loss_curves_combined.csv", index=False)
        logger.info(f"Wrote {len(loss)} loss rows → {out_dir / 'loss_curves_combined.csv'}")
    else:
        logger.warning("No loss data from either source.")

    # 2. Priming surprisals.
    beetle_priming = load_beetle_priming(beetle_dir)
    arnett_priming = load_arnett_priming(arnett_dir)
    priming = pd.concat([beetle_priming, arnett_priming], ignore_index=True, sort=False)
    if not priming.empty:
        # priming_combined.tsv is gitignored (regularly >100MB on trajectory
        # tiers) and writing it on a 2M-row frame can SIGILL on the kernel
        # we run on. Skip unless explicitly requested.
        if getattr(args, "write_combined_tsv", False):
            priming.to_csv(out_dir / "priming_combined.tsv", sep="\t", index=False)
            logger.info(f"Wrote {len(priming)} priming rows → {out_dir / 'priming_combined.tsv'}")
        else:
            logger.info(f"Skipping priming_combined.tsv ({len(priming)} rows; pass --write-combined-tsv to enable)")

        # 3. Effect-size collapse.
        effects = collapse_priming_effects(priming)
        effects.to_csv(out_dir / "priming_effects_per_model.csv", index=False)
        logger.info(f"Wrote {len(effects)} effect rows → {out_dir / 'priming_effects_per_model.csv'}")
    else:
        logger.warning("No priming data from either source.")

    # 4. Markdown summary.
    summary_path = out_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("# Beetle vs. BGPT priming comparison\n\n")
        fh.write(f"- Beetle priming dir: `{beetle_dir}`\n")
        fh.write(f"- Arnett OSF dir   : `{arnett_dir}`\n\n")
        if not loss.empty:
            fh.write("## Loss curves (mean surprisal)\n\n")
            for src, n in loss.groupby("model_source").size().items():
                fh.write(f"- `{src}`: {n} rows\n")
            fh.write("\n")
        if not priming.empty:
            fh.write("## Priming surprisals\n\n")
            for src, n in priming.groupby("model_source").size().items():
                fh.write(f"- `{src}`: {n} rows\n")
            fh.write("\n## Per-model priming effects\n\n")
            if not effects.empty:
                pivot = effects.pivot_table(
                    index="dataset", columns="model_source",
                    values="priming_effect", aggfunc="mean",
                ).round(4)
                # to_markdown() needs the optional `tabulate` dependency; fall
                # back to a plain pre-formatted block if it's not available.
                try:
                    fh.write(pivot.to_markdown())
                except (ImportError, ModuleNotFoundError):
                    fh.write("```\n")
                    fh.write(pivot.to_string())
                    fh.write("\n```")
                fh.write("\n")
    logger.info(f"Wrote summary → {summary_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare Beetle priming results to Arnett (2025) BGPT numbers",
    )
    p.add_argument("--beetle-dir", default=str(DEFAULT_BEETLE_DIR),
                   help=f"Beetle priming results dir (default: {DEFAULT_BEETLE_DIR})")
    p.add_argument("--arnett-dir", default=str(DEFAULT_ARNETT_DIR),
                   help=f"Arnett OSF dir (default: {DEFAULT_ARNETT_DIR})")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR),
                   help=f"Output dir (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--write-combined-tsv", action="store_true",
                   help="Also write priming_combined.tsv (often >100MB; gitignored).")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
