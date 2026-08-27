#!/usr/bin/env python3
"""
sync_ablation_manifest.py — regenerate analyze/ablation_manifest.json
from BeetleLM's ablation manifest CSV (or synthesize from ABLATION_VARIANTS
if the training run hasn't produced the CSV yet).

Sources (first one that exists wins):
    1. <beetlelm>/logs_ablation_<l1>_<l2>/variants_manifest.csv
    2. Synthesized from BeetleLM's ABLATION_VARIANTS × SEEDS × {nld-eng}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


DEFAULT_VARIANTS = [
    "B1", "B1b", "B2", "B2_ewc", "B2_lamol", "B2_lr", "B3", "B3_full",
]
DEFAULT_SEEDS = [13, 42, 97]
DEFAULT_PAIRS = [("nld", "eng")]


def _variant_slug(variant: str, l1: str, l2: str) -> str:
    """Map an ablation variant code to its full HF slug (without org/seed).

    Mirrors beetlelm/src/config/_repo_naming.py::build_model_run_name.
    """
    root_b1 = f"beetle-bilingual-balanced-b1-humanscale-{l1}-{l2}"
    root_b2 = f"beetle-bilingual-l2-50-simultaneous-b2-humanscale-{l1}-{l2}"
    root_b3 = f"beetle-bilingual-l2-50-sequential-33-67-b3-humanscale-{l1}-{l2}"
    root_b3_full = f"beetle-bilingual-l2-50-sequential-0-100-b3-humanscale-{l1}-{l2}"

    return {
        "B1":       root_b1,
        "B1a":      root_b1,                      # a/b not slug-marked; ignore
        "B1b":      root_b1 + "-episodic",
        "B2":       root_b2,
        "B2_later": root_b2,
        "B2_smooth":root_b2,                      # slug-identical; differs in YAML
        "B2_burst": root_b2,
        "B2_ewc":   root_b2 + "-ewc-l20",
        "B2_lamol": root_b2 + "-lamol-l20",
        "B2_lr":    root_b2 + "-lr-b2",
        "B3":       root_b3,
        "B3_full":  root_b3_full,
        "B3_25_75": root_b3.replace("33-67", "25-75"),
        "B4":       f"beetle-bilingual-l2-50-classroom-20-b4-humanscale-{l1}-{l2}",
        "B4a":      f"beetle-bilingual-l2-50-classroom-20-b4-humanscale-{l1}-{l2}",
        "B4b":      f"beetle-bilingual-l2-50-classroom-20-b4-humanscale-{l1}-{l2}",
        "B4b_lamol":f"beetle-bilingual-l2-50-classroom-20-b4-humanscale-{l1}-{l2}-lamol-l20",
        "B5":       f"beetle-bilingual-l2-80-late-b5-humanscale-{l1}-{l2}",
        "B5_ewc":   f"beetle-bilingual-l2-80-late-b5-humanscale-{l1}-{l2}-ewc-l40",
        "B5_lamol": f"beetle-bilingual-l2-80-late-b5-humanscale-{l1}-{l2}-lamol-l20",
    }.get(variant, "")


def build_from_csv(csv_paths: list[Path]) -> list[str]:
    """Read ``variants_manifest.csv`` rows and translate to HF repo ids."""
    seen: set[str] = set()
    out: list[str] = []
    for csv_path in csv_paths:
        if not csv_path.is_file():
            continue
        with csv_path.open("r", newline="") as f:
            rd = csv.DictReader(f)
            for row in rd:
                slug = (row.get("hf_push_suffix") or "").strip()
                seed = (row.get("seed") or "").strip()
                if not slug:
                    continue
                if seed and not slug.endswith(f"-seed{seed}"):
                    slug = f"{slug}-seed{seed}"
                repo = f"BeetleLM/{slug}"
                if repo not in seen:
                    seen.add(repo)
                    out.append(repo)
    out.sort()
    return out


def synthesize(variants, seeds, pairs) -> list[str]:
    out: set[str] = set()
    for v in variants:
        for l1, l2 in pairs:
            slug = _variant_slug(v, l1, l2)
            if not slug:
                continue
            for s in seeds:
                out.add(f"BeetleLM/{slug}-seed{s}")
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beetlelm", default="../beetlelm",
                    help="Path to sibling BeetleLM repo")
    ap.add_argument("--out", default=None,
                    help="Output JSON path "
                    "(default: ../analyze/ablation_manifest.json)")
    ap.add_argument("--pairs", default="nld-eng",
                    help="Comma-sep L1-L2 pairs to synthesize if no CSV found")
    args = ap.parse_args()

    beetlelm = Path(args.beetlelm).resolve()
    pairs = [tuple(p.split("-")) for p in args.pairs.split(",") if "-" in p]

    csv_candidates = []
    if beetlelm.is_dir():
        for p in beetlelm.glob("logs_ablation_*"):
            csv_candidates.append(p / "variants_manifest.csv")
    repos = build_from_csv(csv_candidates) if csv_candidates else []
    source = "csv"

    if not repos:
        repos = synthesize(DEFAULT_VARIANTS, DEFAULT_SEEDS, pairs)
        source = "synthesized"

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "analyze" / "ablation_manifest.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(repos, indent=2) + "\n")
    print(f"Wrote {len(repos)} ablation repos → {out}  (source={source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
