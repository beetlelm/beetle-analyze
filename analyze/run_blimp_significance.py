"""
run_blimp_significance.py — CLI for BLiMP / BLiMP-NL / ZhoBLiMP significance testing.

Reads the existing per-phenomenon accuracy CSVs (results/blimp_eng_results.csv,
results/blimp_nl_results.csv, results/zhoblimp_results.csv) plus a monolingual
baseline CSV, runs the full statistical pipeline in analyze.blimp_significance,
and renders figures 06–09 via analyze.visualise.

Usage
-----
    python -m analyze.run_blimp_significance \\
        --tasks blimp_eng blimp_nl zhoblimp \\
        --monolingual-baseline goldfish \\
        --output results/blimp_significance/

    # Custom CSV paths (e.g. for a new task):
    python -m analyze.run_blimp_significance \\
        --task-csv multiblimp=results/multiblimp_results.csv \\
        --mono-csv multiblimp=results/multiblimp_monolingual.csv \\
        --output results/blimp_significance/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

try:
    import _path  # noqa: F401
except Exception:
    pass

from analyze.blimp_significance import analyse_task

try:
    from analyze.visualise import render_blimp_figures
    _HAVE_FIGURES = True
except Exception as _e:
    _HAVE_FIGURES = False


# Standard registry of (task, bilingual CSV, monolingual CSV) for the main
# tasks called out in the plan. Extend by simply adding entries.
STANDARD_TASKS = {
    "blimp_eng": {
        "bilingual":   Path("results/blimp_eng_results.csv"),
        "monolingual": Path("results/blimp_eng_monolingual.csv"),
    },
    "blimp_nl": {
        "bilingual":   Path("results/blimp_nl_results.csv"),
        "monolingual": Path("results/blimp_nl_monolingual.csv"),
    },
    "zhoblimp": {
        "bilingual":   Path("results/zhoblimp_results.csv"),
        "monolingual": Path("results/zhoblimp_monolingual.csv"),
    },
    "multiblimp": {
        "bilingual":   Path("results/multiblimp_results.csv"),
        "monolingual": Path("results/multiblimp_monolingual.csv"),
    },
}


def _parse_kv(pairs):
    out = {}
    for p in (pairs or []):
        if "=" not in p:
            raise SystemExit(f"Invalid --task-csv/--mono-csv entry: {p} (expected task=path).")
        k, v = p.split("=", 1)
        out[k.strip()] = Path(v.strip())
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="BLiMP / BLiMP-NL / ZhoBLiMP significance pipeline.",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["blimp_eng", "blimp_nl", "zhoblimp"],
        help="Task names to run (must be in STANDARD_TASKS unless --task-csv is used).",
    )
    parser.add_argument(
        "--task-csv", action="append", default=None,
        help="Per-task bilingual CSV override as 'task=path'. Repeatable.",
    )
    parser.add_argument(
        "--mono-csv", action="append", default=None,
        help="Per-task monolingual baseline CSV override as 'task=path'. Repeatable.",
    )
    parser.add_argument(
        "--monolingual-baseline", default="goldfish",
        help="Label used in output summaries (informational).",
    )
    parser.add_argument(
        "--output", default="results/blimp_significance/",
        help="Output directory for CSV/JSON summaries.",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip figure rendering (useful for CI / headless).",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)

    task_csv_over = _parse_kv(args.task_csv)
    mono_csv_over = _parse_kv(args.mono_csv)

    for task in args.tasks:
        spec = STANDARD_TASKS.get(task, {}).copy()
        if task in task_csv_over:
            spec["bilingual"] = task_csv_over[task]
        if task in mono_csv_over:
            spec["monolingual"] = mono_csv_over[task]
        if "bilingual" not in spec or "monolingual" not in spec:
            print(
                f"[run_blimp] Skipping {task}: need both bilingual + monolingual "
                f"CSVs (via STANDARD_TASKS or --task-csv/--mono-csv overrides)."
            )
            continue
        bil_csv = Path(spec["bilingual"])
        mono_csv = Path(spec["monolingual"])
        if not bil_csv.exists() or not mono_csv.exists():
            print(
                f"[run_blimp] Skipping {task}: missing file(s): "
                f"{bil_csv} exists={bil_csv.exists()}, "
                f"{mono_csv} exists={mono_csv.exists()}"
            )
            continue

        print(f"[run_blimp] Analysing {task} …")
        out_paths = analyse_task(
            task_name=task,
            bilingual_csv=bil_csv,
            monolingual_csv=mono_csv,
            output_dir=output_dir,
        )
        for label, p in out_paths.items():
            print(f"           → {label}: {p}")

        if not args.no_figures and _HAVE_FIGURES:
            figs = render_blimp_figures(
                task=task,
                bilingual_csv=bil_csv,
                monolingual_csv=mono_csv,
                significance_dir=output_dir,
            )
            for k, p in figs.items():
                print(f"           [fig {k}] {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
