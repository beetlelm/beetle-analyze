#!/usr/bin/env python3
"""
scripts/make_beetlelm_blimp_table.py

Consume the CSV emitted by `evals/eval_blimp.py` (one row per (model,
benchmark, language)) and produce a LaTeX table with one row per BeetleLM
model and one accuracy column per evaluated language.

Column order: English (BLiMP) | Bulgarian | Persian | French | Ukrainian |
Indonesian (MultiBLiMP).  Indonesian is emitted as `--` if no rows exist
(jumelet/multiblimp has no `ind` config at time of writing).

Cells left empty when the model's repo name does NOT contain that language.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# (column header, iso_code, benchmark) — column order in output
COLUMNS: list[tuple[str, str, str]] = [
    ("English (BLiMP)",          "eng", "blimp_eng"),
    ("Bulgarian",                "bul", "multiblimp"),
    ("Persian",                  "fas", "multiblimp"),
    ("French",                   "fra", "multiblimp"),
    ("Ukrainian",                "ukr", "multiblimp"),
    ("Indonesian",               "ind", "multiblimp"),
]


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("_", r"\_")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("#", r"\#")
    )


def build_table(df: pd.DataFrame) -> str:
    df = df.copy()
    # Canonicalise types
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")

    # One row per model, aggregate by (benchmark, iso_code)
    models = sorted(df["model"].unique())
    rows: list[list[str]] = []
    for m in models:
        sub = df[df["model"] == m]
        cells: list[str] = []
        for _, iso, bench in COLUMNS:
            hit = sub[(sub["benchmark"] == bench) & (sub["iso_code"] == iso)]
            if hit.empty:
                cells.append("")
            else:
                acc = hit["accuracy"].iloc[0]
                if pd.isna(acc):
                    cells.append("")
                else:
                    cells.append(f"{acc * 100:.1f}")
        rows.append([m, *cells])

    # Build LaTeX
    col_spec = "l" + "r" * len(COLUMNS)
    header = " & ".join([""] + [c for c, _, _ in COLUMNS]) + r" \\"
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{1}{l}{Model} "
                 + "& " + " & ".join(
                     [f"\\textbf{{{c}}}" for c, _, _ in COLUMNS]
                 ) + r" \\")
    lines.append(r"\midrule")
    for r in rows:
        model = _latex_escape(r[0])
        body = " & ".join(c if c else "--" for c in r[1:])
        lines.append(f"{model} & {body} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{BLiMP and MultiBLiMP accuracy (\%) for BeetleLM "
                 r"models. Cells marked `--' indicate the model's training "
                 r"languages do not include that evaluation language, or the "
                 r"language is not available in MultiBLiMP (e.g.\ Indonesian).}")
    lines.append(r"\label{tab:beetlelm-blimp}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="blimp_results.csv path")
    p.add_argument("--output", required=True, help="Where to write .tex")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    required = {"model", "benchmark", "iso_code", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing columns in {args.input}: {missing}",
              file=sys.stderr)
        return 1

    tex = build_table(df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(tex)
    print(f"Wrote LaTeX table → {args.output}  ({len(df)} rows processed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
