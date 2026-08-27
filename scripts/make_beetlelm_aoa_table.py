#!/usr/bin/env python3
"""
scripts/make_beetlelm_aoa_table.py

Consume the CSV emitted by ``evals/eval_aoa.py`` (one row per
(model, language)) and produce a LaTeX table with:

    * one row per model
    * an L1 column (parsed from the model repo name, or ``multi`` for baselines)
    * one Spearman-rho column per evaluation language

Invoked automatically by ``scripts/run_beetlelm_aoa.sh`` (Phase C) — not
intended for manual use, though it can be re-run standalone to rebuild the
table from an existing CSV.

Columns are fixed in a stable order; languages absent from the CSV are
dropped to keep the table compact. Cells are formatted as ``.3f`` or ``--``
when that (model, language) cell has no rho.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# (column header, iso_code) — stable display order matching analysis focus.
CANDIDATE_COLUMNS: list[tuple[str, str]] = [
    ("English",    "eng"),
    ("Dutch",      "nld"),
    ("German",     "deu"),
    ("Chinese",    "zho"),
    ("French",     "fra"),
    ("Persian",    "fas"),
    ("Bulgarian",  "bul"),
    ("Ukrainian",  "ukr"),
    ("Spanish",    "spa"),
    ("Russian",    "rus"),
    ("Portuguese", "por"),
    ("Japanese",   "jpn"),
    ("Italian",    "ita"),
    ("Polish",     "pol"),
    ("Turkish",    "tur"),
    ("Greek",      "ell"),
    ("Hungarian",  "hun"),
    ("Czech",      "ces"),
    ("Catalan",    "cat"),
    ("Danish",     "dan"),
    ("Swedish",    "swe"),
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
    df["spearman_rho"] = pd.to_numeric(df["spearman_rho"], errors="coerce")

    # Keep only columns for languages actually present in the CSV.
    present_iso = set(df["iso_code"].dropna().unique())
    columns = [(h, iso) for h, iso in CANDIDATE_COLUMNS if iso in present_iso]
    if not columns:
        raise ValueError("No known AoA languages found in the input CSV.")

    # One row per model. Pick the L1 from the first row for that model.
    models = sorted(df["model"].unique())

    rows: list[list[str]] = []
    for m in models:
        sub = df[df["model"] == m]
        l1 = sub["l1_code"].dropna().iloc[0] if sub["l1_code"].notna().any() else ""
        cells: list[str] = [m, str(l1)]
        for _, iso in columns:
            hit = sub[sub["iso_code"] == iso]
            if hit.empty:
                cells.append("")
                continue
            rho = hit["spearman_rho"].iloc[0]
            if pd.isna(rho):
                cells.append("")
            else:
                cells.append(f"{rho:.3f}")
        rows.append(cells)

    # Build LaTeX
    col_spec = "ll" + "r" * len(columns)
    header_cols = ["Model", "L1"] + [h for h, _ in columns]
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(f"\\textbf{{{c}}}" for c in header_cols) + r" \\")
    lines.append(r"\midrule")
    for r in rows:
        model = _latex_escape(r[0])
        l1 = _latex_escape(r[1]) if r[1] else "--"
        body = " & ".join(c if c else "--" for c in r[2:])
        lines.append(f"{model} & {l1} & {body} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Final-checkpoint Age-of-Acquisition Spearman $\rho$ "
        r"between per-word mean surprisal and human AoA norms "
        r"(\texttt{aoa/standardized/\{lang\}.csv}). Higher $\rho$ = model "
        r"surprisal tracks human AoA better. The L1 column is parsed from "
        r"the model repo name (\texttt{multi} for language-agnostic "
        r"baselines). Cells marked `--' indicate the model's training "
        r"languages do not include that evaluation language.}"
    )
    lines.append(r"\label{tab:beetlelm-aoa}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="aoa_results.csv path")
    p.add_argument("--output", required=True, help="Where to write .tex")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    required = {"model", "l1_code", "iso_code", "spearman_rho"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing columns in {args.input}: {missing}", file=sys.stderr)
        return 1

    tex = build_table(df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(tex)
    print(f"Wrote LaTeX table → {args.output}  ({len(df)} rows processed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
