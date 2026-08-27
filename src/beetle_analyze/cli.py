"""Command-line interface for beetle-analyze.

    beetle-analyze blimp                 # run the headline analysis on bundled toy data
    beetle-analyze blimp --scoring mean  # change one variable and re-run
    beetle-analyze blimp --input my_evals.jsonl --json-out result.json
    beetle-analyze version
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .blimp import analyze_file, toy_data_path


def _load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _cmd_blimp(args: argparse.Namespace) -> int:
    cfg: dict = {}
    if args.config:
        cfg = _load_config(args.config)

    input_path = args.input or cfg.get("input") or str(toy_data_path())
    scoring = args.scoring or cfg.get("scoring", "sum")
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    resamples = args.bootstrap if args.bootstrap is not None else cfg.get("bootstrap_resamples", 2000)

    result = analyze_file(input_path, scoring=scoring, bootstrap_resamples=resamples, seed=seed)

    print(f"beetle-analyze BLiMP  (input={input_path})")
    print(f"  scoring         : {result.scoring}")
    print(f"  pairs           : {result.n_pairs}")
    print(f"  overall accuracy: {result.overall_accuracy:.4f}  "
          f"[95% CI {result.ci_low:.4f}, {result.ci_high:.4f}]")
    print("  per phenomenon  :")
    for row in result.per_phenomenon.itertuples():
        print(f"      {row.phenomenon:<30} {row.accuracy:6.3f}  (n={row.n_pairs})")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        print(f"  wrote metrics   : {args.json_out}")

    if args.plot_out:
        from .plots import plot_per_phenomenon

        p = plot_per_phenomenon(result, args.plot_out)
        print(f"  wrote figure    : {p}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="beetle-analyze", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("blimp", help="run the BLiMP-style minimal-pair analysis")
    b.add_argument("--input", help="JSONL eval-output file (default: bundled toy data)")
    b.add_argument("--config", help="JSON config file (CLI flags override its values)")
    b.add_argument("--scoring", choices=["sum", "mean"], help="score sentences by total (sum) or per-token (mean) log-prob")
    b.add_argument("--bootstrap", type=int, help="bootstrap resamples for the CI (default 2000)")
    b.add_argument("--seed", type=int, help="RNG seed for the bootstrap")
    b.add_argument("--json-out", help="write the metrics to this JSON file")
    b.add_argument("--plot-out", help="write a per-phenomenon bar chart to this PNG (needs [plots] extra)")
    b.set_defaults(func=_cmd_blimp)

    v = sub.add_parser("version", help="print the installed version")
    v.set_defaults(func=lambda _a: (print(__version__), 0)[1])

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
