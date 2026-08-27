#!/usr/bin/env python3
"""
compute_meco_surprisal.py — Compute surprisal from trained BeetleLM conditions
against MECO L2 reading stimuli.

Scans trained model checkpoints from conditions training runs and computes
word-by-word surprisal on MECO stimuli. Results are saved to a TSV file
compatible with the R analysis pipeline in beetlelm-rts-private.

Usage
-----
    python script/compute_meco_surprisal.py \
        --runs-dir ./runs \
        --stimuli ../beetlelm-rts-private/stimuli/meco_l2_stims.tsv \
        --output ../beetlelm-rts-private/stats/results_conditions.tsv \
        --conditions B1 B2 B3 B4 B5 \
        --checkpoint step-10000

    # Compute for all conditions and all checkpoints
    python script/compute_meco_surprisal.py --all
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def find_model_dirs(
    runs_dir: Path,
    conditions: Optional[List[str]] = None,
    checkpoint: str = "latest",
) -> dict:
    """Find trained model directories matching condition patterns."""
    model_dirs = {}

    if not runs_dir.exists():
        print(f"Warning: runs directory not found: {runs_dir}")
        return model_dirs

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name

        # Check if this run matches any condition
        if conditions:
            matched = any(c in run_name for c in conditions)
            if not matched:
                continue

        # Find checkpoint
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue

        if checkpoint == "latest":
            # Find the latest step_N directory
            step_dirs = sorted(
                [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
                key=lambda d: int(d.name.split("_")[1]),
            )
            if step_dirs:
                model_dirs[run_name] = str(step_dirs[-1])
        else:
            # Look for specific checkpoint
            ckpt_name = checkpoint.replace("step-", "step_")
            target = ckpt_dir / ckpt_name
            if target.exists():
                model_dirs[run_name] = str(target)

    return model_dirs


def compute_surprisal(
    model_path: str,
    stimuli_path: str,
    model_name: str,
    device: str = "cuda",
) -> List[float]:
    """Compute conditional surprisal for each word in the stimuli."""
    try:
        from minicons import scorer
    except ImportError:
        print("minicons required. Install: pip install minicons")
        sys.exit(1)

    model = scorer.IncrementalLMScorer(model_path, device=device)

    surprisals = []
    with open(stimuli_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            full_text = row["FullText"]
            word = row["word"]

            if not full_text or not word:
                surprisals.append(float("nan"))
                continue

            # Compute conditional surprisal: -log P(word | prefix)
            try:
                score = model.conditional_score(full_text, word)
                surprisals.append(-score[0] if isinstance(score, list) else -score)
            except Exception:
                surprisals.append(float("nan"))

    del model
    return surprisals


def main():
    parser = argparse.ArgumentParser(description="Compute MECO surprisal from BeetleLM conditions")
    parser.add_argument("--runs-dir", default="./runs", help="Directory with training runs")
    parser.add_argument("--stimuli", default="../beetlelm-rts-private/stimuli/meco_l2_stims.tsv")
    parser.add_argument("--output", default="../beetlelm-rts-private/stats/results_conditions.tsv")
    parser.add_argument("--conditions", nargs="+", help="Condition codes to include (e.g., B1 B2)")
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint to use (e.g., step-10000)")
    parser.add_argument("--all", action="store_true", help="Process all conditions")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device. 'auto' picks cuda if available, else cpu.",
    )
    args = parser.parse_args()

    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    runs_dir = Path(args.runs_dir)
    stimuli_path = Path(args.stimuli)

    if not stimuli_path.exists():
        print(f"Stimuli file not found: {stimuli_path}")
        sys.exit(1)

    conditions = args.conditions
    if args.all:
        conditions = ["B1", "B2", "B3", "B4", "B5", "T1", "T2", "T3", "T4"]

    # Find models
    model_dirs = find_model_dirs(runs_dir, conditions, args.checkpoint)
    if not model_dirs:
        print("No trained models found matching conditions.")
        print(f"Searched in: {runs_dir}")
        if conditions:
            print(f"Conditions: {conditions}")
        sys.exit(1)

    print(f"Found {len(model_dirs)} models:")
    for name, path in model_dirs.items():
        print(f"  {name}: {path}")

    # Load stimuli
    with open(stimuli_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        stimuli_rows = list(reader)
    n_words = len(stimuli_rows)
    print(f"\nStimuli: {n_words} words")

    # Compute surprisal for each model
    results = {}
    for model_name, model_path in model_dirs.items():
        print(f"\nComputing surprisal: {model_name}...")
        try:
            surprisals = compute_surprisal(model_path, str(stimuli_path), model_name, device=device)
            results[model_name] = surprisals
            print(f"  Done ({len(surprisals)} words)")
        except Exception as e:
            print(f"  FAILED: {e}")
            results[model_name] = [float("nan")] * n_words

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        # Header: stimuli columns + one column per model
        fieldnames = list(stimuli_rows[0].keys()) + sorted(results.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for i, row in enumerate(stimuli_rows):
            out_row = dict(row)
            for model_name, surprisals in results.items():
                out_row[model_name] = f"{surprisals[i]:.6f}" if i < len(surprisals) else ""
            writer.writerow(out_row)

    print(f"\nResults written to: {output_path}")
    print(f"Models: {len(results)}")
    print(f"Words: {n_words}")


if __name__ == "__main__":
    main()
