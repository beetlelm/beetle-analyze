#!/usr/bin/env python3
"""
evals/eval_meco_english_l2.py — MECO surprisal for English-as-L2 BeetleLM models.

Filters models where English is the L2 and the L1 is one of the MECO
Human-Scale languages covered by BeetleLM (nld, deu, zho), scores each
on ``data/meco_l2_stims.tsv`` using the existing
``analyze.meco_surprisal.score_checkpoint`` pipeline (reusing any cached
``results/meco/surprisals_*.csv``), then writes a per-model summary CSV
and a booktabs-style LaTeX table.

Usage
-----
    # Default: all BeetleLM/* repos with L2==eng and L1 in {nld,deu,zho}
    python evals/eval_meco_english_l2.py

    # Specify models explicitly
    python evals/eval_meco_english_l2.py \\
        --models BeetleLM/beetlelm_nld_L1-eng_L2_balanced \\
                 BeetleLM/beetlelm_zho_L1-eng_L2_simultaneous

    # Force re-scoring (ignore cached surprisals_*.csv)
    python evals/eval_meco_english_l2.py --no-reuse-cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable (evals/ sibling to analyze/, models.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# analyze/ uses ``import _path`` from within itself; make that work too.
_ANALYZE_DIR = _PROJECT_ROOT / "analyze"
if str(_ANALYZE_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYZE_DIR))

from analyze.meco_surprisal import (  # noqa: E402
    DEFAULT_STIMULI,
    DEVICE,
    RESULTS_DIR,
    score_checkpoint,
)
from analyze.model_naming import parse_repo  # noqa: E402
from models import ALL_MODELS  # noqa: E402

# Best-effort ledger record (matches pattern in meco_surprisal.run_meco_eval)
try:
    from analyze.eval_ledger import record as _ledger_record  # noqa: E402
except Exception:  # pragma: no cover — ledger is best-effort
    def _ledger_record(*_a, **_kw) -> None:  # type: ignore[misc]
        pass


# L1 codes covered by MECO Human-Scale in BeetleLM
IN_SCOPE_L1 = frozenset({"nld", "deu", "zho"})

DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "results" / "meco" / "english_l2"


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_english_l2_models(repos: list[str]) -> list[tuple[str, str, str]]:
    """
    Return [(repo, l1, condition_label)] for repos where L2 == 'eng' and
    L1 is in IN_SCOPE_L1. Order preserved, duplicates removed.
    """
    seen: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for repo in repos:
        if not repo.startswith("BeetleLM/"):
            continue
        if repo in seen:
            continue
        meta = parse_repo(repo)
        if meta.l2 != "eng" or meta.l1 not in IN_SCOPE_L1:
            continue
        seen.add(repo)
        out.append((repo, meta.l1, meta.condition))
    return out


# ---------------------------------------------------------------------------
# Surprisal cache
# ---------------------------------------------------------------------------

def _slug(repo: str) -> str:
    """Canonical on-disk slug used by analyze.meco_surprisal.run_meco_eval."""
    return repo.split("/")[-1].replace("-", "__")


def _cached_surprisal_path(repo: str, cache_dir: Path) -> Path:
    return cache_dir / f"surprisals_{_slug(repo)}.csv"


def load_or_score(
    repo: str,
    cache_dir: Path,
    stimuli_path: Path,
    device: str,
    reuse_cache: bool,
) -> pd.DataFrame:
    """Return surprisal DataFrame for ``repo``, scoring if no cache present."""
    cache_path = _cached_surprisal_path(repo, cache_dir)
    if reuse_cache and cache_path.exists():
        print(f"[meco-l2] cached → {cache_path}")
        return pd.read_csv(cache_path)

    print(f"[meco-l2] scoring  → {repo}")
    df = score_checkpoint(repo, stimuli_path, device, revision="main")
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"[meco-l2] saved    → {cache_path}")
    try:
        _ledger_record(
            stage="meco", repo=repo, checkpoint="main",
            benchmark="meco_l2", lang=None, status="ok",
        )
    except Exception:
        pass
    return df


# ---------------------------------------------------------------------------
# Summarisation & LaTeX
# ---------------------------------------------------------------------------

def summarise(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["l1", "condition", "model"]).reset_index(drop=True)


_L1_DISPLAY = {"nld": "Dutch", "deu": "German", "zho": "Chinese"}


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def to_latex_table(df: pd.DataFrame) -> str:
    """Booktabs table: Model | L1 | Condition | Mean Surprisal | N words."""
    lines: list[str] = []
    lines.append(r"\begin{tabular}{llcrr}")
    lines.append(r"\toprule")
    lines.append(
        "Model & L1 & Cond. & Mean surprisal & $N$ words \\\\"
    )
    lines.append(r"\midrule")

    # Highlight the minimum mean-surprisal per L1 block (lower is better for
    # likelihood-based surprisal on held-out text).
    mins_by_l1: dict[str, float] = {}
    for l1 in df["l1"].unique():
        sub = df[df["l1"] == l1]
        if not sub.empty and sub["mean_surprisal"].notna().any():
            mins_by_l1[l1] = float(sub["mean_surprisal"].min())

    current_l1: str | None = None
    for _, row in df.iterrows():
        if current_l1 is not None and row["l1"] != current_l1:
            lines.append(r"\midrule")
        current_l1 = row["l1"]
        model_short = row["model"].split("/")[-1]
        mean_val = row["mean_surprisal"]
        mean_str = (
            f"{mean_val:.3f}" if pd.notna(mean_val) else "---"
        )
        if (
            pd.notna(mean_val)
            and row["l1"] in mins_by_l1
            and np.isclose(float(mean_val), mins_by_l1[row["l1"]])
        ):
            mean_str = r"\textbf{" + mean_str + r"}"
        lines.append(
            f"{_tex_escape(model_short)} & "
            f"{_L1_DISPLAY.get(row['l1'], row['l1'])} & "
            f"{row['condition']} & "
            f"{mean_str} & "
            f"{int(row['n_words'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--models", nargs="+", default=None,
        help="HF repo IDs. Default: BeetleLM/* with L2=eng and L1 in "
             "{nld,deu,zho}.",
    )
    ap.add_argument(
        "--stimuli", default=str(DEFAULT_STIMULI),
        help="Path to MECO stimuli TSV.",
    )
    ap.add_argument(
        "--surprisal-cache-dir", default=str(RESULTS_DIR),
        help="Directory for cached surprisals_{slug}.csv files.",
    )
    ap.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for summary CSV and LaTeX table.",
    )
    ap.add_argument(
        "--device", default=DEVICE, help="Device (cuda or cpu).",
    )
    ap.add_argument(
        "--no-reuse-cache", action="store_true",
        help="Ignore existing surprisal CSVs and re-score every model.",
    )
    args = ap.parse_args()

    stimuli_path = Path(args.stimuli)
    cache_dir = Path(args.surprisal_cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model list
    if args.models:
        requested = args.models
    else:
        requested = ALL_MODELS
    selected = select_english_l2_models(requested)
    if not selected:
        print("[meco-l2] No models matched L2=eng, L1 in {nld,deu,zho}.",
              file=sys.stderr)
        return 1
    print(f"[meco-l2] {len(selected)} model(s) selected:")
    for repo, l1, cond in selected:
        print(f"          {repo:80s}  L1={l1}  cond={cond}")

    rows: list[dict] = []
    for repo, l1, cond in selected:
        try:
            df = load_or_score(
                repo, cache_dir, stimuli_path, args.device,
                reuse_cache=not args.no_reuse_cache,
            )
        except Exception as e:
            print(f"[meco-l2]   FAILED {repo}: {e}", file=sys.stderr)
            rows.append({
                "model": repo, "l1": l1, "condition": cond,
                "n_words": 0,
                "mean_surprisal": float("nan"),
                "median_surprisal": float("nan"),
                "std_surprisal": float("nan"),
            })
            continue

        valid = df["surprisal"].dropna() if "surprisal" in df.columns \
            else pd.Series(dtype=float)
        rows.append({
            "model": repo,
            "l1": l1,
            "condition": cond,
            "n_words": int(len(valid)),
            "mean_surprisal": float(valid.mean()) if len(valid) else float("nan"),
            "median_surprisal": float(valid.median()) if len(valid) else float("nan"),
            "std_surprisal": float(valid.std()) if len(valid) else float("nan"),
        })

    summary = summarise(rows)
    summary_path = output_dir / "summary_by_model.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[meco-l2] Summary CSV → {summary_path}")

    tex_path = output_dir / "table_english_l2.tex"
    tex_path.write_text(to_latex_table(summary))
    print(f"[meco-l2] LaTeX table → {tex_path}")

    print("\n[meco-l2] Summary:")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
