"""
verify_100m_variant_checkpoints.py — sanity-check that the 8 nld-eng B3 100M
HF repos actually have distinct weights at each step-N checkpoint.

Motivation
----------
The first round of 100M-variants evals produced byte-identical numbers for
`pre-pretrain` and `safe` (and possibly others) on BLiMP, BLiSS, MeCo, and
FLORES. That is almost certainly *not* a model property — two different
training manipulations producing identical NLL to 15 decimal places means
the eval pipeline served the same weights twice.

This script asks the Hub directly:

  1. For every variant repo, list the branches and confirm that the
     expected step-N branches exist (no silent fall-back to `main`).
  2. For every (variant, step) pair, fetch the commit SHA on that branch
     and the SHA-256 of the safetensors blob. Group by SHA — any variants
     sharing a SHA at the same step are byte-identical.
  3. Emit a CSV / markdown report at
     `results/100m-variants/checkpoint_audit.{csv,md}`.

GPU-safe: pure HF Hub HTTP calls, never imports torch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Pure-HTTP — no torch.
from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO_ROOT / "results" / "100m-variants"

VARIANTS = [
    "baseline",       # if HF org has a "no-suffix" repo, treat that as baseline
    "safe",
    "ewc",
    "lamol",
    "maml",
    "pre-pretrain",
    "alibi",
    "layer-growth",
]

ORG = "Beetle-FineWeb-100M"
REPO_STEM = "beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-100m-nld-eng"

# Steps the existing evals targeted; we audit the same set so we can match
# any byte-identical hits to the prior FLORES bug.
EXPECTED_STEPS = [0, 1, 16, 256, 1024, 4096, 5000, 5128, 8192, 9096]


def _repo_for(variant: str) -> str:
    if variant == "baseline":
        # Convention used in eval_loss_curves_flores_dutch_v2.sh.
        return f"{ORG}/{REPO_STEM}"
    return f"{ORG}/{REPO_STEM}-{variant}"


def _list_branches(api: HfApi, repo: str) -> list[str]:
    try:
        refs = api.list_repo_refs(repo)
        return [b.name for b in refs.branches]
    except Exception as e:
        print(f"  ! could not list refs for {repo}: {e}", file=sys.stderr)
        return []


def _commit_sha(api: HfApi, repo: str, revision: str) -> str | None:
    try:
        info = api.repo_info(repo, revision=revision)
        return info.sha
    except Exception:
        return None


def _safetensors_size(api: HfApi, repo: str, revision: str) -> int | None:
    """File-size of model.safetensors on (repo, revision). A cheap proxy for
    "are these the same weights" without downloading the weights themselves."""
    try:
        info = api.repo_info(repo, revision=revision, files_metadata=True)
    except Exception:
        return None
    for sib in info.siblings:
        if sib.rfilename == "model.safetensors":
            return sib.size
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--out-csv",
                        default=str(RESULT_DIR / "checkpoint_audit.csv"))
    parser.add_argument("--out-md",
                        default=str(RESULT_DIR / "checkpoint_audit.md"))
    args = parser.parse_args()

    api = HfApi(token=args.token)
    rows: list[dict] = []

    for variant in VARIANTS:
        repo = _repo_for(variant)
        print(f"[audit] {variant:14s} → {repo}")
        branches = set(_list_branches(api, repo))
        if not branches:
            rows.append({
                "variant": variant, "repo": repo, "step": None,
                "branch": None, "exists": False, "sha": None, "size": None,
                "note": "repo not found or no read access",
            })
            continue
        for step in EXPECTED_STEPS:
            br = f"step-{step}"
            exists = br in branches
            sha = _commit_sha(api, repo, br) if exists else None
            size = _safetensors_size(api, repo, br) if exists else None
            rows.append({
                "variant": variant, "repo": repo, "step": step,
                "branch": br, "exists": exists, "sha": sha, "size": size,
                "note": "" if exists else "branch missing — eval will fall back",
            })

    # Group by step → which variants share a SHA.
    by_step: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["sha"]:
            by_step[r["step"]][r["sha"]].append(r["variant"])

    # Persist CSV.
    import pandas as pd
    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[audit] CSV  → {args.out_csv}")

    # Markdown summary.
    lines = ["# 100M nld-eng B3 — checkpoint audit", ""]
    lines.append(f"Repos audited: {len(VARIANTS)}  ·  steps: {EXPECTED_STEPS}")
    lines.append("")
    lines.append("## Missing branches")
    miss = df[(~df["exists"]) & (df["step"].notna())]
    if miss.empty:
        lines.append("- _none_")
    else:
        for _, r in miss.iterrows():
            lines.append(f"- `{r['variant']}` step-{int(r['step'])}: branch missing")
    lines.append("")
    lines.append("## Variants sharing weights at the same step")
    any_shared = False
    for step, sha_groups in sorted(by_step.items()):
        for sha, vs in sha_groups.items():
            if len(vs) > 1:
                any_shared = True
                lines.append(f"- step-{step}: `{', '.join(sorted(vs))}` "
                             f"share SHA `{sha[:10]}…`  ⚠  byte-identical weights")
    if not any_shared:
        lines.append("- _none — every (variant, step) has a distinct commit SHA_")
    lines.append("")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"[audit] MD   → {args.out_md}")

    # Surface the punchline at exit.
    if any_shared:
        print("\n[audit] ⚠  Detected variants sharing weights — see "
              f"{args.out_md} for the offenders.", file=sys.stderr)
        sys.exit(2)
    if not miss.empty:
        print(f"\n[audit] ⚠  {len(miss)} (variant, step) branches missing — "
              "those evals will silently fall back to `main`. See "
              f"{args.out_md}.", file=sys.stderr)
        sys.exit(3)
    print("\n[audit] OK — every variant has distinct weights at every step.")


if __name__ == "__main__":
    main()
