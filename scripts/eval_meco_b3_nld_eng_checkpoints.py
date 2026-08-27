"""Score MECO surprisals for missing checkpoints of the universal-best
nld-eng Beetle model: HumanScale B3 sequential 33:67 nld-eng.

Writes one CSV per (model, step) into ``results/evals_checkpoints/meco/``
matching the existing ``surprisals_<orgslug>_<repo>__step-N.csv`` pattern,
which ``compute_delta_loglik_all_models`` already knows how to ingest.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from analyze.meco_surprisal import score_checkpoint

REPO = "Beetle-HumanScale/beetle-bilingual-l2-50-sequential-33-67-b3-humanscale-nld-eng"
OUT_DIR = ROOT / "results" / "evals_checkpoints" / "meco"
STIMULI = ROOT / "data" / "meco_l2_stims.tsv"
ORG_SLUG = "beetle-humanscale"
REPO_NAME = "beetle-bilingual-l2-50-sequential-33-67-b3-humanscale-nld-eng"


def existing_steps() -> set[int]:
    out: set[int] = set()
    for f in OUT_DIR.glob(f"surprisals_{ORG_SLUG}_{REPO_NAME}__step-*.csv"):
        try:
            out.add(int(f.stem.rsplit("step-", 1)[-1]))
        except Exception:
            pass
    return out


def fetch_branches() -> list[str]:
    import urllib.request, json
    url = f"https://huggingface.co/api/models/{REPO}/refs"
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read())
    out = [b["name"] for b in data.get("branches", [])
           if b["name"].startswith("step-")]
    out.sort(key=lambda b: int(b.split("-")[1]))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    have = existing_steps()
    branches = fetch_branches()
    todo = [b for b in branches if int(b.split("-")[1]) not in have]
    print(f"[checkpoint-meco] {len(branches)} step-N branches; "
          f"{len(have)} already scored; {len(todo)} to score")
    for branch in todo:
        step = int(branch.split("-")[1])
        out_csv = OUT_DIR / f"surprisals_{ORG_SLUG}_{REPO_NAME}__step-{step}.csv"
        if out_csv.exists():
            continue
        print(f"[checkpoint-meco] scoring {branch} → {out_csv.name}")
        try:
            df = score_checkpoint(REPO, STIMULI, revision=branch)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        if df is None or df.empty:
            print(f"  empty df, skipped")
            continue
        df.to_csv(out_csv, index=False)
        print(f"  wrote {len(df)} rows")
    print("[checkpoint-meco] done")


if __name__ == "__main__":
    main()
