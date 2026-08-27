#!/usr/bin/env python3
"""
sync_fineweb_manifest.py — regenerate analyze/fineweb_model_manifest.json
from BeetleLM's FineWeb config YAMLs.

Reads every train_*.yaml under
    <beetlelm>/configs/fineweb_bilingual/{monolingual,core/B*}/
and emits the full list of {Beetle-FineWeb}/{run_name} repos that BeetleLM
will publish when those configs are trained.

The JSON manifest is committed so analyze can resolve planned FineWeb
repos without the BeetleLM repo being present at analyze time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


RUN_NAME_RE = re.compile(r"^\s*run_name\s*:\s*([^\s#]+)\s*(?:#.*)?$", re.M)
HF_USER_RE  = re.compile(r"^\s*hf_user\s*:\s*([^\s#]+)\s*(?:#.*)?$", re.M)


def extract_from_yaml(path: Path) -> tuple[str | None, str | None]:
    try:
        text = path.read_text()
    except OSError:
        return None, None
    rn = RUN_NAME_RE.search(text)
    hu = HF_USER_RE.search(text)
    return (rn.group(1).strip() if rn else None,
            hu.group(1).strip() if hu else None)


def build_manifest(beetlelm_root: Path) -> list[str]:
    roots = [
        beetlelm_root / "configs" / "fineweb_bilingual" / "monolingual",
        beetlelm_root / "configs" / "fineweb_bilingual" / "core",
    ]
    repos: list[str] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for yaml in sorted(root.rglob("train_*.yaml")):
            run_name, hf_user = extract_from_yaml(yaml)
            if not run_name:
                continue
            hf_user = hf_user or "Beetle-FineWeb"
            repo = f"{hf_user}/{run_name}"
            if repo not in seen:
                seen.add(repo)
                repos.append(repo)
    repos.sort()
    return repos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beetlelm", default="../beetlelm",
                    help="Path to sibling BeetleLM repo")
    ap.add_argument("--out", default=None,
                    help="Output JSON path "
                    "(default: ../analyze/fineweb_model_manifest.json)")
    args = ap.parse_args()

    beetlelm = Path(args.beetlelm).resolve()
    if not beetlelm.is_dir():
        print(f"ERROR: BeetleLM repo not found at {beetlelm}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "analyze" / "fineweb_model_manifest.json"
    )

    repos = build_manifest(beetlelm)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(repos, indent=2) + "\n")
    print(f"Wrote {len(repos)} FineWeb repos → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
