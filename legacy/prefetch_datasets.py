#!/usr/bin/env python3
"""
prefetch_datasets.py — Download all multi-config benchmark datasets ONCE
                        and save them as plain pickle files.

Run by launch_all.sh before the 8 GPU worker processes start.
Workers load from .pkl directly — no HTTP requests, no 429s.

Usage:
    PKL_DIR=/path/to/results/.pkl_cache python3 prefetch_datasets.py

If a .pkl file already exists it is verified (non-empty) and skipped,
so re-running after a crash is safe.
"""

import os
import pickle
import sys
import time

from datasets import get_dataset_config_names, load_dataset

PKL_DIR = os.environ.get("PKL_DIR", "results/.pkl_cache")
os.makedirs(PKL_DIR, exist_ok=True)

# (hf_id, good_col, bad_col, output_filename)
TARGETS = [
    ("Junrui1202/zhoblimp",  "sentence_good", "sentence_bad", "zhoblimp.pkl"),
    ("juletxara/blimp-nl",   "sentence_good", "sentence_bad", "blimp_nl.pkl"),
    ("nyu-mll/blimp",        "sentence_good", "sentence_bad", "blimp_eng.pkl"),
]


def fetch_all_configs(hf_id: str, good_col: str, bad_col: str) -> list:
    configs = get_dataset_config_names(hf_id)
    print(f"    {len(configs)} configs to fetch", flush=True)
    pairs = []
    for i, cfg in enumerate(configs):
        if i > 0:
            time.sleep(0.15)   # stay well under HF rate limit
        for attempt in range(4):
            try:
                try:
                    ds = load_dataset(hf_id, cfg, split="train")
                except Exception:
                    ds_dict = load_dataset(hf_id, cfg)
                    ds      = ds_dict[list(ds_dict.keys())[0]]
                pairs.extend(zip(ds[good_col], ds[bad_col]))
                del ds
                break
            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    wait = 30 * (2 ** attempt)   # 30s, 60s, 120s
                    print(f"    429 on '{cfg}' — waiting {wait}s …", flush=True)
                    time.sleep(wait)
                else:
                    print(f"    warn: skipping '{cfg}': {e}", flush=True)
                    break
        if (i + 1) % 25 == 0:
            print(f"    … {i+1}/{len(configs)} configs done", flush=True)
    return list(pairs)


def main():
    all_ok = True
    for hf_id, good_col, bad_col, fname in TARGETS:
        out_path = os.path.join(PKL_DIR, fname)

        # Skip if already fetched
        if os.path.exists(out_path):
            with open(out_path, "rb") as fh:
                existing = pickle.load(fh)
            if existing:
                print(f"  {fname}: already cached ({len(existing):,} pairs) — skipping",
                      flush=True)
                continue
            else:
                print(f"  {fname}: found but empty — re-fetching", flush=True)

        print(f"  {hf_id}  →  {fname}", flush=True)
        try:
            pairs = fetch_all_configs(hf_id, good_col, bad_col)
            if not pairs:
                print(f"    ERROR: fetched 0 pairs for {hf_id}", flush=True)
                all_ok = False
                continue
            with open(out_path, "wb") as fh:
                pickle.dump(pairs, fh)
            print(f"    saved {len(pairs):,} pairs", flush=True)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            all_ok = False

    if all_ok:
        print("Pre-fetch complete — all datasets saved.", flush=True)
    else:
        print("Pre-fetch finished with errors — check warnings above.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
