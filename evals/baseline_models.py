"""
evals/baseline_models.py — Registry of baseline multilingual LLMs.

Provides BASELINE_MODELS (grouped by family) and helpers for resolving
the --models CLI argument used by eval_blimp.py, eval_meco.py, and
eval_flores_ppl.py.

Beetle model enumeration is done via the HuggingFace Hub API and cached
locally for 24 hours to avoid repeated network calls.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Baseline model registry
# ---------------------------------------------------------------------------

BASELINE_MODELS: dict[str, list[str]] = {
    "eurollm": [
        "utter-project/EuroLLM-1.7B",
        "utter-project/EuroLLM-9B",
    ],
    "xglm": [
        "facebook/xglm-564M",
        "facebook/xglm-1.7B",
        "facebook/xglm-2.9B",
        "facebook/xglm-4.5B",
        "facebook/xglm-7.5B",
    ],
    "bloom": [
        "bigscience/bloom-560m",
        "bigscience/bloom-1b1",
        "bigscience/bloom-1b7",
        "bigscience/bloom-3b",
        "bigscience/bloom-7b1",
    ],
    # Frontier multilingual baselines (added 2026-04-24)
    "gemma": [
        "google/gemma-2-2b",
        "google/gemma-2-9b",
        "google/gemma-2-27b",
    ],
    "qwen": [
        "Qwen/Qwen3-4B-Base",
        "Qwen/Qwen3-8B-Base",
    ],
    "llama": [
        "meta-llama/Llama-3.1-8B",
    ],
    "apertus": [
        "swiss-ai/Apertus-8B-2509",
        "swiss-ai/Apertus-70B-2509",
    ],
    # Japanese–English cohort (added 2026-04-27): LLM-jp v3 family.
    # 172B-beta1 is xlarge — gated behind SKIP_LLMJP_172B in the launcher.
    "llmjp": [
        "llm-jp/llm-jp-3-150m",
        "llm-jp/llm-jp-3-440m",
        "llm-jp/llm-jp-3-980m",
        "llm-jp/llm-jp-3-1.8b",
        "llm-jp/llm-jp-3-3.7b",
        "llm-jp/llm-jp-3-13b",
        "llm-jp/llm-jp-3-172b-beta1",
    ],
    # Chinese–English cohort (added 2026-04-27).
    # Qwen-14B and Baichuan-7B require trust_remote_code=True (already set
    # in evals/model_loader.py:load_model). ChatGLM-6B uses a non-CausalLM
    # head — model_loader will log SKIP and continue if AutoModelForCausalLM
    # cannot map its architecture.
    "chinese_english": [
        "Qwen/Qwen-14B",
        "baichuan-inc/Baichuan-7B",
        "01-ai/Yi-6B",
        "THUDM/chatglm-6b",
    ],
}

#: Frontier cohort — public multilingual LLMs released 2024-2025.
#: Resolved by ``--models frontier`` on the eval CLIs.
FRONTIER_FAMILIES: tuple[str, ...] = ("gemma", "qwen", "llama", "apertus", "eurollm")

# All baselines flattened
ALL_BASELINES: list[str] = [m for ms in BASELINE_MODELS.values() for m in ms]

# Beetle org identifiers → HuggingFace org names
BEETLE_ORGS: dict[str, str] = {
    "beetle-humanscale": "Beetle-HumanScale",
    "humanscale": "Beetle-HumanScale",
    "beetle-fineweb": "Beetle-FineWeb",
    "fineweb": "Beetle-FineWeb",
    "beetlelm": "BeetleLM",
}

# Cache TTL (seconds)
_CACHE_TTL = 24 * 3600

# ---------------------------------------------------------------------------
# Non-model repo filtering
# ---------------------------------------------------------------------------

# Repo name prefixes that identify tokenizer / dataset artefacts published
# alongside actual language models in the Beetle-HumanScale org.
_BEETLE_NON_MODEL_PREFIXES: tuple[str, ...] = (
    "bpe-",        # Beetle-HumanScale/bpe-humanscale-*  (BPE tokenizer repos)
    "bpe_",        # BeetleLM/bpe_BabyBabel-*, BeetleLM/bpe_babylm-*  (BPE tokenizers)
    "BabyBabel-",  # Beetle-HumanScale/BabyBabel-*        (held-out eval datasets)
)

# Orgs that publish only tokenizers / datasets — never actual LMs.
_TOKENIZER_ONLY_ORGS: frozenset[str] = frozenset({"Beetle-Data"})


def _is_model_repo(org: str, full_repo_id: str) -> bool:
    """Return True iff *full_repo_id* is an evaluable language model.

    Filters out tokenizer repos (bpe-*), held-out dataset repos
    (BabyBabel-*), and any repo belonging to a tokenizer-only org
    (Beetle-Data).
    """
    if org in _TOKENIZER_ONLY_ORGS:
        return False
    name = full_repo_id.split("/", 1)[-1]   # strip "Org/" prefix
    return not any(name.startswith(p) for p in _BEETLE_NON_MODEL_PREFIXES)


# ---------------------------------------------------------------------------
# Org model enumeration
# ---------------------------------------------------------------------------

def _model_cache_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / ".model_list_cache.json"


def _list_org_models(org: str, output_dir: str | Path) -> list[str]:
    """Return model repo IDs for *org* (non-tokenizer repos only), 24h cache."""
    cache_path = _model_cache_path(output_dir)
    now = time.time()

    # Try cache
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            entry = data.get(org, {})
            if now - entry.get("ts", 0) < _CACHE_TTL:
                return entry["models"]
        except Exception:
            pass

    # Fetch from HF Hub and filter out non-model repos
    try:
        from huggingface_hub import list_models as hf_list_models
        repos = [
            m.id for m in hf_list_models(author=org)
            if _is_model_repo(org, m.id)
        ]
    except Exception as exc:
        print(f"[baseline_models] WARNING: could not list {org} models: {exc}")
        repos = []

    # Update cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {}
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
        data[org] = {"ts": now, "models": repos}
        cache_path.write_text(json.dumps(data, indent=2))
    except Exception:
        pass

    return repos


def resolve_models(
    model_specs: list[str],
    output_dir: str | Path = "results/evals",
) -> list[tuple[str, str]]:
    """Resolve a list of model spec strings to (repo_id, family_label) pairs.

    Spec strings may be:
      - "beetle-humanscale" / "humanscale"  → enumerate Beetle-HumanScale org
      - "beetle-fineweb" / "fineweb"        → enumerate Beetle-FineWeb org
      - "baselines"                         → all EuroLLM + XGLM + BLOOM
      - "eurollm" / "xglm" / "bloom"       → specific baseline family
      - any HF repo ID (org/model)          → used as-is, labelled "custom"
    """
    seen: set[str] = set()
    result: list[tuple[str, str]] = []

    def _add(repo: str, label: str) -> None:
        if repo not in seen:
            seen.add(repo)
            result.append((repo, label))

    for spec in model_specs:
        original_spec = spec.strip()
        spec = original_spec.lower()

        if spec in BEETLE_ORGS:
            org = BEETLE_ORGS[spec]
            label = spec.replace("-", "_")
            for repo in _list_org_models(org, output_dir):
                _add(repo, label)

        elif spec == "baselines":
            for family, repos in BASELINE_MODELS.items():
                for repo in repos:
                    _add(repo, family)

        elif spec == "frontier":
            for family in FRONTIER_FAMILIES:
                for repo in BASELINE_MODELS[family]:
                    _add(repo, family)

        elif spec in BASELINE_MODELS:
            for repo in BASELINE_MODELS[spec]:
                _add(repo, spec)

        else:
            # Treat as a raw HF repo ID — preserve original case (HF
            # namespaces and repo names are case-sensitive).
            _add(original_spec if "/" in original_spec else original_spec, "custom")

    return result


def model_family_label(repo: str) -> str:
    """Return a short family label for *repo* (used in output CSVs)."""
    for family, repos in BASELINE_MODELS.items():
        if repo in repos:
            return family
    for key, org in BEETLE_ORGS.items():
        if repo.startswith(org + "/") or repo.startswith(org.lower() + "/"):
            return key.replace("-", "_")
    return "custom"


# ---------------------------------------------------------------------------
# Size tiers + recommended batch sizes
# ---------------------------------------------------------------------------
#
# Why this exists: the parallel rank launcher in
# scripts/eval_frontier_baselines.sh runs N_GPUS copies of the per-model
# orchestrator simultaneously. Each rank loads the *full* model on its own
# GPU at bf16, with no device_map sharding. A static --batch-size-blimp=32
# works for 1-3B models but blows past 80 GB on Gemma-2-27B once KV cache
# is included, and Apertus-70B / LLM-jp-172B exceed a single GPU outright.
#
# We expose a coarse size tier per repo, plus per-tier recommended BLiMP
# and BLiSS batch sizes that the orchestrator can apply automatically.
# Repos not listed here fall through to the small-tier default.

# Repo → tier among {"small", "medium", "large", "xlarge"}.
# Tier definitions (params, bf16):
#   small  : <  4 B   — runs comfortably with batch=32 on a single A100-80G
#   medium : 4 – 10 B — needs batch≤16
#   large  : 10 – 30 B — needs batch≤8, may need device_map="auto" on A100-40G
#   xlarge : >= 30 B  — REQUIRES device_map="auto" + multi-GPU; rank-parallel
#                       launches will OOM. The launcher serialises these.
MODEL_SIZE_HINT: dict[str, str] = {
    # Gemma 2
    "google/gemma-2-2b":           "small",
    "google/gemma-2-9b":           "medium",
    "google/gemma-2-27b":          "large",
    # Qwen 3 base
    "Qwen/Qwen3-4B-Base":          "small",
    "Qwen/Qwen3-8B-Base":          "medium",
    # Llama 3.1
    "meta-llama/Llama-3.1-8B":     "medium",
    # Apertus
    "swiss-ai/Apertus-8B-2509":    "medium",
    "swiss-ai/Apertus-70B-2509":   "xlarge",
    # EuroLLM
    "utter-project/EuroLLM-1.7B":  "small",
    "utter-project/EuroLLM-9B":    "medium",
    # LLM-jp v3
    "llm-jp/llm-jp-3-150m":        "small",
    "llm-jp/llm-jp-3-440m":        "small",
    "llm-jp/llm-jp-3-980m":        "small",
    "llm-jp/llm-jp-3-1.8b":        "small",
    "llm-jp/llm-jp-3-3.7b":        "small",
    "llm-jp/llm-jp-3-13b":         "large",
    "llm-jp/llm-jp-3-172b-beta1":  "xlarge",
    # Chinese–English
    "Qwen/Qwen-14B":               "large",
    "baichuan-inc/Baichuan-7B":    "medium",
    "01-ai/Yi-6B":                 "medium",
    "THUDM/chatglm-6b":            "medium",
}

# Per-tier recommended batch sizes for the BLiMP/BLiSS scorers running on
# A100-80G. Halve again on A100-40G via env overrides.
_TIER_BATCH_SIZES: dict[str, dict[str, int]] = {
    "small":  {"blimp": 32, "bliss": 16},
    "medium": {"blimp": 16, "bliss":  8},
    "large":  {"blimp":  8, "bliss":  4},
    "xlarge": {"blimp":  4, "bliss":  2},
}


def model_size_tier(repo: str) -> str:
    """Return the size tier for *repo*. Defaults to 'small' if unknown."""
    return MODEL_SIZE_HINT.get(repo, "small")


def recommended_batch_sizes(repo: str) -> dict[str, int]:
    """Return {'blimp': int, 'bliss': int} recommended batch sizes for *repo*.

    Used by evals/eval_per_model.py when --auto-batch-size is enabled.
    Callers may still override via --batch-size-blimp / --batch-size-bliss.
    """
    tier = model_size_tier(repo)
    return dict(_TIER_BATCH_SIZES[tier])


def needs_device_map_auto(repo: str) -> bool:
    """True iff *repo* should be loaded with device_map='auto' across all
    visible GPUs (i.e. tier == xlarge). The launcher must run such repos
    in a serial pass with all N_GPUS visible to a single process.
    """
    return model_size_tier(repo) == "xlarge"
