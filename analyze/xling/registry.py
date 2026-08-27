"""
analyze/xling/registry.py
─────────────────────────
Model-agnostic registry mapping (HF repo, revision) → architectural
metadata + per-language token streams.

Why this exists
---------------
The ``xling`` analyses (SVCCA, CKA, probes, CrossEncoder) all need to
know:

  * how many transformer blocks the model has,
  * how many attention heads per block,
  * the hidden / head dimensions,
  * which HF parallel-corpus pairs are valid for the model's tokenizer.

Hard-coding this against the current 5 NLD-ENG bilingual checkpoints
would prevent the framework from being reused for future multilingual
Beetle models.  Instead, we register named *configurations* and look
them up by the same ``family`` / ``condition`` / ``variant`` keys that
``analyze/run.py`` already uses.

The Beetle-FineWeb-2B NLD-ENG B1–B5 checkpoints are pre-registered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Language code constants
# ---------------------------------------------------------------------------

# ISO 639-3 codes used everywhere in beetle-analyze.
LANG_NLD = "nld"
LANG_ENG = "eng"


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

@dataclass
class LanguageConfig:
    """A single language inside a multilingual model.

    ``hf_corpus`` is the HF dataset id for monolingual evaluation text
    (used by SVCCA / CKA to extract activations).  ``flores_code`` maps
    the language to its FLORES-200 script-tagged identifier (e.g.
    ``nld_Latn``) for parallel-corpus probing.
    """
    iso: str
    hf_corpus: str
    flores_code: str


@dataclass
class ModelConfig:
    """Architectural + language metadata for a single (repo, family) entry.

    Layer count / head count / hidden size are needed to drive the
    activation buffer and per-head CKA.  The defaults below match the
    Beetle pico architecture (n_layers=14, n_heads=12, d_model=768).
    Future Beetle variants override as needed.
    """
    repo: str
    family: str
    languages: List[LanguageConfig]
    n_layers: int = 14
    n_heads: int = 12
    d_model: int = 768
    d_head: int = 64
    description: str = ""

    # Optional override: explicit list of HF revisions to analyse.  When
    # None, callers should use ``analyze.convergence.list_checkpoints``
    # to discover ``step-N`` branches dynamically.
    revisions: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, ModelConfig] = {}


def register(repo: str, cfg: ModelConfig) -> None:
    """Register a model.  Subsequent ``get(repo)`` returns ``cfg``."""
    _REGISTRY[repo] = cfg


def get(repo: str) -> ModelConfig:
    """Lookup; raises ``KeyError`` with a helpful message when missing."""
    if repo not in _REGISTRY:
        raise KeyError(
            f"[xling.registry] unknown model: {repo!r}.  "
            f"Register it with xling.registry.register(...).  "
            f"Known repos: {sorted(_REGISTRY)[:5]}..."
        )
    return _REGISTRY[repo]


def list_repos(family: Optional[str] = None) -> List[str]:
    if family is None:
        return sorted(_REGISTRY)
    return sorted(r for r, c in _REGISTRY.items() if c.family == family)


def has(repo: str) -> bool:
    return repo in _REGISTRY


# ---------------------------------------------------------------------------
# Default registrations — Beetle-FineWeb-2B NLD-ENG bilinguals
# ---------------------------------------------------------------------------

_NLD = LanguageConfig(iso=LANG_NLD, hf_corpus="Beetle-Data/nl-2B-pretok",
                      flores_code="nld_Latn")
_ENG = LanguageConfig(iso=LANG_ENG, hf_corpus="Beetle-Data/en-for-nl-2B-pretok",
                      flores_code="eng_Latn")

_BEETLE_FINEWEB_2B_BILINGUALS = [
    "Beetle-FineWeb-2B/beetle-bilingual-balanced-b1-fineweb-2b-nld-eng",
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-simultaneous-b2-fineweb-2b-nld-eng",
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng",
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-classroom-20-b4-fineweb-2b-nld-eng",
    "Beetle-FineWeb-2B/beetle-bilingual-l2-80-late-b5-fineweb-2b-nld-eng",
]

for _repo in _BEETLE_FINEWEB_2B_BILINGUALS:
    register(_repo, ModelConfig(
        repo=_repo,
        family="beetle_fineweb_bilingual",
        languages=[_NLD, _ENG],
        description=(
            "Beetle pico (0.2B) bilingual NLD-ENG, 2B-token FineWeb sweep; "
            "B-curriculum embedded in the repo name (b1..b5)."
        ),
    ))


def language_codes(repo: str) -> List[str]:
    """Convenience: ISO codes for the given repo."""
    return [lc.iso for lc in get(repo).languages]
