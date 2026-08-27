"""
analyze/units/stimuli.py — Minimal-pair stimuli for functional localization.

A ``MinimalPairSet`` is the unit of currency for the whole ``analyze.units``
package: a named linguistic phenomenon together with aligned lists of
grammatical (``good``) and ungrammatical (``bad``) sentences that differ in
exactly the property of interest. Both the *localizer* contrast (which units
respond to the phenomenon) and the *ablation* benchmark (does knocking those
units out hurt accuracy) are expressed as ``MinimalPairSet`` objects.

Loaders mirror the benchmark table in ``evals/eval_blimp.py`` so the same HF
datasets drive both the behavioural eval and the mechanistic localization:

  * ``load_blimp``       — nyu-mll/blimp        (sentence_good / sentence_bad)
  * ``load_multiblimp``  — jumelet/multiblimp   (sen / wrong_sen), 9 languages
  * ``load_jsonl``       — any local JSONL with good/bad columns
  * ``demo_pairs``       — tiny hand-written fixture (offline; used by tests)

The coarse ``category`` on each set follows BLiMP's ``linguistics_term`` field.
``AGREEMENT_CATEGORIES`` is the subset studied by Kryvosheieva et al. (2026),
whose replication code this package wraps.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Category constants (BLiMP `linguistics_term` values)
# ---------------------------------------------------------------------------

# The three agreement families that the source paper localizes and shows to
# recruit *overlapping* units. Passing these to ``load_blimp(categories=...)``
# reproduces the paper's phenomenon selection.
AGREEMENT_CATEGORIES: Tuple[str, ...] = (
    "anaphor_agreement",
    "determiner_noun_agreement",
    "subject_verb_agreement",
)


@dataclass
class MinimalPairSet:
    """Aligned grammatical / ungrammatical sentences for one phenomenon.

    Parameters
    ----------
    phenomenon : str
        Fine-grained identifier — e.g. the BLiMP config name
        ``"distractor_agreement_relational_noun"``.
    category : str
        Coarse family used for cross-phenomenon overlap, e.g.
        ``"subject_verb_agreement"``.
    good, bad : list[str]
        Same length; ``good[i]`` and ``bad[i]`` are a minimal pair.
    language : str
        ISO 639-3 code (default ``"eng"``).
    """

    phenomenon: str
    category: str
    good: List[str]
    bad: List[str]
    language: str = "eng"
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.good) != len(self.bad):
            raise ValueError(
                f"[MinimalPairSet:{self.phenomenon}] good/bad length mismatch "
                f"({len(self.good)} vs {len(self.bad)})"
            )

    def __len__(self) -> int:
        return len(self.good)

    def subsample(self, n: int, *, seed: int = 0) -> "MinimalPairSet":
        """Return a deterministic size-``n`` subsample (no-op if ``n >= len``)."""
        if n >= len(self):
            return self
        import numpy as np

        idx = np.random.default_rng(seed).permutation(len(self))[:n]
        return MinimalPairSet(
            self.phenomenon,
            self.category,
            [self.good[i] for i in idx],
            [self.bad[i] for i in idx],
            self.language,
            dict(self.meta, subsampled_from=len(self)),
        )

    def split(self, frac: float = 0.5, *, seed: int = 0) -> Tuple["MinimalPairSet", "MinimalPairSet"]:
        """Split into two disjoint sets (for cross-validation of unit masks)."""
        import numpy as np

        perm = np.random.default_rng(seed).permutation(len(self))
        cut = int(round(frac * len(self)))
        a, b = perm[:cut], perm[cut:]

        def _take(ix):
            return MinimalPairSet(
                self.phenomenon, self.category,
                [self.good[i] for i in ix], [self.bad[i] for i in ix],
                self.language, dict(self.meta),
            )

        return _take(a), _take(b)

    def folds(self, k: int, *, seed: int = 0) -> List["MinimalPairSet"]:
        """Partition into ``k`` roughly equal folds."""
        import numpy as np

        perm = np.random.default_rng(seed).permutation(len(self))
        out = []
        for chunk in np.array_split(perm, k):
            out.append(MinimalPairSet(
                self.phenomenon, self.category,
                [self.good[i] for i in chunk], [self.bad[i] for i in chunk],
                self.language, dict(self.meta),
            ))
        return out


def merge(sets: Sequence[MinimalPairSet], *, phenomenon: str, category: str) -> MinimalPairSet:
    """Concatenate several sets into one (e.g. all subject-verb configs)."""
    good: List[str] = []
    bad: List[str] = []
    lang = sets[0].language if sets else "eng"
    for s in sets:
        good.extend(s.good)
        bad.extend(s.bad)
    return MinimalPairSet(phenomenon, category, good, bad, lang,
                          {"merged_from": [s.phenomenon for s in sets]})


# ---------------------------------------------------------------------------
# BLiMP loader (nyu-mll/blimp and Dutch/Chinese variants)
# ---------------------------------------------------------------------------

def load_blimp(
    *,
    categories: Optional[Iterable[str]] = None,
    phenomena: Optional[Iterable[str]] = None,
    max_pairs: Optional[int] = None,
    hf_id: str = "nyu-mll/blimp",
    language: str = "eng",
    hf_token: Optional[str] = None,
) -> List[MinimalPairSet]:
    """Load BLiMP configs as ``MinimalPairSet``s (one per config = phenomenon).

    ``categories`` filters by ``linguistics_term`` (see ``AGREEMENT_CATEGORIES``);
    ``phenomena`` filters by exact config name. Pass neither to load all 67.
    """
    from datasets import get_dataset_config_names, load_dataset

    cats = set(categories) if categories is not None else None
    phen = set(phenomena) if phenomena is not None else None

    configs = get_dataset_config_names(hf_id, token=hf_token)
    if phen is not None:
        configs = [c for c in configs if c in phen]

    out: List[MinimalPairSet] = []
    for cfg in configs:
        ds = load_dataset(hf_id, cfg, split="train", token=hf_token)
        cols = ds.column_names
        good_col = "sentence_good" if "sentence_good" in cols else cols[0]
        bad_col = "sentence_bad" if "sentence_bad" in cols else cols[1]
        term = (
            ds[0].get("linguistics_term", cfg) if "linguistics_term" in cols else cfg
        )
        if cats is not None and term not in cats:
            continue
        good = list(ds[good_col])
        bad = list(ds[bad_col])
        s = MinimalPairSet(cfg, term, good, bad, language, {"hf_id": hf_id})
        if max_pairs is not None:
            s = s.subsample(max_pairs)
        out.append(s)
    return out


def load_multiblimp(
    langs: Sequence[str],
    *,
    max_pairs: Optional[int] = None,
    hf_id: str = "jumelet/multiblimp",
    hf_token: Optional[str] = None,
) -> List[MinimalPairSet]:
    """Load MultiBLiMP per-language configs (``sen`` / ``wrong_sen``).

    One ``MinimalPairSet`` per language; ``category`` is taken from the
    dataset's ``phenomenon`` column when present, else ``"agreement"``.
    """
    from datasets import load_dataset

    out: List[MinimalPairSet] = []
    for lang in langs:
        ds = load_dataset(hf_id, lang, split="train", token=hf_token)
        cols = ds.column_names
        good = list(ds["sen"]) if "sen" in cols else list(ds[cols[0]])
        bad = list(ds["wrong_sen"]) if "wrong_sen" in cols else list(ds[cols[1]])
        cat = ds[0].get("phenomenon", "agreement") if "phenomenon" in cols else "agreement"
        s = MinimalPairSet(f"multiblimp_{lang}", str(cat), good, bad, lang,
                           {"hf_id": hf_id})
        if max_pairs is not None:
            s = s.subsample(max_pairs)
        out.append(s)
    return out


def load_jsonl(
    path: str | Path,
    *,
    phenomenon: Optional[str] = None,
    category: str = "custom",
    good_key: str = "sentence_good",
    bad_key: str = "sentence_bad",
    language: str = "eng",
) -> MinimalPairSet:
    """Load a local JSONL where each line has a good/bad sentence pair."""
    p = Path(path)
    good: List[str] = []
    bad: List[str] = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            good.append(row[good_key])
            bad.append(row[bad_key])
    return MinimalPairSet(phenomenon or p.stem, category, good, bad, language,
                          {"source": str(p)})


# ---------------------------------------------------------------------------
# Offline demo fixture (used by tests; no network / HF access)
# ---------------------------------------------------------------------------

def demo_pairs() -> List[MinimalPairSet]:
    """Three tiny agreement sets — enough to exercise the whole pipeline."""
    sv = MinimalPairSet(
        "demo_subject_verb", "subject_verb_agreement",
        good=[
            "The dog runs in the park.",
            "The teachers grade the exams.",
            "My sister sings every morning.",
            "Those children play outside.",
        ],
        bad=[
            "The dog run in the park.",
            "The teachers grades the exams.",
            "My sister sing every morning.",
            "Those children plays outside.",
        ],
    )
    anaphor = MinimalPairSet(
        "demo_anaphor", "anaphor_agreement",
        good=[
            "The boy hurt himself.",
            "The women blamed themselves.",
            "Sarah trusts herself.",
            "The men defended themselves.",
        ],
        bad=[
            "The boy hurt herself.",
            "The women blamed himself.",
            "Sarah trusts themselves.",
            "The men defended herself.",
        ],
    )
    detn = MinimalPairSet(
        "demo_det_noun", "determiner_noun_agreement",
        good=[
            "She bought these apples.",
            "He read that book.",
            "They painted those walls.",
            "I want this cookie.",
        ],
        bad=[
            "She bought this apples.",
            "He read those book.",
            "They painted that walls.",
            "I want these cookie.",
        ],
    )
    return [sv, anaphor, detn]
