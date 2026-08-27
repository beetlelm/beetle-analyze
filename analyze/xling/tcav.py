"""
analyze/xling/tcav.py
─────────────────────
TCAV (Kim et al., ICML 2018) cross-lingual concept activation vectors
for Set B4.

What this measures
------------------
For each linguistic concept ``C`` (negation, number, tense, …) we
collect:

  * ``X_pos`` — activations on examples where the concept is present.
  * ``X_neg`` — activations on examples where it is absent.

A linear classifier separating these two sets defines a *concept
activation vector* (CAV) — the unit normal to the separating
hyperplane.  TCAV asks: does the CAV computed in language A align
with the CAV for the same concept computed in language B?

If the same direction encodes the same concept across languages we
have direct evidence for cross-lingual conceptual structure — a
stronger claim than per-task probe accuracy because the comparison
is over the *direction* in activation space, not just the decision
boundary's accuracy.

We expose two operations:

  * :func:`compute_cav` — fit a CAV per (concept, language).
  * :func:`crosslingual_alignment` — cosine-similarity matrix between
    CAVs computed in different languages.

The TCAV "directional derivative" used in the original paper is not
implemented here: it requires gradients of a downstream task head,
which Beetle decoder-only models do not have a single canonical
classification head for.  Cosine alignment of CAVs is the
language-model-friendly analogue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CAV computation
# ---------------------------------------------------------------------------

def compute_cav(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Return the unit-norm CAV separating positive and negative examples.

    Uses a linear SVM (sklearn ``LinearSVC``); the CAV is the (normalised)
    coefficient vector.  Returns shape ``(D,)`` matching the input
    activation dimensionality.
    """
    from sklearn.svm import LinearSVC

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([
        np.ones(X_pos.shape[0], dtype=np.int64),
        np.zeros(X_neg.shape[0], dtype=np.int64),
    ])
    clf = LinearSVC(C=1.0, max_iter=2000, random_state=seed, dual=False)
    clf.fit(X, y)
    cav = clf.coef_[0].astype(np.float64)
    n = np.linalg.norm(cav)
    if n == 0:
        return cav
    return cav / n


# ---------------------------------------------------------------------------
# Cross-lingual alignment
# ---------------------------------------------------------------------------

def crosslingual_alignment(
    cavs: Dict[Tuple[str, str], np.ndarray],
) -> Dict[str, np.ndarray]:
    """Cosine similarity between CAVs of the same concept across languages.

    ``cavs`` is keyed by ``(language_iso, concept_name)``; we return a
    dict ``{concept_name: M}`` where ``M`` is a square ``L × L`` matrix
    of cosine similarities (``L`` = number of languages registered for
    that concept).
    """
    by_concept: Dict[str, Dict[str, np.ndarray]] = {}
    for (lang, concept), v in cavs.items():
        by_concept.setdefault(concept, {})[lang] = v

    out: Dict[str, np.ndarray] = {}
    for concept, lang_map in by_concept.items():
        langs = sorted(lang_map)
        M = np.zeros((len(langs), len(langs)))
        vs = [lang_map[l] for l in langs]
        for i, vi in enumerate(vs):
            for j, vj in enumerate(vs):
                ni = np.linalg.norm(vi)
                nj = np.linalg.norm(vj)
                if ni == 0 or nj == 0:
                    M[i, j] = 0.0
                else:
                    M[i, j] = float(vi @ vj / (ni * nj))
        out[concept] = M
    out["_lang_order"] = np.array(sorted({l for l, _ in cavs}))  # type: ignore
    return out


# ---------------------------------------------------------------------------
# Convenience: end-to-end (concept, lang) → score
# ---------------------------------------------------------------------------

def cav_alignment_score(
    pos_neg_by_lang_concept: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """One-shot helper: build CAVs, then compute alignment.

    ``pos_neg_by_lang_concept`` maps ``(lang, concept)`` to
    ``(X_pos, X_neg)``.  Returns the same shape as
    :func:`crosslingual_alignment`.
    """
    cavs = {
        key: compute_cav(pos, neg)
        for key, (pos, neg) in pos_neg_by_lang_concept.items()
    }
    return crosslingual_alignment(cavs)
