"""
embedding_drift.py — Embedding space analysis across BeetleLM conditions.

Outputs
-------
results/embeddings/
  vocab_overlap_<iso>.csv        — per-model vocab overlap stats
  cka_matrix_<iso>.csv           — pairwise CKA between all models for lang
  probe_pca_<iso>.csv            — 2D PCA coordinates for probe words
"""

from __future__ import annotations
from analyze import _path  # noqa: F401  — adds repo root to sys.path

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.nn.functional import cosine_similarity

from models import MODEL_GROUPS, get_bilingual_type
from ppl_utils import (
    LOAD_FAILURES,
    clear_model_cache,
    get_best_revision,
    load_model_and_tokenizer,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("results/embeddings")

PROBE_WORDS: dict[str, list[str]] = {
    "nld": ["de", "het", "een", "van", "en", "in", "is", "dat", "op", "te"],
    "deu": ["die", "der", "und", "in", "den", "ist", "das", "des", "mit", "zu"],
    "zho": ["的", "是", "在", "了", "不", "有", "我", "他", "这", "人"],
    "fra": ["le", "de", "et", "à", "les", "des", "en", "un", "du", "que"],
    "fas": ["و", "در", "به", "از", "که", "را", "با", "این", "است", "بود"],
    "bul": ["на", "и", "в", "да", "се", "е", "не", "с", "за", "от"],
    "eng": ["the", "of", "and", "to", "a", "in", "is", "it", "that", "was"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_input_embeddings(repo: str):
    """
    Return (embedding_matrix [vocab, hidden], vocab_tokens) for *repo*.
    Returns None if the model failed to load.
    """
    result = load_model_and_tokenizer(repo)
    if result is None:
        return None
    model, tokenizer = result
    emb   = model.get_input_embeddings().weight.detach().float().cpu()
    vocab = tokenizer.convert_ids_to_tokens(range(emb.size(0)))
    return emb, vocab


def token_ids_for_words(words: list[str], tokenizer) -> dict[str, list[int]]:
    """Map surface words to single token-ids (multi-piece words excluded)."""
    result: dict[str, list[int]] = {}
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            result[word] = ids
    return result


# ---------------------------------------------------------------------------
# 1. Vocabulary overlap & embedding drift
# ---------------------------------------------------------------------------

def compute_vocab_overlap(iso_code: str, overwrite: bool = False) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"vocab_overlap_{iso_code}.csv"

    if out_path.exists() and not overwrite:
        print(f"[embed_drift] Loading cached {out_path}")
        return pd.read_csv(out_path)

    mono_repo = f"BeetleLM/beetlelm_{iso_code}_mono"
    print(f"[embed_drift] Loading mono baseline: {mono_repo}")

    mono_result = get_input_embeddings(mono_repo)
    if mono_result is None:
        print(f"[embed_drift] SKIPPED {iso_code} — mono baseline failed to load.")
        return pd.DataFrame()
    mono_emb, mono_vocab = mono_result
    mono_vocab_set       = {t: i for i, t in enumerate(mono_vocab)}

    mono_load = load_model_and_tokenizer(mono_repo)
    _, mono_tokenizer = mono_load
    probe_words = PROBE_WORDS.get(iso_code, [])

    candidates = [m for m in MODEL_GROUPS[iso_code] if "mono" not in m]
    rows: list[dict] = []

    for repo in candidates:
        # Drop the previous repo's (model, tokenizer) before resolving the
        # next pair so we never hold two distinct models resident at once.
        # The bounded module cache eventually evicts on its own, but explicit
        # eviction here avoids a transient 2x host-RAM peak that has tipped
        # low-RAM nodes into swap thrash during long sweeps.
        clear_model_cache()
        revision   = get_best_revision(repo)
        print(f"[embed_drift] Comparing {repo} @ {revision} …")

        bi_result = get_input_embeddings(repo)
        if bi_result is None:
            print(f"[embed_drift] SKIPPED {repo} — model failed to load.")
            continue
        bi_emb, bi_vocab = bi_result

        bi_load = load_model_and_tokenizer(repo)
        _, bi_tokenizer = bi_load
        bi_vocab_set    = {t: i for i, t in enumerate(bi_vocab)}

        shared_tokens = set(mono_vocab_set.keys()) & set(bi_vocab_set.keys())
        shared_pct    = len(shared_tokens) / len(mono_vocab_set) * 100

        mono_ids  = [mono_vocab_set[t] for t in shared_tokens]
        bi_ids    = [bi_vocab_set[t]   for t in shared_tokens]
        mono_vecs = mono_emb[mono_ids]
        bi_vecs   = bi_emb[bi_ids]
        cos_dists = (1.0 - cosine_similarity(mono_vecs, bi_vecs, dim=1)).numpy()

        row = {
            "iso_code":           iso_code,
            "repo":               repo,
            "revision":           revision,
            "bilingual_type":     get_bilingual_type(repo),
            "shared_vocab_pct":   round(shared_pct, 2),
            "n_shared":           len(shared_tokens),
            "mean_cosine_dist":   float(np.mean(cos_dists)),
            "std_cosine_dist":    float(np.std(cos_dists)),
            "median_cosine_dist": float(np.median(cos_dists)),
        }

        probe_map = token_ids_for_words(probe_words, mono_tokenizer)
        for word, (mono_id,) in {k: v for k, v in probe_map.items() if len(v) == 1}.items():
            bi_probe = token_ids_for_words([word], bi_tokenizer)
            if word in bi_probe and len(bi_probe[word]) == 1:
                sim = cosine_similarity(
                    mono_emb[mono_id].unsqueeze(0),
                    bi_emb[bi_probe[word][0]].unsqueeze(0),
                ).item()
                row[f"probe_{word}_cosine_dist"] = round(1.0 - sim, 4)

        rows.append(row)

        # Drop bilingual model + tokenizer + embedding tensors before the
        # next iteration; otherwise they outlive the loop body and stack up.
        del bi_emb, bi_vocab, bi_load, bi_tokenizer

    # Final eviction so the function does not retain the last loaded model
    # in the module cache when the caller invokes us across many ISO codes.
    clear_model_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  → saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# 2. Pairwise CKA
# ---------------------------------------------------------------------------

def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    K_X = X @ X.T
    K_Y = Y @ Y.T
    n   = K_X.shape[0]
    H   = np.eye(n) - np.ones((n, n)) / n
    cKX = H @ K_X @ H
    cKY = H @ K_Y @ H
    return float(np.sum(cKX * cKY) / (np.linalg.norm(cKX, "fro") * np.linalg.norm(cKY, "fro") + 1e-10))


def compute_cka_matrix(iso_code: str, overwrite: bool = False) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"cka_matrix_{iso_code}.csv"

    if out_path.exists() and not overwrite:
        print(f"[embed_drift] Loading cached {out_path}")
        return pd.read_csv(out_path, index_col=0)

    repos       = MODEL_GROUPS[iso_code]
    probe_words = PROBE_WORDS.get(iso_code, [])

    print("[embed_drift] Finding shared single-token probe words across all models …")
    probe_in_all: set[str] | None = None
    tokenizers: dict[str, object] = {}
    valid_repos: list[str] = []

    for repo in repos:
        result = load_model_and_tokenizer(repo)
        if result is None:
            print(f"[embed_drift] SKIPPED {repo} — model failed to load.")
            continue
        _, tok = result
        tokenizers[repo] = tok
        valid_repos.append(repo)
        single = set(token_ids_for_words(probe_words, tok).keys())
        probe_in_all = single if probe_in_all is None else probe_in_all & single

    if not valid_repos:
        print("[embed_drift] No models loaded — skipping CKA.")
        return pd.DataFrame()

    if not probe_in_all:
        print("[embed_drift] WARNING: no shared single-token probe words — skipping CKA.")
        return pd.DataFrame()

    probe_list = sorted(probe_in_all)
    print(f"  {len(probe_list)} shared probe words: {probe_list}")

    model_reps: dict[str, np.ndarray] = {}
    for repo in valid_repos:
        emb_result = get_input_embeddings(repo)
        if emb_result is None:
            continue
        emb, _   = emb_result
        tok      = tokenizers[repo]
        pmap     = token_ids_for_words(probe_list, tok)
        model_reps[repo] = np.stack([emb[pmap[w][0]].numpy() for w in probe_list])

    if len(model_reps) < 2:
        print("[embed_drift] Fewer than 2 models loaded — skipping CKA.")
        return pd.DataFrame()

    repo_list = list(model_reps.keys())
    cka_vals  = np.zeros((len(repo_list), len(repo_list)))
    for i, r1 in enumerate(repo_list):
        for j, r2 in enumerate(repo_list):
            cka_vals[i, j] = _linear_cka(model_reps[r1], model_reps[r2])

    cka_df = pd.DataFrame(cka_vals, index=repo_list, columns=repo_list)
    cka_df.to_csv(out_path)
    print(f"  → saved {out_path}")
    return cka_df


# ---------------------------------------------------------------------------
# 3. Probe-word PCA (mono-anchored)
# ---------------------------------------------------------------------------

def compute_probe_pca(
    iso_code: str,
    n_components: int = 2,
    overwrite: bool = False,
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"probe_pca_{iso_code}.csv"

    if out_path.exists() and not overwrite:
        print(f"[embed_drift] Loading cached {out_path}")
        return pd.read_csv(out_path)

    probe_words = PROBE_WORDS.get(iso_code, [])
    repos       = MODEL_GROUPS[iso_code]
    mono_repo   = f"BeetleLM/beetlelm_{iso_code}_mono"

    mono_result = get_input_embeddings(mono_repo)
    if mono_result is None:
        print(f"[embed_drift] SKIPPED {iso_code} PCA — mono baseline failed to load.")
        return pd.DataFrame()
    mono_emb, _ = mono_result

    mono_load = load_model_and_tokenizer(mono_repo)
    _, mono_tok = mono_load
    mono_pmap   = token_ids_for_words(probe_words, mono_tok)
    avail_words = sorted(mono_pmap.keys())
    mono_vecs   = np.stack([mono_emb[mono_pmap[w][0]].numpy() for w in avail_words])

    pca = PCA(n_components=n_components)
    pca.fit(mono_vecs)
    print(f"[embed_drift] PCA variance explained: {pca.explained_variance_ratio_}")

    rows: list[dict] = []
    for repo in repos:
        revision   = get_best_revision(repo)
        emb_result = get_input_embeddings(repo)
        if emb_result is None:
            print(f"[embed_drift] SKIPPED {repo} in PCA — model failed to load.")
            continue
        emb, _ = emb_result

        load_result = load_model_and_tokenizer(repo)
        _, tok = load_result
        pmap   = token_ids_for_words(avail_words, tok)

        for word in avail_words:
            if word not in pmap:
                continue
            coords = pca.transform(emb[pmap[word][0]].numpy().reshape(1, -1))[0]
            rows.append({
                "repo":           repo,
                "revision":       revision,
                "iso_code":       iso_code,
                "bilingual_type": get_bilingual_type(repo),
                "word":           word,
                "pc1":            float(coords[0]),
                "pc2":            float(coords[1]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  → saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def run_embedding_analysis(
    lang_codes: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, dict]:
    if lang_codes is None:
        lang_codes = list(MODEL_GROUPS.keys())

    results = {}
    for code in lang_codes:
        print(f"\n{'='*60}\n Embedding analysis: {code}\n{'='*60}")
        results[code] = {
            "vocab_overlap": compute_vocab_overlap(code, overwrite),
            "cka_matrix":    compute_cka_matrix(code, overwrite),
            "probe_pca":     compute_probe_pca(code, overwrite),
        }

    if LOAD_FAILURES:
        print(f"\n[embed_drift] {'='*56}")
        print(f"[embed_drift] SKIPPED {len(LOAD_FAILURES)} model(s) due to load errors:")
        for repo, reason in LOAD_FAILURES.items():
            print(f"  ✗  {repo}")
            print(f"     {reason}")
        print(f"[embed_drift] {'='*56}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run embedding drift analysis.")
    parser.add_argument("--langs", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_embedding_analysis(lang_codes=args.langs, overwrite=args.overwrite)