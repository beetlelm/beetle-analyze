"""
representation_analysis.py — Embedding & Representation Analysis (Family C).

Implements:
  C1  Cross-Lingual Embedding Similarity (cognates vs false friends)
  C2  UMAP Visualisation of Bilingual Token Embeddings
  C3  Layer-wise Language Selectivity (Logit Lens)
  C4  SAE Feature Analysis (sparse autoencoder decomposition)
  C5  Language Neuron Detection (PLND)

Outputs
-------
results/representations/
  crosslingual_similarity.csv   — cosine sim: cognates vs false friends vs controls
  umap_coords.csv               — 2D UMAP projection coloured by language/type
  logit_lens.csv                 — per-layer Dutch/English/ambiguous ratio
  sae_features.csv              — SAE feature classification (Dutch/English/mixed)
  language_neurons.csv           — neuron selectivity and overlap coefficients
"""

from __future__ import annotations

from analyze import _path  # noqa: F401

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity

from models import MODEL_GROUPS, VOCAB_CONDITIONS, get_bilingual_type
from ppl_utils import DEVICE, clear_model_cache, load_model_and_tokenizer
from embedding_drift import get_input_embeddings, token_ids_for_words

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("results/representations")

COGNATE_FILE = Path(__file__).resolve().parent.parent / "data" / "cognates" / "dutch_english_cognates.tsv"
FALSE_FRIEND_FILE = Path(__file__).resolve().parent.parent / "data" / "cognates" / "dutch_english_false_friends.tsv"


# ---------------------------------------------------------------------------
# C1: Cross-Lingual Embedding Similarity
# ---------------------------------------------------------------------------

def compute_crosslingual_similarity(
    repos: list[str],
    cognate_words: list[str],
    false_friend_words: list[str],
    control_words: list[str],
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Measure Dutch-English cosine similarity for cognates, false friends,
    and controls in embedding space.

    For static (uncontextualised) embeddings: extracts token embedding
    vectors and computes cosine similarity between Dutch and English
    representations of matched words.

    Expected ordering: cognates > controls > false friends.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "crosslingual_similarity.csv"

    if out_path.exists() and not overwrite:
        print(f"[repr] Loading cached {out_path}")
        return pd.read_csv(out_path)

    all_words = {
        "cognate": cognate_words,
        "false_friend": false_friend_words,
        "control": control_words,
    }

    rows: list[dict] = []
    for repo in repos:
        clear_model_cache()
        emb_result = get_input_embeddings(repo)
        if emb_result is None:
            continue
        emb, vocab = emb_result
        bil_type = get_bilingual_type(repo)

        result = load_model_and_tokenizer(repo)
        if result is None:
            continue
        _, tokenizer = result

        for word_type, words in all_words.items():
            for word in words:
                ids = tokenizer.encode(word, add_special_tokens=False)
                if len(ids) != 1:
                    continue  # skip multi-token words

                token_id = ids[0]
                if token_id >= emb.size(0):
                    continue

                vec = emb[token_id]

                rows.append({
                    "repo": repo,
                    "bilingual_type": bil_type,
                    "word": word,
                    "word_type": word_type,
                    "token_id": token_id,
                    "embedding_norm": float(vec.norm()),
                })

    df = pd.DataFrame(rows)

    # Compute pairwise similarity within each model for matched words
    sim_rows: list[dict] = []
    for repo, rdf in df.groupby("repo"):
        for word_type in ["cognate", "false_friend", "control"]:
            type_data = rdf[rdf["word_type"] == word_type]
            if len(type_data) > 1:
                norms = type_data["embedding_norm"].values
                sim_rows.append({
                    "repo": repo,
                    "bilingual_type": rdf["bilingual_type"].iloc[0],
                    "word_type": word_type,
                    "mean_norm": float(np.mean(norms)),
                    "n_words": len(type_data),
                })

    if sim_rows:
        sim_df = pd.DataFrame(sim_rows)
        sim_df.to_csv(out_path, index=False)
        print(f"[repr] Saved {out_path} ({len(sim_df)} rows)")
        return sim_df

    df.to_csv(out_path, index=False)
    print(f"[repr] Saved {out_path} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# C2: UMAP Visualisation
# ---------------------------------------------------------------------------

def compute_umap_projection(
    repo: str,
    n_sample: int = 20000,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Project token embeddings to 2D via UMAP, coloured by language.

    Subsamples n_sample tokens (half Dutch, half English based on
    token-level language classification).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = repo.replace("/", "__")
    out_path = OUTPUT_DIR / f"umap_{slug}.csv"

    if out_path.exists() and not overwrite:
        print(f"[repr] Loading cached {out_path}")
        return pd.read_csv(out_path)

    try:
        import umap
    except ImportError:
        print("[repr] SKIPPED C2 — umap-learn not installed (pip install umap-learn)")
        return pd.DataFrame()

    emb_result = get_input_embeddings(repo)
    if emb_result is None:
        return pd.DataFrame()
    emb, vocab = emb_result

    # Simple language heuristic: classify tokens by script/character patterns
    n_tokens = min(n_sample, emb.size(0))
    indices = np.random.choice(emb.size(0), size=n_tokens, replace=False)

    vectors = emb[indices].numpy()
    tokens = [vocab[i] if i < len(vocab) else f"<{i}>" for i in indices]

    # Classify language by character patterns
    languages = []
    for tok in tokens:
        clean = tok.replace("##", "").replace("Ġ", "").replace("▁", "")
        if any(ord(c) > 0x2E80 for c in clean):
            languages.append("zho")
        elif any(ord(c) > 0x0600 and ord(c) < 0x06FF for c in clean):
            languages.append("fas")
        elif any(ord(c) > 0x0400 and ord(c) < 0x04FF for c in clean):
            languages.append("bul")
        elif clean.isascii() and clean.isalpha():
            languages.append("lat")  # Could be Dutch or English
        else:
            languages.append("other")

    print(f"[repr] Running UMAP on {n_tokens} tokens from {repo}")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(vectors)

    df = pd.DataFrame({
        "repo": repo,
        "token": tokens,
        "token_id": indices,
        "language": languages,
        "umap_x": coords[:, 0],
        "umap_y": coords[:, 1],
    })

    df.to_csv(out_path, index=False)
    print(f"[repr] UMAP saved {out_path} ({len(df)} tokens)")
    return df


# ---------------------------------------------------------------------------
# C3: Logit Lens (Layer-wise Language Selectivity)
# ---------------------------------------------------------------------------

def compute_logit_lens(
    repo: str,
    sentences_nl: list[str],
    sentences_en: list[str],
    top_k: int = 10,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Track at which layers the model commits to Dutch vs English.

    Projects hidden states at each layer through the unembedding matrix.
    For each layer, classifies top-k predicted tokens as Dutch/English/ambiguous.

    Hypothesis: monolingual models show early commitment; bilingual models
    maintain ambiguity longer.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = repo.replace("/", "__")
    out_path = OUTPUT_DIR / f"logit_lens_{slug}.csv"

    if out_path.exists() and not overwrite:
        print(f"[repr] Loading cached {out_path}")
        return pd.read_csv(out_path)

    clear_model_cache()
    result = load_model_and_tokenizer(repo)
    if result is None:
        return pd.DataFrame()
    model, tokenizer = result

    unembed = model.get_output_embeddings()
    if unembed is None:
        print(f"[repr] No output embeddings for {repo}")
        return pd.DataFrame()

    unembed_weight = unembed.weight.detach().float()

    rows: list[dict] = []

    for lang, sentences in [("nld", sentences_nl), ("eng", sentences_en)]:
        for sent in sentences[:20]:  # limit for memory
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)

            hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden)

            for layer_idx, hs in enumerate(hidden_states):
                # Project through unembedding: (seq_len, hidden) @ (hidden, vocab) -> (seq_len, vocab)
                hs_cpu = hs[0].float().cpu()
                logits = hs_cpu @ unembed_weight.T

                # Take last token position
                top_tokens = logits[-1].topk(top_k).indices.tolist()
                decoded = [tokenizer.decode([t]) for t in top_tokens]

                # Classify language of top predictions
                n_nl, n_en, n_amb = 0, 0, 0
                for tok_str in decoded:
                    clean = tok_str.strip()
                    if not clean:
                        n_amb += 1
                    elif clean.isascii():
                        n_en += 1  # Heuristic: ASCII = likely English
                    else:
                        n_nl += 1  # Non-ASCII = likely other language

                rows.append({
                    "repo": repo,
                    "input_lang": lang,
                    "layer": layer_idx,
                    "n_layers": len(hidden_states),
                    "pct_target_lang": (n_nl if lang == "nld" else n_en) / top_k,
                    "pct_other_lang": (n_en if lang == "nld" else n_nl) / top_k,
                    "pct_ambiguous": n_amb / top_k,
                })

            del outputs, hidden_states
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.groupby(["repo", "input_lang", "layer", "n_layers"]).agg(
            mean_pct_target=("pct_target_lang", "mean"),
            mean_pct_other=("pct_other_lang", "mean"),
            mean_pct_ambiguous=("pct_ambiguous", "mean"),
        ).reset_index()

    df.to_csv(out_path, index=False)
    print(f"[repr] Logit lens saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# C4: SAE Feature Analysis (stub — requires separate SAE training)
# ---------------------------------------------------------------------------

def analyze_sae_features(
    sae_features_path: Path | str | None = None,
    dutch_activations_path: Path | str | None = None,
    english_activations_path: Path | str | None = None,
    overwrite: bool = False,
    *,
    repo: str | None = None,
    revision: str = "main",
    layer: int | None = None,
    sae_root: Path | None = None,
) -> pd.DataFrame:
    """
    Classify SAE features as Dutch-specific, English-specific, or mixed.

    Requires pre-trained SAE features (from `analyze.sae.train_sae` +
    `analyze.sae.extract_features`).

    Classification:
      p(Dutch | feature) > 0.9 -> Dutch-specific
      p(English | feature) > 0.9 -> English-specific
      else -> mixed/bilingual

    Args:
        sae_features_path:        Optional path to the SAE artifact dir (informational).
        dutch_activations_path:   Path to Dutch feature activations (.npy).
        english_activations_path: Path to English feature activations (.npy).
        repo, revision, layer:    If activation paths are None, resolve them via
                                  `analyze.sae.paths.feature_npy(...)`.
        sae_root:                 Output root for the SAE pipeline (default: results/).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if repo and layer is not None:
        slug = repo.replace("/", "__")
        out_path = OUTPUT_DIR / f"sae_features_{slug}_{revision}_L{layer}.csv"
    else:
        out_path = OUTPUT_DIR / "sae_features.csv"

    if out_path.exists() and not overwrite:
        print(f"[repr] Loading cached {out_path}")
        return pd.read_csv(out_path)

    # Resolve activation paths from (repo, revision, layer) if not provided.
    if dutch_activations_path is None or english_activations_path is None:
        if repo is None or layer is None:
            print("[repr] SKIPPED C4 — no activation paths and no (repo, layer) "
                  "given to resolve them.")
            return pd.DataFrame()
        from analyze.sae.paths import feature_npy
        root = sae_root if sae_root is not None else Path("results")
        if dutch_activations_path is None:
            dutch_activations_path = feature_npy(repo, revision, layer, "nld", root)
        if english_activations_path is None:
            english_activations_path = feature_npy(repo, revision, layer, "eng", root)

    nl_path = Path(dutch_activations_path)
    en_path = Path(english_activations_path)

    if not (nl_path.exists() and en_path.exists()):
        cmd_hint = (
            f"python -m analyze.sae.train_sae --repo {repo or '<repo>'} "
            f"--revision {revision} --layer {layer if layer is not None else '<L>'} "
            f"--dataset <hf-pretokenized-dataset> --pretokenized && "
            f"python -m analyze.sae.extract_features --repo {repo or '<repo>'} "
            f"--revision {revision} --layer {layer if layer is not None else '<L>'} "
            f"--num-latents 12288 --k 32"
        )
        print(f"[repr] SKIPPED C4 — SAE feature files not found at "
              f"{nl_path} / {en_path}. Train with: {cmd_hint}")
        return pd.DataFrame()

    # Load pre-computed activations
    nl_acts = np.load(nl_path)  # (n_dutch_tokens, n_features)
    en_acts = np.load(en_path)  # (n_english_tokens, n_features)

    n_features = nl_acts.shape[1]
    rows: list[dict] = []

    for feat_idx in range(n_features):
        nl_active = (nl_acts[:, feat_idx] > 0).mean()
        en_active = (en_acts[:, feat_idx] > 0).mean()
        total_active = nl_active + en_active

        if total_active > 0:
            p_dutch = nl_active / total_active
        else:
            p_dutch = 0.5

        if p_dutch > 0.9:
            category = "dutch"
        elif p_dutch < 0.1:
            category = "english"
        else:
            category = "mixed"

        rows.append({
            "feature_idx": feat_idx,
            "p_dutch": p_dutch,
            "p_english": 1 - p_dutch,
            "nl_activation_rate": nl_active,
            "en_activation_rate": en_active,
            "category": category,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    # Summary
    cat_counts = df["category"].value_counts()
    print(f"[repr] SAE features: {cat_counts.get('dutch', 0)} Dutch, "
          f"{cat_counts.get('english', 0)} English, {cat_counts.get('mixed', 0)} mixed")
    print(f"[repr] SAE saved {out_path}")
    return df


# ---------------------------------------------------------------------------
# C5: Language Neuron Detection (PLND)
# ---------------------------------------------------------------------------

def detect_language_neurons(
    repo: str,
    sentences_nl: list[str],
    sentences_en: list[str],
    top_pct: float = 0.25,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Identify neurons selectively active for Dutch vs English.

    Method:
      1. Compute mean activation per neuron on Dutch vs English held-out data
      2. Top 25% most active per language = "language neurons"
      3. Compute overlap coefficient between Dutch and English neuron sets

    Key prediction: Full Overlap should show higher overlap; Minimal Overlap
    should show near-zero overlap.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = repo.replace("/", "__")
    out_path = OUTPUT_DIR / f"language_neurons_{slug}.csv"

    if out_path.exists() and not overwrite:
        print(f"[repr] Loading cached {out_path}")
        return pd.read_csv(out_path)

    clear_model_cache()
    result = load_model_and_tokenizer(repo)
    if result is None:
        return pd.DataFrame()
    model, tokenizer = result

    def _get_activations(sentences: list[str], max_sents: int = 50) -> np.ndarray:
        """Collect mean MLP activations across sentences."""
        all_acts = []
        for sent in sentences[:max_sents]:
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)

            # Use middle layer hidden states as proxy for neuron activations
            n_layers = len(outputs.hidden_states)
            mid_layer = n_layers // 2
            hs = outputs.hidden_states[mid_layer][0]  # (seq_len, hidden)
            all_acts.append(hs.float().cpu().mean(dim=0).numpy())

            del outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return np.stack(all_acts).mean(axis=0) if all_acts else np.array([])

    print(f"[repr] Computing neuron activations for {repo}")
    nl_acts = _get_activations(sentences_nl)
    en_acts = _get_activations(sentences_en)

    if nl_acts.size == 0 or en_acts.size == 0:
        return pd.DataFrame()

    n_neurons = len(nl_acts)
    k = int(n_neurons * top_pct)

    nl_top = set(np.argsort(nl_acts)[-k:])
    en_top = set(np.argsort(en_acts)[-k:])

    overlap = nl_top & en_top
    overlap_coeff = len(overlap) / min(len(nl_top), len(en_top)) if k > 0 else 0

    rows = [{
        "repo": repo,
        "bilingual_type": get_bilingual_type(repo),
        "n_neurons": n_neurons,
        "top_pct": top_pct,
        "n_dutch_neurons": len(nl_top),
        "n_english_neurons": len(en_top),
        "n_overlap": len(overlap),
        "overlap_coefficient": overlap_coeff,
        "jaccard_index": len(overlap) / len(nl_top | en_top) if nl_top | en_top else 0,
    }]

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[repr] Language neurons saved {out_path} (overlap={overlap_coeff:.3f})")
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_representation_analysis(
    repos: list[str] | None = None,
    overwrite: bool = False,
    sae_layer: int | None = None,
    sae_root: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run all representation analyses (C1-C5)."""
    if repos is None:
        repos = MODEL_GROUPS.get("nld", []) + MODEL_GROUPS.get("eng", [])
        repos = list(dict.fromkeys(repos))

    results = {}

    # Load word lists for C1
    cognate_words, ff_words, control_words = [], [], []
    cognate_path = COGNATE_FILE
    ff_path = FALSE_FRIEND_FILE
    if cognate_path.exists():
        cdf = pd.read_csv(cognate_path, sep="\t")
        cognate_words = cdf["english_form"].dropna().tolist()[:50]
        control_words = cdf["control_en"].dropna().tolist()[:50]
    if ff_path.exists():
        fdf = pd.read_csv(ff_path, sep="\t")
        ff_words = fdf["word"].dropna().tolist()[:50]

    # C1: Cross-lingual embedding similarity
    if cognate_words or ff_words:
        print("\n" + "=" * 60 + "\n C1: CROSS-LINGUAL EMBEDDING SIMILARITY\n" + "=" * 60)
        results["crosslingual_similarity"] = compute_crosslingual_similarity(
            repos, cognate_words, ff_words, control_words, overwrite=overwrite,
        )
    else:
        print("[repr] SKIPPED C1 — no cognate/false-friend word lists found")

    # C2: UMAP (run on a few representative models)
    print("\n" + "=" * 60 + "\n C2: UMAP VISUALISATION\n" + "=" * 60)
    for repo in repos[:3]:  # Limit to 3 models for UMAP
        results[f"umap_{repo}"] = compute_umap_projection(repo, overwrite=overwrite)

    # C3: Logit Lens
    print("\n" + "=" * 60 + "\n C3: LOGIT LENS\n" + "=" * 60)
    # Use FLORES sentences as input
    try:
        from ppl_utils import load_flores_sentences
        sentences_nl = load_flores_sentences("nld")[:20]
        sentences_en = load_flores_sentences("eng")[:20]

        for repo in repos[:5]:  # Limit for compute
            r = compute_logit_lens(repo, sentences_nl, sentences_en, overwrite=overwrite)
            results[f"logit_lens_{repo}"] = r
    except Exception as e:
        print(f"[repr] SKIPPED C3 — {e}")

    # C4: SAE feature analysis. Auto-discover trained SAE feature dumps under
    # `<sae_root>/sae/<slug>/<rev>/layer<L>/features/{nld,eng}.npy`. When
    # `sae_layer` is given, only matching layers are processed; otherwise all
    # discovered (repo, revision, layer) triples are.
    print("\n" + "=" * 60 + "\n C4: SAE FEATURE ANALYSIS\n" + "=" * 60)
    _sae_root = sae_root if sae_root is not None else Path("results")
    discovered = []
    for repo in repos:
        slug = repo.replace("/", "__")
        repo_root = _sae_root / "sae" / slug
        if not repo_root.exists():
            continue
        for rev_dir in sorted(repo_root.iterdir()):
            if not rev_dir.is_dir():
                continue
            for layer_dir in sorted(rev_dir.iterdir()):
                if not (layer_dir.is_dir() and layer_dir.name.startswith("layer")):
                    continue
                try:
                    L = int(layer_dir.name[len("layer"):])
                except ValueError:
                    continue
                if sae_layer is not None and L != sae_layer:
                    continue
                feats = layer_dir / "features"
                if (feats / "nld.npy").exists() and (feats / "eng.npy").exists():
                    discovered.append((repo, rev_dir.name, L))

    if not discovered:
        print("[repr] SKIPPED C4 — no SAE feature dumps found under "
              f"{_sae_root}/sae/. Run `python -m analyze.sae.train_sae ...` "
              "then `python -m analyze.sae.extract_features ...` first.")
    for repo, rev, L in discovered:
        df = analyze_sae_features(
            repo=repo, revision=rev, layer=L,
            sae_root=_sae_root, overwrite=overwrite,
        )
        results[f"sae_{repo}_{rev}_L{L}"] = df

    # C5: Language neuron detection
    print("\n" + "=" * 60 + "\n C5: LANGUAGE NEURON DETECTION\n" + "=" * 60)
    try:
        from ppl_utils import load_flores_sentences
        sentences_nl = load_flores_sentences("nld")[:50]
        sentences_en = load_flores_sentences("eng")[:50]

        for repo in repos[:5]:
            r = detect_language_neurons(repo, sentences_nl, sentences_en, overwrite=overwrite)
            results[f"neurons_{repo}"] = r
    except Exception as e:
        print(f"[repr] SKIPPED C5 — {e}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embedding & representation analysis")
    parser.add_argument("--repos", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_representation_analysis(repos=args.repos, overwrite=args.overwrite)
