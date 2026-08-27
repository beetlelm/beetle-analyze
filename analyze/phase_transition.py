"""
analyze/phase_transition.py — Phase-Transition Diagnostics for BeetleLM.

Implements the two structural-development metrics used by Aoyama et al. and
Saphra et al. as developmental-interpretability signals during pretraining:

    PS  — Prefix-Matching Score  (Olsson et al. 2022 Eq. 2)
                Average attention paid by an attention head to the (i-1)-th
                token of the *previous* occurrence of the current token in a
                random-50-token-repeat sequence.  Maximum across heads.

    UAS — Unlabeled Attachment Score  (Clark et al. 2019; Saphra et al. 2024)
                Best-head-per-relation accuracy on whether the per-head
                attention pattern recovers a parser-labelled head→dependent
                edge.  Aggregated across UD relations.

Both scores are reduced over the model to a single scalar per checkpoint;
breakthrough detection follows Aoyama et al.:

    breakthrough(f) = min { c  :  f(c) > 0.1 }

Saphra et al.'s paired *structure onset / capabilities onset* are computed by
applying the same break-detector to (i) the UAS trajectory and (ii) a
capabilities curve such as BLiMP accuracy or per-token loss.  The function
``detect_dual_onsets`` returns both.

Bilingual probe data
--------------------
For RQ-A2 the trainer emits monolingual L1 / L2 token streams; this module
accepts a ``language`` argument that determines which probe corpus is
sampled.  Token-id frequency tables are produced on first call and cached.

CLI
---
    python -m analyze.phase_transition \
        --models   Beetle-2B/eng_mono Beetle-2B/zho_L1-eng_L2_B5 \
        --languages eng zho \
        --all-checkpoints

Outputs go to ``../results/phase_transition/<slug>_<lang>.csv``.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from eval_ledger import record as _ledger_record
except Exception:  # pragma: no cover — ledger is best-effort
    def _ledger_record(*_a, **_kw) -> None:
        pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "phase_transition"
PROBE_DIR   = Path(__file__).resolve().parent.parent / "data" / "phase_probes"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_PS_SEQ_LEN = 50          # half-length of the (X X) random-repeat probe
DEFAULT_PS_BATCH   = 16
DEFAULT_PS_BATCHES = 8
DEFAULT_UAS_SENTS  = 200
BREAKTHROUGH_THRESH = 0.10       # Aoyama et al. f(c) > 0.1

# UD relations against which UAS is evaluated. Saphra et al. used these eight.
UD_RELATIONS = (
    "nsubj", "obj", "iobj", "amod",
    "advmod", "det", "case", "obl",
)


def _cleanup_model() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Probe construction
# ---------------------------------------------------------------------------

def _build_random_repeat_batch(
    vocab_size: int,
    batch_size: int = DEFAULT_PS_BATCH,
    seq_len: int = DEFAULT_PS_SEQ_LEN,
    seed: int = 0,
    forbid_token_ids: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Sample (B, 2T) random-repeat sequences  [X X] with no token repeats *within* X.

    The *induction-head* check looks at the second half: position 2t is a copy
    of position t, so a head that copies its prefix-1 neighbour from the
    matching first-half position scores 1.0 on the diagonal of the
    attention block (Olsson et al. Fig 3).

    forbid_token_ids
        Optional set of vocabulary ids to exclude from sampling — used in the
        bilingual setting to mask the *other* language's tokens.
    """
    rng = np.random.default_rng(seed)
    if forbid_token_ids is not None:
        forbidden = np.fromiter(forbid_token_ids, dtype=np.int64)
        allowed = np.setdiff1d(np.arange(vocab_size, dtype=np.int64), forbidden)
        if len(allowed) < seq_len:
            raise ValueError(
                f"After masking, only {len(allowed)} tokens remain — need >= {seq_len}."
            )
    else:
        allowed = np.arange(vocab_size, dtype=np.int64)

    out = np.empty((batch_size, 2 * seq_len), dtype=np.int64)
    for b in range(batch_size):
        first = rng.choice(allowed, size=seq_len, replace=False)
        out[b, :seq_len] = first
        out[b, seq_len:] = first
    return torch.from_numpy(out)


def _load_language_token_mask(
    language: str,
    vocab_size: int,
    cache_dir: Path = PROBE_DIR,
) -> set[int]:
    """Return the set of token-ids *frequently* used by `language`.

    Built once from the language-tagged FineWeb pretokenized stream by
    counting token occurrences over a fixed-size sample (1M tokens).  The
    result is cached at ``cache_dir/lang_mask_<language>.json``.

    Used in two ways:
      • monolingual probing of L1 in a bilingual model:
            forbid_token_ids = vocab \ language_mask
      • L2 leakage check:
            forbid_token_ids = language_mask of L1
    """
    cache = Path(cache_dir) / f"lang_mask_{language}.json"
    if cache.exists():
        return set(json.loads(cache.read_text())["token_ids"])
    raise FileNotFoundError(
        f"[phase_transition] Token mask for '{language}' not found at {cache}. "
        f"Build it offline with `analyze.phase_transition.build_lang_mask`."
    )


def build_lang_mask(
    pretokenized_path: str | Path,
    language: str,
    n_tokens: int = 1_000_000,
    threshold: int = 5,
    cache_dir: Path = PROBE_DIR,
) -> Path:
    """Count token-ids in a monolingual pretokenized shard, persist > threshold.

    Reads up to `n_tokens` tokens (concatenated across documents) and writes
    a JSON mapping {token_id: count} restricted to ids whose count exceeds
    `threshold`.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[int, int] = {}

    # Lazy-import datasets — only needed when *building* the mask.
    from datasets import load_from_disk  # type: ignore

    ds = load_from_disk(str(pretokenized_path))
    seen = 0
    for ex in ds:
        ids = ex["input_ids"] if isinstance(ex, dict) else ex
        for t in ids:
            counts[int(t)] = counts.get(int(t), 0) + 1
            seen += 1
            if seen >= n_tokens:
                break
        if seen >= n_tokens:
            break

    kept = sorted(t for t, c in counts.items() if c >= threshold)
    out = cache_dir / f"lang_mask_{language}.json"
    out.write_text(json.dumps({
        "language": language,
        "n_tokens": seen,
        "threshold": threshold,
        "n_kept": len(kept),
        "token_ids": kept,
    }, indent=2))
    print(f"[phase_transition] Wrote {language} mask → {out}  ({len(kept)} ids)")
    return out


# ---------------------------------------------------------------------------
# Prefix-Matching Score (Olsson et al. 2022 Eq. 2)
# ---------------------------------------------------------------------------

@torch.no_grad()
def prefix_matching_score(
    model: torch.nn.Module,
    vocab_size: int,
    *,
    batch_size: int = DEFAULT_PS_BATCH,
    seq_len: int = DEFAULT_PS_SEQ_LEN,
    n_batches: int = DEFAULT_PS_BATCHES,
    forbid_token_ids: Optional[Iterable[int]] = None,
    device: str = DEVICE,
    seed: int = 0,
) -> dict[str, float]:
    """
    Compute Olsson et al.'s prefix-matching score for every (layer, head)
    and return the per-checkpoint summary statistics.

    For a head h at layer l with attention matrix A ∈ R^{2T × 2T} on the
    random-repeat probe:

        PS_lh  =  mean_{t in [T, 2T)}   A[t, t - T + 1]

    ie the attention from the second-half position t to the position
    *immediately after* its first-occurrence match in the first half.
    A perfect induction head scores 1.0; random scores ≈ 1/T.

    The reported scalar is **max over heads** (Aoyama et al. Eq. 1).

    The function expects ``model`` to expose attention probabilities via the
    PicoDecoder ``return_attentions`` hook.  If not available, falls back to
    monkey-patching ``torch.nn.functional.scaled_dot_product_attention`` to
    capture the softmax weights.
    """
    model.eval()
    raw = getattr(model, "module", model)

    # Discover layout. PicoDecoder exposes attention_n_heads / n_layers in
    # model_config; pico-style returns (logits, extras).
    cfg = getattr(raw, "model_config", None)
    n_layers = int(getattr(cfg, "n_layers", getattr(cfg, "num_layers", 0))) if cfg else 0
    n_heads  = int(getattr(cfg, "attention_n_heads", getattr(cfg, "n_heads", 0))) if cfg else 0
    if n_layers == 0 or n_heads == 0:
        raise RuntimeError(
            "[phase_transition] Could not infer (n_layers, n_heads) from model_config."
        )

    # Allocate accumulator: (layers, heads)
    score_sum = torch.zeros(n_layers, n_heads, device=device)
    n_seen    = torch.zeros(n_layers, n_heads, device=device)

    # Capture attention weights via a forward hook on every attention module.
    captured: list[torch.Tensor] = []

    def _attn_hook(_mod, _inp, output):
        # Many PicoDecoder variants expose (output, attn_weights) when
        # `output_attentions=True`.  Otherwise the user must register a custom
        # hook upstream; we silently skip if the second element is missing.
        if isinstance(output, tuple) and len(output) >= 2 and torch.is_tensor(output[1]):
            captured.append(output[1].detach())

    handles = []
    # Heuristic: find every nn.Module whose class name ends in "Attention".
    for mod in raw.modules():
        if mod.__class__.__name__.lower().endswith("attention"):
            handles.append(mod.register_forward_hook(_attn_hook))

    try:
        T = seq_len
        for k in range(n_batches):
            ids = _build_random_repeat_batch(
                vocab_size=vocab_size,
                batch_size=batch_size,
                seq_len=T,
                seed=seed + k,
                forbid_token_ids=forbid_token_ids,
            ).to(device)
            captured.clear()
            try:
                _ = raw(ids, output_attentions=True)
            except TypeError:
                _ = raw(ids)

            if not captured:
                raise RuntimeError(
                    "[phase_transition] No attention tensors captured. "
                    "Either pass output_attentions=True manually or use a "
                    "model that exposes per-layer attention weights."
                )

            # Each captured tensor: (B, H, S, S)
            for layer_idx, A in enumerate(captured[:n_layers]):
                # Indices for the second-half positions and their matches.
                idx_t   = torch.arange(T, 2 * T, device=A.device)
                idx_src = idx_t - T + 1                       # prefix-1 match
                # Clip src for the first second-half position (t=T → src=1).
                idx_src = idx_src.clamp(min=0, max=2 * T - 1)
                # Gather A[..., t, src] for each t,src pair.
                A_sel = A[..., idx_t, idx_src]                # (B, H, T)
                ps    = A_sel.mean(dim=(0, 2))                # (H,)
                score_sum[layer_idx, :ps.shape[0]] += ps.to(score_sum.device)
                n_seen[layer_idx, :ps.shape[0]]    += 1
    finally:
        for h in handles:
            h.remove()

    per_head = (score_sum / n_seen.clamp(min=1)).cpu().numpy()
    return {
        "ps_max": float(np.nanmax(per_head)),
        "ps_mean": float(np.nanmean(per_head)),
        "ps_p95": float(np.nanpercentile(per_head, 95)),
        "ps_per_head": per_head.tolist(),
    }


# ---------------------------------------------------------------------------
# Unlabeled Attachment Score (Clark et al. 2019; Saphra et al. 2024)
# ---------------------------------------------------------------------------

@torch.no_grad()
def unlabeled_attachment_score(
    model: torch.nn.Module,
    sentences: list[str],
    *,
    parser: str = "spacy",
    language: str = "en",
    relations: tuple[str, ...] = UD_RELATIONS,
    device: str = DEVICE,
) -> dict[str, float]:
    """
    Per-relation best-head UAS.

    For each UD relation r:
        for each (layer, head) (l, h):
            UAS_lh,r = fraction of (head_word, dep_word) pairs in `sentences`
                       for which argmax_j A[l,h,dep,j] == head
        UAS_r = max_{l,h} UAS_lh,r
    UAS = mean_r UAS_r

    Parsing is delegated to spaCy / Stanza based on ``parser``; this lets us
    use UD-consistent labels across the 9 Beetle languages.

    NB: the function expects ``model`` to take token-ids and return per-layer
    attention.  We tokenize the parsed sentence with the model's own
    tokenizer (assumed available as ``model.tokenizer`` or via
    ``minicons.scorer.IncrementalLMScorer``).
    """
    raw = getattr(model, "module", model)
    tokenizer = getattr(raw, "tokenizer", None) or getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            "[phase_transition] No tokenizer attached. Pass an "
            "IncrementalLMScorer.model + .tokenizer pair."
        )

    # Lazy-import the parser of choice.
    if parser == "spacy":
        import spacy
        nlp = spacy.load(_spacy_model_for(language))
    elif parser == "stanza":
        import stanza
        nlp = stanza.Pipeline(lang=language, processors="tokenize,pos,depparse")
    else:
        raise ValueError(f"Unknown parser: {parser}")

    # Pre-extract gold (head, dep, rel) tuples.
    gold_edges: list[tuple[str, str, int, int]] = []  # (sent_id, rel, head_word_idx, dep_word_idx)
    parsed_token_lists: list[list[str]] = []
    for s_id, sent in enumerate(sentences):
        doc = nlp(sent)
        tokens = [t.text for t in (doc if parser == "spacy" else doc.sentences[0].words)]
        parsed_token_lists.append(tokens)
        if parser == "spacy":
            for t in doc:
                if t.dep_ in relations and t.head.i != t.i:
                    gold_edges.append((s_id, t.dep_, t.head.i, t.i))
        else:
            for t in doc.sentences[0].words:
                if t.deprel in relations and t.head != 0:
                    gold_edges.append((s_id, t.deprel, t.head - 1, t.id - 1))

    cfg = getattr(raw, "model_config", None)
    n_layers = int(getattr(cfg, "n_layers", getattr(cfg, "num_layers", 0)))
    n_heads  = int(getattr(cfg, "attention_n_heads", getattr(cfg, "n_heads", 0)))
    correct_per_rel = {r: torch.zeros(n_layers, n_heads) for r in relations}
    total_per_rel   = {r: 0 for r in relations}

    captured: list[torch.Tensor] = []
    def _hook(_m, _i, o):
        if isinstance(o, tuple) and len(o) >= 2 and torch.is_tensor(o[1]):
            captured.append(o[1].detach())
    handles = [m.register_forward_hook(_hook) for m in raw.modules()
               if m.__class__.__name__.lower().endswith("attention")]

    try:
        # Score each parsed sentence once.
        for s_id, tokens in enumerate(parsed_token_lists):
            # Tokenize as a single string. We need word→subword alignment;
            # use the standard "first sub-token of the word" alignment.
            enc = tokenizer(" ".join(tokens), return_tensors="pt",
                            return_offsets_mapping=True, add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            offsets = enc["offset_mapping"][0].tolist()
            # Map word index → first subword index using cumulative char positions.
            char_starts = [0]
            for tok in tokens[:-1]:
                char_starts.append(char_starts[-1] + len(tok) + 1)
            word_to_sub = []
            for ws in char_starts:
                cand = next((i for i, (a, _) in enumerate(offsets) if a == ws), None)
                if cand is None:
                    cand = next((i for i, (a, _) in enumerate(offsets) if a >= ws), 0)
                word_to_sub.append(cand)

            captured.clear()
            try:
                _ = raw(ids, output_attentions=True)
            except TypeError:
                _ = raw(ids)
            if not captured:
                continue

            # Score every gold edge from this sentence.
            for (sid, rel, head_w, dep_w) in gold_edges:
                if sid != s_id:
                    continue
                head_sub = word_to_sub[head_w] if head_w < len(word_to_sub) else None
                dep_sub  = word_to_sub[dep_w] if dep_w < len(word_to_sub) else None
                if head_sub is None or dep_sub is None:
                    continue
                total_per_rel[rel] += 1
                for L, A in enumerate(captured[:n_layers]):
                    pred = A[0, :, dep_sub, :].argmax(dim=-1)   # (H,)
                    correct_per_rel[rel][L] += (pred.cpu() == head_sub).float()
    finally:
        for h in handles:
            h.remove()

    per_rel_uas: dict[str, float] = {}
    for r in relations:
        if total_per_rel[r] == 0:
            per_rel_uas[r] = float("nan")
        else:
            per_rel_uas[r] = float((correct_per_rel[r] / total_per_rel[r]).max().item())
    uas_macro = float(np.nanmean(list(per_rel_uas.values())))
    return {"uas": uas_macro, "uas_per_rel": per_rel_uas}


def _spacy_model_for(language: str) -> str:
    """Map ISO-639-3 / common code to a spaCy model name."""
    return {
        "eng": "en_core_web_sm", "en": "en_core_web_sm",
        "deu": "de_core_news_sm", "de": "de_core_news_sm",
        "ita": "it_core_news_sm", "it": "it_core_news_sm",
        "nld": "nl_core_news_sm", "nl": "nl_core_news_sm",
        "por": "pt_core_news_sm", "pt": "pt_core_news_sm",
        "spa": "es_core_news_sm", "es": "es_core_news_sm",
        "rus": "ru_core_news_sm", "ru": "ru_core_news_sm",
        "tur": "xx_ent_wiki_sm",  "tr": "xx_ent_wiki_sm",
        "zho": "zh_core_web_sm",  "zh": "zh_core_web_sm",
    }.get(language, "xx_ent_wiki_sm")


# ---------------------------------------------------------------------------
# Breakthrough detection (Aoyama et al. + Saphra et al.)
# ---------------------------------------------------------------------------

def detect_breakthrough(
    steps: Iterable[int],
    values: Iterable[float],
    threshold: float = BREAKTHROUGH_THRESH,
) -> Optional[int]:
    """min{c : f(c) > threshold}.  Returns None if never crossed."""
    for s, v in zip(steps, values):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if v > threshold:
            return int(s)
    return None


def detect_dual_onsets(
    df: pd.DataFrame,
    structure_col: str = "uas",
    capabilities_col: str = "blimp",
    structure_threshold: float = 0.30,
    capabilities_threshold: float = 0.55,
) -> dict[str, Optional[int]]:
    """
    Saphra et al.'s structure-onset / capabilities-onset pair (Eq. 2).

    Returns dict with:
        structure_onset     — first step at which structure_col > threshold
        capabilities_onset  — first step at which capabilities_col > threshold
        gap                 — steps from structure to capabilities (None if either missing)
    """
    df = df.sort_values("step")
    s_on = detect_breakthrough(df["step"], df[structure_col], structure_threshold)
    c_on = detect_breakthrough(df["step"], df[capabilities_col], capabilities_threshold)
    return {
        "structure_onset": s_on,
        "capabilities_onset": c_on,
        "gap": (c_on - s_on) if (s_on is not None and c_on is not None) else None,
    }


# ---------------------------------------------------------------------------
# Checkpoint-trajectory driver
# ---------------------------------------------------------------------------

def score_checkpoint(
    model_path: str,
    revision: str = "main",
    language: str = "eng",
    *,
    sentences: Optional[list[str]] = None,
    device: str = DEVICE,
    forbid_other_lang: Optional[str] = None,
    rank: int = 0,
    world_size: int = 1,
) -> dict:
    """Score one checkpoint: returns {step, ps_max, uas, uas_per_rel, ...}."""
    from minicons import scorer

    load_kw = {"revision": revision, "trust_remote_code": True}
    ilm = scorer.IncrementalLMScorer(
        model_path, device, tokenizer=model_path, **load_kw,
    )
    raw = getattr(ilm.model, "module", ilm.model)
    vocab_size = int(getattr(raw.model_config, "vocab_size", 0))

    forbid_ids: Optional[set[int]] = None
    if forbid_other_lang is not None:
        forbid_ids = _load_language_token_mask(forbid_other_lang, vocab_size)

    ps = prefix_matching_score(
        ilm.model, vocab_size,
        forbid_token_ids=forbid_ids,
        device=device,
    )

    if sentences:
        # attach tokenizer for UAS
        setattr(raw, "tokenizer", ilm.tokenizer)
        uas = unlabeled_attachment_score(
            ilm.model, sentences, language=language, device=device,
        )
    else:
        uas = {"uas": float("nan"), "uas_per_rel": {}}

    del ilm, raw
    _cleanup_model()

    return {
        "model": model_path,
        "revision": revision,
        "language": language,
        **ps,
        **uas,
    }


def score_all_checkpoints(
    repo: str,
    *,
    language: str = "eng",
    sentences: Optional[list[str]] = None,
    forbid_other_lang: Optional[str] = None,
    device: str = DEVICE,
) -> pd.DataFrame:
    from huggingface_hub import list_repo_refs
    try:
        refs = list_repo_refs(repo)
        step_branches = sorted(
            [(int(m.group(1)), r.name)
             for r in refs.branches
             if (m := re.match(r"^step-(\d+)$", r.name))],
            key=lambda t: t[0],
        )
    except Exception as e:
        print(f"[phase_transition] Could not list refs for {repo}: {e}")
        step_branches = [(0, "main")]
    if not step_branches:
        step_branches = [(0, "main")]

    rows = []
    for step_num, branch in tqdm(step_branches,
                                 desc=f"phase[{repo.split('/')[-1]}]",
                                 unit="ckpt", leave=False):
        try:
            row = score_checkpoint(
                repo, branch, language,
                sentences=sentences,
                forbid_other_lang=forbid_other_lang,
                device=device,
            )
        except Exception as e:
            print(f"[phase_transition]   step {step_num} failed: {e}")
            continue
        row["step"] = step_num
        rows.append(row)
    return pd.DataFrame(rows)


def run_phase_transition_eval(
    models: list[str],
    languages: list[str],
    *,
    output_dir: str | Path = RESULTS_DIR,
    device: str = DEVICE,
    sentences_for: Optional[dict[str, list[str]]] = None,
    forbid_other_lang_for: Optional[dict[str, str]] = None,
    rank: int = 0,
    world_size: int = 1,
    output_root: "Path | str" = "results",
) -> Path:
    """Score (model, language) pairs and write CSVs.  Mirrors run_meco_eval."""
    from _rank_barrier import mark_done, reset, wait_for_all
    if rank == 0:
        reset("phase_transition", world_size, output_root)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(models) != len(languages):
        raise ValueError("models and languages must have equal length.")

    pairs = list(zip(models, languages))
    pairs = pairs[rank::world_size] if world_size > 1 else pairs

    for repo, lang in tqdm(pairs, desc="phase models", unit="model"):
        slug = repo.split("/")[-1].replace("-", "__")
        sents = (sentences_for or {}).get(lang)
        forbid = (forbid_other_lang_for or {}).get(repo)
        df = score_all_checkpoints(
            repo, language=lang, sentences=sents,
            forbid_other_lang=forbid, device=device,
        )
        out_path = output_dir / f"{slug}_{lang}.csv"
        df.to_csv(out_path, index=False)
        try:
            for ck in df.get("revision", pd.Series([])).unique():
                _ledger_record(
                    stage="phase_transition", repo=repo, checkpoint=str(ck),
                    benchmark="ps_uas", lang=lang, status="ok",
                    rank=rank, output_root=output_root,
                )
        except Exception:
            pass

    mark_done("phase_transition", rank, world_size, output_root)
    if rank == 0 and world_size > 1:
        wait_for_all("phase_transition", world_size, output_root)
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Phase-transition diagnostics (PS + UAS).")
    p.add_argument("--models",    nargs="+", required=True)
    p.add_argument("--languages", nargs="+", required=True,
                   help="ISO-639-3 code per model.")
    p.add_argument("--probe-sentences", type=str, default=None,
                   help="Path to a JSON {lang: [sent, ...]} for UAS scoring.")
    p.add_argument("--forbid-other-lang", type=str, default=None,
                   help="Path to a JSON {model: lang} of L1 to mask in PS.")
    p.add_argument("--output-dir", default=str(RESULTS_DIR))
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--smoke-test", action="store_true",
                   help="Run on Pythia-70M and assert PS@2B > 0.1.")
    args = p.parse_args()

    if args.smoke_test:
        df = score_all_checkpoints("EleutherAI/pythia-70m", language="eng")
        ok = (df["ps_max"] > 0.1).any()
        print(f"[phase_transition][smoke] PS>0.1 ever:  {ok}")
        raise SystemExit(0 if ok else 1)

    sents = json.loads(open(args.probe_sentences).read()) if args.probe_sentences else None
    forbid = json.loads(open(args.forbid_other_lang).read()) if args.forbid_other_lang else None

    run_phase_transition_eval(
        models=args.models,
        languages=args.languages,
        output_dir=args.output_dir,
        device=args.device,
        sentences_for=sents,
        forbid_other_lang_for=forbid,
    )
