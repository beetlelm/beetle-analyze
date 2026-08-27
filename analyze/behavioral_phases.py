"""
analyze/behavioral_phases.py — Michaelov & Levy NeurIPS 2025 phase regression.

Replicates the regression described in Michaelov & Levy (2025) §3:

    LM_logprob(token | context)
        ~  β_uni · log p_unigram(token)
         + β_n   · log p_ngram(token | context_n)        for n ∈ {1..5}
         + β_sim · cosine(fastText(token), fastText(context))

Tracks the coefficients β_* across pretraining checkpoints; the three
behavioural phases are:

    Phase 1 — β_uni and β_sim peak,  β_5gram still ≈ 0.
    Phase 2 — β_5gram rises sharply, β_uni declines.
    Phase 3 — coefficients stabilise.

Onsets are detected with the same break-detector as
`phase_transition.detect_breakthrough` so the *behavioural* phases can be
overlaid on the *structural* phases (PS / UAS).

Required external resources:
    • infini-gram index over the model's pretraining corpus
        (https://github.com/liujch1998/infini-gram)
    • fastText vectors per language (Wiki + CC pretrained, 300-dim).

The infini-gram and fastText backends are wrapped behind a thin
`NgramProvider` / `EmbeddingProvider` interface so they can be swapped
for a small smoke-test back-end (literal-count + dummy similarity) used
in CI replication of the published Pythia trajectory.

CLI
---
    python -m analyze.behavioral_phases \
        --model        Beetle-2B/eng_mono \
        --probe-tokens results/probes/eng_nawoco.tsv \
        --infini-gram  /path/to/infini-gram/eng \
        --fasttext     /path/to/cc.eng.300.bin \
        --output       results/behavioral_phases/eng_mono.csv
"""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "behavioral_phases"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Provider interfaces
# ---------------------------------------------------------------------------

class NgramProvider:
    """Wraps an infini-gram (or smoke-test) backend for n-gram queries."""

    def __init__(self, kind: str = "smoke", path: Optional[str] = None):
        self.kind = kind
        self._engine = None
        if kind == "infini-gram":
            from infini_gram.engine import InfiniGramEngine  # type: ignore
            assert path is not None
            self._engine = InfiniGramEngine(index_dir=path)
        elif kind != "smoke":
            raise ValueError(f"Unknown ngram backend '{kind}'.")

    def logp(self, context_tokens: list[str], target: str, n: int) -> float:
        """log p(target | context's last n-1 tokens)."""
        if self.kind == "smoke":
            # Always return -log(vocab) ≈ uniform
            return -float(np.log(50_000))
        ctx = " ".join(context_tokens[-(n - 1):]) if n > 1 else ""
        try:
            return float(self._engine.logp_ngram(ctx, target))
        except Exception:
            return float("nan")


class EmbeddingProvider:
    """Cosine similarity between target token and average-context fastText vector."""

    def __init__(self, kind: str = "smoke", path: Optional[str] = None):
        self.kind = kind
        self._model = None
        if kind == "fasttext":
            import fasttext  # type: ignore
            assert path is not None
            self._model = fasttext.load_model(path)
        elif kind != "smoke":
            raise ValueError(f"Unknown embedding backend '{kind}'.")

    def cosine_to_context(self, context_tokens: list[str], target: str) -> float:
        if self.kind == "smoke":
            return 0.0
        if not context_tokens:
            return 0.0
        v_t = np.array(self._model.get_word_vector(target), dtype=np.float32)
        v_c = np.mean(
            [self._model.get_word_vector(w) for w in context_tokens],
            axis=0,
        )
        n_t = np.linalg.norm(v_t) + 1e-8
        n_c = np.linalg.norm(v_c) + 1e-8
        return float(v_t @ v_c / (n_t * n_c))


# ---------------------------------------------------------------------------
# Per-checkpoint regression
# ---------------------------------------------------------------------------

@torch.no_grad()
def _lm_logprob_for_targets(
    ilm,
    contexts: list[str],
    targets: list[str],
) -> list[float]:
    """log p_LM(target | context) for each pair, using minicons IncrementalLMScorer."""
    out = []
    for ctx, tgt in zip(contexts, targets):
        try:
            score = ilm.conditional_score(ctx, tgt, bos_token=False)
            first = score[0] if score else None
            if first is None:
                out.append(float("nan"))
                continue
            if hasattr(first, "__iter__") and not isinstance(first, (str, bytes)):
                out.append(float(sum(first)))
            else:
                out.append(float(first))
        except Exception:
            out.append(float("nan"))
    return out


def _build_design_matrix(
    contexts: list[str],
    targets: list[str],
    ngram_provider: NgramProvider,
    emb_provider: EmbeddingProvider,
    ngram_orders: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> pd.DataFrame:
    rows = []
    for ctx, tgt in zip(contexts, targets):
        ctx_tokens = ctx.split()
        row = {"target": tgt, "context_len": len(ctx_tokens)}
        for n in ngram_orders:
            row[f"ngram_{n}"] = ngram_provider.logp(ctx_tokens, tgt, n)
        row["sim"] = emb_provider.cosine_to_context(ctx_tokens, tgt)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_phase_regression(
    lm_logp: list[float],
    design: pd.DataFrame,
    ngram_orders: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, float]:
    """OLS LM_logp ~ Σ β_n · ngram_n + β_sim · sim.  Returns β_n, β_sim, R²."""
    import statsmodels.formula.api as smf
    df = design.copy()
    df["lm_logp"] = lm_logp
    use = ["lm_logp", "sim"] + [f"ngram_{n}" for n in ngram_orders]
    df = df[use].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 50:
        return {f"beta_ngram_{n}": float("nan") for n in ngram_orders} | {
            "beta_sim": float("nan"), "r2": float("nan"), "n": int(len(df)),
        }
    formula = "lm_logp ~ sim + " + " + ".join(f"ngram_{n}" for n in ngram_orders)
    fit = smf.ols(formula=formula, data=df).fit()
    out = {f"beta_ngram_{n}": float(fit.params.get(f"ngram_{n}", np.nan))
           for n in ngram_orders}
    out["beta_sim"] = float(fit.params.get("sim", np.nan))
    out["r2"] = float(fit.rsquared)
    out["n"] = int(len(df))
    return out


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

def score_checkpoint(
    model_repo: str,
    revision: str,
    *,
    contexts: list[str],
    targets: list[str],
    ngram_provider: NgramProvider,
    emb_provider: EmbeddingProvider,
    device: str = DEVICE,
) -> dict:
    from minicons import scorer
    ilm = scorer.IncrementalLMScorer(
        model_repo, device, tokenizer=model_repo,
        revision=revision, trust_remote_code=True,
    )
    lm_lp = _lm_logprob_for_targets(ilm, contexts, targets)
    del ilm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    design = _build_design_matrix(contexts, targets, ngram_provider, emb_provider)
    out = fit_phase_regression(lm_lp, design)
    out["model"] = model_repo
    out["revision"] = revision
    return out


def score_all_checkpoints(
    repo: str,
    *,
    contexts: list[str],
    targets: list[str],
    ngram_provider: NgramProvider,
    emb_provider: EmbeddingProvider,
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
    except Exception:
        step_branches = [(0, "main")]
    if not step_branches:
        step_branches = [(0, "main")]

    rows = []
    for step, branch in tqdm(step_branches, desc=f"behav[{repo.split('/')[-1]}]",
                             unit="ckpt", leave=False):
        try:
            row = score_checkpoint(
                repo, branch,
                contexts=contexts, targets=targets,
                ngram_provider=ngram_provider, emb_provider=emb_provider,
                device=device,
            )
        except Exception as e:
            print(f"[behavioral_phases]   step {step} failed: {e}")
            continue
        row["step"] = step
        rows.append(row)
    return pd.DataFrame(rows)


def detect_phase_onsets(traj: pd.DataFrame) -> dict[str, Optional[int]]:
    """Return Phase-1 → Phase-2 onset (5-gram coefficient rise) and stabilization."""
    from analyze.phase_transition import detect_breakthrough
    traj = traj.sort_values("step")
    p2 = detect_breakthrough(traj["step"], traj["beta_ngram_5"].abs(), threshold=0.10)
    # Stabilization heuristic: first step from which |Δβ_5gram| < 0.01 for 3 ckpts.
    p3 = None
    if "beta_ngram_5" in traj.columns:
        diffs = traj["beta_ngram_5"].diff().abs()
        for i in range(2, len(diffs)):
            if (diffs.iloc[i - 2:i + 1] < 0.01).all():
                p3 = int(traj["step"].iloc[i])
                break
    return {"phase2_onset": p2, "phase3_onset": p3}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Behavioral-phases regression (M&L 2025).")
    p.add_argument("--model", required=True)
    p.add_argument("--probe-tokens", required=True,
                   help="TSV with columns (context, target).")
    p.add_argument("--infini-gram", default=None,
                   help="Path to an infini-gram index dir; if omitted use smoke backend.")
    p.add_argument("--fasttext", default=None,
                   help="Path to a fastText .bin model; if omitted use smoke backend.")
    p.add_argument("--output", default=None)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    if args.smoke_test:
        # Sanity: 50 fake (ctx, tgt) pairs through smoke backends and Pythia-160m.
        contexts = ["the cat sat on the"] * 50
        targets  = ["mat"] * 50
        ng = NgramProvider("smoke")
        eb = EmbeddingProvider("smoke")
        df = score_all_checkpoints(
            "EleutherAI/pythia-160m",
            contexts=contexts, targets=targets,
            ngram_provider=ng, emb_provider=eb,
        )
        print(df.head())
        raise SystemExit(0)

    df_probe = pd.read_csv(args.probe_tokens, sep="\t")
    contexts = df_probe["context"].astype(str).tolist()
    targets  = df_probe["target"].astype(str).tolist()
    ng = NgramProvider("infini-gram" if args.infini_gram else "smoke", args.infini_gram)
    eb = EmbeddingProvider("fasttext" if args.fasttext else "smoke", args.fasttext)

    traj = score_all_checkpoints(
        args.model,
        contexts=contexts, targets=targets,
        ngram_provider=ng, emb_provider=eb,
        device=args.device,
    )
    out = Path(args.output) if args.output else (
        RESULTS_DIR / f"{args.model.replace('/', '__')}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    traj.to_csv(out, index=False)
    print(f"[behavioral_phases] wrote → {out}")
    print("Onsets:", detect_phase_onsets(traj))
