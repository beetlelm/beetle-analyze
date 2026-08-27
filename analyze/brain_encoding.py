"""
analyze/brain_encoding.py — Voxel-wise fMRI encoding (Nakagi et al. 2025).

For Paper-B RQ-B3 we replicate Nakagi et al.'s three-phase brain-alignment
trajectory.  The pipeline:

    1.  Extract per-token hidden states from a model checkpoint over a
        stimulus passage (text or audio transcript) aligned to the fMRI BOLD
        time-course.
    2.  Optionally pool to TR (~ 2 s) using a haemodynamic-response delay.
    3.  Train an L2-regularised ridge regression per voxel mapping
        hidden states → BOLD on a train fold; evaluate Pearson r on a held-
        out fold.
    4.  Aggregate r over voxels in language-selective ROIs (Fedorenko atlas
        + temporal-cortex ROIs used by Nakagi et al.).
    5.  Repeat across pretraining checkpoints and detect the three-phase
        trajectory:
            Phase 1 — alignment surge (~10⁹–10¹⁰ tokens)
            Phase 2 — detachment/stagnation (~10¹⁰–3·10¹¹)
            Phase 3 — realignment (>3·10¹¹) — only emerges if the language
                      is *sufficiently learned*.

Supported corpora (loader stubs; user must download under licence):
    pereira_2018         — passage-reading fMRI (English).
    lebel_2023           — Moth Radio Hour natural listening (English).
    schoffelen_2019      — MEG, NL + EN (Beetle deu/nld/eng coverage).
    malik_moraleda_2022  — multilingual sentence-level fMRI (45 languages).

All loaders return a dict:
    {
      "stimulus_text":      list[str]  (one entry per TR or sentence),
      "fmri":               np.ndarray (n_TR, n_voxels),
      "roi_mask":           dict[str, np.ndarray[bool]]   per ROI
      "tr_seconds":         float,
      "language":           ISO-639-3,
    }

CLI
---
    python -m analyze.brain_encoding \
        --model        Beetle-2B/eng_mono \
        --corpus       pereira_2018 \
        --output       results/brain/eng_mono_pereira.csv
"""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "brain_encoding"
DATA_DIR    = Path(__file__).resolve().parent.parent / "data" / "brain"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Corpus loader stubs
# ---------------------------------------------------------------------------

def _expect_npz(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"[brain_encoding] {path} not found.  This corpus requires a "
            f"licensed download (see Paper-B plan)."
        )
    return dict(np.load(path, allow_pickle=True))


def _load_pereira_2018():
    return _expect_npz(DATA_DIR / "pereira_2018.npz")


def _load_lebel_2023():
    return _expect_npz(DATA_DIR / "lebel_2023.npz")


def _load_schoffelen_2019():
    return _expect_npz(DATA_DIR / "schoffelen_2019.npz")


def _load_malik_moraleda_2022():
    return _expect_npz(DATA_DIR / "malik_moraleda_2022.npz")


_CORPORA: dict[str, Callable[[], dict]] = {
    "pereira_2018":         _load_pereira_2018,
    "lebel_2023":           _load_lebel_2023,
    "schoffelen_2019":      _load_schoffelen_2019,
    "malik_moraleda_2022":  _load_malik_moraleda_2022,
}


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_hidden_states(
    model_repo: str,
    revision: str,
    sentences: list[str],
    *,
    pooling: str = "last_token",
    device: str = DEVICE,
) -> np.ndarray:
    """Return (N, hidden_dim) feature matrix, one row per sentence."""
    from minicons import scorer
    ilm = scorer.IncrementalLMScorer(
        model_repo, device, tokenizer=model_repo,
        revision=revision, trust_remote_code=True,
    )
    raw = getattr(ilm.model, "module", ilm.model)
    feats = []
    for s in sentences:
        ids = ilm.tokenizer(s, return_tensors="pt").input_ids.to(device)
        try:
            out = raw(ids, output_hidden_states=True)
            hs = out.hidden_states[-1] if hasattr(out, "hidden_states") else out[1][-1]
        except TypeError:
            # PicoDecoder may not implement output_hidden_states; fall back to
            # logits' mean as a coarse representation.
            logits, _ = raw(ids)
            hs = logits  # (B, T, V); pool below by reducing over T
        if pooling == "last_token":
            feats.append(hs[0, -1].cpu().numpy())
        elif pooling == "mean":
            feats.append(hs[0].mean(dim=0).cpu().numpy())
        else:
            raise ValueError(pooling)
    del ilm, raw
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.stack(feats)


# ---------------------------------------------------------------------------
# Ridge encoding (Nakagi §3.2)
# ---------------------------------------------------------------------------

def ridge_encoding_score(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_folds: int = 5,
    alpha_grid: tuple[float, ...] = (1e2, 1e3, 1e4, 1e5),
) -> np.ndarray:
    """5-fold CV ridge per voxel.  Returns Pearson r per voxel (length = Y.shape[1])."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    n, _ = X.shape
    n_vox = Y.shape[1]
    r_per_voxel = np.zeros(n_vox)
    counts = np.zeros(n_vox)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    for tr, te in kf.split(X):
        model = RidgeCV(alphas=alpha_grid).fit(X[tr], Y[tr])
        Yhat = model.predict(X[te])
        for v in range(n_vox):
            yt = Y[te, v]
            yp = Yhat[:, v]
            if np.std(yt) < 1e-9 or np.std(yp) < 1e-9:
                continue
            r_per_voxel[v] += np.corrcoef(yt, yp)[0, 1]
            counts[v] += 1
    counts[counts == 0] = 1
    return r_per_voxel / counts


def aggregate_roi(r_per_voxel: np.ndarray, roi_mask: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        name: float(np.nanmean(r_per_voxel[mask]))
        for name, mask in roi_mask.items()
    }


# ---------------------------------------------------------------------------
# Trajectory + phase detection
# ---------------------------------------------------------------------------

def score_checkpoint(
    model_repo: str,
    revision: str,
    *,
    corpus_data: dict,
    pooling: str = "last_token",
    device: str = DEVICE,
) -> dict:
    sentences = list(corpus_data["stimulus_text"])
    Y = np.asarray(corpus_data["fmri"])
    X = _extract_hidden_states(model_repo, revision, sentences,
                               pooling=pooling, device=device)
    r_vox = ridge_encoding_score(X, Y)
    rois = corpus_data.get("roi_mask", {})
    if isinstance(rois, np.ndarray) and rois.dtype == object:
        rois = rois.item()
    roi_r = aggregate_roi(r_vox, rois) if rois else {}
    return {
        "model": model_repo,
        "revision": revision,
        "global_r_mean": float(np.nanmean(r_vox)),
        **{f"roi_{k}": v for k, v in roi_r.items()},
    }


def score_all_checkpoints(
    repo: str,
    corpus: str,
    *,
    device: str = DEVICE,
) -> pd.DataFrame:
    from huggingface_hub import list_repo_refs
    corpus_data = _CORPORA[corpus]()

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
    for step, branch in tqdm(step_branches, desc=f"brain[{repo.split('/')[-1]}]",
                             unit="ckpt", leave=False):
        try:
            row = score_checkpoint(repo, branch, corpus_data=corpus_data, device=device)
        except Exception as e:
            print(f"[brain_encoding]  step {step} failed: {e}")
            continue
        row["step"] = step
        row["corpus"] = corpus
        rows.append(row)
    return pd.DataFrame(rows)


def detect_three_phase(
    traj: pd.DataFrame,
    metric_col: str = "global_r_mean",
) -> dict[str, Optional[int]]:
    """Detect Nakagi et al.'s three-phase trajectory by piecewise slope changes."""
    traj = traj.sort_values("step")
    s = traj["step"].to_numpy()
    y = traj[metric_col].to_numpy()
    if len(y) < 6:
        return {"alignment_onset": None, "detachment_onset": None, "realignment_onset": None}

    # Phase 1: first idx where y exceeds 50% of peak.
    peak = np.nanmax(y)
    align_idx = int(np.argmax(y > 0.5 * peak)) if peak > 0 else None

    # Phase 2: argmax of diff < 0 plateau after Phase 1.
    detach_idx = None
    if align_idx is not None and align_idx + 2 < len(y):
        diffs = np.diff(y[align_idx:])
        for i in range(len(diffs) - 1):
            if diffs[i] < 0 and diffs[i + 1] < 0:
                detach_idx = align_idx + i + 1
                break

    # Phase 3: first idx after detachment where y > local minimum + 5%.
    realign_idx = None
    if detach_idx is not None and detach_idx + 1 < len(y):
        local_min = np.nanmin(y[detach_idx:])
        for j in range(detach_idx + 1, len(y)):
            if y[j] > local_min + 0.05 * peak:
                realign_idx = j
                break

    return {
        "alignment_onset":   int(s[align_idx]) if align_idx is not None else None,
        "detachment_onset":  int(s[detach_idx]) if detach_idx is not None else None,
        "realignment_onset": int(s[realign_idx]) if realign_idx is not None else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="fMRI encoding three-phase trajectory.")
    p.add_argument("--model", required=True)
    p.add_argument("--corpus", required=True, choices=sorted(_CORPORA))
    p.add_argument("--output", default=None)
    p.add_argument("--device", default=DEVICE)
    args = p.parse_args()
    df = score_all_checkpoints(args.model, args.corpus, device=args.device)
    out = Path(args.output) if args.output else (
        RESULTS_DIR / f"{args.model.replace('/', '__')}_{args.corpus}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[brain_encoding] wrote → {out}")
    print("Three-phase onsets:", detect_three_phase(df))
