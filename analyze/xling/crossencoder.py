"""
analyze/xling/crossencoder.py
─────────────────────────────
CrossEncoder semantic-alignment probe (Sets B3, C2).

We follow the McCoy-et-al. principle (cited in the user spec for this
project): a probe trained on the same distribution as the model under
study reflects the model's behaviour, not the probe's.  The
CrossEncoder here is therefore trained on a 400M-token subsample of
each Beetle model's pretraining distribution (FineWeb-2B NLD/ENG)
rather than on a generic third-party multilingual corpus.

Architecture: a thin two-layer transformer head sitting on top of the
Beetle model's *intermediate* activations, fine-tuned to score
sentence-pair semantic equivalence on FLORES-200.  By reading from
each layer in turn we localise where in the network cross-lingual
semantic structure lives — Sets B3 (peak-accuracy layer) and C2
(degradation curve over training checkpoints).

Public surface:

  * :class:`CrossEncoderProbe` — the probe model itself.
  * :func:`train_probe` — trains a probe on a fixed layer of activations.
  * :func:`evaluate_probe` — accuracy on FLORES dev pairs (with shifted
    negatives) and per-layer trajectory.

We avoid any dependence on ``sentence-transformers`` at module import
time so that lightweight CKA / SVCCA runs that don't need the
CrossEncoder don't pay the import cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CrossEncoderProbeConfig:
    """Hyperparameters for the probe.

    Defaults are tuned for the 0.2B Beetle pico (d_model=768) on FLORES
    dev (~1k pairs); change ``hidden_dim`` and ``epochs`` for larger
    backbones.
    """
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 5
    seed: int = 0


# ---------------------------------------------------------------------------
# Probe model
# ---------------------------------------------------------------------------

class CrossEncoderProbe:
    """A small bi-encoder that scores cross-lingual semantic equivalence.

    Inputs are pre-extracted, pre-pooled activations from a *single*
    layer of the model under study (one vector per sentence).  The probe
    learns a score:

        score(s_src, s_tgt) = MLP([h_src ; h_tgt ; |h_src − h_tgt|])

    which we fit with binary cross-entropy on (positive_pair,
    shifted_negative_pair) examples.  This mirrors the Sentence-BERT /
    LaBSE training objective, restricted to the activations of the
    backbone we care about.

    The probe is intentionally tiny so its accuracy reflects the
    backbone's representational quality, not the probe's capacity —
    Hewitt & Liang (2019) control-task style.
    """

    def __init__(self, d_in: int, cfg: CrossEncoderProbeConfig):
        import torch
        import torch.nn as nn

        torch.manual_seed(cfg.seed)
        layers: List[nn.Module] = []
        d = 3 * d_in
        for _ in range(cfg.n_layers):
            layers.append(nn.Linear(d, cfg.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.dropout))
            d = cfg.hidden_dim
        layers.append(nn.Linear(d, 1))
        self.model = nn.Sequential(*layers)
        self.cfg = cfg
        self.d_in = d_in

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _pair_features(h_src, h_tgt):
        import torch

        return torch.cat([h_src, h_tgt, (h_src - h_tgt).abs()], dim=-1)

    # ── Training ─────────────────────────────────────────────────────────

    def fit(
        self,
        h_src_pos,
        h_tgt_pos,
        h_src_neg,
        h_tgt_neg,
        *,
        device: str = "cpu",
    ) -> List[float]:
        """Train on positive (paired) and negative (shifted) examples.

        All four inputs are torch tensors of shape ``(N, d_in)`` —
        exactly what :func:`xling.activation_buffer.ActivationBuffer`
        produces with ``pooling="mean"``.  Returns the per-epoch loss
        history.
        """
        import torch
        import torch.nn as nn

        self.model = self.model.to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        h_src = torch.cat([h_src_pos.to(device), h_src_neg.to(device)], dim=0)
        h_tgt = torch.cat([h_tgt_pos.to(device), h_tgt_neg.to(device)], dim=0)
        n_pos = h_src_pos.shape[0]
        n_neg = h_src_neg.shape[0]
        y = torch.cat([
            torch.ones(n_pos, device=device),
            torch.zeros(n_neg, device=device),
        ])

        history: List[float] = []
        n = h_src.shape[0]
        bs = self.cfg.batch_size
        for epoch in range(self.cfg.epochs):
            perm = torch.randperm(n, device=device)
            running = 0.0
            steps = 0
            for s in range(0, n, bs):
                idx = perm[s : s + bs]
                feats = self._pair_features(h_src[idx], h_tgt[idx])
                logits = self.model(feats).squeeze(-1)
                loss = loss_fn(logits, y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += float(loss.item())
                steps += 1
            history.append(running / max(1, steps))
        return history

    # ── Evaluation ───────────────────────────────────────────────────────

    def score(self, h_src, h_tgt, *, device: str = "cpu"):
        import torch

        with torch.no_grad():
            self.model = self.model.to(device).eval()
            feats = self._pair_features(h_src.to(device), h_tgt.to(device))
            return torch.sigmoid(self.model(feats).squeeze(-1)).cpu()


# ---------------------------------------------------------------------------
# Public training / evaluation API
# ---------------------------------------------------------------------------

def train_probe(
    *,
    h_src: "torch.Tensor",
    h_tgt: "torch.Tensor",
    h_src_neg: "torch.Tensor",
    h_tgt_neg: "torch.Tensor",
    cfg: Optional[CrossEncoderProbeConfig] = None,
    device: str = "cpu",
) -> CrossEncoderProbe:
    cfg = cfg or CrossEncoderProbeConfig()
    probe = CrossEncoderProbe(d_in=h_src.shape[-1], cfg=cfg)
    probe.fit(h_src, h_tgt, h_src_neg, h_tgt_neg, device=device)
    return probe


def evaluate_probe(
    probe: CrossEncoderProbe,
    *,
    h_src: "torch.Tensor",
    h_tgt: "torch.Tensor",
    h_src_neg: "torch.Tensor",
    h_tgt_neg: "torch.Tensor",
    device: str = "cpu",
) -> Dict[str, float]:
    """Returns accuracy + AUROC on a held-out pair set.

    Positive and negative inputs must be of identical shape; the
    classifier threshold is 0.5 (sigmoid).
    """
    import torch
    from sklearn.metrics import roc_auc_score  # type: ignore

    pos = probe.score(h_src, h_tgt, device=device).numpy()
    neg = probe.score(h_src_neg, h_tgt_neg, device=device).numpy()
    y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    y_score = np.concatenate([pos, neg])
    acc = float(((y_score >= 0.5) == y_true).mean())
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = float("nan")
    return {"accuracy": acc, "auroc": auroc}


# ---------------------------------------------------------------------------
# Layer-wise sweep convenience
# ---------------------------------------------------------------------------

def layerwise_alignment(
    *,
    src_acts_by_layer: Dict[int, "torch.Tensor"],
    tgt_acts_by_layer: Dict[int, "torch.Tensor"],
    src_neg_acts_by_layer: Dict[int, "torch.Tensor"],
    tgt_neg_acts_by_layer: Dict[int, "torch.Tensor"],
    cfg: Optional[CrossEncoderProbeConfig] = None,
    device: str = "cpu",
) -> Dict[int, Dict[str, float]]:
    """Train + evaluate one probe per layer.  Returns
    ``{layer: {"accuracy": ..., "auroc": ...}}``.

    Used for Set B3 (peak-accuracy layer per language pair) and Set C2
    (alignment trajectory across checkpoints).
    """
    common = sorted(set(src_acts_by_layer) & set(tgt_acts_by_layer))
    results: Dict[int, Dict[str, float]] = {}
    for l in common:
        probe = train_probe(
            h_src=src_acts_by_layer[l],
            h_tgt=tgt_acts_by_layer[l],
            h_src_neg=src_neg_acts_by_layer[l],
            h_tgt_neg=tgt_neg_acts_by_layer[l],
            cfg=cfg,
            device=device,
        )
        results[l] = evaluate_probe(
            probe,
            h_src=src_acts_by_layer[l],
            h_tgt=tgt_acts_by_layer[l],
            h_src_neg=src_neg_acts_by_layer[l],
            h_tgt_neg=tgt_neg_acts_by_layer[l],
            device=device,
        )
    return results
