"""TopK Sparse Autoencoder subpackage for beetle-analyze.

Adapted from llm-jp-sae (https://github.com/llm-jp/llm-jp-sae) for BeetleLM
PicoDecoder models. Original LLM-jp authors retain copyright on the
adapted training/hooking primitives.

Public API:
  TopKSAE, TopKSAEConfig          — model + config
  train_sae_for_checkpoint        — train an SAE on one beetle checkpoint × layer
  extract_features_for_checkpoint — extract feature activations on Dutch + English
  sae_dir, feature_npy            — output-path resolvers (single source of truth)
"""

from .model import TopKSAE, TopKSAEConfig
from .paths import sae_dir, feature_npy
from .train_sae import train_sae_for_checkpoint
from .extract_features import extract_features_for_checkpoint

__all__ = [
    "TopKSAE",
    "TopKSAEConfig",
    "train_sae_for_checkpoint",
    "extract_features_for_checkpoint",
    "sae_dir",
    "feature_npy",
]
