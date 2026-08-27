"""Training-config dataclass for SAE training (defaults from llm-jp-sae)."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    num_warmup_steps: int = 1000
    ctx_len: int = 512                 # BeetleLM max_seq_len = 512 (was 1024 for LLM-jp)
    inf_bs_expansion: int = 2
    logging_step: int = 50
    lr: float = 1e-3
    bs_lm: int = 32
    bs_sae: int = 32768
    num_workers: int = 12
    buffer_size: int = 3_000_000
    tokens: int = 100_000_000
    tokens_eval: int = 10_000_000
