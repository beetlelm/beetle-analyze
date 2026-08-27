"""ActivationBuffer — streams activations from a hooked LM layer.

Adapted from llm-jp-sae/utils/activation_buffer.py with these changes:
  * relative import of Trace
  * d_submodule resolves hidden_size OR d_model (PicoDecoderHF stores d_model)
  * tensor-or-tuple guard on trace.output (PicoDecoder layers return tensors)
  * pretokenized=True branch reads `input_ids` directly (no tokenizer step)
  * padding_side awareness in BOS-removal logic
"""

import gc

import torch

from .nethook import Trace


def _resolve_d_submodule(model) -> int:
    cfg = model.config
    d = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
    if d is None:
        raise AttributeError("model.config has no hidden_size or d_model")
    return d


class ActivationBuffer:
    def __init__(
        self,
        model,
        submodule: str,
        dataset,
        buffer_size: int,
        refresh_batch_size: int,
        batch_size: int,
        device: str = "cpu",
        remove_bos: bool = True,
        pretokenized: bool = False,
        padding_side: str = "right",
    ):
        self.d_submodule = _resolve_d_submodule(model)
        self.activations = torch.empty(0, self.d_submodule, device=device)
        self.read = torch.zeros(0).bool()

        self.dataset = dataset
        self.model = model
        self.submodule = submodule
        self.buffer_size = buffer_size
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = batch_size
        self.device = device
        self.remove_bos = remove_bos
        self.pretokenized = pretokenized
        self.padding_side = padding_side

        self._iterator = iter(self.dataset.iter(batch_size=refresh_batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        with torch.no_grad():
            if (~self.read).sum() < self.refresh_batch_size:
                self.refresh()
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[torch.randperm(len(unreads), device=unreads.device)[: self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]

    def tokenized_batch(self):
        batch = next(self._iterator)
        input_ids = torch.tensor(batch["input_ids"]).to(self.device)
        if "attention_mask" in batch:
            attention_mask = torch.tensor(batch["attention_mask"]).to(self.device)
        else:
            # Pretokenized streams may omit attention_mask; assume all-valid.
            attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    def refresh(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        new_activations = torch.empty(
            self.buffer_size,
            self.d_submodule,
            device=self.device,
            dtype=self.model.dtype,
        )
        new_activations[: len(self.activations)] = self.activations
        self.activations = new_activations

        with torch.no_grad():
            while current_idx < self.buffer_size:
                input_ids, attention_mask = self.tokenized_batch()
                with Trace(self.model, self.submodule, stop=True) as trace:
                    _ = self.model(input_ids)
                # PicoDecoder layers return Tensor; HF Llama-style return tuple.
                hidden_states = (
                    trace.output if torch.is_tensor(trace.output) else trace.output[0]
                )  # B x T x D

                if self.remove_bos:
                    if self.padding_side == "left":
                        # Strip first valid token of each sequence (left-padded BOS).
                        cumsum = (attention_mask == 1).float().cumsum(dim=-1)
                        attention_mask = attention_mask.clone()
                        attention_mask[cumsum == 1] = 0
                    else:
                        # Right-padded: BOS is at position 0 of each row.
                        attention_mask = attention_mask.clone()
                        attention_mask[:, 0] = 0
                hidden_states = hidden_states[attention_mask != 0]

                remaining_space = self.buffer_size - current_idx
                hidden_states = hidden_states[:remaining_space]

                self.activations[current_idx : current_idx + len(hidden_states)] = hidden_states
                current_idx += len(hidden_states)

        self.read = torch.zeros(len(self.activations), dtype=torch.bool, device=self.device)
