"""Component enumeration helpers for susceptibility analysis.

Wraps ``devinterp.slt.weight_restrictions.create_param_masks`` with a
schema that returns a list of ``(name, mask)`` pairs for every component
of interest in a Beetle / Pico transformer:

  - ``embed`` and ``unembed``
  - per-layer ``MLP`` blocks
  - per-(layer, head) attention heads (Q/O per-head, KV per-group under GQA)

The Beetle and Pico backbones are both derived from
``LlamaForCausalLM`` (HF) which devinterp supports natively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import torch

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentSpec:
    """A named weight-restriction passed to ``susceptibilities()``.

    Attributes
    ----------
    name : str
        Stable identifier used as a column label in response matrices
        (e.g. ``"l3h7"``, ``"l3 mlp"``, ``"embed"``).
    restriction : str
        ``devinterp`` weight-restriction string. ``None`` is reserved for
        the all-parameters baseline ("full").
    kind : str
        ``"head" | "mlp" | "embed" | "unembed" | "full"``.
    layer : int | None
        Layer index, or ``None`` for top-level components.
    head : int | None
        Head index, or ``None`` for non-attention components.
    """

    name: str
    restriction: str | None
    kind: str
    layer: int | None = None
    head: int | None = None


def _num_layers(model: torch.nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "num_hidden_layers"):
        return int(cfg.num_hidden_layers)
    # Fallback: inspect ``.model.layers`` if it exists.
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError("Cannot determine number of layers for this model.")
    return len(layers)


def _num_heads(model: torch.nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "num_attention_heads"):
        return int(cfg.num_attention_heads)
    raise ValueError("Cannot determine number of attention heads.")


def enumerate_components(
    model: torch.nn.Module,
    *,
    include_full: bool = True,
    include_embed: bool = True,
    include_per_head: bool = True,
    include_per_mlp: bool = True,
) -> list[ComponentSpec]:
    """Return the canonical component list for a Beetle / Pico model.

    For a 12-layer, 12-head model: 1 (full) + 2 (embed/unembed) + 12 (mlp)
    + 144 (heads) = 159 components by default.
    """
    n_layers = _num_layers(model)
    n_heads = _num_heads(model)
    out: list[ComponentSpec] = []

    if include_full:
        out.append(ComponentSpec("full", None, "full"))
    if include_embed:
        out.append(ComponentSpec("embed", "embed", "embed"))
        out.append(ComponentSpec("unembed", "unembed", "unembed"))
    if include_per_mlp:
        for layer in range(n_layers):
            out.append(
                ComponentSpec(
                    name=f"l{layer}_mlp",
                    restriction=f"l{layer} mlp",
                    kind="mlp",
                    layer=layer,
                )
            )
    if include_per_head:
        for layer in range(n_layers):
            for head in range(n_heads):
                out.append(
                    ComponentSpec(
                        name=f"l{layer}h{head}",
                        restriction=f"l{layer}h{head}",
                        kind="head",
                        layer=layer,
                        head=head,
                    )
                )

    log.info(
        "enumerate_components: %d total (layers=%d, heads/layer=%d)",
        len(out), n_layers, n_heads,
    )
    return out


def build_devinterp_masks(
    model: torch.nn.Module,
    components: list[ComponentSpec],
) -> dict[str, "ParamMasks | None"]:
    """Materialise ``ComponentSpec``s into devinterp ``ParamMasks`` dicts.

    Returns a dict ``{name: mask | None}`` ready to pass as
    ``weight_restrictions=`` to ``devinterp.slt.susceptibilities``.
    """
    from devinterp.slt.weight_restrictions import create_param_masks  # type: ignore

    out: dict[str, "ParamMasks | None"] = {}
    for c in components:
        if c.restriction is None:
            out[c.name] = None  # "full" — entire parameter vector
        else:
            out[c.name] = create_param_masks(model, c.restriction)
    return out


def iter_component_groups(
    components: list[ComponentSpec],
) -> Iterator[tuple[str, list[ComponentSpec]]]:
    """Yield ``(group_name, [component, ...])`` partitions by ``kind``.

    Used by aggregate.py and plot_slt.py to colour / label components
    grouped by attention-head vs MLP vs embed in figures.
    """
    by_kind: dict[str, list[ComponentSpec]] = {}
    for c in components:
        by_kind.setdefault(c.kind, []).append(c)
    for kind, lst in by_kind.items():
        yield kind, lst
