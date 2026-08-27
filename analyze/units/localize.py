"""
analyze/units/localize.py — Functional localization of syntactic units.

This is the core of the ``analyze.units`` package and the faithful generalization
of ``localize.py`` from the two source repos (Kryvosheieva et al. 2026,
arXiv:2512.03676; the ``syntax-units`` and ``syntax-interpretability`` releases).

Method (functional localization, borrowed from systems neuroscience)
--------------------------------------------------------------------
1. Run the model over a *localizer* ``MinimalPairSet`` and record, for every
   candidate **unit**, its activation on the grammatical vs. ungrammatical
   sentences (reduced over the token axis).
2. Score each unit by how strongly it *separates* the two conditions — a Welch
   t-statistic by default (the paper's ``--method ttest``), or Cohen's d /
   mean-difference.
3. Select the top ``top_frac`` most selective units. This set (a ``UnitMask``)
   is the phenomenon's functional localizer. ``ablate.py`` then knocks it out;
   ``overlap.py`` compares masks across phenomena; ``trajectory.py`` tracks it
   across training checkpoints.

A **unit** is a coordinate at a chosen ``site``:

  * ``site="resid"`` (default) — a residual-stream dimension at each block
    output. Robust across architectures, including the custom Beetle/Pico
    decoders whose ``forward`` ignores ``output_hidden_states``.
  * ``site="mlp"``  — an MLP hidden ("intermediate") neuron, captured as the
    input to each block's down-projection. This matches the classic
    single-neuron reading of the paper but requires a recognizable MLP.

CLI
---
    python -m analyze.units.localize \\
        --repo Beetle-FineWeb-2B/beetle-bilingual-balanced-b1-fineweb-2b-nld-eng \\
        --categories subject_verb_agreement anaphor_agreement \\
        --top-frac 0.01 --method ttest --out results/units/masks.json
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .stimuli import MinimalPairSet

# Reused, architecture-agnostic block finder.
from analyze._interp_helpers import _find_layer_module_list


SELECTIVITY_METHODS = ("ttest", "welch", "student", "cohend", "meandiff")
SITES = ("resid", "mlp")
REDUCTIONS = ("last", "mean", "sum")

# Submodule name hints for locating the MLP down-projection (its *input* is the
# intermediate neuron activation we treat as a unit).
_MLP_DOWN_HINTS = ("down_proj", "fc2", "c_proj", "w2", "wo", "proj_out", "output_proj")


# ---------------------------------------------------------------------------
# Site resolution
# ---------------------------------------------------------------------------

def _resolve_blocks(model) -> List:
    blocks = _find_layer_module_list(model)
    if blocks is None:
        raise RuntimeError(
            "[units.localize] could not locate the transformer block ModuleList; "
            "pass a standard causal-LM or extend analyze._interp_helpers."
        )
    return list(blocks)


def _find_mlp_down(block):
    """Return the down-projection Linear inside ``block`` (or ``None``)."""
    import torch.nn as nn

    best = None
    for name, mod in block.named_modules():
        low = name.lower()
        if isinstance(mod, nn.Linear) and any(h in low for h in _MLP_DOWN_HINTS):
            # Prefer the one with the largest in_features (the d_ff → d_model map).
            if best is None or mod.in_features > best.in_features:
                best = mod
    return best


def _resolve_site_modules(model, site: str) -> List[Tuple[int, object]]:
    """Return ``[(layer_idx, module)]`` whose captured tensor holds the units."""
    blocks = _resolve_blocks(model)
    if site == "resid":
        return list(enumerate(blocks))
    if site == "mlp":
        mods = []
        for i, b in enumerate(blocks):
            down = _find_mlp_down(b)
            if down is None:
                raise RuntimeError(
                    f"[units.localize] site='mlp': no down-projection found in "
                    f"block {i}. Use site='resid' for this architecture."
                )
            mods.append((i, down))
        return mods
    raise ValueError(f"[units.localize] unknown site {site!r}; choose from {SITES}")


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

def _reduce_tokens(h, mask, reduction: str, padding_side: str):
    """Reduce (B, S, D) hidden states over the token axis using ``mask`` (B, S).

    ``last`` (the source paper's default "last-token" pooling) takes the final
    non-padding position; ``mean``/``sum`` pool over attended tokens.
    """
    import torch

    w = mask.unsqueeze(-1).to(h.dtype)
    if reduction == "mean":
        return (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
    if reduction == "sum":
        return (h * w).sum(dim=1)
    if reduction == "last":
        if padding_side == "left":
            return h[:, -1, :]
        lengths = mask.long().sum(dim=1) - 1
        return h[torch.arange(h.size(0), device=h.device), lengths.clamp(min=0)]
    raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")


def collect_site_activations(
    model,
    tokenizer,
    sentences: Sequence[str],
    *,
    site: str = "resid",
    layers: Optional[Sequence[int]] = None,
    reduction: str = "last",
    batch_size: int = 16,
    max_length: int = 128,
) -> Dict[int, np.ndarray]:
    """Collect per-layer, per-unit activations reduced to one vector per sentence.

    Returns ``{layer_idx: array(n_sentences, n_units)}`` as CPU float32. For
    ``site='resid'`` a unit is a residual-stream dim; for ``site='mlp'`` it is
    an intermediate neuron (the down-projection input).
    """
    import torch

    device = next(model.parameters()).device
    padding_side = getattr(tokenizer, "padding_side", "right")
    want = set(layers) if layers is not None else None
    site_mods = _resolve_site_modules(model, site)
    if want is not None:
        site_mods = [(i, m) for (i, m) in site_mods if i in want]

    out: Dict[int, List[np.ndarray]] = {i: [] for i, _ in site_mods}
    sents = list(sentences)

    was_training = model.training
    model.eval()
    for start in range(0, len(sents), batch_size):
        batch = sents[start : start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        captured: Dict[int, "torch.Tensor"] = {}
        handles = []

        def _make_hook(idx, is_pre):
            if is_pre:  # forward_pre_hook on down_proj → grab its input
                def _pre(_m, args):
                    captured[idx] = args[0].detach()
                return _pre

            def _fwd(_m, _inp, output):  # forward_hook on block → grab hidden
                h = output[0] if isinstance(output, tuple) else output
                captured[idx] = h.detach()
            return _fwd

        for idx, mod in site_mods:
            if site == "mlp":
                handles.append(mod.register_forward_pre_hook(_make_hook(idx, True)))
            else:
                handles.append(mod.register_forward_hook(_make_hook(idx, False)))

        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn)
        finally:
            for h in handles:
                h.remove()

        for idx in out:
            h = captured[idx]
            red = _reduce_tokens(h, attn, reduction, padding_side)
            out[idx].append(red.float().cpu().numpy())

    if was_training:
        model.train()
    return {idx: np.concatenate(chunks, axis=0) for idx, chunks in out.items()}


# ---------------------------------------------------------------------------
# Per-unit selectivity
# ---------------------------------------------------------------------------

def selectivity(good: np.ndarray, bad: np.ndarray, method: str = "ttest",
                *, abs_activations: bool = True,
                return_pvalues: bool = False):
    """Per-unit selectivity magnitude separating ``good`` from ``bad``.

    ``good``/``bad`` are ``(n, d)``; returns a length-``d`` non-negative score
    (or ``(score, pvalues)`` when ``return_pvalues=True``). Higher = more
    sharply the unit distinguishes grammatical from ungrammatical.

    ``abs_activations=True`` (the source paper's convention) takes the absolute
    value of each activation before the test, so a unit counts as selective
    whether it is driven up or down by grammaticality.

    ``method="ttest"``/``"welch"`` both use a Welch (unequal-variance) t-test —
    matching the reference ``ttest_ind(..., equal_var=False)``; ``"student"``
    forces the equal-variance variant.
    """
    good = np.asarray(good, dtype=np.float64)
    bad = np.asarray(bad, dtype=np.float64)
    if abs_activations:
        good, bad = np.abs(good), np.abs(bad)
    if method in ("ttest", "welch", "student"):
        from scipy import stats

        t, p = stats.ttest_ind(good, bad, axis=0,
                               equal_var=(method == "student"), nan_policy="omit")
        score = np.abs(np.nan_to_num(np.asarray(t, dtype=np.float64)))
        if return_pvalues:
            return score, np.nan_to_num(np.asarray(p, dtype=np.float64),
                                        nan=1.0, posinf=1.0, neginf=1.0)
        return score
    if method == "cohend":
        mg, mb = good.mean(0), bad.mean(0)
        ng, nb = good.shape[0], bad.shape[0]
        vg, vb = good.var(0, ddof=1), bad.var(0, ddof=1)
        pooled = np.sqrt(((ng - 1) * vg + (nb - 1) * vb) / max(ng + nb - 2, 1))
        d = (mg - mb) / np.where(pooled == 0, np.nan, pooled)
        return np.abs(np.nan_to_num(d))
    if method == "meandiff":
        s = np.abs(good.mean(0) - bad.mean(0))
        return (s, np.ones_like(s)) if return_pvalues else s
    raise ValueError(f"method must be one of {SELECTIVITY_METHODS}, got {method!r}")


def fdr_control(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (matches the paper's FDR pass)."""
    from scipy.stats import false_discovery_control

    flat = np.clip(np.asarray(pvalues, dtype=np.float64).ravel(), 0, 1)
    flat = np.nan_to_num(flat, nan=1.0, posinf=1.0, neginf=1.0)
    return false_discovery_control(flat).reshape(np.asarray(pvalues).shape)


# ---------------------------------------------------------------------------
# UnitMask
# ---------------------------------------------------------------------------

@dataclass
class UnitMask:
    """A localized set of units, indexed by ``(layer, unit)``.

    ``selected[layer]`` is a boolean array over that layer's units;
    ``scores[layer]`` the underlying selectivity used to threshold.
    """

    selected: Dict[int, np.ndarray]
    scores: Dict[int, np.ndarray]
    top_frac: float
    method: str
    site: str
    phenomenon: str = ""
    category: str = ""
    meta: Dict = field(default_factory=dict)

    # -- basic accessors ---------------------------------------------------
    def count(self) -> int:
        return int(sum(int(v.sum()) for v in self.selected.values()))

    def total_units(self) -> int:
        return int(sum(v.size for v in self.selected.values()))

    def indices(self) -> Set[Tuple[int, int]]:
        """Set of selected ``(layer, unit)`` coordinates."""
        out: Set[Tuple[int, int]] = set()
        for layer, sel in self.selected.items():
            for u in np.nonzero(sel)[0]:
                out.add((int(layer), int(u)))
        return out

    def per_layer_counts(self) -> Dict[int, int]:
        return {L: int(v.sum()) for L, v in self.selected.items()}

    # -- set algebra -------------------------------------------------------
    def jaccard(self, other: "UnitMask") -> float:
        return jaccard(self, other)

    # -- (de)serialization -------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "top_frac": self.top_frac,
            "method": self.method,
            "site": self.site,
            "phenomenon": self.phenomenon,
            "category": self.category,
            "meta": self.meta,
            "selected": {str(L): np.nonzero(v)[0].tolist()
                         for L, v in self.selected.items()},
            "n_units": {str(L): int(v.size) for L, v in self.selected.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "UnitMask":
        selected: Dict[int, np.ndarray] = {}
        for L, idxs in d["selected"].items():
            n = int(d["n_units"][L])
            arr = np.zeros(n, dtype=bool)
            arr[np.asarray(idxs, dtype=int)] = True
            selected[int(L)] = arr
        return cls(
            selected=selected,
            scores={},
            top_frac=d["top_frac"], method=d["method"], site=d["site"],
            phenomenon=d.get("phenomenon", ""), category=d.get("category", ""),
            meta=d.get("meta", {}),
        )


def _topk_mask(values: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask selecting exactly ``k`` highest ``values`` (ties broken by
    index, deterministically). Matches the reference ``is_topk`` count of
    ``num_units`` rather than a ``>= cutoff`` rule, which would over-select when
    many units share the boundary score (e.g. degenerate zero t-values)."""
    k = int(np.clip(k, 1, values.size))
    order = np.argsort(-values, kind="stable")[:k]
    out = np.zeros(values.size, dtype=bool)
    out[order] = True
    return out


def _threshold(scores: Dict[int, np.ndarray], top_frac: float,
               per_layer: bool) -> Dict[int, np.ndarray]:
    """Boolean mask keeping the top ``top_frac`` units by score."""
    if per_layer:
        return {L: _topk_mask(s, max(1, int(math.ceil(top_frac * s.size))))
                for L, s in scores.items()}
    # Global: one budget across all units (the paper's --percentage semantics:
    # num_units = round(top_frac * hidden_dim * n_layers)).
    layers = sorted(scores)
    sizes = [scores[L].size for L in layers]
    flat = np.concatenate([scores[L].ravel() for L in layers])
    k = max(1, int(round(top_frac * flat.size)))
    flat_mask = _topk_mask(flat, k)
    out, off = {}, 0
    for L, n in zip(layers, sizes):
        out[L] = flat_mask[off:off + n]
        off += n
    return out


def localize(
    model,
    tokenizer,
    pair_set: MinimalPairSet,
    *,
    top_frac: float = 0.01,
    method: str = "ttest",
    site: str = "resid",
    layers: Optional[Sequence[int]] = None,
    reduction: str = "last",
    per_layer: bool = False,
    abs_activations: bool = True,
    batch_size: int = 16,
    max_length: int = 128,
) -> UnitMask:
    """Localize the units most selective for ``pair_set``.

    ``per_layer=False`` (default) selects the top ``top_frac`` units *globally*
    across all layers, matching the source paper's ``--percentage`` semantics
    (note: their ``--percentage`` is a *percent*, so ``0.01`` here == ``1.0``
    there). ``abs_activations`` / ``reduction="last"`` reproduce the paper's
    absolute-value + last-token functional-localization contrast.
    """
    good = collect_site_activations(model, tokenizer, pair_set.good, site=site,
                                    layers=layers, reduction=reduction,
                                    batch_size=batch_size, max_length=max_length)
    bad = collect_site_activations(model, tokenizer, pair_set.bad, site=site,
                                   layers=layers, reduction=reduction,
                                   batch_size=batch_size, max_length=max_length)
    scores, pvals = {}, {}
    for L in good:
        s, p = selectivity(good[L], bad[L], method,
                           abs_activations=abs_activations, return_pvalues=True)
        scores[L], pvals[L] = s, p
    selected = _threshold(scores, top_frac, per_layer)
    # BH-adjusted p-values over the full (layer × unit) grid, paper-style.
    stacked = np.stack([pvals[L] for L in sorted(pvals)])
    fdr = fdr_control(stacked)
    return UnitMask(
        selected=selected, scores=scores, top_frac=top_frac, method=method,
        site=site, phenomenon=pair_set.phenomenon, category=pair_set.category,
        meta={"language": pair_set.language, "n_pairs": len(pair_set),
              "per_layer": per_layer, "reduction": reduction,
              "abs_activations": abs_activations,
              "min_fdr_p": float(np.min(fdr)) if fdr.size else 1.0},
    )


# Import here to avoid a cycle at module load (overlap imports UnitMask).
from .overlap import jaccard  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Iterable[str]] = None) -> None:
    from analyze.convergence import _load_checkpoint, _free
    from analyze._interp_helpers import device_for
    from . import stimuli as _stim

    p = argparse.ArgumentParser(description="Localize syntactic units in a checkpoint.")
    p.add_argument("--repo", required=True, help="HF repo id of the model.")
    p.add_argument("--revision", default="main")
    p.add_argument("--categories", nargs="*", default=list(_stim.AGREEMENT_CATEGORIES))
    p.add_argument("--phenomena", nargs="*", default=None,
                   help="Exact BLiMP config names (overrides --categories).")
    p.add_argument("--blimp-id", default="nyu-mll/blimp")
    p.add_argument("--language", default="eng")
    p.add_argument("--max-pairs", type=int, default=200)
    p.add_argument("--top-frac", type=float, default=0.01)
    p.add_argument("--method", default="ttest", choices=SELECTIVITY_METHODS)
    p.add_argument("--site", default="resid", choices=SITES)
    p.add_argument("--reduction", default="last", choices=list(REDUCTIONS))
    p.add_argument("--per-layer", action="store_true")
    p.add_argument("--demo", action="store_true",
                   help="Use the offline demo stimuli instead of BLiMP.")
    p.add_argument("--out", default="results/units/masks.json")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.demo:
        sets = _stim.demo_pairs()
    else:
        sets = _stim.load_blimp(categories=args.categories, phenomena=args.phenomena,
                                max_pairs=args.max_pairs, hf_id=args.blimp_id,
                                language=args.language)

    loaded = _load_checkpoint(args.repo, args.revision, device=device_for())
    if loaded is None:
        raise SystemExit(f"[units.localize] failed to load {args.repo}@{args.revision}")
    model, tok = loaded

    masks = {}
    for s in sets:
        m = localize(model, tok, s, top_frac=args.top_frac, method=args.method,
                     site=args.site, reduction=args.reduction, per_layer=args.per_layer)
        masks[s.phenomenon] = m.to_dict()
        print(f"[units.localize] {s.phenomenon:45s} → {m.count():5d} units "
              f"({m.category})")

    _free(model, tok)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "repo": args.repo, "revision": args.revision,
        "top_frac": args.top_frac, "method": args.method, "site": args.site,
        "masks": masks,
    }, indent=2))
    print(f"[units.localize] wrote {len(masks)} masks → {out}")


if __name__ == "__main__":
    main()
