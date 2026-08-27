"""
analyze/units/ablate.py — Causal ablation of localized units.

Wraps ``ablation.py`` from the source repos. Once ``localize.py`` has produced a
``UnitMask``, this module *causally* tests it: knock the units out (zero- or
mean-ablation) and measure how much the model's minimal-pair accuracy drops. A
large drop on the phenomenon the units were localized for — and, per the paper,
on *other* agreement phenomena too — is the evidence that the units are
functionally responsible for agreement.

Behavioural metric
-------------------
``minimal_pair_accuracy`` = fraction of pairs where the model assigns higher
total log-probability to the grammatical sentence than the ungrammatical one —
the standard BLiMP scoring, computed directly from next-token logits.

Ablation is applied with forward hooks at the same ``site`` the mask was
localized at:

  * ``site="resid"`` — zero / mean-substitute residual-stream dims at each
    block output (a forward hook on the block).
  * ``site="mlp"``   — zero / mean-substitute intermediate neurons via a
    forward *pre*-hook on the down-projection (editing its input).

CLI
---
    python -m analyze.units.ablate \\
        --repo google/gemma-3-4b-pt --demo \\
        --mode zero --top-frac 0.01 --out results/units/ablation.csv
"""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .stimuli import MinimalPairSet
from .localize import UnitMask, _resolve_site_modules


# ---------------------------------------------------------------------------
# Sequence log-probability scoring
# ---------------------------------------------------------------------------

def sequence_logprobs(
    model,
    tokenizer,
    sentences: Sequence[str],
    *,
    batch_size: int = 16,
    max_length: int = 128,
) -> np.ndarray:
    """Total (summed) next-token log-probability of each sentence."""
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    scores: List[float] = []
    model.eval()
    for start in range(0, len(sentences), batch_size):
        batch = list(sentences[start : start + batch_size])
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        tgt = input_ids[:, 1:]
        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        # Mask padding: a target position counts only if it is attended to.
        tgt_mask = attn[:, 1:].to(tok_lp.dtype)
        scores.extend((tok_lp * tgt_mask).sum(dim=1).cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def minimal_pair_accuracy(
    model, tokenizer, pair_set: MinimalPairSet, **kw
) -> float:
    """Fraction of pairs with logprob(good) > logprob(bad)."""
    g = sequence_logprobs(model, tokenizer, pair_set.good, **kw)
    b = sequence_logprobs(model, tokenizer, pair_set.bad, **kw)
    return float(np.mean(g > b))


# ---------------------------------------------------------------------------
# Ablation hooks
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def ablated(model, mask: UnitMask, *, mode: str = "zero",
            mean_vals: Optional[Dict[int, np.ndarray]] = None,
            fill_value: Optional[float] = None):
    """Context manager that knocks out ``mask``'s units for the duration.

    * ``mode="zero"`` sets the selected units to 0.
    * ``mode="mean"`` substitutes a mean activation. Pass a single ``fill_value``
      scalar for the source paper's *global*-mean ablation, or a per-unit
      ``mean_vals`` dict (see ``compute_mean_vals``) for the finer variant.
    """
    import torch

    if mode == "mean" and mean_vals is None and fill_value is None:
        raise ValueError(
            "mode='mean' needs either fill_value (global mean) or mean_vals "
            "(per-unit means from compute_mean_vals).")

    site_mods = dict(_resolve_site_modules(model, mask.site))
    handles = []

    def _edit(tensor, layer):
        sel = mask.selected.get(layer)
        if sel is None or not sel.any():
            return tensor
        idx = torch.as_tensor(np.nonzero(sel)[0], device=tensor.device, dtype=torch.long)
        tensor = tensor.clone()
        if mode == "zero":
            tensor[..., idx] = 0.0
        elif mean_vals is not None:
            mv = torch.as_tensor(mean_vals[layer][sel], device=tensor.device,
                                 dtype=tensor.dtype)
            tensor[..., idx] = mv
        else:
            tensor[..., idx] = float(fill_value)
        return tensor

    for layer, mod in site_mods.items():
        if not mask.selected.get(layer, np.zeros(0, bool)).any():
            continue
        if mask.site == "mlp":
            def _pre(_m, args, _layer=layer):
                return (_edit(args[0], _layer),) + tuple(args[1:])
            handles.append(mod.register_forward_pre_hook(_pre))
        else:
            def _fwd(_m, _inp, output, _layer=layer):
                if isinstance(output, tuple):
                    return (_edit(output[0], _layer),) + tuple(output[1:])
                return _edit(output, _layer)
            handles.append(mod.register_forward_hook(_fwd))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


def compute_mean_vals(model, tokenizer, sentences: Sequence[str], *, site: str,
                      layers=None, **kw) -> Dict[int, np.ndarray]:
    """Per-unit mean activation (for the finer ``mode='mean'`` ablation)."""
    from .localize import collect_site_activations

    acts = collect_site_activations(model, tokenizer, sentences, site=site,
                                    layers=layers, **kw)
    return {L: a.mean(axis=0) for L, a in acts.items()}


def global_activation_mean(model, tokenizer, sentences: Sequence[str], *,
                           site: str = "resid", layers=None, **kw) -> float:
    """Single scalar mean activation across all units/sentences.

    Port of the reference ``global_activation_mean`` — the fill value used by
    the paper's global-mean ablation (``model.set_ablation_mode('mean', mu)``).
    """
    from .localize import collect_site_activations

    acts = collect_site_activations(model, tokenizer, sentences, site=site,
                                    layers=layers, **kw)
    tot = float(sum(a.sum() for a in acts.values()))
    cnt = int(sum(a.size for a in acts.values()))
    return tot / cnt if cnt else 0.0


def random_mask_from_remaining(mask: UnitMask, *, seed: int = 0) -> UnitMask:
    """A same-size mask drawn at random from the *unselected* units.

    Port of the reference ``random_mask_from_remaining``: the control condition
    that shows the localized units matter more than an equal number of random
    ones. Sampling is global across the (layer × unit) grid.
    """
    rng = np.random.default_rng(seed)
    layers = sorted(mask.selected)
    sizes = {L: mask.selected[L].size for L in layers}
    offsets, off = {}, 0
    for L in layers:
        offsets[L] = off
        off += sizes[L]
    total = off

    remaining = []
    for L in layers:
        base = offsets[L]
        for u in np.nonzero(~mask.selected[L])[0]:
            remaining.append(base + int(u))
    k = min(mask.count(), len(remaining))
    chosen = set(rng.choice(np.asarray(remaining), size=k, replace=False).tolist()) \
        if k else set()

    selected = {L: np.zeros(sizes[L], dtype=bool) for L in layers}
    for L in layers:
        base = offsets[L]
        for u in range(sizes[L]):
            if (base + u) in chosen:
                selected[L][u] = True
    return UnitMask(selected=selected, scores={}, top_frac=mask.top_frac,
                    method=mask.method, site=mask.site,
                    phenomenon=f"random({mask.phenomenon})", category="random",
                    meta={"random_seed": seed, "n_units": k})


# ---------------------------------------------------------------------------
# Ablation effect
# ---------------------------------------------------------------------------

def ablation_effect(
    model,
    tokenizer,
    eval_set: MinimalPairSet,
    mask: UnitMask,
    *,
    mode: str = "zero",
    mean_scope: str = "global",
    batch_size: int = 16,
    max_length: int = 128,
) -> Dict[str, float]:
    """Accuracy on ``eval_set`` before vs. after ablating ``mask``.

    ``mode="mean"`` with ``mean_scope="global"`` (default, the paper's variant)
    fills ablated units with one global-mean scalar; ``mean_scope="per_unit"``
    uses each unit's own mean. Returns ``{baseline, ablated, delta, n_units,
    eval_phenomenon, mask_phenomenon, ...}``. Negative ``delta`` = ablation hurt
    (the expected causal signature).
    """
    kw = dict(batch_size=batch_size, max_length=max_length)
    base = minimal_pair_accuracy(model, tokenizer, eval_set, **kw)

    mean_vals = fill_value = None
    if mode == "mean":
        sents = list(eval_set.good) + list(eval_set.bad)
        if mean_scope == "per_unit":
            mean_vals = compute_mean_vals(model, tokenizer, sents, site=mask.site,
                                          layers=sorted(mask.selected), **kw)
        else:
            fill_value = global_activation_mean(model, tokenizer, sents,
                                                site=mask.site,
                                                layers=sorted(mask.selected), **kw)

    with ablated(model, mask, mode=mode, mean_vals=mean_vals, fill_value=fill_value):
        abl = minimal_pair_accuracy(model, tokenizer, eval_set, **kw)

    return {
        "baseline": base,
        "ablated": abl,
        "delta": abl - base,
        "n_units": mask.count(),
        "eval_phenomenon": eval_set.phenomenon,
        "eval_category": eval_set.category,
        "mask_phenomenon": mask.phenomenon,
        "mask_category": mask.category,
        "mode": mode,
    }


def ablation_grid(
    model, tokenizer, masks: Dict[str, UnitMask],
    eval_sets: Sequence[MinimalPairSet], *, mode: str = "zero", **kw,
):
    """Cross the localized ``masks`` against ``eval_sets`` → tidy ``DataFrame``.

    The off-diagonal (mask localized for phenomenon A, evaluated on B) is the
    paper's key generalization test: agreement units ablate agreement broadly.
    """
    import pandas as pd

    rows = []
    for name, mask in masks.items():
        for es in eval_sets:
            r = ablation_effect(model, tokenizer, es, mask, mode=mode, **kw)
            r["mask_name"] = name
            rows.append(r)
    return pd.DataFrame(rows)


def run_ablation(
    model,
    tokenizer,
    pair_set: MinimalPairSet,
    *,
    top_frac: float = 0.01,
    method: str = "ttest",
    site: str = "resid",
    mode: str = "zero",
    mean_scope: str = "global",
    seeds: Sequence[int] = (42, 123, 456, 789),
    batch_size: int = 16,
    max_length: int = 128,
    **localize_kwargs,
) -> Dict[str, float]:
    """Faithful single-phenomenon ablation with a random-mask control.

    Mirrors the reference ``ablation.py`` main loop: **localize on the first
    half** of the pairs, then measure minimal-pair accuracy **on the held-out
    second half** for three conditions —

      * ``none``     — unablated baseline,
      * ``language`` — the localized ``UnitMask`` ablated,
      * ``random``   — an equal-size mask from the *remaining* units, averaged
        over ``seeds``.

    A localized mask that hurts accuracy substantially more than the random
    control is the paper's causal evidence. Returns a dict with the three
    accuracies and their deltas.
    """
    from .localize import localize as _localize

    first, second = pair_set.split(0.5, seed=0)
    kw = dict(batch_size=batch_size, max_length=max_length)

    mask = _localize(model, tokenizer, first, top_frac=top_frac, method=method,
                     site=site, **localize_kwargs)

    acc_none = minimal_pair_accuracy(model, tokenizer, second, **kw)

    mean_vals = fill_value = None
    if mode == "mean":
        sents = list(second.good) + list(second.bad)
        if mean_scope == "per_unit":
            mean_vals = compute_mean_vals(model, tokenizer, sents, site=site,
                                          layers=sorted(mask.selected), **kw)
        else:
            fill_value = global_activation_mean(model, tokenizer, sents, site=site,
                                                layers=sorted(mask.selected), **kw)

    with ablated(model, mask, mode=mode, mean_vals=mean_vals, fill_value=fill_value):
        acc_lang = minimal_pair_accuracy(model, tokenizer, second, **kw)

    rand_scores = []
    for s in seeds:
        rmask = random_mask_from_remaining(mask, seed=s)
        rmv = rfv = None
        if mode == "mean":
            rmv, rfv = mean_vals, fill_value
        with ablated(model, rmask, mode=mode, mean_vals=rmv, fill_value=rfv):
            rand_scores.append(minimal_pair_accuracy(model, tokenizer, second, **kw))
    acc_rand = float(np.mean(rand_scores)) if rand_scores else float("nan")

    return {
        "phenomenon": pair_set.phenomenon,
        "category": pair_set.category,
        "n_units": mask.count(),
        "acc_none": acc_none,
        "acc_language": acc_lang,
        "acc_random": acc_rand,
        "delta_language": acc_lang - acc_none,
        "delta_random": acc_rand - acc_none,
        "selectivity_gap": acc_rand - acc_lang,   # >0 ⇒ localized units matter more
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Iterable[str]] = None) -> None:
    from analyze.convergence import _load_checkpoint, _free
    from analyze._interp_helpers import device_for, write_csv_atomic
    from .localize import localize, SELECTIVITY_METHODS, SITES
    from . import stimuli as _stim

    p = argparse.ArgumentParser(description="Ablate localized units and score BLiMP.")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--categories", nargs="*", default=list(_stim.AGREEMENT_CATEGORIES))
    p.add_argument("--max-pairs", type=int, default=200)
    p.add_argument("--top-frac", type=float, default=0.01)
    p.add_argument("--method", default="ttest", choices=SELECTIVITY_METHODS)
    p.add_argument("--site", default="resid", choices=SITES)
    p.add_argument("--mode", default="zero", choices=["zero", "mean"])
    p.add_argument("--mean-scope", default="global", choices=["global", "per_unit"])
    p.add_argument("--grid", action="store_true",
                   help="Cross every mask against every eval set (generalization "
                        "test) instead of the faithful per-phenomenon random-control run.")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--out", default="results/units/ablation.csv")
    args = p.parse_args(list(argv) if argv is not None else None)

    sets = (_stim.demo_pairs() if args.demo else
            _stim.load_blimp(categories=args.categories, max_pairs=args.max_pairs))

    loaded = _load_checkpoint(args.repo, args.revision, device=device_for())
    if loaded is None:
        raise SystemExit(f"[units.ablate] failed to load {args.repo}@{args.revision}")
    model, tok = loaded

    if args.grid:
        masks = {s.phenomenon: localize(model, tok, s, top_frac=args.top_frac,
                                        method=args.method, site=args.site)
                 for s in sets}
        df = ablation_grid(model, tok, masks, sets, mode=args.mode,
                           mean_scope=args.mean_scope)
    else:
        import pandas as pd
        df = pd.DataFrame([
            run_ablation(model, tok, s, top_frac=args.top_frac, method=args.method,
                         site=args.site, mode=args.mode, mean_scope=args.mean_scope)
            for s in sets
        ])
    _free(model, tok)

    write_csv_atomic(df, args.out)
    print(df.to_string(index=False))
    print(f"[units.ablate] wrote {len(df)} rows → {args.out}")


if __name__ == "__main__":
    main()
