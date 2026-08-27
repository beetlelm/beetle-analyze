"""
run.py — Master runner for the beetle-analyze pipeline.

Usage examples
--------------
# Full HumanScale pipeline on 4 languages (primary paper target):
python analyze/run.py --all --family humanscale --langs deu nld zho eng

# FineWeb-only (after scripts/sync_fineweb_manifest.py):
python analyze/run.py --all --family fineweb --langs deu nld zho eng

# Just the 9 ACL main-body figures from existing CSVs:
python analyze/run.py --paper-figures --family all --langs deu nld zho eng

# PPL + MECO on a single family, minimal ablation sweep (seeded):
python analyze/run.py --ppl --meco --family ablations --langs nld eng

# Comparative figures (HumanScale vs FineWeb):
python analyze/run.py --figures --compare family-vs-family \
    --langs deu nld zho eng

Key concepts
------------
--family : which repo family to iterate. One of
    humanscale | fineweb | ablations | beetlelm_legacy | all | <path-to-file>
--condition : filter by B-code (B1..B5, mono, all). Repeatable.
--variant   : filter by ablation variant (B1b, B2_ewc, ...). Repeatable.
--compare   : comparison mode for figures
    (none | pairs | family-vs-family)
--cache-mode : HF cache lifecycle (keep | rotate | clear). Rotate recommended
    on GPU clusters; it trims old revisions between models.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(stage: str, output_root: Path) -> None:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{stage}_{time.strftime('%Y-%m-%d')}.log"
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    # Reset any prior handlers so successive _setup_logging calls don't pile up.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a"),
        ],
    )
    logging.getLogger().info("=" * 60)
    logging.getLogger().info("STAGE: %s", stage)
    logging.getLogger().info("LOG FILE: %s", log_file)


# ---------------------------------------------------------------------------
# Family / condition / variant resolution
# ---------------------------------------------------------------------------

def _resolve_repos(args) -> list:
    """Return the filtered list of model repos for this run."""
    from models import MODEL_GROUPS
    from analyze.model_naming import parse_repo

    # Step 1: family selection.
    fam = args.family or "all"
    if fam == "humanscale":
        pool = list(MODEL_GROUPS["humanscale"])
    elif fam == "fineweb":
        pool = list(MODEL_GROUPS["fineweb_planned"])
    elif fam == "ablations":
        pool = list(MODEL_GROUPS["ablation_seeded"])
    elif fam == "beetlelm_legacy":
        pool = [m for m in MODEL_GROUPS.get("nld", [])
                if parse_repo(m).family == "beetlelm_legacy"]
        # Union over all legacy language groups
        for k in ("nld", "deu", "zho", "fra", "fas", "bul", "eng"):
            for m in MODEL_GROUPS.get(k, []):
                if parse_repo(m).family == "beetlelm_legacy" and m not in pool:
                    pool.append(m)
    elif fam == "all":
        pool = (list(MODEL_GROUPS["humanscale"])
                + list(MODEL_GROUPS["fineweb_planned"])
                + list(MODEL_GROUPS["ablation_seeded"]))
        # Also include legacy BeetleLM models.
        for k in ("nld", "deu", "zho", "fra", "fas", "bul", "eng"):
            for m in MODEL_GROUPS.get(k, []):
                if m not in pool:
                    pool.append(m)
    elif fam and Path(fam).is_file():
        pool = [ln.strip() for ln in Path(fam).read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
    else:
        raise SystemExit(f"Unknown --family: {fam!r}")

    # Step 2: condition / variant filters.
    if args.condition:
        cset = set(args.condition)
        pool = [m for m in pool if parse_repo(m).condition in cset]
    if args.variant:
        vset = set(args.variant)
        pool = [m for m in pool if parse_repo(m).variant in vset]

    # Step 3: language filter — STRICT SUBSET. A model passes only when its
    # language set is a subset of ``--langs``. This is the behaviour the
    # paper scripts assume: ``--langs nld eng`` returns only Dutch-English
    # bilinguals and the monolinguals of those languages, not cross-pair
    # models such as ``beetlelm_fas-nld`` (which used to leak through under
    # the earlier OR-based filter).
    if args.langs:
        langs = set(args.langs)

        def _in_langs(r: str) -> bool:
            m = parse_repo(r)
            model_langs = set()
            if m.l1:
                model_langs.add(m.l1)
            if m.l2:
                model_langs.add(m.l2)
            return bool(model_langs) and model_langs <= langs
        pool = [m for m in pool if _in_langs(m)]

    # Step 4: HF-org filter. When ``--orgs`` is set, restrict to repos whose
    # org prefix is in that list. This is the main mechanism for excluding
    # ``BeetleLM/*`` legacy repos from Beetle-HumanScale / Beetle-FineWeb
    # sweeps without modifying ``models.py``.
    orgs = getattr(args, "orgs", None)
    if orgs:
        orgs_set = set(orgs)
        pool = [m for m in pool if m.split("/", 1)[0] in orgs_set]

    return list(dict.fromkeys(pool))  # dedup, keep order


# ---------------------------------------------------------------------------
# Submodule dispatchers (lazy imports)
# ---------------------------------------------------------------------------

def _stage_cleanup(stage: str, args, dump_failures=()) -> None:
    """Stage-boundary cleanup: collect garbage, clear CUDA cache, dump
    diagnostic failure registries to disk so they don't accumulate in process.

    ``dump_failures`` is an iterable of ``(name, dict)`` pairs; each is
    written to ``results/.failures/<stage>_rank<rank>_<name>.json``.
    """
    import gc
    import json
    log = logging.getLogger(stage)
    try:
        gc.collect()
    except Exception:
        log.exception("gc.collect failed during %s cleanup", stage)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        log.exception("torch.cuda.empty_cache failed during %s cleanup", stage)
    if dump_failures:
        try:
            out_dir = Path(args.output_root) / ".failures"
            out_dir.mkdir(parents=True, exist_ok=True)
            for name, payload in dump_failures:
                if not payload:
                    continue
                fp = out_dir / f"{stage}_rank{args.rank}_{name}.json"
                fp.write_text(json.dumps(dict(payload), indent=2, default=str))
        except Exception:
            log.exception("failure-dump persist failed during %s cleanup", stage)


def _run_ppl(args):
    _setup_logging("ppl", args.output_root)
    from ppl_eval import run_all_languages
    from ppl_utils import LOAD_FAILURES, clear_failures as _clear_ppl_failures
    log = logging.getLogger("ppl")
    log.info("PPL EVALUATION — family=%s orgs=%s langs=%s (rank=%d/%d)",
             args.family, getattr(args, "orgs", None),
             args.langs, args.rank, args.world_size)

    # Route PPL through the same _resolve_repos() used by every other stage
    # so --family / --orgs / strict --langs all take effect. Previously
    # run_all_languages() iterated MODEL_GROUPS[lang] directly, which
    # silently merged BeetleLM/* legacy repos into the pool.
    repos = _resolve_repos(args)
    log.info("PPL pool after filters: %d repos", len(repos))
    if not repos:
        log.warning("PPL pool is empty after filters — skipping.")
        return

    _clear_ppl_failures()
    try:
        run_all_languages(
            lang_codes=args.langs,
            overwrite=args.overwrite,
            rank=args.rank,
            world_size=args.world_size,
            output_root=args.output_root,
            repos=repos,
            all_checkpoints=getattr(args, "all_checkpoints", False),
        )
    except Exception as exc:  # pragma: no cover
        log.exception("PPL stage failed: %s", exc)
    finally:
        _stage_cleanup("ppl", args, dump_failures=[("load_failures", LOAD_FAILURES)])


def _run_forgetting(args):
    # Writes per-language aggregated CSVs (forgetting_raw_<iso>.csv) that
    # require all bilingual candidates; not shardable. Run on rank 0 only.
    if args.rank != 0:
        logging.getLogger("forgetting").info(
            "rank=%d — skipping forgetting (rank-0-only stage)", args.rank)
        return
    _setup_logging("forgetting", args.output_root)
    from forgetting import run_forgetting_analysis
    log = logging.getLogger("forgetting")
    langs = args.langs or ["deu", "nld", "zho"]
    log.info("CATASTROPHIC FORGETTING — langs=%s", langs)
    run_forgetting_analysis(lang_codes=langs, overwrite=args.overwrite)


def _run_embeddings(args):
    # compute_cka_matrix needs the full model set for a pairwise matrix
    # per language; not shardable. Run on rank 0 only.
    if args.rank != 0:
        logging.getLogger("embeddings").info(
            "rank=%d — skipping embeddings (rank-0-only stage)", args.rank)
        return
    _setup_logging("embeddings", args.output_root)
    from embedding_drift import run_embedding_analysis
    log = logging.getLogger("embeddings")
    log.info("EMBEDDING DRIFT — langs=%s", args.langs)
    run_embedding_analysis(lang_codes=args.langs, overwrite=args.overwrite)


def _run_convergence(args):
    _setup_logging("convergence", args.output_root)
    from convergence import (
        run_convergence_analysis,
        FOCUS_REPOS,
        CONVERGENCE_FAILURES,
        clear_failures as _clear_conv_failures,
    )
    from _rank_barrier import shard

    log = logging.getLogger("convergence")
    log.info("CONVERGENCE / CHECKPOINT DRIFT (rank=%d/%d)",
             args.rank, args.world_size)
    repos = _resolve_repos(args) or FOCUS_REPOS["forgetting_focus"]
    my_repos = shard(repos, args.rank, args.world_size)
    log.info("Total %d repos; this rank owns %d: %s",
             len(repos), len(my_repos), my_repos)

    _clear_conv_failures()
    try:
        run_convergence_analysis(
            my_repos,
            hf_token=getattr(args, "hf_token", None),
            overwrite=args.overwrite,
            signals=["ppl", "drift", "cka"],
            rank=args.rank,
            world_size=args.world_size,
            all_repos=repos,
            output_root=args.output_root,
            all_checkpoints=bool(getattr(args, "all_checkpoints", False)),
        )
    except Exception as exc:  # pragma: no cover
        log.exception("Convergence stage failed: %s", exc)
    finally:
        _stage_cleanup(
            "convergence", args,
            dump_failures=[("convergence_failures", CONVERGENCE_FAILURES)],
        )


def _run_meco(args):
    _setup_logging("meco", args.output_root)
    from meco_surprisal import run_meco_eval, DEFAULT_STIMULI, RESULTS_DIR
    from _rank_barrier import shard
    log = logging.getLogger("meco")
    log.info("MECO EYE-TRACKING EVALUATION (rank=%d/%d)",
             args.rank, args.world_size)

    models = getattr(args, "meco_models", None) or _resolve_repos(args)
    if not models:
        log.warning("No models matched the selection; skipping MECO")
        return

    # Sharding policy:
    #   * When the pool is >= world_size, shard models across ranks.
    #   * Otherwise every rank processes every model and the sentence loop
    #     inside score_checkpoint does the within-stage sharding.
    all_checkpoints_flag = (
        getattr(args, "meco_checkpoints", False)
        or getattr(args, "all_checkpoints", False)
    )
    try:
        if args.world_size <= 1 or len(models) >= args.world_size or all_checkpoints_flag:
            my_models = shard(models, args.rank, args.world_size)
            log.info("Total %d models; this rank owns %d (model-level sharding)",
                     len(models), len(my_models))
            run_meco_eval(
                models=my_models,
                stimuli_path=str(DEFAULT_STIMULI),
                output_dir=str(RESULTS_DIR),
                all_checkpoints=all_checkpoints_flag,
                rank=args.rank,
                world_size=1,
                all_models=models,
                output_root=args.output_root,
            )
        else:
            log.info("Total %d models < world_size=%d; every rank processes all "
                     "models with within-stage stimulus sharding.",
                     len(models), args.world_size)
            run_meco_eval(
                models=models,
                stimuli_path=str(DEFAULT_STIMULI),
                output_dir=str(RESULTS_DIR),
                all_checkpoints=all_checkpoints_flag,
                rank=args.rank,
                world_size=args.world_size,
                all_models=models,
                output_root=args.output_root,
            )
    except Exception as exc:  # pragma: no cover
        log.exception("MeCo stage failed: %s", exc)
    finally:
        _stage_cleanup("meco", args)


def _run_meco_sweep(args):
    _setup_logging("meco_sweep", args.output_root)
    from meco_surprisal import score_sweep_results, DEFAULT_STIMULI, RESULTS_DIR
    log = logging.getLogger("meco_sweep")

    org = args.sweep_org or "Beetle-HumanScale"
    log.info("MECO SWEEP — org=%s", org)
    score_sweep_results(
        hf_org=org,
        stimuli_path=str(DEFAULT_STIMULI),
        output_dir=str(RESULTS_DIR / "sweep"),
    )


def _run_aoa(args):
    _setup_logging("aoa", args.output_root)
    log = logging.getLogger("aoa")
    log.info("AOA / LEARNING CURVES (Chang & Bergen 2024) — family=%s orgs=%s langs=%s (rank=%d/%d)",
             args.family, getattr(args, "orgs", None), args.langs,
             args.rank, args.world_size)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from aoa.run_learning_curves import run_pipeline

    # Honour --family / --orgs / --langs the same way every other stage does.
    # --aoa-model (if provided) overrides the family-derived pool.
    explicit_models = getattr(args, "aoa_model", None)
    if explicit_models:
        # argparse gives a list (nargs="+"); accept a bare str defensively too.
        repos = [explicit_models] if isinstance(explicit_models, str) else list(explicit_models)
    else:
        repos = _resolve_repos(args)
    log.info("AoA pool after filters: %d repos", len(repos))
    if not repos:
        log.warning("AoA pool is empty after filters — skipping.")
        return

    run_pipeline(
        langs=args.langs or ["all"],
        output_dir=Path("results/aoa"),
        models=repos,
        resume=getattr(args, "resume", False),
        batch_size=getattr(args, "batch_size", 32),
        all_checkpoints=bool(getattr(args, "all_checkpoints", False)),
        debug=False,
        rank=args.rank,
        world_size=args.world_size,
    )


def _run_aoa_ficarra(args):
    """Ficarra-style signature + convergence AoA (rank-0 only post-processor).

    Consumes the per-token surprisals produced by ``_run_aoa`` and writes
    signature trajectories, ϵ-convergence AoA, Figure 1/2 and Table A6
    into ``results/aoa/ficarra/…`` + ``results/figures/ficarra/…``.
    """
    if args.rank != 0:
        logging.getLogger("aoa_ficarra").info(
            "rank=%d — skipping aoa-ficarra (rank-0-only stage)", args.rank)
        return
    _setup_logging("aoa_ficarra", args.output_root)
    log = logging.getLogger("aoa_ficarra")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from aoa.ficarra.pipeline import (
        run_ficarra_pipeline,
        DEFAULT_EPSILON,
        FULL_EPSILON_SWEEP,
    )

    explicit_models = getattr(args, "aoa_model", None)
    if explicit_models:
        repos = [explicit_models] if isinstance(explicit_models, str) else list(explicit_models)
    else:
        repos = _resolve_repos(args)
    log.info(
        "AOA-FICARRA — family=%s orgs=%s langs=%s regime=%s repos=%d "
        "all_signatures=%s full_sweep=%s heatmap=%s",
        args.family, getattr(args, "orgs", None), args.langs,
        getattr(args, "ficarra_regime", "humanscale"),
        len(repos),
        bool(getattr(args, "ficarra_all_signatures", False)),
        bool(getattr(args, "ficarra_full_sweep", False)),
        bool(getattr(args, "ficarra_heatmap", False)),
    )
    if not repos:
        log.warning("AoA-Ficarra pool is empty after filters — skipping.")
        return

    epsilons = (list(FULL_EPSILON_SWEEP)
                if getattr(args, "ficarra_full_sweep", False)
                else [DEFAULT_EPSILON])

    # _run_aoa hardcodes Path("results/aoa"); match that so Ficarra finds
    # the surprisals written by the --aoa stage regardless of --output-root.
    run_ficarra_pipeline(
        langs=args.langs or ["deu", "nld", "zho", "eng"],
        output_dir=Path("results/aoa"),
        models=repos,
        regime=getattr(args, "ficarra_regime", "humanscale"),
        epsilons=epsilons,
        all_signatures=bool(getattr(args, "ficarra_all_signatures", False)),
        rank=args.rank,
        world_size=args.world_size,
        overwrite=getattr(args, "overwrite", False),
        render_figures=True,
        render_heatmap=bool(getattr(args, "ficarra_heatmap", False)),
    )


def _run_cognates(args):
    if args.rank != 0:
        return
    _setup_logging("cognates", args.output_root)
    from cognate_surprisal import run_cognate_analysis
    logging.getLogger("cognates").info("COGNATE & FALSE-FRIEND (A1, A4, A5, E1)")
    run_cognate_analysis(overwrite=args.overwrite)


def _run_meco_extended(args):
    if args.rank != 0:
        return
    _setup_logging("meco_extended", args.output_root)
    from meco_extended import run_meco_extended
    logging.getLogger("meco_extended").info("MECO EXTENDED (A2)")
    run_meco_extended(overwrite=args.overwrite)


def _run_cli(args):
    if args.rank != 0:
        return
    _setup_logging("cli", args.output_root)
    from cli_eval import run_cli_evaluation
    logging.getLogger("cli").info("CROSS-LINGUAL INTERFERENCE (B1–B5)")
    run_cli_evaluation(overwrite=args.overwrite)


def _run_representations(args):
    if args.rank != 0:
        return
    _setup_logging("representations", args.output_root)
    from representation_analysis import run_representation_analysis
    logging.getLogger("representations").info("REPRESENTATION ANALYSIS (C1–C5)")
    run_representation_analysis(
        overwrite=args.overwrite,
        sae_layer=getattr(args, "sae_layer", None),
        sae_root=args.output_root,
    )


def _run_sae(args):
    """Train SAE + extract Dutch/English features for selected (repo, revision) pairs.

    Rank-shards across repos × revisions using the same _resolve_repos pool
    that other stages use. Each rank trains its own slice; subsequent
    `--representations` reads the .npy files via the new auto-discovery.
    """
    _setup_logging("sae", args.output_root)
    log = logging.getLogger("sae")

    from analyze.sae.train_sae import train_sae_for_checkpoint
    from analyze.sae.extract_features import extract_features_for_checkpoint
    from _rank_barrier import shard

    repos = _resolve_repos(args)
    if not repos:
        log.warning("SAE pool empty after filters — skipping.")
        return

    revisions = list(getattr(args, "sae_revisions", None) or ["main"])
    pairs = [(r, rev) for r in repos for rev in revisions]
    my_pairs = shard(pairs, args.rank, args.world_size)
    log.info("SAE: %d (repo, revision) pairs total; rank %d/%d owns %d",
             len(pairs), args.rank, args.world_size, len(my_pairs))

    do_train = bool(getattr(args, "sae", False) or getattr(args, "sae_train", False))
    do_extract = bool(getattr(args, "sae", False) or getattr(args, "sae_extract", False))

    for repo, rev in my_pairs:
        log.info("SAE [%s @ %s] layer=%d", repo, rev, args.sae_layer)
        if do_train:
            if not getattr(args, "sae_dataset", None):
                log.error("--sae-dataset is required for SAE training; skipping %s@%s",
                          repo, rev)
                continue
            train_sae_for_checkpoint(
                repo=repo,
                revision=rev,
                layer=args.sae_layer,
                num_latents=args.sae_num_latents,
                k=args.sae_k,
                lr=getattr(args, "sae_lr", 1e-3),
                tokens=args.sae_tokens,
                dataset_repo=args.sae_dataset,
                dataset_name=getattr(args, "sae_dataset_name", None),
                pretokenized=bool(getattr(args, "sae_pretokenized", True)),
                ctx_len=getattr(args, "sae_ctx_len", 512),
                output_root=args.output_root,
                overwrite=args.overwrite,
            )
        if do_extract:
            extract_features_for_checkpoint(
                repo=repo,
                revision=rev,
                layer=args.sae_layer,
                langs=tuple(args.langs) if args.langs else ("nld", "eng"),
                n_tokens_per_lang=getattr(args, "sae_extract_tokens", 200_000),
                num_latents=args.sae_num_latents,
                k=args.sae_k,
                lr=getattr(args, "sae_lr", 1e-3),
                output_root=args.output_root,
            )


# ---------------------------------------------------------------------------
# Cross-lingual mechanistic stack — xling/*
# ---------------------------------------------------------------------------
#
# Default layer policy: when a repo is registered in ``xling.registry``
# we pick early / mid / late thirds; otherwise we let the activation
# buffer auto-discover all transformer blocks.
#
# Default language pair: the first two ``--langs`` codes (NLD-ENG by
# default for the Beetle bilingual demo).

def _xling_default_layers(n_layers: int) -> list:
    if n_layers <= 3:
        return list(range(n_layers))
    return [n_layers // 4, n_layers // 2, (3 * n_layers) // 4]


def _xling_pair(args) -> tuple:
    langs = args.langs or ["nld", "eng"]
    if len(langs) < 2:
        raise SystemExit("xling stages require at least two --langs")
    return langs[0], langs[1]


def _xling_layers_for(repo: str) -> list:
    """Return the list of layer indices to analyse for ``repo``.

    Falls back to a 5-layer default ([0, 2, 4, 6, 8]) when the repo is
    not in the xling registry; the activation buffer will raise IndexError
    if any of these are out of range and the caller can shrink.
    """
    try:
        from xling import registry
        cfg = registry.get(repo)
        return _xling_default_layers(cfg.n_layers)
    except Exception:
        return [0, 2, 4, 6, 8]


def _flores_code_for(iso: str) -> str:
    """Map ISO-639-3 to FLORES-200 script-tagged code.

    Looks up via xling.registry first; falls back to ``{iso}_Latn`` for
    Latin-script languages we know about.
    """
    try:
        from xling import registry
        for repo in registry.list_repos():
            for lc in registry.get(repo).languages:
                if lc.iso == iso:
                    return lc.flores_code
    except Exception:
        pass
    _DEFAULT = {
        "nld": "nld_Latn", "eng": "eng_Latn", "deu": "deu_Latn",
        "fra": "fra_Latn", "zho": "zho_Hans", "fas": "pes_Arab",
        "bul": "bul_Cyrl",
    }
    return _DEFAULT.get(iso, f"{iso}_Latn")


def _xling_extract(
    *, repo, revision, texts, dataset_id, layers, output_root, device, pooling="mean",
):
    from xling.activation_buffer import BufferConfig, extract_or_load
    cfg = BufferConfig(
        cache_root=Path(output_root) / "xling" / "_cache",
        layers=tuple(layers),
        pooling=pooling,
        max_seq_len=256,
        batch_size=8,
        device=device,
    )
    return extract_or_load(
        repo=repo, revision=revision, dataset_id=dataset_id,
        texts=texts, cfg=cfg,
    )


def _xling_write_csv(path: Path, rows: list) -> None:
    import csv
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _run_svcca(args):
    _setup_logging("svcca", args.output_root)
    log = logging.getLogger("svcca")
    log.info("XLING SVCCA (rank=%d/%d)", args.rank, args.world_size)

    import torch
    from convergence import list_checkpoints
    from xling import svcca
    from xling.datasets import load_flores_pairs
    from _rank_barrier import shard

    repos = _resolve_repos(args)
    my_repos = shard(repos, args.rank, args.world_size)
    if not my_repos:
        log.warning("svcca: empty pool — skipping")
        return

    src_lang, tgt_lang = _xling_pair(args)
    method = getattr(args, "svcca_method", "svcca")
    n_pairs = int(getattr(args, "xling_max_pairs", 256))

    try:
        pairs = load_flores_pairs(
            _flores_code_for(src_lang), _flores_code_for(tgt_lang),
            split="dev", max_pairs=n_pairs,
        )
    except Exception as e:
        log.exception("svcca: FLORES load failed: %s — skipping stage", e)
        _stage_cleanup("svcca", args)
        return

    src = [p.src_text for p in pairs]
    tgt = [p.tgt_text for p in pairs]
    out_root = Path(args.output_root) / "xling" / "svcca"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        for repo in my_repos:
            layers = _xling_layers_for(repo)
            revisions = list_checkpoints(
                repo, hf_token=getattr(args, "hf_token", None),
                all_checkpoints=bool(getattr(args, "all_checkpoints", False)),
            )
            rows = []
            for rev in revisions:
                try:
                    acts_a = _xling_extract(
                        repo=repo, revision=rev, texts=src,
                        dataset_id=f"flores200-{src_lang}", layers=layers,
                        output_root=args.output_root, device=device,
                    )
                    acts_b = _xling_extract(
                        repo=repo, revision=rev, texts=tgt,
                        dataset_id=f"flores200-{tgt_lang}", layers=layers,
                        output_root=args.output_root, device=device,
                    )
                    scores = svcca.layerwise_trajectory(
                        acts_a, acts_b, method=method,
                    )
                    for layer_idx, score in scores.items():
                        rows.append({
                            "repo": repo, "revision": rev,
                            "layer": layer_idx, "method": method,
                            "score": float(score),
                        })
                except Exception as e:
                    log.exception("svcca: %s @ %s failed: %s", repo, rev, e)
            _xling_write_csv(
                out_root / f"{repo.replace('/', '__')}.csv", rows,
            )
            log.info("svcca: %s — %d rows", repo, len(rows))
    finally:
        _stage_cleanup("svcca", args)


def _run_cka_full(args):
    _setup_logging("cka_full", args.output_root)
    log = logging.getLogger("cka_full")
    log.info("XLING FULL-ACTIVATION CKA (rank=%d/%d)", args.rank, args.world_size)

    import torch
    from convergence import list_checkpoints
    from xling import cka_full
    from xling.datasets import load_flores_pairs
    from _rank_barrier import shard

    repos = _resolve_repos(args)
    my_repos = shard(repos, args.rank, args.world_size)
    if not my_repos:
        log.warning("cka_full: empty pool — skipping")
        return

    src_lang, tgt_lang = _xling_pair(args)
    n_pairs = int(getattr(args, "xling_max_pairs", 256))
    kernel = getattr(args, "cka_kernel", "linear")
    unbiased = bool(getattr(args, "cka_unbiased", False))

    try:
        pairs = load_flores_pairs(
            _flores_code_for(src_lang), _flores_code_for(tgt_lang),
            split="dev", max_pairs=n_pairs,
        )
    except Exception as e:
        log.exception("cka_full: FLORES load failed: %s — skipping", e)
        _stage_cleanup("cka_full", args)
        return

    src = [p.src_text for p in pairs]
    tgt = [p.tgt_text for p in pairs]
    out_root = Path(args.output_root) / "xling" / "cka_full"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        for repo in my_repos:
            layers = _xling_layers_for(repo)
            revisions = list_checkpoints(
                repo, hf_token=getattr(args, "hf_token", None),
                all_checkpoints=bool(getattr(args, "all_checkpoints", False)),
            )
            rows = []
            for rev in revisions:
                try:
                    acts_a = _xling_extract(
                        repo=repo, revision=rev, texts=src,
                        dataset_id=f"flores200-{src_lang}", layers=layers,
                        output_root=args.output_root, device=device,
                    )
                    acts_b = _xling_extract(
                        repo=repo, revision=rev, texts=tgt,
                        dataset_id=f"flores200-{tgt_lang}", layers=layers,
                        output_root=args.output_root, device=device,
                    )
                    common = sorted(set(acts_a) & set(acts_b))
                    for layer_idx in common:
                        if unbiased:
                            # Split along sample dim into ~64-sample chunks
                            # for the unbiased minibatch estimator.
                            import numpy as np
                            xa = acts_a[layer_idx]
                            xb = acts_b[layer_idx]
                            xa = xa.numpy() if hasattr(xa, "numpy") else np.asarray(xa)
                            xb = xb.numpy() if hasattr(xb, "numpy") else np.asarray(xb)
                            n = xa.shape[0]
                            chunk = max(4, min(64, n // 4 if n >= 16 else n))
                            batches_a = [xa[i:i+chunk] for i in range(0, n, chunk)
                                         if xa[i:i+chunk].shape[0] >= 4]
                            batches_b = [xb[i:i+chunk] for i in range(0, n, chunk)
                                         if xb[i:i+chunk].shape[0] >= 4]
                            score = cka_full.cka_minibatch_unbiased(
                                batches_a, batches_b, kernel=kernel,
                            )
                        else:
                            score = cka_full.cka_score(
                                acts_a[layer_idx], acts_b[layer_idx],
                                kernel=kernel,
                            )
                        rows.append({
                            "repo": repo, "revision": rev,
                            "layer": layer_idx, "kernel": kernel,
                            "unbiased": unbiased, "score": float(score),
                        })
                except Exception as e:
                    log.exception("cka_full: %s @ %s failed: %s", repo, rev, e)
            _xling_write_csv(
                out_root / f"{repo.replace('/', '__')}.csv", rows,
            )
            log.info("cka_full: %s — %d rows", repo, len(rows))
    finally:
        _stage_cleanup("cka_full", args)


def _run_crossencoder(args):
    _setup_logging("crossencoder", args.output_root)
    log = logging.getLogger("crossencoder")
    log.info("XLING CROSSENCODER PROBE (rank=%d/%d)", args.rank, args.world_size)

    import torch
    from convergence import list_checkpoints
    from xling import crossencoder
    from xling.datasets import load_flores_pairs, iter_negatives
    from _rank_barrier import shard

    repos = _resolve_repos(args)
    my_repos = shard(repos, args.rank, args.world_size)
    if not my_repos:
        log.warning("crossencoder: empty pool — skipping")
        return

    src_lang, tgt_lang = _xling_pair(args)
    n_pairs = int(getattr(args, "xling_max_pairs", 512))

    try:
        pairs = load_flores_pairs(
            _flores_code_for(src_lang), _flores_code_for(tgt_lang),
            split="dev", max_pairs=n_pairs,
        )
    except Exception as e:
        log.exception("crossencoder: FLORES load failed: %s — skipping", e)
        _stage_cleanup("crossencoder", args)
        return

    out_root = Path(args.output_root) / "xling" / "crossencoder"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        for repo in my_repos:
            layers = _xling_layers_for(repo)
            revisions = list_checkpoints(
                repo, hf_token=getattr(args, "hf_token", None),
                all_checkpoints=bool(getattr(args, "all_checkpoints", False)),
            )
            rows = []
            for rev in revisions:
                try:
                    src_texts = [p.src_text for p in pairs]
                    tgt_texts = [p.tgt_text for p in pairs]
                    # iter_negatives consumes ParallelPair[]; extract shifted
                    # texts for the contrastive baseline.
                    neg_pairs_tgt = list(iter_negatives(pairs, shift=1))
                    neg_pairs_src = list(iter_negatives(pairs, shift=2))
                    neg_src = [p.src_text for p in neg_pairs_src]
                    neg_tgt = [p.tgt_text for p in neg_pairs_tgt]
                    acts_src = _xling_extract(
                        repo=repo, revision=rev, texts=src_texts,
                        dataset_id=f"flores200-{src_lang}", layers=layers,
                        output_root=args.output_root, device=device,
                    )
                    acts_tgt = _xling_extract(
                        repo=repo, revision=rev, texts=tgt_texts,
                        dataset_id=f"flores200-{tgt_lang}", layers=layers,
                        output_root=args.output_root, device=device,
                    )
                    acts_src_neg = _xling_extract(
                        repo=repo, revision=rev, texts=neg_src,
                        dataset_id=f"flores200-{src_lang}-neg", layers=layers,
                        output_root=args.output_root, device=device,
                    )
                    acts_tgt_neg = _xling_extract(
                        repo=repo, revision=rev, texts=neg_tgt,
                        dataset_id=f"flores200-{tgt_lang}-neg", layers=layers,
                        output_root=args.output_root, device=device,
                    )
                    scores_by_layer = crossencoder.layerwise_alignment(
                        src_acts_by_layer=acts_src,
                        tgt_acts_by_layer=acts_tgt,
                        src_neg_acts_by_layer=acts_src_neg,
                        tgt_neg_acts_by_layer=acts_tgt_neg,
                        device=device,
                    )
                    for layer_idx, metrics in scores_by_layer.items():
                        for k, v in metrics.items():
                            rows.append({
                                "repo": repo, "revision": rev,
                                "layer": layer_idx, "metric": k,
                                "score": float(v),
                            })
                except Exception as e:
                    log.exception("crossencoder: %s @ %s failed: %s", repo, rev, e)
            _xling_write_csv(
                out_root / f"{repo.replace('/', '__')}.csv", rows,
            )
            log.info("crossencoder: %s — %d rows", repo, len(rows))
    finally:
        _stage_cleanup("crossencoder", args)


def _run_probes(args):
    _setup_logging("probes", args.output_root)
    log = logging.getLogger("probes")
    log.info("XLING LINEAR PROBES (rank=%d/%d)", args.rank, args.world_size)

    import torch
    from convergence import list_checkpoints
    from xling import probes
    from xling.datasets import load_ud_pos
    from _rank_barrier import shard

    repos = _resolve_repos(args)
    my_repos = shard(repos, args.rank, args.world_size)
    if not my_repos:
        log.warning("probes: empty pool — skipping")
        return

    src_lang, _ = _xling_pair(args)
    out_root = Path(args.output_root) / "xling" / "probes"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        try:
            tagged = load_ud_pos(
                src_lang, split="validation", max_sentences=1024,
            )
        except Exception as e:
            log.exception("probes: UD load failed: %s — skipping", e)
            return
        if not tagged:
            log.warning("probes: no UD data for %s — skipping", src_lang)
            return

        # Sentence-level evaluation: join tokens, predict the predominant
        # POS tag of the first content word as a coarse signal.
        texts = [" ".join(t.tokens) for t in tagged]
        for repo in my_repos:
            layers = _xling_layers_for(repo)
            revisions = list_checkpoints(
                repo, hf_token=getattr(args, "hf_token", None),
                all_checkpoints=bool(getattr(args, "all_checkpoints", False)),
            )
            rows = []
            for rev in revisions:
                try:
                    acts = _xling_extract(
                        repo=repo, revision=rev, texts=texts,
                        dataset_id=f"ud-pos-{src_lang}", layers=layers,
                        output_root=args.output_root, device=device,
                        pooling="mean",
                    )
                    # Sentence-level POS — predominant tag per sentence.
                    import numpy as np
                    y = np.array(
                        [t.tags[0] if t.tags else "X" for t in tagged]
                    )
                    for layer_idx, X in acts.items():
                        Xn = X.numpy() if hasattr(X, "numpy") else np.asarray(X)
                        # 80/20 train/test split for the probe.
                        n = Xn.shape[0]
                        cut = max(1, int(0.8 * n))
                        gap = probes.control_gap(
                            Xn[:cut], y[:cut], Xn[cut:], y[cut:],
                        )
                        rows.append({
                            "repo": repo, "revision": rev,
                            "layer": layer_idx,
                            "accuracy": float(gap["accuracy"]),
                            "control_accuracy": float(gap["control_accuracy"]),
                            "gap": float(gap["gap"]),
                        })
                except Exception as e:
                    log.exception("probes: %s @ %s failed: %s", repo, rev, e)
            _xling_write_csv(
                out_root / f"{repo.replace('/', '__')}.csv", rows,
            )
            log.info("probes: %s — %d rows", repo, len(rows))
    finally:
        _stage_cleanup("probes", args)


def _run_tcav(args):
    _setup_logging("tcav", args.output_root)
    log = logging.getLogger("tcav")
    log.info("XLING TCAV (rank=%d/%d)", args.rank, args.world_size)

    import torch
    import numpy as np
    from convergence import list_checkpoints
    from xling import tcav
    from xling.datasets import load_flores_pairs
    from _rank_barrier import shard

    repos = _resolve_repos(args)
    my_repos = shard(repos, args.rank, args.world_size)
    if not my_repos:
        log.warning("tcav: empty pool — skipping")
        return

    src_lang, tgt_lang = _xling_pair(args)
    n_pairs = int(getattr(args, "xling_max_pairs", 256))
    try:
        pairs = load_flores_pairs(
            _flores_code_for(src_lang), _flores_code_for(tgt_lang),
            split="dev", max_pairs=n_pairs,
        )
    except Exception as e:
        log.exception("tcav: FLORES load failed: %s — skipping", e)
        _stage_cleanup("tcav", args)
        return

    out_root = Path(args.output_root) / "xling" / "tcav"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Build a binary "language" concept: src=1, tgt=0.
        src_texts = [p.src_text for p in pairs]
        tgt_texts = [p.tgt_text for p in pairs]
        for repo in my_repos:
            layers = _xling_layers_for(repo)
            revisions = list_checkpoints(
                repo, hf_token=getattr(args, "hf_token", None),
                all_checkpoints=bool(getattr(args, "all_checkpoints", False)),
            )
            rows = []
            for rev in revisions:
                try:
                    acts_s = _xling_extract(
                        repo=repo, revision=rev, texts=src_texts,
                        dataset_id=f"flores200-{src_lang}", layers=layers,
                        output_root=args.output_root, device=device,
                    )
                    acts_t = _xling_extract(
                        repo=repo, revision=rev, texts=tgt_texts,
                        dataset_id=f"flores200-{tgt_lang}", layers=layers,
                        output_root=args.output_root, device=device,
                    )
                    common = sorted(set(acts_s) & set(acts_t))
                    for layer_idx in common:
                        Xs = acts_s[layer_idx]
                        Xt = acts_t[layer_idx]
                        Xs = Xs.numpy() if hasattr(Xs, "numpy") else np.asarray(Xs)
                        Xt = Xt.numpy() if hasattr(Xt, "numpy") else np.asarray(Xt)
                        # Build per-language CAVs for the same "language"
                        # concept (positive=this language, negative=other).
                        cav_src = tcav.compute_cav(Xs, Xt)
                        cav_tgt = tcav.compute_cav(Xt, Xs)
                        cos = float(cav_src @ cav_tgt)
                        rows.append({
                            "repo": repo, "revision": rev,
                            "layer": layer_idx,
                            "concept": f"{src_lang}_vs_{tgt_lang}",
                            "cav_cosine": cos,
                        })
                except Exception as e:
                    log.exception("tcav: %s @ %s failed: %s", repo, rev, e)
            _xling_write_csv(
                out_root / f"{repo.replace('/', '__')}.csv", rows,
            )
            log.info("tcav: %s — %d rows", repo, len(rows))
    finally:
        _stage_cleanup("tcav", args)


def _run_causal(args):
    """Set D — head knockout (D1) and subspace patching (D2).

    Requires upstream xling artifacts: per-head CKA (from --cka-full) and
    a fitted SVCCA basis (from --svcca). When those CSVs are missing,
    skips the affected sub-stage with a warning. D3/D4 are placeholders;
    they raise NotImplementedError until the corresponding modules are
    fleshed out.
    """
    _setup_logging("causal", args.output_root)
    log = logging.getLogger("causal")
    requested = set(getattr(args, "causal", None) or [])
    if not requested:
        requested = {"d1", "d2"}
    log.info("XLING CAUSAL %s (rank=%d/%d)",
             sorted(requested), args.rank, args.world_size)

    if "d3" in requested:
        log.warning("causal d3: early-finetune module not yet wired — skipping")
    if "d4" in requested:
        log.warning("causal d4: replay/EWC/MAD-X module not yet wired — skipping")

    if not (requested & {"d1", "d2"}):
        return

    import torch
    from _rank_barrier import shard
    from convergence import list_checkpoints
    repos = _resolve_repos(args)
    my_repos = shard(repos, args.rank, args.world_size)
    if not my_repos:
        log.warning("causal: empty pool — skipping")
        return

    out_root = Path(args.output_root) / "xling" / "causal"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if "d1" in requested:
            _run_causal_d1(args, log, my_repos, out_root, device)
        if "d2" in requested:
            _run_causal_d2(args, log, my_repos, out_root, device)
    finally:
        _stage_cleanup("causal", args)


def _run_causal_d1(args, log, my_repos, out_root, device):
    """D1 head knockout — uses --cka-full output to rank universal heads."""
    import csv
    from xling.causal import head_knockout

    cka_dir = Path(args.output_root) / "xling" / "cka_full"
    if not cka_dir.exists():
        log.warning(
            "causal d1: %s missing — run --cka-full first; skipping", cka_dir,
        )
        return

    # Per-head CKA support requires upstream `per_head_cka` output; for
    # the default-on demo we use a uniform head ranking when per-head
    # data is not available.
    log.warning(
        "causal d1: per-head CKA file not produced by --cka-full pipeline; "
        "skipping head knockout. Implement per-head extraction via "
        "xling.cka_full.per_head_cka and feed into this stage."
    )


def _run_causal_d2(args, log, my_repos, out_root, device):
    """D2 subspace patching — verify identity-projection no-op contract."""
    from xling.causal import subspace_patch
    from ppl_utils import load_model_and_tokenizer  # type: ignore

    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for repo in my_repos:
        try:
            model, tokenizer = load_model_and_tokenizer(repo, "main")
            model = model.to(device).eval()
            sample = tokenizer(
                "the quick brown fox", return_tensors="pt",
            )["input_ids"].to(device)
            try:
                subspace_patch.assert_identity_is_noop(
                    model, layer_idx=0, sample_input=sample,
                )
                rows.append({"repo": repo, "test": "identity_noop", "passed": True})
                log.info("causal d2: %s identity-noop OK", repo)
            except AssertionError as e:
                rows.append({
                    "repo": repo, "test": "identity_noop",
                    "passed": False, "error": str(e),
                })
                log.warning("causal d2: %s identity-noop FAILED: %s", repo, e)
        except Exception as e:
            log.exception("causal d2: %s failed: %s", repo, e)
    _xling_write_csv(out_root / "d2_identity_noop.csv", rows)


def _run_polca(args):
    _setup_logging("polca", args.output_root)
    log = logging.getLogger("polca")
    log.info("POLCA POST-HOC SHIM (rank=%d/%d)", args.rank, args.world_size)

    from polca_shim import POLCAShimConfig, extract_polca_artifact
    from convergence import list_checkpoints
    from _rank_barrier import shard

    repos = _resolve_repos(args)
    my_repos = shard(repos, args.rank, args.world_size)
    if not my_repos:
        log.warning("polca: empty pool — skipping")
        return

    holdout_texts = list(getattr(args, "polca_holdout", None) or [
        "The quick brown fox jumps over the lazy dog.",
        "De snelle bruine vos springt over de luie hond heen.",
    ] * 4)

    out_root = Path(args.output_root) / "polca"
    try:
        for repo in my_repos:
            revisions = list_checkpoints(
                repo, hf_token=getattr(args, "hf_token", None),
                all_checkpoints=bool(getattr(args, "all_checkpoints", False)),
            )
            for rev in revisions:
                try:
                    cfg = POLCAShimConfig(
                        repo=repo,
                        revision=rev,
                        runs_dir=out_root,
                        k_eigenvectors=int(getattr(args, "polca_k", 32)),
                        basis_rank=int(getattr(args, "polca_rank", 32)),
                        holdout_batch_size=8,
                        holdout_seq_len=512,
                    )
                    path = extract_polca_artifact(
                        cfg, holdout_texts=holdout_texts,
                    )
                    log.info("polca: %s @ %s → %s", repo, rev, path)
                except Exception as e:
                    log.exception("polca: %s @ %s failed: %s", repo, rev, e)
    finally:
        _stage_cleanup("polca", args)


def _run_dynamics(args):
    if args.rank != 0:
        return
    _setup_logging("dynamics", args.output_root)
    from training_dynamics import run_training_dynamics
    logging.getLogger("dynamics").info("TRAINING DYNAMICS (D1–D4)")
    run_training_dynamics(overwrite=args.overwrite)


def _run_novel(args):
    if args.rank != 0:
        return
    _setup_logging("novel", args.output_root)
    from novel_analyses import run_novel_analyses
    logging.getLogger("novel").info("NOVEL ANALYSES (E2–E5)")
    run_novel_analyses(overwrite=args.overwrite)


def _run_learning_dynamics(args):
    if args.rank != 0:
        return
    _setup_logging("learning_dynamics", args.output_root)
    from learning_dynamics import run_learning_dynamics
    logging.getLogger("learning_dynamics").info("LEARNING DYNAMICS")
    run_learning_dynamics(overwrite=args.overwrite)


def _run_baselines(args):
    if args.rank != 0:
        return
    _setup_logging("baselines", args.output_root)
    from learning_dynamics import evaluate_baseline_models
    logging.getLogger("baselines").info("BASELINE MODEL COMPARISON")
    evaluate_baseline_models(overwrite=args.overwrite)


# Figures whose nature is a trajectory across checkpoints.
_TRAJECTORY_GLOBAL_FNS = (
    "ppl_convergence_curves",
    "cka_convergence_plot",
    "meco_checkpoint_trajectory_plot",
)
_TRAJECTORY_PER_LANG_FNS = (
    "ppl_convergence_overlay",
    "drift_trajectory_plot",
    "forgetting_inflection_plot",
)

# Figures whose nature is a final-checkpoint snapshot.
_FINAL_GLOBAL_FNS = (
    "ppl_heatmap", "forgetting_barplot",
    "meco_correlation_heatmap", "meco_condition_comparison",
)
_FINAL_PER_LANG_FNS = (
    "forgetting_scatter",
    "vocab_overlap_plot",
    "probe_pca_plot",
    "cka_heatmap",
    "condition_summary",
)


def _run_figures(args):
    if args.rank != 0:
        return
    _setup_logging("figures", args.output_root)
    from plot_style import apply_acl_style
    apply_acl_style()
    import visualise as viz

    log = logging.getLogger("figures")
    scope = getattr(args, "ckpt_scope", "both")
    log.info("GENERATING FIGURES — compare=%s ckpt_scope=%s",
             args.compare or "none", scope)
    langs = args.langs or ["deu", "nld", "zho", "eng"]

    emit_final = scope in ("latest", "both")
    emit_trajectory = scope in ("all", "both")

    global_fns: list[tuple[str, str]] = []
    per_lang_fns: list[tuple[str, str]] = []
    if emit_final:
        global_fns   += [(fn, "final") for fn in _FINAL_GLOBAL_FNS]
        per_lang_fns += [(fn, "final") for fn in _FINAL_PER_LANG_FNS]
    if emit_trajectory:
        global_fns   += [(fn, "trajectory") for fn in _TRAJECTORY_GLOBAL_FNS]
        per_lang_fns += [(fn, "trajectory") for fn in _TRAJECTORY_PER_LANG_FNS]

    # Expose the current variant to visualise.py so ``_save_pub`` can
    # append it to output stems. A module-level variable is used rather
    # than threading a kwarg through every figure function because many
    # of them are exported as parameter-less convenience wrappers.
    def _with_variant(variant):
        class _Ctx:
            def __enter__(self_):
                viz.CURRENT_FIGURE_VARIANT = variant
                return self_
            def __exit__(self_, *a):
                viz.CURRENT_FIGURE_VARIANT = None
        return _Ctx()

    for fn_name, variant in global_fns:
        fn = getattr(viz, fn_name, None)
        if fn is None:
            log.warning("figures: %s not available — skip", fn_name)
            continue
        try:
            with _with_variant(variant):
                fn()
        except FileNotFoundError as e:
            log.warning("figures: %s (%s) skipped — %s", fn_name, variant, e)
        except Exception as e:  # pragma: no cover
            log.exception("figures: %s (%s) ERROR — %s", fn_name, variant, e)

    for lang in langs:
        for fn_name, variant in per_lang_fns:
            fn = getattr(viz, fn_name, None)
            if fn is None:
                continue
            try:
                with _with_variant(variant):
                    fn(lang=lang)
            except FileNotFoundError as e:
                log.warning("figures: %s(%s, %s) skipped — %s",
                            fn_name, lang, variant, e)
            except Exception as e:  # pragma: no cover
                log.exception("figures: %s(%s, %s) ERROR — %s",
                              fn_name, lang, variant, e)


def _run_paper_figures(args):
    if args.rank != 0:
        return
    # Paper figures are final-snapshot by nature. Skip them entirely when
    # ``--ckpt-scope all`` is requested — only trajectory figures make sense
    # in that mode. Under ``latest`` and ``both`` (default) they are emitted.
    scope = getattr(args, "ckpt_scope", "both")
    if scope == "all":
        logging.getLogger("paper_figures").info(
            "SKIPPING paper figures — ckpt_scope=all (trajectory-only mode)")
        return
    _setup_logging("paper_figures", args.output_root)
    from plot_style import apply_acl_style
    apply_acl_style()
    import paper_figures as pf
    from paper_figures import render_all as render_paper
    log = logging.getLogger("paper_figures")
    log.info("RENDERING ACL PAPER FIGURES — ckpt_scope=%s", scope)
    langs = args.langs or ["deu", "nld", "zho", "eng"]
    # Paper figures gain the "_final" suffix uniformly (all are snapshots).
    pf.CURRENT_FIGURE_VARIANT = "final"
    try:
        render_paper(langs=langs)
    finally:
        pf.CURRENT_FIGURE_VARIANT = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="beetle-analyze pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--langs", nargs="+",
                   help="ISO-639-3 language codes (default: all)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-compute even if cached")

    # Family / condition / variant selection.
    # NOTE: ``--family`` no longer uses argparse ``choices`` because it also
    # accepts a path to a newline-separated repo list. Values are validated
    # inside ``_resolve_repos``.
    p.add_argument("--family", default="all",
                   help="Model family (humanscale | fineweb | ablations | "
                        "beetlelm_legacy | all), or a path to a file with "
                        "one repo per line. Default: all.")
    p.add_argument("--orgs", nargs="+", default=None,
                   help="Restrict to these HF org prefixes (e.g. "
                        "'Beetle-HumanScale Beetle-FineWeb'). Excludes "
                        "BeetleLM/* legacy repos by default when set.")
    p.add_argument("--condition", nargs="+", default=None,
                   choices=["B1", "B2", "B3", "B4", "B5", "mono"],
                   help="Restrict to these B-conditions.")
    p.add_argument("--variant", nargs="+", default=None,
                   help="Restrict to these ablation variants (B1b, B2_ewc, …).")
    p.add_argument("--compare", default=None,
                   choices=["pairs", "family-vs-family"],
                   help="Comparison figure mode.")

    # Priority / ordering.
    p.add_argument("--priority", default=None,
                   choices=["P0", "P1", "P2", "all"],
                   help="Run the stage-set for this priority tier. "
                        "P0 = figure-critical core evals (ppl, convergence, "
                        "meco, meco-checkpoints). "
                        "P1 = P0 + baselines. "
                        "P2 = P1 + representations, cognates, cli. "
                        "'all' = every registered stage.")
    p.add_argument("--order", default="per-model",
                   choices=["per-model", "per-eval"],
                   help="Execution order across the repo pool. 'per-model' "
                        "(default) finishes every stage for one repo before "
                        "moving on, so interrupted jobs leave whole models "
                        "complete. 'per-eval' matches the historical "
                        "behaviour (one stage across all models, then the "
                        "next).")

    # Cache management.
    p.add_argument("--cache-mode", default="rotate",
                   choices=["keep", "rotate", "clear"],
                   help="HF cache lifecycle (default: rotate).")
    p.add_argument("--hf-cache-dir", default=None,
                   help="Override BEETLE_HF_CACHE.")
    p.add_argument("--output-root", type=Path, default=Path("results"),
                   help="Root for outputs + logs (default: results/).")

    # Analysis switches.
    p.add_argument("--all",           action="store_true")
    p.add_argument("--ppl",           action="store_true")
    p.add_argument("--forgetting",    action="store_true")
    p.add_argument("--embeddings",    action="store_true")
    p.add_argument("--convergence",   action="store_true")
    p.add_argument("--figures",       action="store_true")
    p.add_argument("--paper-figures", action="store_true",
                   help="Render the ACL main-body figures from existing CSVs")
    p.add_argument("--meco",          action="store_true")
    p.add_argument("--meco-only",     action="store_true")
    p.add_argument("--meco-checkpoints", action="store_true",
                   help="Score MeCo at every step-N branch. Shadowed by "
                        "--ckpt-scope when that flag is set to all/both.")
    p.add_argument("--ckpt-scope", default="both",
                   choices=["all", "latest", "both"],
                   help="Checkpoint iteration scope for data & figures. "
                        "'all'    — every step-N branch iterated; only "
                        "           trajectory figures are emitted. "
                        "'latest' — only the highest step-N (or main) is "
                        "           evaluated; only final-snapshot figures "
                        "           are emitted. "
                        "'both'   — (default) every step-N branch iterated "
                        "           AND both trajectory and final figure "
                        "           variants are emitted.")
    p.add_argument("--meco-models",   nargs="+", default=None)
    p.add_argument("--meco-sweep",    action="store_true")
    p.add_argument("--sweep-org",     default=None)
    p.add_argument("--aoa",           action="store_true")
    p.add_argument("--aoa-model",     nargs="+", default=None,
                   help="Explicit model repo(s) for AoA (overrides --family).")
    p.add_argument("--aoa-ficarra",   action="store_true",
                   help="Ficarra-style signature + ϵ-convergence AoA "
                        "(runs on top of --aoa surprisals).")
    p.add_argument("--ficarra-all-signatures", action="store_true",
                   help="Enable σ_I (intrinsic) and σ_R (extrinsic) "
                        "signatures in addition to σ+/σ-/σ±.")
    p.add_argument("--ficarra-full-sweep", action="store_true",
                   help="Sweep ϵ ∈ [0.01, 0.05, 0.07, 0.1, 0.15] "
                        "instead of ϵ = 0.07 only.")
    p.add_argument("--ficarra-heatmap", action="store_true",
                   help="Also render the Appendix-J cross-signature "
                        "Pearson heatmap.")
    p.add_argument("--ficarra-regime", default="humanscale",
                   choices=["humanscale", "fineweb"],
                   help="Training regime for Ficarra AoA (default: humanscale).")
    p.add_argument("--resume",        action="store_true")
    p.add_argument("--batch-size",    type=int, default=32)
    p.add_argument("--cognates",      action="store_true")
    p.add_argument("--meco-extended", action="store_true")
    p.add_argument("--cli",           action="store_true")
    p.add_argument("--representations", action="store_true")
    p.add_argument("--dynamics",      action="store_true")
    p.add_argument("--novel",         action="store_true")
    p.add_argument("--learning-dynamics", action="store_true")
    p.add_argument("--baselines",     action="store_true")
    p.add_argument("--hf_token",      default=None)

    # SAE (TopK-SAE training + Dutch/English feature extraction).
    # Activate either with --sae (train + extract), or split via
    # --sae-train and --sae-extract.
    p.add_argument("--sae",           action="store_true",
                   help="Train SAE and extract Dutch/English features for "
                        "each (repo, revision) in the resolved pool.")
    p.add_argument("--sae-train",     action="store_true",
                   help="SAE training only (no feature extraction).")
    p.add_argument("--sae-extract",   action="store_true",
                   help="SAE feature extraction only (assumes trained SAE).")
    p.add_argument("--sae-layer",     type=int, default=7,
                   help="Hidden-layer index to hook (0-indexed; default 7).")
    p.add_argument("--sae-num-latents", type=int, default=12_288)
    p.add_argument("--sae-k",         type=int, default=32)
    p.add_argument("--sae-lr",        type=float, default=1e-3)
    p.add_argument("--sae-tokens",    type=int, default=100_000_000,
                   help="Total activation tokens for SAE training.")
    p.add_argument("--sae-extract-tokens", type=int, default=200_000,
                   help="Per-language tokens for feature extraction.")
    p.add_argument("--sae-dataset",   default=None,
                   help="HF dataset for activation source during training "
                        "(e.g. Beetle-HumanScale/nld-50M-nld-eng).")
    p.add_argument("--sae-dataset-name", default=None,
                   help="HF dataset config name (optional).")
    p.add_argument("--sae-pretokenized", action="store_true", default=True,
                   help="Dataset already has input_ids (default: True).")
    p.add_argument("--sae-raw-text", dest="sae_pretokenized", action="store_false",
                   help="Dataset is raw text; tokenize on the fly.")
    p.add_argument("--sae-ctx-len",   type=int, default=512)
    p.add_argument("--sae-revisions", nargs="+", default=None,
                   help="Revisions to process for SAE (default: ['main']). "
                        "Use e.g. 'main step-50000' for final + intermediate.")

    # Cross-lingual mechanistic stack (xling/*).
    p.add_argument("--svcca", action="store_true",
                   help="Run SVCCA / PWCCA layer-wise similarity (Sets A1, A3, C1).")
    p.add_argument("--svcca-method", default="svcca",
                   choices=["svcca", "pwcca"],
                   help="Variant for --svcca (default: svcca).")
    p.add_argument("--cka-full", action="store_true",
                   help="Run full-activation CKA (Sets B1, B2).")
    p.add_argument("--cka-kernel", default="linear",
                   choices=["linear", "rbf"],
                   help="Kernel for --cka-full (default: linear).")
    p.add_argument("--cka-unbiased", action="store_true",
                   help="Use the Nguyen et al. 2021 unbiased minibatch estimator.")
    p.add_argument("--crossencoder", action="store_true",
                   help="Run CrossEncoder semantic alignment probe (Sets B3, C2).")
    p.add_argument("--probes", action="store_true",
                   help="Run linear probes with control tasks (Sets A2, B4).")
    p.add_argument("--tcav", action="store_true",
                   help="Run TCAV cross-lingual concept vectors (Set B4).")
    p.add_argument("--causal", nargs="+", default=None,
                   choices=["d1", "d2", "d3", "d4"],
                   help="Causal-validation sub-stages. d1=head knockout, "
                        "d2=subspace patching, d3=early-finetune (gated), "
                        "d4=replay/EWC/MAD-X (gated).")
    p.add_argument("--polca", action="store_true",
                   help="Run POLCA post-hoc artifact extraction.")
    p.add_argument("--polca-k", type=int, default=32,
                   help="Hessian top-k eigenvectors for --polca (default: 32).")
    p.add_argument("--polca-rank", type=int, default=32,
                   help="Gradient SVD basis rank for --polca (default: 32).")
    p.add_argument("--xling-max-pairs", type=int, default=256,
                   help="Max FLORES parallel pairs sampled per checkpoint "
                        "for SVCCA / CKA / CrossEncoder / TCAV (default: 256).")

    # SLT geometry analysis (slt/*).
    p.add_argument("--llc", action="store_true",
                   help="Estimate LLC / RLCT per (model, checkpoint).")
    p.add_argument("--fisher", action="store_true",
                   help="Block-diagonal Fisher + Lanczos top-k eigenvalues.")
    p.add_argument("--susceptibilities", action="store_true",
                   help="Per-component susceptibilities + structural-inference PCA.")
    p.add_argument("--bridge", action="store_true",
                   help="A1: cross-lingual probe ↔ susceptibility-PC1 regression.")
    p.add_argument("--falsification", action="store_true",
                   help="A2: vocab-disjoint and token-permutation controls.")
    p.add_argument("--per-lang-llc", action="store_true",
                   help="S3: per-language LLC decomposition.")
    p.add_argument("--beta-sweep", action="store_true",
                   help="S2: 4-temperature β-sweep validation on anchor cells.")
    p.add_argument("--slt-aggregate", action="store_true",
                   help="Merge SLT shards → numbers.tex + tables.")
    p.add_argument("--slt-plot", action="store_true",
                   help="Render SLT paper figures.")
    p.add_argument("--all-checkpoints", action="store_true",
                   help="Iterate every step-N branch (S1 trajectory) for SLT stages.")
    p.add_argument("--llc-batch-size", type=int, default=8)
    p.add_argument("--llc-num-chains", type=int, default=4)
    p.add_argument("--llc-num-draws", type=int, default=200)
    p.add_argument("--llc-num-burnin", type=int, default=100)
    p.add_argument("--sus-num-chains", type=int, default=4)
    p.add_argument("--sus-num-draws", type=int, default=200)
    p.add_argument("--sus-num-burnin", type=int, default=100)
    p.add_argument("--sus-batch-size", type=int, default=8)
    p.add_argument("--sus-n-per-probe", type=int, default=256)
    p.add_argument("--fisher-samples", type=int, default=8192)
    p.add_argument("--fisher-topk", type=int, default=128)
    p.add_argument("--falsification-a2a-repos", nargs="+", default=None,
                   help="Repos for A2a vocab-disjoint LLC.")
    p.add_argument("--falsification-a2b-repos", nargs="+", default=None,
                   help="Repos for A2b token-permutation LLC.")
    p.add_argument("--beta-sweep-anchors", nargs="+", default=None,
                   help="Anchor repos for the four-temperature β-sweep.")

    # Multi-rank fan-out (one process per GPU; see analyze/_rank_barrier.py).
    import os as _os
    p.add_argument("--rank", type=int,
                   default=int(_os.environ.get("RANK", "0")),
                   help="This process's rank in [0, world_size). "
                        "Default: $RANK or 0.")
    p.add_argument("--world-size", type=int,
                   default=int(_os.environ.get("WORLD_SIZE", "1")),
                   help="Number of parallel ranks sharing the output tree. "
                        "Default: $WORLD_SIZE or 1.")

    return p


_PRIORITY_STAGE_SETS: dict[str, tuple[str, ...]] = {
    # Stage names match the attribute flags on ``argparse.Namespace``.
    "P0": ("ppl", "convergence", "meco", "meco_checkpoints"),
    "P1": ("ppl", "convergence", "meco", "meco_checkpoints", "baselines"),
    "P2": ("ppl", "convergence", "meco", "meco_checkpoints", "baselines",
           "representations", "cognates", "cli", "aoa_ficarra"),
    "all": ("all",),
}


def _apply_priority(args) -> None:
    """Set the per-stage flags implied by ``--priority``.

    Stages already explicitly requested on the command line are preserved.
    Adds the figure regeneration at the end when ``--priority`` is given
    so the scripts can chain ``--priority P0 --figures`` naturally.
    """
    pr = getattr(args, "priority", None)
    if not pr:
        return
    enabled = _PRIORITY_STAGE_SETS.get(pr, ())
    if pr == "all":
        args.all = True
        return
    for stage in enabled:
        if not getattr(args, stage, False):
            setattr(args, stage, True)


def _apply_ckpt_scope(args) -> None:
    """Derive ``args.all_checkpoints`` from ``args.ckpt_scope``.

    When ``--ckpt-scope`` is ``all`` or ``both``, every PPL / MeCo call
    iterates every ``step-N`` branch plus ``main``. When ``latest``, the
    legacy single-best-checkpoint behavior is preserved. The derived flag
    shadows ``--meco-checkpoints`` (they are OR'd together to preserve
    back-compat with older shell invocations).
    """
    scope = getattr(args, "ckpt_scope", "both")
    args.all_checkpoints = scope in ("all", "both")
    if args.all_checkpoints:
        args.meco_checkpoints = True


def _per_model_order(args) -> None:
    """Run the requested stages once per repo, looping over the resolved pool.

    The resolved pool is computed on rank 0 and each rank deterministically
    shards it by ``args.world_size``. For each repo we call the same stage
    dispatchers used by the default order, but pass a one-element pool via
    a scoped override on ``args.family`` (a tmp file containing just that
    repo). This keeps the existing stage code untouched while giving the
    "finish one model at a time" guarantee.
    """
    import tempfile

    repos = _resolve_repos(args)
    if not repos:
        logging.getLogger("run").warning("per-model: resolved pool is empty")
        return

    my_repos = repos[args.rank :: args.world_size]
    log = logging.getLogger("run")
    log.info("per-model: rank %d/%d owns %d/%d repos",
             args.rank, args.world_size, len(my_repos), len(repos))

    orig_family = args.family
    for idx, repo in enumerate(my_repos, 1):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, prefix="beetle_pool_",
        ) as fh:
            fh.write(repo + "\n")
            tmp = fh.name
        try:
            args.family = tmp
            log.info("per-model [%d/%d] %s", idx, len(my_repos), repo)
            _dispatch_stages(args, per_model=True)
        finally:
            args.family = orig_family
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _dispatch_stages(args, per_model: bool = False) -> None:
    """Invoke the stage functions selected by ``args``.

    When ``per_model=True``, rank-0-only stages (forgetting, embeddings,
    figures) are skipped in the inner loop and run once at the end.
    """
    # Core evals — safe per-model.
    if args.all or args.ppl:               _run_ppl(args)
    if args.all or args.convergence:       _run_convergence(args)
    if args.all or args.meco:              _run_meco(args)
    if args.all or args.aoa:               _run_aoa(args)
    if args.all or getattr(args, "aoa_ficarra", False):
        _run_aoa_ficarra(args)

    # Cross-lingual mechanistic stack — also safe per-model since each
    # stage shards repos via _resolve_repos and writes per-repo CSVs.
    if args.all or getattr(args, "svcca", False):        _run_svcca(args)
    if args.all or getattr(args, "cka_full", False):     _run_cka_full(args)
    if args.all or getattr(args, "crossencoder", False): _run_crossencoder(args)
    if args.all or getattr(args, "probes", False):       _run_probes(args)
    if args.all or getattr(args, "tcav", False):         _run_tcav(args)
    if args.all or getattr(args, "causal", None):        _run_causal(args)
    if args.all or getattr(args, "polca", False):        _run_polca(args)

    # SLT geometry stack — per-cell, rank-sharded.
    if args.all or getattr(args, "llc", False):
        from slt.run_llc import run_stage as _slt_llc; _slt_llc(args)
    if args.all or getattr(args, "fisher", False):
        from slt.run_fisher import run_stage as _slt_fisher; _slt_fisher(args)
    if args.all or getattr(args, "susceptibilities", False):
        from slt.run_susceptibilities import run_stage as _slt_sus; _slt_sus(args)
    if args.all or getattr(args, "per_lang_llc", False):
        from slt.per_language_llc import run_stage as _slt_pll; _slt_pll(args)
    if args.all or getattr(args, "falsification", False):
        from slt.falsification import run_stage as _slt_fals; _slt_fals(args)
    if args.all or getattr(args, "beta_sweep", False):
        from slt.beta_sweep import run_stage as _slt_beta; _slt_beta(args)

    if per_model:
        return

    # Stages that need the full pool or rank-0 only.
    if args.all or args.forgetting:        _run_forgetting(args)
    if args.all or args.embeddings:        _run_embeddings(args)
    if args.meco_sweep:                    _run_meco_sweep(args)
    if args.all or args.cognates:          _run_cognates(args)
    if args.meco_extended:                 _run_meco_extended(args)
    if args.all or args.cli:               _run_cli(args)
    if (getattr(args, "sae", False)
            or getattr(args, "sae_train", False)
            or getattr(args, "sae_extract", False)):
        _run_sae(args)
    if args.representations:               _run_representations(args)
    if args.dynamics:                      _run_dynamics(args)
    if args.novel:                         _run_novel(args)
    if args.learning_dynamics:             _run_learning_dynamics(args)
    if args.baselines:                     _run_baselines(args)
    # SLT rank-0-only tail: bridge regression, aggregation, plots.
    if args.all or getattr(args, "bridge", False):
        from slt.bridge import run_stage as _slt_bridge; _slt_bridge(args)
    if args.all or getattr(args, "slt_aggregate", False):
        from slt.aggregate import run_stage as _slt_agg; _slt_agg(args)
    if args.all or getattr(args, "slt_plot", False):
        from slt.plot_slt import run_stage as _slt_plot; _slt_plot(args)

    # Figures must run last.
    if args.all or args.figures:           _run_figures(args)
    if args.all or args.paper_figures:     _run_paper_figures(args)


def main():
    import os
    parser = build_parser()
    args = parser.parse_args()

    if args.hf_cache_dir:
        os.environ["BEETLE_HF_CACHE"] = args.hf_cache_dir

    # Expand --priority into the corresponding per-stage flags before
    # checking whether the user asked for anything to run.
    _apply_priority(args)
    _apply_ckpt_scope(args)

    if not any([args.all, args.ppl, args.forgetting, args.embeddings,
                args.convergence, args.figures, args.paper_figures,
                args.meco, args.meco_only, args.meco_sweep, args.aoa,
                getattr(args, "aoa_ficarra", False),
                args.cognates, args.meco_extended, args.cli,
                args.representations, args.dynamics, args.novel,
                args.learning_dynamics, args.baselines,
                getattr(args, "sae", False),
                getattr(args, "sae_train", False),
                getattr(args, "sae_extract", False),
                getattr(args, "svcca", False),
                getattr(args, "cka_full", False),
                getattr(args, "crossencoder", False),
                getattr(args, "probes", False),
                getattr(args, "tcav", False),
                getattr(args, "causal", None),
                getattr(args, "polca", False),
                getattr(args, "llc", False),
                getattr(args, "fisher", False),
                getattr(args, "susceptibilities", False),
                getattr(args, "bridge", False),
                getattr(args, "falsification", False),
                getattr(args, "per_lang_llc", False),
                getattr(args, "beta_sweep", False),
                getattr(args, "slt_aggregate", False),
                getattr(args, "slt_plot", False)]):
        parser.print_help()
        sys.exit(0)

    t0 = time.time()

    if args.meco_only:
        _run_meco(args)
        print(f"\nPipeline complete in {(time.time()-t0)/60:.1f} min")
        return

    if args.order == "per-model":
        # Run each core stage once per repo (for the per-model "whole
        # models finish together" guarantee), then the rank-0/global tail
        # stages once at the end.
        _per_model_order(args)
        # Rank-0-only tail (forgetting, embeddings, figures, …).
        _dispatch_stages(args, per_model=False)
    else:
        _dispatch_stages(args, per_model=False)

    # Merge any ledger shards written this run so follow-up invocations
    # see the fresh state without needing a separate ``eval_ledger merge``.
    try:
        from eval_ledger import merge as _ledger_merge
        _ledger_merge(args.output_root)
    except Exception:  # pragma: no cover — ledger is best-effort
        logging.getLogger("run").exception("ledger merge failed")

    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed/60:.1f} min")
    print(f"  Outputs in: {args.output_root}/")


if __name__ == "__main__":
    main()
