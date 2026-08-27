"""
model_naming.py — Parser for HuggingFace repo identifiers produced by BeetleLM.

Mirrors the templates in ``beetlelm/src/config/_repo_naming.py`` and the ablation
codes in ``beetlelm/configs/generate_conditions_with_best.py``.

Single source of truth for mapping a repo id to
    (family, condition, variant, l1, l2, cl_variant, cl_param, perlang, seed).

The round-trip property ``reverse_repo(parse_repo(r)) == r`` is enforced by
tests/test_model_naming.py against every manifest entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Iterable, Literal, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Constants mirrored from BeetleLM
# ─────────────────────────────────────────────────────────────────────────────

Family = Literal["humanscale", "fineweb", "fineweb-100m", "fineweb-2b", "beetlelm_legacy", "unknown"]
Condition = Literal["B1", "B2", "B3", "B4", "B5", "mono", "unknown"]
CLVariant = Literal["ewc", "lamol", "lr", "maml", "episodic", "none"]

#: Ablation sub-codes. Mirrors ABLATION_VARIANTS in
#: beetlelm/configs/generate_conditions_with_best.py.
ABLATION_VARIANTS: tuple[str, ...] = (
    "B1a", "B1b",
    "B2_later", "B2_smooth", "B2_burst",
    "B2_ewc", "B2_lamol", "B2_lr",
    "B3_full", "B3_25_75",
    "B4a", "B4b", "B4b_lamol",
    "B5_ewc", "B5_lamol",
)

#: Seeds used by BeetleLM's minimal ablation sweep
#: (script/ablation_minimal_nld_eng.sh).
ABLATION_SEEDS: tuple[int, ...] = (13, 42, 97)

#: Orgs the three publishing namespaces.
HUMAN_SCALE_ORG = "Beetle-HumanScale"
FINEWEB_ORG = "Beetle-FineWeb"
BEETLELM_ORG = "BeetleLM"

#: Canonical six-factor axes from Table `tab:taxonomy` in the paper.
#: Keyed by {B1..B5}, with ablation variants overriding individual axes.
_BASE_FACTORS: dict[str, dict[str, object]] = {
    "B1": dict(onset="early", l2_ratio=0.50, transition="none",
               burst=False, episodic=False, cl="none"),
    "B2": dict(onset="mid", l2_ratio=0.50, transition="sharp",
               burst=False, episodic=False, cl="none"),
    "B3": dict(onset="mid", l2_ratio=0.67, transition="sharp",
               burst=False, episodic=False, cl="none"),
    "B4": dict(onset="mid", l2_ratio=0.20, transition="gradual",
               burst=True, episodic=True, cl="ewc"),
    "B5": dict(onset="late", l2_ratio=0.50, transition="sharp",
               burst=False, episodic=False, cl="lr"),
    "mono": dict(onset=None, l2_ratio=0.0, transition=None,
                 burst=False, episodic=False, cl="none"),
}

_VARIANT_OVERRIDES: dict[str, dict[str, object]] = {
    "B1a":      dict(onset="very_early"),  # B1 with earlier onset (reserved)
    "B1b":      dict(episodic=True),
    "B2_later": dict(onset="late"),
    "B2_smooth":dict(transition="gradual"),
    "B2_burst": dict(burst=True),
    "B3_full":  dict(l2_ratio=1.0),
    "B3_25_75": dict(l2_ratio=0.75),
    "B4a":      dict(cl="none"),           # B4 without EWC
    "B4b":      dict(cl="none"),           # B4 no continual-learning
    # CL overlays handled via cl_variant field, not a variant override.
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RepoMeta:
    raw_name: str
    org: str
    family: Family
    condition: Condition
    variant: Optional[str]   # e.g. "B1b", "B3_full", "B2_smooth", or None
    l1: Optional[str]
    l2: Optional[str]
    cl_variant: CLVariant
    cl_param: Optional[int]
    perlang_vocab: bool
    seed: Optional[int]
    #: For slugs matched via the generic fallback, keeps the descriptor so
    #: reverse_repo round-trips the original string. Empty for canonical slugs.
    descriptor_override: Optional[str] = None
    #: Architectural / replay intervention suffixes (Tier-5):
    #: "safe", "alibi", "layer-growth", "pre-pretrain", or None.
    arch_variant: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Regexes
# ─────────────────────────────────────────────────────────────────────────────

# Optional trailing feature suffixes. Order matters because some variants
# (e.g. "-episodic") must be matched before and stripped before parsing
# curriculum. We split into two classes:
#
#   _SEED_SUFFIX: only seed (always last, handled first)
#   _FEATURE_SUFFIX: CL + episodic + perlang (in any order, parsed greedily)

_SEED_SUFFIX = re.compile(r"-seed(\d+)$")

# Feature fragments — order in regex does not matter, we parse them with
# individual searches after trimming the seed.
#
# CL variants ship in two flavors: an early form that bakes the strength
# parameter into the slug (-ewc-l20, -lamol-l20, -maml-r4, -lr-b2) and a
# bare form that omits it (-ewc, -lamol, -maml). We accept both — the bare
# form leaves cl_param=None.
_FEATURE_PATTERNS: dict[str, re.Pattern] = {
    "ewc":      re.compile(r"-ewc(?:-l(\d+))?\b"),
    "lamol":    re.compile(r"-lamol(?:-l(\d+))?\b"),
    "lr":       re.compile(r"-lr-b(\d+)\b"),
    "maml":     re.compile(r"-maml(?:-r(\d+))?\b"),
    "episodic": re.compile(r"-episodic\b"),
    "perlang":  re.compile(r"-perlang\b"),
}

# Architectural / replay interventions that appear as terminal suffixes on
# Tier-5-style repos (e.g. ``-fineweb-100m-nld-eng-alibi``). These are
# orthogonal to the CL variants above; a repo carries at most one of them.
_ARCH_VARIANTS: tuple[str, ...] = (
    "safe", "alibi", "layer-growth", "pre-pretrain",
)
_ARCH_PATTERN = re.compile(
    r"-(?P<arch>safe|alibi|layer-growth|pre-pretrain)\b"
)

# Core slug regexes. {dataset} ∈ {humanscale, fineweb}.
# We allow *either* `-` or `_` internally but require the published form.

# NB: `fineweb-100m` and `fineweb-2b` MUST come before plain `fineweb` in the
# alternation. The alternatives are tried left-to-right; if `fineweb` matched
# first against `fineweb-100m-nld-eng`, the language group `[a-z]{3}` would
# then try to consume `100`, fail, and the overall match would too.
_DATASET = r"(?P<dataset>humanscale|fineweb-100m|fineweb-2b|fineweb)"

_RE_MONO = re.compile(
    rf"^beetle-monolingual-{_DATASET}-(?P<l1>[a-z]{{3}})$"
)

_RE_B1 = re.compile(
    rf"^beetle-bilingual-balanced-b1-{_DATASET}-"
    r"(?P<l1>[a-z]{3})-(?P<l2>[a-z]{3})$"
)

_RE_B2 = re.compile(
    rf"^beetle-bilingual-l2-50-simultaneous-b2-{_DATASET}-"
    r"(?P<l1>[a-z]{3})-(?P<l2>[a-z]{3})$"
)

_RE_B3 = re.compile(
    rf"^beetle-bilingual-l2-50-sequential-(?P<p>\d+)-(?P<q>\d+)-b3-{_DATASET}-"
    r"(?P<l1>[a-z]{3})-(?P<l2>[a-z]{3})$"
)

_RE_B4 = re.compile(
    rf"^beetle-bilingual-l2-50-classroom-20-b4-{_DATASET}-"
    r"(?P<l1>[a-z]{3})-(?P<l2>[a-z]{3})$"
)

_RE_B5 = re.compile(
    rf"^beetle-bilingual-l2-80-late-b5-{_DATASET}-"
    r"(?P<l1>[a-z]{3})-(?P<l2>[a-z]{3})$"
)

# Fallback: BeetleLM's FineWeb configs ship B4/B5 with a "balanced" descriptor
# (e.g. ``beetle-bilingual-balanced-b4-fineweb-deu-eng``). This generic regex
# catches any ``beetle-bilingual-<descriptor>-b<N>-<dataset>-<l1>-<l2>`` slug
# that the condition-specific regexes above didn't match.
_RE_GENERIC = re.compile(
    rf"^beetle-bilingual-(?P<descriptor>[a-z0-9-]+?)-b(?P<n>[1-5])-{_DATASET}-"
    r"(?P<l1>[a-z]{3})-(?P<l2>[a-z]{3})$"
)


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

def _split_org(repo: str) -> tuple[str, str]:
    if "/" in repo:
        org, name = repo.split("/", 1)
    else:
        org, name = "", repo
    return org, name


def _strip_features(name: str) -> tuple[str, dict]:
    """Strip feature suffixes (CL variant, arch variant, episodic, perlang,
    seed) from a slug. Return the stripped slug and a dict of parsed feature
    values."""
    features: dict = {
        "cl_variant": "none",
        "cl_param": None,
        "arch_variant": None,
        "perlang_vocab": False,
        "episodic": False,
        "seed": None,
    }

    m = _SEED_SUFFIX.search(name)
    if m:
        features["seed"] = int(m.group(1))
        name = name[: m.start()]

    for key in ("ewc", "lamol", "lr", "maml"):
        m = _FEATURE_PATTERNS[key].search(name)
        if m:
            features["cl_variant"] = key
            features["cl_param"] = int(m.group(1)) if m.group(1) else None
            name = name[: m.start()] + name[m.end():]
            break  # only one CL variant can be active

    m = _ARCH_PATTERN.search(name)
    if m:
        features["arch_variant"] = m.group("arch")
        name = name[: m.start()] + name[m.end():]

    m = _FEATURE_PATTERNS["episodic"].search(name)
    if m:
        features["episodic"] = True
        name = name[: m.start()] + name[m.end():]

    m = _FEATURE_PATTERNS["perlang"].search(name)
    if m:
        features["perlang_vocab"] = True
        name = name[: m.start()] + name[m.end():]

    return name, features


def _parse_new_scheme(name: str) -> Optional[dict]:
    """Try the new `beetle-…-{humanscale,fineweb}-…` slug scheme.
    Returns None if not a match."""
    stripped, feat = _strip_features(name)

    for pattern, condition in (
        (_RE_MONO, "mono"),
        (_RE_B1,   "B1"),
        (_RE_B2,   "B2"),
        (_RE_B4,   "B4"),
        (_RE_B5,   "B5"),
    ):
        m = pattern.match(stripped)
        if m:
            gd = m.groupdict()
            return {
                "condition": condition,
                "dataset":   gd["dataset"],
                "l1":        gd.get("l1"),
                "l2":        gd.get("l2"),
                **feat,
            }

    m = _RE_B3.match(stripped)
    if m:
        gd = m.groupdict()
        return {
            "condition":   "B3",
            "dataset":     gd["dataset"],
            "l1":          gd["l1"],
            "l2":          gd["l2"],
            "_b3_ratio":   (int(gd["p"]), int(gd["q"])),
            **feat,
        }

    # Fallback: generic ``beetle-bilingual-<desc>-b<N>-<dataset>-<l1>-<l2>``.
    # Covers BeetleLM's FineWeb B4/B5 configs that use "balanced" as the
    # descriptor rather than "classroom-20" / "late".
    m = _RE_GENERIC.match(stripped)
    if m:
        gd = m.groupdict()
        cond = f"B{gd['n']}"
        return {
            "condition": cond,
            "dataset":   gd["dataset"],
            "l1":        gd["l1"],
            "l2":        gd["l2"],
            # Preserve the original descriptor so reverse_repo can round-trip.
            "_descriptor_override": gd["descriptor"],
            **feat,
        }

    return None


def _resolve_variant(d: dict) -> Optional[str]:
    """Map a parsed dict to a canonical ablation variant code, or None."""
    c = d["condition"]
    if c == "B1":
        if d.get("episodic"):
            return "B1b"
    if c == "B3":
        p, q = d.get("_b3_ratio", (33, 67))
        if (p, q) == (0, 100):
            return "B3_full"
        if (p, q) == (25, 75):
            return "B3_25_75"
    # B2/B4/B5 variants are identifiable only from CL overlays
    if c == "B2" and d.get("cl_variant") == "ewc":
        return "B2_ewc"
    if c == "B2" and d.get("cl_variant") == "lamol":
        return "B2_lamol"
    if c == "B2" and d.get("cl_variant") == "lr":
        return "B2_lr"
    if c == "B4" and d.get("cl_variant") == "lamol":
        return "B4b_lamol"
    # Note: plain "-classroom-20-b4-" with no CL suffix is B4 canonical
    # (EWC is baked into the condition preset). Explicit B4a/B4b variants
    # only arise from ablation slugs that are not currently produced by
    # BeetleLM's _repo_naming.py.
    if c == "B5" and d.get("cl_variant") == "ewc":
        return "B5_ewc"
    if c == "B5" and d.get("cl_variant") == "lamol":
        return "B5_lamol"
    return None


def _legacy_condition(name: str) -> Condition:
    """Fallback for BeetleLM-legacy slugs (``beetlelm_<pair>_<type>``)."""
    # Check most-specific compound suffixes first.
    if "sequential_ewc" in name and "maml" in name:
        return "B3"  # approximate; see legacy get_bilingual_type for detail
    for tag, cond in (
        ("mono", "mono"),
        ("balanced", "B1"),
        ("simultaneous", "B2"),
        ("sequential", "B3"),
        ("heritage", "B5"),
        ("late", "B5"),
        ("part_time", "B4"),
    ):
        if tag in name:
            return cond  # type: ignore[return-value]
    return "unknown"


def parse_repo(repo: str) -> RepoMeta:
    """Parse an HF repo ID into structured metadata.

    Examples
    --------
    >>> parse_repo("Beetle-HumanScale/beetle-monolingual-humanscale-nld").condition
    'mono'
    >>> parse_repo(
    ...     "BeetleLM/beetle-bilingual-l2-50-simultaneous-b2-"
    ...     "humanscale-nld-eng-ewc-l20-seed13"
    ... ).cl_variant
    'ewc'
    """
    org, name = _split_org(repo)

    parsed = _parse_new_scheme(name)
    if parsed is not None:
        dataset = parsed["dataset"]
        family: Family = dataset if dataset in ("humanscale", "fineweb", "fineweb-100m", "fineweb-2b") else "unknown"
        variant = _resolve_variant(parsed)
        return RepoMeta(
            raw_name=repo,
            org=org,
            family=family,
            condition=parsed["condition"],
            variant=variant,
            l1=parsed.get("l1"),
            l2=parsed.get("l2"),
            cl_variant=parsed["cl_variant"],
            cl_param=parsed["cl_param"],
            perlang_vocab=parsed["perlang_vocab"],
            seed=parsed["seed"],
            descriptor_override=parsed.get("_descriptor_override"),
            arch_variant=parsed.get("arch_variant"),
        )

    # Fallback: legacy BeetleLM slugs (beetlelm_<pair>_<type>).
    if name.startswith("beetlelm_"):
        cond = _legacy_condition(name)
        # Parse language pair: beetlelm_{l1}_mono, beetlelm_{pair}_{type},
        # or beetlelm_{l1}_L1-{l2}_L2_{type}.
        l1 = l2 = None
        m = re.match(r"^beetlelm_([a-z]{3})_mono", name)
        if m:
            l1 = m.group(1)
        else:
            m = re.match(r"^beetlelm_([a-z]{3})_L1-([a-z]{3})_L2", name)
            if m:
                l1, l2 = m.group(1), m.group(2)
            else:
                m = re.match(r"^beetlelm_([a-z]{3})-([a-z]{3})", name)
                if m:
                    l1, l2 = m.group(1), m.group(2)
        return RepoMeta(
            raw_name=repo, org=org, family="beetlelm_legacy",
            condition=cond, variant=None,
            l1=l1, l2=l2,
            cl_variant="none", cl_param=None,
            perlang_vocab=False, seed=None,
        )

    return RepoMeta(
        raw_name=repo, org=org, family="unknown",
        condition="unknown", variant=None,
        l1=None, l2=None,
        cl_variant="none", cl_param=None,
        perlang_vocab=False, seed=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reverse (meta → repo string)
# ─────────────────────────────────────────────────────────────────────────────

def _format_features(meta: RepoMeta) -> str:
    tail = ""
    if meta.cl_variant == "ewc":
        tail += f"-ewc-l{meta.cl_param}" if meta.cl_param is not None else "-ewc"
    elif meta.cl_variant == "lamol":
        tail += f"-lamol-l{meta.cl_param}" if meta.cl_param is not None else "-lamol"
    elif meta.cl_variant == "lr" and meta.cl_param is not None:
        tail += f"-lr-b{meta.cl_param}"
    elif meta.cl_variant == "maml":
        tail += f"-maml-r{meta.cl_param}" if meta.cl_param is not None else "-maml"

    if meta.arch_variant:
        tail += f"-{meta.arch_variant}"

    if meta.variant == "B1b" or (meta.condition == "B1" and meta.cl_variant == "episodic"):
        # episodic handled as a feature flag; B1b = B1 + episodic
        if "-episodic" not in tail:
            tail += "-episodic"

    if meta.perlang_vocab:
        tail += "-perlang"
    if meta.seed is not None:
        tail += f"-seed{meta.seed}"
    return tail


def reverse_repo(meta: RepoMeta) -> str:
    """Format a RepoMeta back to its canonical HF repo id.

    Round-trips with :func:`parse_repo` for all well-formed BeetleLM slugs.
    """
    if meta.family == "beetlelm_legacy":
        return meta.raw_name  # legacy formatting is not round-tripped

    if meta.family not in ("humanscale", "fineweb"):
        return meta.raw_name

    dataset = meta.family
    if meta.descriptor_override:
        # Slug didn't match a canonical condition template; reproduce it
        # verbatim from the parsed descriptor + B-code.
        n = meta.condition[-1]  # "B1" → "1"
        core = (
            f"beetle-bilingual-{meta.descriptor_override}-b{n}-"
            f"{dataset}-{meta.l1}-{meta.l2}"
        )
    elif meta.condition == "mono":
        core = f"beetle-monolingual-{dataset}-{meta.l1}"
    elif meta.condition == "B1":
        core = (
            f"beetle-bilingual-balanced-b1-{dataset}-{meta.l1}-{meta.l2}"
        )
    elif meta.condition == "B2":
        core = (
            f"beetle-bilingual-l2-50-simultaneous-b2-{dataset}-"
            f"{meta.l1}-{meta.l2}"
        )
    elif meta.condition == "B3":
        if meta.variant == "B3_full":
            ratio_slug = "0-100"
        elif meta.variant == "B3_25_75":
            ratio_slug = "25-75"
        else:
            ratio_slug = "33-67"
        core = (
            f"beetle-bilingual-l2-50-sequential-{ratio_slug}-b3-"
            f"{dataset}-{meta.l1}-{meta.l2}"
        )
    elif meta.condition == "B4":
        core = (
            f"beetle-bilingual-l2-50-classroom-20-b4-{dataset}-"
            f"{meta.l1}-{meta.l2}"
        )
    elif meta.condition == "B5":
        core = (
            f"beetle-bilingual-l2-80-late-b5-{dataset}-"
            f"{meta.l1}-{meta.l2}"
        )
    else:
        return meta.raw_name

    slug = core + _format_features(meta)
    return f"{meta.org}/{slug}" if meta.org else slug


# ─────────────────────────────────────────────────────────────────────────────
# Derived helpers
# ─────────────────────────────────────────────────────────────────────────────

def condition_label(meta: RepoMeta) -> str:
    """Short paper-friendly label for a model (seed-stripped)."""
    if meta.condition == "mono":
        lang = meta.l1 or "?"
        return f"mono({lang})"

    base = meta.variant or meta.condition
    if meta.cl_variant == "ewc":
        return f"{base}+EWC(λ={meta.cl_param})"
    if meta.cl_variant == "lamol":
        val = f"{meta.cl_param / 100:.2f}" if meta.cl_param else "?"
        return f"{base}+LAMOL({val})"
    if meta.cl_variant == "lr":
        return f"{base}+LR(b{meta.cl_param})"
    if meta.cl_variant == "maml":
        return f"{base}+MAML(r{meta.cl_param})"
    return str(base)


def grouping_key(meta: RepoMeta) -> str:
    """Aggregation key for mean ± SE across seeds (seed-stripped label)."""
    tag = condition_label(meta)
    pair = "-".join(x for x in (meta.l1, meta.l2) if x)
    return f"{meta.family}:{tag}:{pair}" if pair else f"{meta.family}:{tag}"


def factor_axes(meta: RepoMeta) -> dict:
    """Six-factor vector per Table `tab:taxonomy`."""
    base = dict(_BASE_FACTORS.get(meta.condition, _BASE_FACTORS["mono"]))
    if meta.variant and meta.variant in _VARIANT_OVERRIDES:
        base.update(_VARIANT_OVERRIDES[meta.variant])
    if meta.cl_variant != "none":
        base["cl"] = meta.cl_variant
    return base


def group_by_seed(repos: Iterable[str]) -> dict[str, list[str]]:
    """Group seeded repos by their grouping_key for mean-across-seeds plots."""
    groups: dict[str, list[str]] = {}
    for r in repos:
        key = grouping_key(parse_repo(r))
        groups.setdefault(key, []).append(r)
    # Keep alphabetical order inside each group for stable plotting.
    for key, lst in groups.items():
        lst.sort()
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Training-budget mapping
# ─────────────────────────────────────────────────────────────────────────────
#
# Maps a parsed RepoMeta to the L1 / L2 training-token budget prescribed by
# ``beetlelm/README.md`` (sections "Human-scale" and "FineWeb-scale"):
#
#   family=humanscale   → 50M  tokens per language (mono=L1 only; bilingual
#                         splits the 100M budget per ``condition`` onset/ratio).
#   family=fineweb      → 12B  tokens per language (bilingual splits the 24B
#                         budget equivalently).
#
# These are the values figures A, C, E, F use for x-axis / line-style faceting.
# ``condition`` and ``variant`` refine the split (e.g. B4 = L1 80M / L2 20M at
# human scale; B5 late-onset starts L2 at 80% of training).
#
# Values are returned in raw tokens (int) so callers can format as "400M", "4B"
# or normalise to the x-axis of the learning curves freely.

#: Per-language budget per training family.
_FAMILY_TOKEN_BUDGET: dict[str, int] = {
    "humanscale":       50_000_000,
    "fineweb":      12_000_000_000,
}


def training_budget(meta: RepoMeta) -> tuple[int, int]:
    """Return ``(l1_tokens, l2_tokens)`` in raw tokens for this model.

    Follows the table in ``beetlelm/README.md``:

    * ``mono``: ``(l1=family_budget, l2=0)``.
    * ``B1`` (balanced 50:50): both languages at the family budget. The repo
      was trained on 2×budget total tokens, half to each language.
    * ``B2`` (simultaneous): 100% L1 for half the run, then 50:50 — net budget
      is still ``(budget, budget)`` because the step count doubles.
    * ``B3`` (sequential): 100% L1 phase then 33:67 L1:L2 (canonical) or the
      ablation ratio. For the default case, returns ``(budget, budget)``.
    * ``B4`` (classroom, humanscale 80M/20M): ``(80M, 20M)`` at human scale;
      ``(budget, budget * 0.25)`` scaled up at FineWeb.
    * ``B5`` (late-onset): L2 introduced at 80% of training — the L2 still
      reaches ``budget`` tokens by the end; return ``(budget, budget)``.
    * ``unknown``: return ``(0, 0)``.

    Ablation ratio overrides from ``_VARIANT_OVERRIDES`` are honoured for B3.
    """
    fam = meta.family
    if fam not in _FAMILY_TOKEN_BUDGET:
        return (0, 0)

    budget = _FAMILY_TOKEN_BUDGET[fam]
    cond = meta.condition

    if cond == "mono":
        return (budget, 0)

    if cond == "B4":
        # Human-scale spec: L1 80M / L2 20M (= 1.6× L1, 0.4× budget L2 at HS).
        # At FineWeb scale the ratio scales linearly.
        if fam == "humanscale":
            return (80_000_000, 20_000_000)
        return (int(budget * 1.6), int(budget * 0.4))

    if cond == "B3":
        # l2_ratio comes from _BASE_FACTORS / _VARIANT_OVERRIDES.
        factors = factor_axes(meta)
        ratio = float(factors.get("l2_ratio") or 0.67)
        # Total effective tokens ≈ 2×budget; split by ratio (L1, L2).
        total = 2 * budget
        l2 = int(round(total * ratio))
        l1 = total - l2
        return (l1, l2)

    if cond in ("B1", "B2", "B5"):
        # Balanced / simultaneous / late-onset all net to (budget, budget)
        # by end of training, modulo a late-onset cutoff that still targets
        # the same total on the L2 side.
        return (budget, budget)

    return (budget, 0)


def _fmt_token_budget(tokens: int) -> str:
    """Format a token count for figure legends, e.g. 4_000_000_000 → '4B'."""
    if tokens >= 1_000_000_000:
        v = tokens / 1_000_000_000
        return f"{v:g}B"
    if tokens >= 1_000_000:
        v = tokens / 1_000_000
        return f"{v:g}M"
    if tokens >= 1_000:
        v = tokens / 1_000
        return f"{v:g}K"
    return str(tokens)


def training_budget_label(meta: RepoMeta) -> str:
    """Human-readable label for ``training_budget``, e.g. ``'50M/50M'``."""
    l1, l2 = training_budget(meta)
    return f"{_fmt_token_budget(l1)}/{_fmt_token_budget(l2)}"


# ─────────────────────────────────────────────────────────────────────────────
# Language inference for trajectory eval drivers
# ─────────────────────────────────────────────────────────────────────────────

#: BGPT repo names use 2-letter ISO-639-1; map to the 3-letter codes that
#: ``ppl_utils.FLORES_LANG_MAP`` keys on. Limited to the BGPT pairs.
_BGPT_ISO2_TO_3: dict[str, str] = {
    "en": "eng", "nl": "nld", "es": "spa", "el": "ell", "pl": "pol",
    "de": "deu", "zh": "zho",
}

_RE_BGPT = re.compile(
    r"^(?:catherinearnett/)?B-GPT_"
    r"(?P<l1>[a-z]{2})_(?P<l2>[a-z]{2})_"
    r"(?P<mode>simultaneous|sequential)$"
)


def parse_bgpt(repo: str) -> Optional[dict]:
    """Parse a Catherine Arnett B-GPT repo id.

    Returns ``{"l1": "eng", "l2": "nld", "mode": "simultaneous"}`` or
    ``None`` if the slug does not match the BGPT naming convention.
    """
    m = _RE_BGPT.match(repo)
    if not m:
        return None
    gd = m.groupdict()
    l1 = _BGPT_ISO2_TO_3.get(gd["l1"], gd["l1"])
    l2 = _BGPT_ISO2_TO_3.get(gd["l2"], gd["l2"])
    return {"l1": l1, "l2": l2, "mode": gd["mode"], "family": "bgpt"}


def eval_languages_from_repo(repo: str) -> list[str]:
    """Infer the ISO-639-3 evaluation languages for a model repo.

    Used by trajectory drivers to decide which two languages each model
    should be scored on. Order: L1 first, L2 second; deduped, ``None``
    filtered. Falls back to an empty list when the repo is unrecognised.

    * ``catherinearnett/B-GPT_en_nl_simultaneous`` → ``["eng", "nld"]``
    * ``Beetle-FineWeb/beetle-bilingual-...-eng-nld`` → ``["eng", "nld"]``
    * ``Beetle-HumanScale/beetle-monolingual-humanscale-nld`` → ``["nld"]``
    """
    bg = parse_bgpt(repo)
    if bg is not None:
        return [c for c in (bg["l1"], bg["l2"]) if c]
    meta = parse_repo(repo)
    out: list[str] = []
    for c in (meta.l1, meta.l2):
        if c and c not in out:
            out.append(c)
    return out


__all__ = [
    "Family", "Condition", "CLVariant",
    "ABLATION_VARIANTS", "ABLATION_SEEDS",
    "HUMAN_SCALE_ORG", "FINEWEB_ORG", "BEETLELM_ORG",
    "RepoMeta",
    "parse_repo", "reverse_repo", "parse_bgpt",
    "eval_languages_from_repo",
    "condition_label", "grouping_key", "factor_axes", "group_by_seed",
    "training_budget", "training_budget_label",
]
