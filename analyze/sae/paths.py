"""Single source of truth for SAE artifact paths."""

from __future__ import annotations

from pathlib import Path


def slug(repo: str) -> str:
    return repo.replace("/", "__")


def sae_dir(
    repo: str,
    revision: str,
    layer: int,
    num_latents: int,
    k: int,
    lr: float,
    root: Path | str = Path("results"),
) -> Path:
    """`<root>/sae/<slug>/<rev>/layer<L>/n<N>_k<K>/lr<lr>`"""
    return (
        Path(root)
        / "sae"
        / slug(repo)
        / revision
        / f"layer{layer}"
        / f"n{num_latents}_k{k}"
        / f"lr{lr}"
    )


def features_dir(
    repo: str,
    revision: str,
    layer: int,
    root: Path | str = Path("results"),
) -> Path:
    """`<root>/sae/<slug>/<rev>/layer<L>/features`"""
    return Path(root) / "sae" / slug(repo) / revision / f"layer{layer}" / "features"


def feature_npy(
    repo: str,
    revision: str,
    layer: int,
    lang: str,
    root: Path | str = Path("results"),
) -> Path:
    """`<root>/sae/<slug>/<rev>/layer<L>/features/<lang>.npy`"""
    return features_dir(repo, revision, layer, root) / f"{lang}.npy"
