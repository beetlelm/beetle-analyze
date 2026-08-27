"""Optional plotting helpers (requires the ``plots`` extra: ``pip install beetle-analyze[plots]``)."""
from __future__ import annotations

from pathlib import Path

from .blimp import BlimpResult


def plot_per_phenomenon(result: BlimpResult, out_path: str | Path, title: str | None = None) -> Path:
    """Write a bar chart of per-phenomenon accuracy to ``out_path``."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install beetle-analyze[plots]"
        ) from exc

    per = result.per_phenomenon
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(per["phenomenon"], per["accuracy"], color="#4c72b0")
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.axhline(result.overall_accuracy, color="#c44e52", lw=1.5, label="overall")
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title(title or f"BLiMP accuracy ({result.scoring} scoring)")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
