"""EDA figures for the Cats vs Dogs dataset.

Logic lives here rather than in the notebook so it is importable and testable;
``notebooks/01_eda.ipynb`` is a thin driver that calls these functions. Same
split of responsibility used on Assignment 1.

Three figures, deliberately not more — M1 grades versioning and tracking, not
exploratory depth, and the 10-day plan spends the saved time on cross-validation
instead (see the cuts table in tracker/PROGRESS.md).

Run:
    python -m src.eda
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # no display in CI or over SSH
import matplotlib.pyplot as plt
import numpy as np

from src.data import CLASS_NAMES, ROOT, discover_images, load_corrupt_list
from src.preprocess import SPLITS, load_image, read_manifest

FIG_DIR = ROOT / "reports" / "figures"

# Own palette, not a matplotlib default — cat/dog/accent/ink.
PALETTE = {
    "cat": "#D1495B",
    "dog": "#30638E",
    "accent": "#EDAE49",
    "ink": "#003D5B",
}


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "font.size": 10,
    })


def plot_class_balance(save_dir: Path = FIG_DIR) -> Path:
    """Per-class counts overall and per split.

    Two panels rather than one: the left shows the dataset is balanced at source,
    the right shows stratification preserved that balance in every split — which
    is the claim audit check 46 verifies numerically.
    """
    counts = dict.fromkeys(CLASS_NAMES, 0)
    for _, label in discover_images():
        counts[CLASS_NAMES[label]] += 1
    total = sum(counts.values())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    bars = axes[0].bar(list(counts), list(counts.values()),
                       color=[PALETTE[c] for c in counts], edgecolor="white", linewidth=1.5)
    for bar, n in zip(bars, counts.values(), strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, n,
                     f"{n:,}\n{100 * n / total:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("images")
    axes[0].set_title(f"Class balance at source (n={total:,})")
    axes[0].set_ylim(0, max(counts.values()) * 1.18)

    width = 0.38
    xs = np.arange(len(SPLITS))
    for i, cls in enumerate(CLASS_NAMES):
        vals = []
        for split in SPLITS:
            _, labels = read_manifest(split)
            vals.append(100 * sum(1 for y in labels if y == i) / len(labels))
        axes[1].bar(xs + (i - 0.5) * width, vals, width,
                    label=cls, color=PALETTE[cls], edgecolor="white", linewidth=1.2)
    axes[1].axhline(50, color=PALETTE["ink"], linestyle=":", linewidth=1, label="50%")
    axes[1].set_xticks(xs, [f"{s}\n(n={len(read_manifest(s)[1]):,})" for s in SPLITS])
    axes[1].set_ylabel("% of split")
    axes[1].set_title("Stratification held across the 80/10/10 split")
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 100)

    out = save_dir / "class_balance.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_sample_grid(n_per_class: int = 4, save_dir: Path = FIG_DIR) -> Path:
    """A row per class of pre-processed 224x224 images, as the model sees them."""
    items = discover_images()
    corrupt = {Path(c).name for c in load_corrupt_list()}
    by_class: dict[int, list[Path]] = {0: [], 1: []}
    for p, label in items:
        if p.name not in corrupt and len(by_class[label]) < n_per_class:
            by_class[label].append(p)

    fig, axes = plt.subplots(2, n_per_class, figsize=(2.3 * n_per_class, 5))
    for row, cls in enumerate(CLASS_NAMES):
        for col in range(n_per_class):
            ax = axes[row, col]
            ax.axis("off")
            paths = by_class[row]
            if col < len(paths):
                ax.imshow(load_image(paths[col]))
                if col == 0:
                    ax.set_title(cls, color=PALETTE[cls], loc="left", fontsize=11)
    fig.suptitle("Pre-processed samples (224x224 RGB, scaled to [0,1])",
                 fontsize=12, fontweight="bold")
    out = save_dir / "sample_grid.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_augmentation_grid(n: int = 5, save_dir: Path = FIG_DIR) -> Path:
    """One image put through the augmentation stack n times.

    Evidence that augmentation is actually active and that its output stays
    inside [0,1] after the clip layer — the bug found on Day 2 was a max of 1.02
    before clipping was added.
    """
    from src.preprocess import build_augmentation

    items = discover_images()
    corrupt = {Path(c).name for c in load_corrupt_list()}
    src = next(p for p, _ in items if p.name not in corrupt)
    base = load_image(src)

    aug = build_augmentation()
    batch = np.repeat(base[None, ...], n, axis=0)
    out_imgs = np.asarray(aug(batch, training=True))

    fig, axes = plt.subplots(1, n + 1, figsize=(2.1 * (n + 1), 2.6))
    axes[0].imshow(base)
    axes[0].set_title("original", fontsize=9)
    axes[0].axis("off")
    for i in range(n):
        axes[i + 1].imshow(out_imgs[i])
        axes[i + 1].set_title(f"aug {i + 1}", fontsize=9)
        axes[i + 1].axis("off")

    lo, hi = float(out_imgs.min()), float(out_imgs.max())
    fig.suptitle(
        f"Augmentation: flip / rotate / zoom / contrast, clipped  "
        f"(output range [{lo:.3f}, {hi:.3f}])",
        fontsize=11, fontweight="bold",
    )
    out = save_dir / "augmentation_grid.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def make_all(save_dir: Path = FIG_DIR) -> list[Path]:
    _setup_style()
    save_dir.mkdir(parents=True, exist_ok=True)
    made = [
        plot_class_balance(save_dir),
        plot_sample_grid(save_dir=save_dir),
        plot_augmentation_grid(save_dir=save_dir),
    ]
    for p in made:
        print(f"[eda] wrote {p.relative_to(ROOT)} ({p.stat().st_size // 1024}KB)")
    return made


if __name__ == "__main__":
    make_all()
    print("[eda] PASS")
