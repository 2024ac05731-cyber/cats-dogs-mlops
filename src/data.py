"""Dataset acquisition and auditing for the Cats vs Dogs classifier.

Owns every path constant and the canonical seed, so nothing downstream
re-defines them (audit check 19 enforces the single definition).

The Kaggle archive has been republished in more than one directory shape, so
``find_image_root`` locates the per-class folders by searching for known class
directory names at any depth instead of hard-coding ``PetImages/Cat``.

This dataset family is well known for truncated and zero-byte JPEGs. They are
found here by ``audit_images`` and written to ``data/corrupt_files.txt``, then
excluded at manifest-build time in ``src.preprocess`` — not skipped with a
try/except inside the training loop, which would fail mid-epoch.

Run:
    python -m src.data
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MONITORING_DIR = DATA_DIR / "monitoring"
CORRUPT_REPORT = DATA_DIR / "corrupt_files.txt"

# --- Canonical constants. Import these; never redefine them. ---
IMG_SIZE = (224, 224)          # spec: 224x224 RGB for standard CNNs
IMG_CHANNELS = 3
CLASS_NAMES = ["cat", "dog"]   # index == label: cat=0, dog=1
CLASS_INDICES = {name: i for i, name in enumerate(CLASS_NAMES)}
RANDOM_STATE = 42
TARGET = "label"

KAGGLE_DATASET = "bhavikjikadara/dog-and-cat-classification-dataset"

# Split fractions (spec: 80/10/10).
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.8, 0.1, 0.1

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Directory names that have appeared for each class across republications.
_CLASS_DIR_ALIASES = {
    "cat": {"cat", "cats", "cat_images"},
    "dog": {"dog", "dogs", "dog_images"},
}


def find_image_root(base: Path = RAW_DIR) -> dict[str, Path]:
    """Locate the per-class image directories under ``base``.

    Returns ``{"cat": Path, "dog": Path}``. Searches by directory name at any
    depth so the function survives the archive being repackaged, and prefers the
    shallowest match when a name appears more than once.
    """
    if not base.exists():
        raise FileNotFoundError(
            f"{base} does not exist. Fetch the dataset first: bash scripts/download.sh"
        )

    found: dict[str, Path] = {}
    for cls, aliases in _CLASS_DIR_ALIASES.items():
        candidates = [
            d for d in base.rglob("*")
            if d.is_dir() and d.name.lower() in aliases
        ]
        if not candidates:
            continue
        # Shallowest, then alphabetical, for determinism.
        found[cls] = min(candidates, key=lambda p: (len(p.parts), str(p)))

    missing = [c for c in CLASS_NAMES if c not in found]
    if missing:
        tree = "\n".join(f"    {d.relative_to(base)}"
                         for d in sorted(base.rglob("*"))[:25] if d.is_dir())
        raise FileNotFoundError(
            f"Could not find image directories for {missing} under {base}.\n"
            f"Directories present (first 25):\n{tree or '    (none)'}"
        )
    return found


def discover_images(base: Path = RAW_DIR) -> list[tuple[Path, int]]:
    """Return ``[(path, label), ...]`` for every image file, sorted for determinism."""
    out: list[tuple[Path, int]] = []
    for cls, d in find_image_root(base).items():
        label = CLASS_INDICES[cls]
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                out.append((p, label))
    return sorted(out, key=lambda t: str(t[0]))


def audit_images(
    items: list[tuple[Path, int]] | None = None,
    write_report: bool = True,
) -> tuple[list[tuple[Path, int]], list[Path]]:
    """Split images into (usable, corrupt).

    Verification is two-pass on purpose: ``Image.verify()`` catches structural
    damage but consumes the file handle and does *not* catch truncation during
    decode, so each candidate is then fully loaded and converted. Zero-byte
    files are rejected up front.
    """
    from PIL import Image

    if items is None:
        items = discover_images()

    good: list[tuple[Path, int]] = []
    bad: list[Path] = []

    for path, label in items:
        try:
            if path.stat().st_size == 0:
                bad.append(path)
                continue
            with Image.open(path) as im:
                im.verify()                     # pass 1: structural
            with Image.open(path) as im:
                im.convert("RGB").load()        # pass 2: full decode
            good.append((path, label))
        except Exception:
            bad.append(path)

    if write_report:
        CORRUPT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# Corrupt / unreadable image files found by src.data.audit_images.\n"
            "# Excluded from the split manifests by src.preprocess (never skipped\n"
            "# at training time, which would fail mid-epoch).\n"
        )
        body = "".join(f"{p.relative_to(ROOT) if ROOT in p.parents else p}\n" for p in bad)
        CORRUPT_REPORT.write_text(header + body)

    return good, bad


def load_corrupt_list() -> set[str]:
    """Return the set of corrupt file paths recorded by a previous audit."""
    if not CORRUPT_REPORT.exists():
        return set()
    return {
        line.strip()
        for line in CORRUPT_REPORT.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def inspect(base: Path = RAW_DIR) -> dict:
    """Print shape/balance/corruption summary. Returns the stats dict."""
    roots = find_image_root(base)
    print("\n=== class directories ===")
    for cls, d in roots.items():
        print(f"  {cls:4s} -> {d.relative_to(base)}")

    items = discover_images(base)
    print(f"\n=== discovered {len(items)} candidate image files ===")

    print("\n=== auditing (verify + full decode; this walks every file) ===")
    good, bad = audit_images(items)

    per_class = {c: 0 for c in CLASS_NAMES}
    for _, label in good:
        per_class[CLASS_NAMES[label]] += 1

    total = len(good)
    print("\n=== usable images per class ===")
    for cls, n in per_class.items():
        pct = 100 * n / total if total else 0
        print(f"  {cls:4s} {n:>6d}  ({pct:.1f}%)")

    print(f"\n=== corrupt / unreadable: {len(bad)} ===")
    for p in bad[:10]:
        print(f"  {p.name}")
    if len(bad) > 10:
        print(f"  ... and {len(bad) - 10} more (full list in {CORRUPT_REPORT.name})")

    suffixes: dict[str, int] = {}
    for p, _ in good:
        suffixes[p.suffix.lower()] = suffixes.get(p.suffix.lower(), 0) + 1
    print(f"\n=== file types === {suffixes}")

    stats = {
        "total_candidates": len(items),
        "usable": total,
        "corrupt": len(bad),
        "per_class": per_class,
        "suffixes": suffixes,
    }
    balance = {c: round(100 * n / total, 1) for c, n in per_class.items()} if total else {}
    print(f"\n[data] usable={total}, corrupt={len(bad)}, balance={balance}")
    return stats


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    inspect()
