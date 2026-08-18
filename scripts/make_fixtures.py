"""Generate a synthetic stand-in dataset and the committed test fixtures.

Two jobs:

  1. ``--fixtures`` writes the small images ``tests/fixtures/`` needs so the test
     suite (and therefore CI) runs **without** the 25k-image Kaggle download.
     CI has no ``data/``, so fixtures are the only images it ever sees.

  2. ``--synthetic N`` writes a fake ``data/raw`` tree with N images per class.
     This is *not* training data — it exists so the full pipeline (manifests,
     tf.data, augmentation, model fit, save/load, API, container) can be
     validated end-to-end before the real dataset is available, and so a
     pipeline bug is never confused with a data problem.

The synthetic classes are linearly separable by colour channel on purpose: a
correct pipeline should reach near-perfect accuracy on them within an epoch or
two, which makes this a sharp smoke test. Real accuracy numbers only ever come
from the real dataset.

    python -m scripts.make_fixtures --fixtures
    python -m scripts.make_fixtures --synthetic 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "fixtures"
SYNTH_ROOT = ROOT / "data" / "raw" / "SyntheticPetImages"
RNG_SEED = 42


def _blobby(rgb_bias: tuple[int, int, int], size=(224, 224), seed=0) -> Image.Image:
    """A textured image biased toward one colour channel, with soft blobs.

    Not noise: blobs give the convolutions actual spatial structure to learn,
    so a working model separates the classes and a broken pipeline does not.
    """
    rng = np.random.default_rng(seed)
    w, h = size
    arr = rng.integers(60, 120, size=(h, w, 3), dtype=np.int16)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(rng.integers(3, 7)):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        r = rng.integers(h // 10, h // 3)
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < r * r
        arr[mask] += rng.integers(20, 60)
    arr += np.array(rgb_bias, dtype=np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), mode="RGB")


def make_fixtures() -> None:
    """Write the committed test fixtures, including deliberate edge cases."""
    FIXTURES.mkdir(parents=True, exist_ok=True)

    # NOTE: cat_sample.jpg and dog_sample.jpg are NOT generated here. They are
    # real 160x160 crops from the dataset's test split, committed to git.
    #
    # They used to be synthetic colour blobs, which was a mistake: a model trained
    # on real pets classified both as "dog", so fixture-based tests could only
    # assert that two probabilities differed — not that the predicted LABEL was
    # correct. Real fixtures let tests catch an inverted class mapping.
    #
    # To regenerate them, see the snippet in tracker/DAILY_LOG.md (Day 6); it picks
    # the most confidently-correct example per class from data/processed/test.csv.
    if not (FIXTURES / "cat_sample.jpg").exists():
        print("[fixtures] WARNING: cat_sample.jpg/dog_sample.jpg missing — these are"
              " real dataset crops, not generated. See tracker/DAILY_LOG.md (Day 6).")

    # Edge case 1: greyscale (1 channel) must survive the RGB conversion.
    _blobby((0, 0, 0), size=(96, 96), seed=3).convert("L").save(FIXTURES / "greyscale.jpg")

    # Edge case 2: RGBA (4 channels) must also collapse to 3.
    rgba = _blobby((0, 60, 0), size=(96, 96), seed=4).convert("RGBA")
    rgba.save(FIXTURES / "rgba.png")

    # Edge case 3: truncated JPEG — the failure mode this dataset is known for.
    # Derived from a generated image so it does not depend on the real fixtures.
    _blobby((70, 0, 0), size=(96, 96), seed=1).save(FIXTURES / "_trunc_src.jpg", quality=80)
    good = (FIXTURES / "_trunc_src.jpg").read_bytes()
    (FIXTURES / "_trunc_src.jpg").unlink()
    (FIXTURES / "corrupt.jpg").write_bytes(good[: len(good) // 3])

    # Edge case 4: zero-byte file.
    (FIXTURES / "empty.jpg").write_bytes(b"")

    # Edge case 5: not an image at all, despite the extension.
    (FIXTURES / "not_an_image.jpg").write_bytes(b"this is plain text, not a JPEG\n")

    for p in sorted(FIXTURES.iterdir()):
        print(f"[fixtures] {p.relative_to(ROOT)} ({p.stat().st_size} bytes)")


def make_synthetic(per_class: int) -> None:
    """Write a fake data/raw tree: cat=red-biased, dog=blue-biased."""
    if SYNTH_ROOT.exists():
        import shutil
        shutil.rmtree(SYNTH_ROOT)

    for cls, bias in (("Cat", (70, 0, 0)), ("Dog", (0, 0, 70))):
        d = SYNTH_ROOT / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            _blobby(bias, size=(224, 224), seed=hash((cls, i)) % 10_000).save(
                d / f"{cls.lower()}_{i:05d}.jpg", quality=85
            )
        print(f"[synthetic] wrote {per_class} images -> {d.relative_to(ROOT)}")

    # Plant two corrupt files so the audit has something real to catch.
    victim = SYNTH_ROOT / "Cat" / "cat_00000.jpg"
    data = victim.read_bytes()
    (SYNTH_ROOT / "Cat" / "cat_truncated.jpg").write_bytes(data[: len(data) // 3])
    (SYNTH_ROOT / "Dog" / "dog_empty.jpg").write_bytes(b"")
    print("[synthetic] planted 2 corrupt files (truncated + zero-byte)")

    # Marker so scripts/download.sh can tell fixture data from the real dataset.
    # Without it, download.sh's "already populated" guard counts these images and
    # skips the real download — and worse, a training run could silently fit on
    # synthetic images and report meaningless accuracy.
    (SYNTH_ROOT.parent / ".synthetic").write_text(
        "This data/raw tree holds SYNTHETIC smoke-test images from\n"
        "scripts/make_fixtures.py --synthetic, NOT the Kaggle dataset.\n"
        "scripts/download.sh replaces it when fetching the real data.\n"
    )
    print(f"[synthetic] wrote marker {(SYNTH_ROOT.parent / '.synthetic').relative_to(ROOT)}")
    print("[synthetic] NOTE: this is a pipeline smoke fixture, NOT training data.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures", action="store_true", help="write tests/fixtures/")
    ap.add_argument("--synthetic", type=int, metavar="N",
                    help="write N synthetic images per class into data/raw/")
    a = ap.parse_args()
    if not a.fixtures and not a.synthetic:
        ap.error("pass --fixtures and/or --synthetic N")
    if a.fixtures:
        make_fixtures()
    if a.synthetic:
        make_synthetic(a.synthetic)
    return 0


if __name__ == "__main__":
    sys.exit(main())
