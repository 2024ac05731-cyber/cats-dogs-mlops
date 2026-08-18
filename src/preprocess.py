"""Pre-processing: 224x224 RGB, stratified 80/10/10 splits, augmentation.

This module is the **shared seam between training and serving**. The same
``load_image`` runs in ``src.train`` and in ``api.main`` via ``src.predict``, so
train and serve cannot drift apart. A bug here breaks both, which is exactly why
``tests/test_preprocess.py`` targets it (M3's required pre-processing test).

Design notes:
  * Splits are built as **file manifests** (CSV), not by copying images. Cheap,
    inspectable, and DVC-friendly.
  * Corrupt files from the Day-1 audit are excluded at manifest-build time, so a
    training run can never hit one mid-epoch.
  * Augmentation is a Keras layer stack gated behind ``augment=True``, so it is
    structurally impossible for it to touch val/test.

Run:
    python -m src.preprocess
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from src.data import (
    CLASS_NAMES,
    IMG_CHANNELS,
    IMG_SIZE,
    PROCESSED_DIR,
    RANDOM_STATE,
    ROOT,
    TEST_FRAC,
    TRAIN_FRAC,
    audit_images,
    discover_images,
    load_corrupt_list,
)

SPLITS = ("train", "val", "test")
BATCH_SIZE = 32
AUTOTUNE = -1  # replaced with tf.data.AUTOTUNE lazily, to keep import cheap


def manifest_path(split: str) -> Path:
    return PROCESSED_DIR / f"{split}.csv"


def _display(path: Path) -> str:
    """Repo-relative path for logging, falling back to absolute.

    ``Path.relative_to`` raises when the target is outside ROOT, which happens
    whenever PROCESSED_DIR is redirected (tests, or a caller writing elsewhere).
    A cosmetic log line must not be able to abort manifest building.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------- image loading

def load_image(source, target_size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Load one image as a float32 ``(H, W, 3)`` array scaled to [0, 1].

    ``source`` may be a path or raw bytes, so the serving path can hand an
    uploaded file straight in. Conversion to RGB is unconditional: the dataset
    contains greyscale and RGBA files, and without it their channel counts would
    be 1 or 4 and break the model's input shape.

    Raises ``ValueError`` on anything that cannot be decoded, so the API can turn
    that into a clean 422 instead of a stack trace.

    Truncated JPEGs are treated as undecodable. Pillow does not raise on them —
    it warns and pads the missing scanlines with grey — so the truncation warning
    is escalated to an error here. Only that warning is escalated; escalating all
    of them would reject valid images over benign EXIF complaints. This keeps the
    serving path consistent with ``src.data.audit_images``, so an image the audit
    rejected can't be silently accepted at inference time.
    """
    import io
    import warnings

    from PIL import Image

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*[Tt]runcated.*")
            if isinstance(source, (bytes, bytearray)):
                im = Image.open(io.BytesIO(bytes(source)))
            else:
                im = Image.open(source)
            with im:
                im = im.convert("RGB")                      # greyscale/RGBA -> 3 channels
                im = im.resize(target_size, Image.BILINEAR)
                arr = np.asarray(im, dtype="float32") / 255.0
    except Exception as exc:   # any Pillow failure -> a clean ValueError for the API
        raise ValueError(f"could not decode image: {exc}") from exc

    if arr.shape != (target_size[1], target_size[0], IMG_CHANNELS):
        raise ValueError(f"unexpected shape {arr.shape}, want "
                         f"{(target_size[1], target_size[0], IMG_CHANNELS)}")
    return arr


# --------------------------------------------------------------- split manifests

def build_split_manifests(
    items: list[tuple[Path, int]] | None = None,
    random_state: int = RANDOM_STATE,
    run_audit: bool = False,
) -> dict[str, list[tuple[str, int]]]:
    """Stratified 80/10/10 split, written to ``data/processed/{split}.csv``.

    Stratification is done per class so each split carries the same class
    balance. Corrupt files are removed first — either from a previous audit
    report, or by running the audit inline when ``run_audit=True``.
    """
    from sklearn.model_selection import train_test_split

    if items is None:
        items = discover_images()

    if run_audit:
        items, _ = audit_images(items)
    else:
        corrupt = load_corrupt_list()
        if corrupt:
            # Match on the repo-relative path ONLY, never the bare filename.
            # PetImages numbers files per class, so Cat/9041.jpg and Dog/9041.jpg
            # both exist — a basename match discarded the healthy cat image along
            # with the truncated dog one.
            def is_corrupt(p: Path) -> bool:
                rel = str(p.relative_to(ROOT)) if ROOT in p.parents else str(p)
                return rel in corrupt
            items = [(p, y) for p, y in items if not is_corrupt(p)]

    if not items:
        raise ValueError("no usable images — run scripts/download.sh and python -m src.data first")

    paths = [str(p) for p, _ in items]
    labels = [y for _, y in items]

    # 80 / 20, then split the 20 evenly into val / test -> 80 / 10 / 10.
    train_p, rest_p, train_y, rest_y = train_test_split(
        paths, labels,
        train_size=TRAIN_FRAC, stratify=labels, random_state=random_state,
    )
    rel_test = TEST_FRAC / (1.0 - TRAIN_FRAC)   # 0.1 / 0.2 = 0.5
    val_p, test_p, val_y, test_y = train_test_split(
        rest_p, rest_y,
        test_size=rel_test, stratify=rest_y, random_state=random_state,
    )

    out = {
        "train": list(zip(train_p, train_y, strict=True)),
        "val": list(zip(val_p, val_y, strict=True)),
        "test": list(zip(test_p, test_y, strict=True)),
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split, recs in out.items():
        with manifest_path(split).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["filepath", "label"])
            w.writerows(recs)
        print(f"[preprocess] wrote {_display(manifest_path(split))} ({len(recs)} rows)")

    return out


def read_manifest(split: str) -> tuple[list[str], list[int]]:
    """Return ``(filepaths, labels)`` for a split."""
    p = manifest_path(split)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing. Build the manifests with `python -m src.preprocess`."
        )
    paths, labels = [], []
    with p.open() as fh:
        for row in csv.DictReader(fh):
            paths.append(row["filepath"])
            labels.append(int(row["label"]))
    return paths, labels


# --------------------------------------------------------------- augmentation

def build_augmentation():
    """Keras layer stack for training-time augmentation.

    Chosen for cat-vs-dog specifically (ADR-010): horizontal flip is safe because
    a mirrored pet is still the same species; small rotation and zoom mimic
    framing variation in user-submitted photos; mild contrast covers lighting.
    Deliberately no vertical flip — upside-down pets do not occur in adoption
    listings and would only add label-preserving noise the model must waste
    capacity on.

    The trailing clip is not cosmetic: ``RandomContrast`` and ``RandomZoom`` can
    push values outside [0, 1] (measured max 1.02 on the smoke fixture), and
    ``build_transfer_model``'s Rescaling layer assumes a clean [0, 1] input when
    mapping to MobileNetV2's [-1, 1] range. Clipping keeps the augmented
    distribution inside the contract the models were built against.

    Applied in the ``tf.data`` pipeline, never as a model layer, so it is not
    serialized into the served artifact.
    """
    import keras

    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=RANDOM_STATE),
            keras.layers.RandomRotation(0.10, seed=RANDOM_STATE),
            keras.layers.RandomZoom(0.10, seed=RANDOM_STATE),
            keras.layers.RandomContrast(0.10, seed=RANDOM_STATE),
            keras.layers.Lambda(lambda t: keras.ops.clip(t, 0.0, 1.0), name="clip_to_unit"),
        ],
        name="augmentation",
    )


# --------------------------------------------------------------- tf.data pipeline

def make_dataset(
    split: str | None = None,
    paths: list[str] | None = None,
    labels: list[int] | None = None,
    batch_size: int = BATCH_SIZE,
    augment: bool = False,
    shuffle: bool | None = None,
):
    """Build a batched, prefetched ``tf.data.Dataset`` of ``(image, label)``.

    Either pass ``split`` to read a manifest, or pass ``paths``/``labels``
    directly (cross-validation needs the latter, since its folds don't
    correspond to any manifest on disk).
    """
    import tensorflow as tf

    if paths is None or labels is None:
        if split is None:
            raise ValueError("pass either split= or both paths= and labels=")
        paths, labels = read_manifest(split)

    if shuffle is None:
        shuffle = augment or split == "train"

    def _decode(path, label):
        def _py(p):
            return load_image(p.numpy().decode())
        img = tf.py_function(_py, [path], tf.float32)
        img.set_shape((IMG_SIZE[1], IMG_SIZE[0], IMG_CHANNELS))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, list(labels)))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 2048), seed=RANDOM_STATE,
                        reshuffle_each_iteration=True)
    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)

    if augment:
        aug = build_augmentation()
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------- smoke test

def _smoke_test() -> None:
    """Build manifests, check ratios/stratification/disjointness, pull one batch."""
    out = build_split_manifests()
    total = sum(len(v) for v in out.values())

    print(f"\n[smoke] total usable images: {total}")
    for split, recs in out.items():
        pct = 100 * len(recs) / total
        pos = 100 * sum(1 for _, y in recs if y == 1) / len(recs)
        print(f"[smoke] {split:5s} n={len(recs):>6d} ({pct:5.1f}%)  "
              f"{CLASS_NAMES[1]}={pos:.1f}%")

    sets = {s: {p for p, _ in recs} for s, recs in out.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        assert not overlap, f"LEAKAGE: {len(overlap)} files shared by {a} and {b}"
    print("[smoke] splits are disjoint, OK")

    ds = make_dataset("train", batch_size=4, augment=True)
    x, y = next(iter(ds))
    print(f"[smoke] batch x={tuple(x.shape)} dtype={x.dtype.name} "
          f"range=[{float(x.numpy().min()):.3f}, {float(x.numpy().max()):.3f}]")
    print(f"[smoke] batch y={y.numpy().tolist()}")
    assert tuple(x.shape)[1:] == (IMG_SIZE[1], IMG_SIZE[0], IMG_CHANNELS)
    print("[smoke] PASS")


if __name__ == "__main__":
    _smoke_test()
