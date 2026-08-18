"""Unit tests for the pre-processing functions (M3 — required data test).

``src/preprocess.py`` is the shared train/serve seam, so these are the highest
value tests in the suite: a bug here corrupts training *and* inference at once.

Everything runs off committed fixtures in ``tests/fixtures/``, never the Kaggle
dataset — CI has no ``data/``. Tests that genuinely need the full dataset are
marked ``needs_data`` and skip when it is absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data import (
    CLASS_INDICES,
    CLASS_NAMES,
    IMG_CHANNELS,
    IMG_SIZE,
    RANDOM_STATE,
    TEST_FRAC,
    TRAIN_FRAC,
    audit_images,
)
from src.preprocess import build_split_manifests, load_image

FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------- load_image

def test_resize_shape():
    """A 96x96 fixture must come out at exactly 224x224x3."""
    arr = load_image(FIXTURES / "cat_sample.jpg")
    assert arr.shape == (IMG_SIZE[1], IMG_SIZE[0], IMG_CHANNELS)


def test_output_dtype_and_range():
    """float32 scaled to [0,1] — build_transfer_model's Rescaling depends on it."""
    arr = load_image(FIXTURES / "cat_sample.jpg")
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


@pytest.mark.parametrize("fixture", ["greyscale.jpg", "rgba.png"])
def test_non_rgb_inputs_become_three_channel(fixture):
    """Greyscale (1ch) and RGBA (4ch) must both collapse to 3 channels.

    Without the unconditional convert("RGB") these produce (224,224) and
    (224,224,4), which fail at the model's input layer — and the real dataset
    contains both.
    """
    arr = load_image(FIXTURES / fixture)
    assert arr.shape == (IMG_SIZE[1], IMG_SIZE[0], IMG_CHANNELS)


def test_accepts_raw_bytes():
    """The API hands uploaded bytes straight in, so path and bytes must agree."""
    raw = (FIXTURES / "cat_sample.jpg").read_bytes()
    from_bytes = load_image(raw)
    from_path = load_image(FIXTURES / "cat_sample.jpg")
    assert from_bytes.shape == from_path.shape
    np.testing.assert_allclose(from_bytes, from_path)


def test_deterministic():
    """Same input twice must give an identical array."""
    np.testing.assert_array_equal(
        load_image(FIXTURES / "dog_sample.jpg"),
        load_image(FIXTURES / "dog_sample.jpg"),
    )


@pytest.mark.parametrize("fixture", ["corrupt.jpg", "empty.jpg", "not_an_image.jpg"])
def test_undecodable_inputs_raise_valueerror(fixture):
    """Bad input must raise ValueError, which the API turns into a clean 422."""
    with pytest.raises(ValueError, match="could not decode"):
        load_image(FIXTURES / fixture)


def test_truncated_jpeg_is_rejected_not_padded():
    """Regression test for a bug found on the real dataset.

    Pillow does NOT raise on a truncated JPEG — it emits
    ``UserWarning: Truncated File Read`` and pads the missing scanlines with
    grey. Catching only exceptions therefore let a partially-garbage image
    through into training. ``PetImages/Dog/9041.jpg`` is one such file.
    """
    with pytest.raises(ValueError, match="could not decode"):
        load_image(FIXTURES / "corrupt.jpg")


def test_benign_warnings_do_not_reject_valid_images():
    """Only the truncation warning is escalated, not all warnings.

    Escalating every warning would reject valid images over benign EXIF
    complaints, so this guards the narrowness of that filter.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn("a benign unrelated warning", UserWarning, stacklevel=1)
        arr = load_image(FIXTURES / "cat_sample.jpg")
    assert arr.shape == (IMG_SIZE[1], IMG_SIZE[0], IMG_CHANNELS)


# --------------------------------------------------------------- audit

def test_audit_separates_good_from_corrupt(tmp_path):
    """The audit must catch all four bad fixtures and keep the four good ones."""
    good_names = ["cat_sample.jpg", "dog_sample.jpg", "greyscale.jpg", "rgba.png"]
    bad_names = ["corrupt.jpg", "empty.jpg", "not_an_image.jpg"]
    items = [(FIXTURES / n, 0) for n in good_names + bad_names]

    good, bad = audit_images(items, write_report=False)

    assert {p.name for p, _ in good} == set(good_names)
    assert {p.name for p in bad} == set(bad_names)


# --------------------------------------------------------------- splits

def _fake_items(n_per_class: int = 100):
    """Synthetic (path, label) pairs — split logic needs no real files."""
    return (
        [(Path(f"/fake/cat/{i:05d}.jpg"), 0) for i in range(n_per_class)]
        + [(Path(f"/fake/dog/{i:05d}.jpg"), 1) for i in range(n_per_class)]
    )


@pytest.fixture()
def manifests(tmp_path, monkeypatch):
    """Build split manifests into a temp dir so the real ones aren't clobbered."""
    import src.preprocess as pp

    monkeypatch.setattr(pp, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(pp, "load_corrupt_list", lambda: set())
    return pp.build_split_manifests(items=_fake_items())


def test_split_ratios_are_80_10_10(manifests):
    total = sum(len(v) for v in manifests.values())
    assert total == 200
    assert len(manifests["train"]) / total == pytest.approx(TRAIN_FRAC, abs=0.02)
    assert len(manifests["test"]) / total == pytest.approx(TEST_FRAC, abs=0.02)


def test_splits_are_disjoint(manifests):
    """No file may appear in two splits — this is train/test leakage."""
    sets = {k: {p for p, _ in v} for k, v in manifests.items()}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]


def test_stratification_preserves_class_balance(manifests):
    """Each split keeps the 50/50 source balance, within its own granularity."""
    for split, recs in manifests.items():
        pos = sum(1 for _, y in recs if y == 1) / len(recs)
        tol = max(0.02, 1.0 / len(recs))
        assert pos == pytest.approx(0.5, abs=tol), f"{split} balance {pos:.3f}"


def test_split_is_reproducible(tmp_path, monkeypatch):
    """Same seed must give the same split — RANDOM_STATE is the contract."""
    import src.preprocess as pp

    monkeypatch.setattr(pp, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(pp, "load_corrupt_list", lambda: set())
    a = pp.build_split_manifests(items=_fake_items(), random_state=RANDOM_STATE)
    b = pp.build_split_manifests(items=_fake_items(), random_state=RANDOM_STATE)
    assert a == b


def test_corrupt_files_excluded_from_manifests(tmp_path, monkeypatch):
    """Corrupt files must be dropped at manifest time, not skipped mid-epoch."""
    import src.preprocess as pp

    items = _fake_items(50)
    doomed = {str(items[0][0]), str(items[-1][0])}
    monkeypatch.setattr(pp, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(pp, "load_corrupt_list", lambda: doomed)

    out = pp.build_split_manifests(items=items)
    listed = {p for recs in out.values() for p, _ in recs}
    assert not (listed & doomed)
    assert len(listed) == len(items) - 2


def test_corrupt_exclusion_matches_full_path_not_basename(tmp_path, monkeypatch):
    """Regression test for a bug found on the real dataset.

    PetImages numbers files per class, so ``Cat/9041.jpg`` and ``Dog/9041.jpg``
    both exist. Matching the corrupt list on bare filenames discarded the healthy
    cat image alongside the truncated dog one, silently losing a good sample and
    skewing the class balance.
    """
    import src.preprocess as pp

    # Needs enough members per class for a stratified split to be possible;
    # the collision pair is what the test is actually about.
    items = (
        [(Path("/fake/PetImages/Cat/9041.jpg"), 0), (Path("/fake/PetImages/Dog/9041.jpg"), 1)]
        + [(Path(f"/fake/PetImages/Cat/{i:05d}.jpg"), 0) for i in range(30)]
        + [(Path(f"/fake/PetImages/Dog/{i:05d}.jpg"), 1) for i in range(30)]
    )
    monkeypatch.setattr(pp, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(pp, "load_corrupt_list",
                        lambda: {"/fake/PetImages/Dog/9041.jpg"})

    out = pp.build_split_manifests(items=items)
    listed = {p for recs in out.values() for p, _ in recs}

    assert "/fake/PetImages/Dog/9041.jpg" not in listed, "corrupt file was not excluded"
    assert "/fake/PetImages/Cat/9041.jpg" in listed, (
        "healthy Cat/9041.jpg was discarded because it shares a basename "
        "with the corrupt Dog/9041.jpg"
    )
    assert len(listed) == len(items) - 1


def test_empty_input_raises():
    with pytest.raises(ValueError, match="no usable images"):
        build_split_manifests(items=[])


# --------------------------------------------------------------- constants

def test_img_size_matches_spec():
    """The assignment mandates 224x224 RGB."""
    assert IMG_SIZE == (224, 224)
    assert IMG_CHANNELS == 3


def test_class_indices_are_consistent():
    """cat=0, dog=1 — the API's label mapping depends on this ordering."""
    assert CLASS_NAMES == ["cat", "dog"]
    assert CLASS_INDICES == {"cat": 0, "dog": 1}
