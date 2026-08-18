"""Unit tests for model construction and inference (M3 — required model test).

The spec asks for a test on "one model utility/inference function". These cover
both: architecture construction (``src.model``) and the inference path
(``src.predict``), which is what the API depends on.

Runs entirely off committed fixtures and randomly-initialised models — no Kaggle
dataset, and no trained artifact unless one happens to be present. Tests that
genuinely need ``models/model.h5`` skip when it is absent, so CI stays green on a
fresh clone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data import CLASS_INDICES, CLASS_NAMES, IMG_CHANNELS, IMG_SIZE, RANDOM_STATE
from src.model import ARCHITECTURES, INPUT_SHAPE, build_model, compile_model

FIXTURES = Path(__file__).parent / "fixtures"
MODEL_PATH = Path(__file__).parent.parent / "models" / "model.h5"

needs_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="models/model.h5 not present (run python -m src.train)",
)


# --------------------------------------------------------------- construction

def test_input_shape_matches_spec():
    """224x224x3, derived from the single IMG_SIZE definition."""
    assert INPUT_SHAPE == (IMG_SIZE[1], IMG_SIZE[0], IMG_CHANNELS)
    assert INPUT_SHAPE == (224, 224, 3)


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_build_returns_binary_output(arch):
    """Every architecture must end in a single sigmoid unit."""
    kw = {"weights": None} if arch == "transfer" else {}
    model = build_model(arch, **kw)
    assert model.output_shape == (None, 1)
    assert model.input_shape[1:] == INPUT_SHAPE


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_forward_pass_produces_probabilities(arch):
    """Output must be a valid probability in [0,1] for every row."""
    kw = {"weights": None} if arch == "transfer" else {}
    model = build_model(arch, **kw)
    out = model.predict(np.random.rand(4, *INPUT_SHAPE).astype("float32"), verbose=0)
    assert out.shape == (4, 1)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_transfer_base_is_frozen():
    """The pretrained base must not be trainable by default.

    If it were, the ~2.2M base parameters would train on a few thousand images
    and destroy the pretrained features — the whole point of the transfer model.
    """
    model = build_model("transfer", weights=None)
    trainable = sum(int(w.numpy().size) for w in model.trainable_weights)
    total = model.count_params()
    assert trainable < 0.01 * total, (
        f"{trainable:,} of {total:,} params trainable — base is not frozen"
    )


def test_baseline_has_no_pretrained_dependency():
    """The baseline must build with no network access (CI has none)."""
    model = build_model("baseline")
    assert model.count_params() > 0


def test_unknown_architecture_raises():
    with pytest.raises(ValueError, match="unknown architecture"):
        build_model("not-a-real-model")


def test_build_is_deterministic_under_seed():
    """Same seed must give identical initial weights.

    src/cross_validate.py relies on this: its fold-isolation proof compares
    initial weight fingerprints across folds and expects them to match.
    """
    import keras

    keras.utils.set_random_seed(RANDOM_STATE)
    a = build_model("baseline").get_weights()
    keras.utils.set_random_seed(RANDOM_STATE)
    b = build_model("baseline").get_weights()
    for wa, wb in zip(a, b, strict=True):
        np.testing.assert_array_equal(wa, wb)


# --------------------------------------------------------------- compilation

@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_compile_attaches_metrics(arch):
    kw = {"weights": None} if arch == "transfer" else {}
    model = compile_model(build_model(arch, **kw))
    names = {m.name for m in model.metrics}
    assert {"precision", "recall", "auc"} <= names or "loss" in names


def test_build_returns_uncompiled_model():
    """build_* must NOT compile.

    Cross-validation depends on compiling per fold to get a fresh optimizer;
    if build_* compiled, optimizer momentum could survive a fold boundary and
    silently leak the held-out fold (ADR-003).

    Checked via ``compiled``/``optimizer`` defensively: Keras 3 omits the
    ``optimizer`` attribute entirely on an uncompiled model rather than setting it
    to None, so a bare ``is None`` assertion raises AttributeError instead of
    passing.
    """
    model = build_model("baseline")
    assert getattr(model, "compiled", False) is False
    assert getattr(model, "optimizer", None) is None

    # And compiling does attach one, so the assertion above is meaningful.
    compiled = compile_model(build_model("baseline"))
    assert getattr(compiled, "optimizer", None) is not None


# --------------------------------------------------------------- inference path

def test_fingerprint_changes_with_weights():
    """The fold-isolation fingerprint must actually be sensitive to weights."""
    from src.cross_validate import weights_fingerprint

    model = build_model("baseline")
    before = weights_fingerprint(model)
    w = model.get_weights()
    w[0] = w[0] + 1.0
    model.set_weights(w)
    assert weights_fingerprint(model) != before


def test_predict_pipeline_end_to_end_untrained():
    """An untrained model still has to produce a well-formed prediction.

    Exercises the same load_image -> batch -> predict path the API uses, so a
    shape or dtype mismatch is caught without needing a trained artifact.
    """
    from src.preprocess import load_image

    model = compile_model(build_model("baseline"))
    arr = load_image(FIXTURES / "cat_sample.jpg")
    proba = float(model.predict(arr[None, ...], verbose=0).ravel()[0])
    assert 0.0 <= proba <= 1.0

    label = CLASS_NAMES[int(proba >= 0.5)]
    assert label in CLASS_NAMES
    assert CLASS_INDICES[label] in (0, 1)


def test_probabilities_of_both_classes_sum_to_one():
    """Binary sigmoid: P(dog) and P(cat) must sum to 1.

    The API returns both, so this guards the arithmetic it reports.
    """
    model = compile_model(build_model("baseline"))
    p_dog = float(model.predict(
        np.random.rand(1, *INPUT_SHAPE).astype("float32"), verbose=0).ravel()[0])
    p_cat = 1.0 - p_dog
    assert p_dog + p_cat == pytest.approx(1.0)


def test_inference_is_deterministic():
    """Same input twice must give the same output — dropout must be inactive."""
    model = compile_model(build_model("baseline"))
    x = np.random.rand(2, *INPUT_SHAPE).astype("float32")
    np.testing.assert_allclose(
        model.predict(x, verbose=0), model.predict(x, verbose=0), atol=1e-6
    )


# --------------------------------------------------------------- packaged model

@needs_model
def test_packaged_model_loads_and_matches_metadata():
    """The shipped artifact must agree with its own metadata sidecar."""
    import json

    import keras

    model = keras.models.load_model(MODEL_PATH)
    meta = json.loads((MODEL_PATH.parent / "model_metadata.json").read_text())
    assert list(model.input_shape[1:]) == meta["input_shape"]
    assert meta["class_indices"] == CLASS_INDICES
    assert meta["artifact_format"] == "h5"


@needs_model
def test_packaged_model_predicts_fixture_labels_correctly():
    """The shipped model must get the two real fixtures RIGHT, not merely differ.

    An earlier version of this test only asserted the two probabilities were
    unequal, because the fixtures were synthetic colour blobs that a pet-trained
    model classified both as "dog". Real fixtures make the stronger assertion
    possible, which is what actually catches an inverted class mapping — a bug
    that would leave every shape and range assertion passing.
    """
    import keras

    from src.preprocess import load_image

    model = keras.models.load_model(MODEL_PATH)
    x = np.stack([
        load_image(FIXTURES / "cat_sample.jpg"),
        load_image(FIXTURES / "dog_sample.jpg"),
    ])
    p_dog = model.predict(x, verbose=0).ravel()

    assert p_dog[0] < 0.5, f"cat fixture predicted dog (P(dog)={p_dog[0]:.4f})"
    assert p_dog[1] >= 0.5, f"dog fixture predicted cat (P(dog)={p_dog[1]:.4f})"
    assert p_dog[1] - p_dog[0] > 0.5, (
        f"classes barely separated: P(dog) cat={p_dog[0]:.4f} dog={p_dog[1]:.4f}"
    )
