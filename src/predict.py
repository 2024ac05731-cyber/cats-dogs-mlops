"""Load the packaged model and score images.

The serving artifact is a single `.h5` file holding the full Keras model,
produced by ``python -m src.train``. Loading it needs only TensorFlow — not
MLflow, DVC, or the dataset — which is what keeps `requirements-serve.txt`
small and the container self-contained.

Pre-processing deliberately reuses ``src.preprocess.load_image``, the same
function training uses. That shared seam is why train and serve cannot drift:
if the resize, the RGB conversion, or the [0,1] scaling changed here but not
there, every prediction would be subtly wrong while every test still passed.

    from src.predict import load_model, predict_image
    model = load_model()
    predict_image(open("cat.jpg", "rb").read(), model)

Run a self-check against the bundled fixtures:
    python -m src.predict
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from src.data import CLASS_NAMES, ROOT
from src.preprocess import load_image

MODEL_PATH = ROOT / "models" / "model.h5"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

DEFAULT_THRESHOLD = 0.5


def load_model(path: Path = MODEL_PATH):
    """Load the packaged Keras model.

    Raises ``FileNotFoundError`` with the command that produces the artifact,
    rather than a bare Keras error — the API turns this into a clean 503.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Build it with `python -m src.train --model transfer`."
        )
    import keras

    return keras.models.load_model(path)


def load_metadata(path: Path = METADATA_PATH) -> dict:
    """Return the model card. Empty dict if absent, so the API can still serve."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def predict_proba(source, model) -> float:
    """Return P(dog) for one image (path or raw bytes).

    ``load_image`` raises ``ValueError`` on undecodable input, including
    truncated JPEGs — see its docstring for why that case needs special
    handling. Callers should let that propagate as a 4xx, not a 500.
    """
    arr = load_image(source)
    return float(model.predict(arr[None, ...], verbose=0).ravel()[0])


def predict_image(source, model=None, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Score one image and return a JSON-ready result.

    ``probabilities`` carries both classes explicitly rather than only P(dog).
    The spec asks for "class probabilities/label", and a single sigmoid output is
    easy to misread — returning both makes the response self-describing.
    """
    if model is None:
        model = load_model()

    p_dog = predict_proba(source, model)
    p_cat = 1.0 - p_dog
    idx = int(p_dog >= threshold)

    return {
        "prediction": idx,
        "label": CLASS_NAMES[idx],
        "probabilities": {
            CLASS_NAMES[0]: round(p_cat, 6),
            CLASS_NAMES[1]: round(p_dog, 6),
        },
        "confidence": round(max(p_cat, p_dog), 6),
        "threshold": threshold,
    }


def predict_batch(sources, model=None, threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """Score several images in one forward pass.

    Used by ``scripts/replay_batch.py`` for the M5 post-deployment check. Batched
    on purpose: one ``predict`` call over N images is markedly faster than N
    calls, and the per-image path is already covered by ``predict_image``.
    """
    if model is None:
        model = load_model()

    arrays, errors = [], {}
    for i, src in enumerate(sources):
        try:
            arrays.append(load_image(src))
        except ValueError as exc:
            errors[i] = str(exc)
            arrays.append(np.zeros((224, 224, 3), dtype="float32"))

    proba = model.predict(np.stack(arrays), verbose=0).ravel()

    out = []
    for i, p_dog in enumerate(proba):
        if i in errors:
            out.append({"error": errors[i]})
            continue
        p_dog = float(p_dog)
        p_cat = 1.0 - p_dog
        idx = int(p_dog >= threshold)
        out.append({
            "prediction": idx,
            "label": CLASS_NAMES[idx],
            "probabilities": {CLASS_NAMES[0]: round(p_cat, 6),
                              CLASS_NAMES[1]: round(p_dog, 6)},
            "confidence": round(max(p_cat, p_dog), 6),
        })
    return out


def model_summary() -> dict:
    """Compact description of what is serving, for the API's ``GET /`` route."""
    meta = load_metadata()
    cv = meta.get("cross_validation", {})
    return {
        "model_family": meta.get("model_family"),
        "artifact_format": meta.get("artifact_format"),
        "input_shape": meta.get("input_shape"),
        "class_names": meta.get("class_names", CLASS_NAMES),
        "test_accuracy": meta.get("metrics", {}).get("test", {}).get("accuracy"),
        "test_roc_auc": meta.get("metrics", {}).get("test", {}).get("roc_auc"),
        "cv_accuracy_mean": cv.get("accuracy_mean"),
        "tensorflow": meta.get("package_versions", {}).get("tensorflow"),
    }


if __name__ == "__main__":
    fixtures = ROOT / "tests" / "fixtures"
    model = load_model()
    meta = load_metadata()
    print(f"[predict] model: {meta.get('model_family')} "
          f"(tensorflow {meta.get('package_versions', {}).get('tensorflow')})")

    for name in ("cat_sample.jpg", "dog_sample.jpg"):
        r = predict_image(fixtures / name, model)
        print(f"[predict] {name:18s} -> {r['label']:4s} "
              f"cat={r['probabilities']['cat']:.4f} dog={r['probabilities']['dog']:.4f}")

    # Undecodable input must raise, not return a wrong answer.
    try:
        predict_image(fixtures / "corrupt.jpg", model)
        print("[predict] FAIL: corrupt image did not raise")
        raise SystemExit(1)
    except ValueError as exc:
        print(f"[predict] corrupt.jpg correctly rejected: {str(exc)[:52]}")

    print("[predict] PASS")
