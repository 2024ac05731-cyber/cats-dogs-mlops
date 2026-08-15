"""Model architectures for the Cats vs Dogs classifier.

Two architectures, so cross-validation is a real selection instrument rather than
a formality (ADR-003):

  * ``build_baseline_cnn`` — a small CNN written from scratch. Satisfies the
    spec's "at least one baseline model".
  * ``build_transfer_model`` — MobileNetV2 with a frozen ImageNet base. Chosen
    partly *because* it converges in few epochs, which is what makes 5-fold CV
    affordable on CPU.

Both return an **uncompiled** model. Compilation happens in the caller, because
cross-validation must recompile per fold to get fresh optimizer state — if the
optimizer were baked in here, momentum would leak across folds and silently
invalidate the whole exercise.

Run:
    python -m src.model
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.data import IMG_CHANNELS, IMG_SIZE, RANDOM_STATE

INPUT_SHAPE = (IMG_SIZE[1], IMG_SIZE[0], IMG_CHANNELS)
ARCHITECTURES = ("baseline", "transfer")


def build_baseline_cnn(input_shape: tuple[int, int, int] = INPUT_SHAPE):
    """A small from-scratch CNN.

    Four Conv-BN-Pool blocks with widths 32->64->128->128. Design choices
    (ADR-009):
      * BatchNorm after each conv — the inputs are only [0,1]-scaled, so
        normalising activations keeps the deeper blocks trainable at a
        reasonable learning rate.
      * GlobalAveragePooling instead of Flatten — collapses 14x14x128 to 128
        features, which cuts the head from ~3.2M parameters to ~16k and is the
        single biggest overfitting guard on a dataset this size.
      * One dropout before the classifier, sigmoid output for binary.
    """
    import keras

    keras.utils.set_random_seed(RANDOM_STATE)
    layers = keras.layers

    model = keras.Sequential(name="baseline_cnn")
    model.add(layers.Input(shape=input_shape))
    for width in (32, 64, 128, 128):
        model.add(layers.Conv2D(width, 3, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.3, seed=RANDOM_STATE))
    model.add(layers.Dense(1, activation="sigmoid", name="prediction"))
    return model


def build_transfer_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    trainable_base: bool = False,
    weights: str | None = "imagenet",
):
    """MobileNetV2 with an ImageNet-pretrained, frozen base and a custom head.

    The base expects inputs in [-1, 1] while our pipeline yields [0, 1], so a
    Rescaling layer bridges the two *inside the model*. That keeps the
    preprocessing contract identical for both architectures — the serving path
    does not need to know which model it loaded, which is what lets
    ``src.predict`` stay architecture-agnostic.

    ``weights="imagenet"`` triggers a one-time ~9MB download from
    ``storage.googleapis.com``, cached under ``~/.keras/models/``. Pass
    ``weights=None`` to build the same architecture with random initialisation —
    useful for validating the pipeline on a network that blocks the download,
    but useless for real accuracy.
    """
    import keras

    keras.utils.set_random_seed(RANDOM_STATE)
    layers = keras.layers

    try:
        base = keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights=weights,
        )
    except Exception as exc:
        if weights is None:
            raise
        raise RuntimeError(
            "Could not fetch MobileNetV2 ImageNet weights.\n"
            f"  cause: {exc}\n\n"
            "This is a network problem, not a code problem. Keras downloads them "
            "from storage.googleapis.com on first use and caches them in "
            "~/.keras/models/.\n"
            "Options:\n"
            "  1. Run once on an unrestricted network to populate the cache.\n"
            "  2. Download the weights file by hand into ~/.keras/models/.\n"
            "  3. Pass weights=None to build the architecture untrained "
            "(pipeline validation only — accuracy will be meaningless).\n"
            "  4. Use --model baseline, which needs no pretrained weights."
        ) from exc

    base.trainable = trainable_base

    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="to_mobilenet_range")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2, seed=RANDOM_STATE)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2_transfer")


def build_model(architecture: str = "transfer", **kw):
    """Dispatch by name. Used by train.py and cross_validate.py."""
    if architecture == "baseline":
        return build_baseline_cnn(**kw)
    if architecture == "transfer":
        return build_transfer_model(**kw)
    raise ValueError(f"unknown architecture {architecture!r}, want one of {ARCHITECTURES}")


def compile_model(model, learning_rate: float = 1e-3):
    """Compile for binary classification.

    Kept separate from ``build_*`` so cross-validation can construct a *fresh*
    optimizer per fold. See ADR-003.
    """
    import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


if __name__ == "__main__":
    import sys

    # Allow validating the architecture without the ImageNet download:
    #   python -m src.model --no-pretrained
    weights = None if "--no-pretrained" in sys.argv else "imagenet"

    for arch in ARCHITECTURES:
        kw = {"weights": weights} if arch == "transfer" else {}
        try:
            m = compile_model(build_model(arch, **kw))
        except RuntimeError as exc:
            print(f"[model] {arch:9s} SKIPPED\n{exc}")
            continue
        trainable = sum(int(w.numpy().size) for w in m.trainable_weights)
        tag = "" if arch == "baseline" else f" weights={weights}"
        print(f"[model] {arch:9s} output={m.output_shape} "
              f"params={m.count_params():,} trainable={trainable:,}{tag}")
    print("[model] PASS")
