"""Train, evaluate, and package a Cats vs Dogs classifier.

Fits one architecture on the train split, validates on val, evaluates on the
held-out test split, writes the figures, and packages ``models/model.h5`` plus a
``model_metadata.json`` sidecar.

The sidecar is the contract between training and serving: it records the package
versions the model was fitted under, the class-index mapping, the input shape,
and the metrics. ``api/main.py`` reports from it and ``scripts/daily_audit.py``
cross-checks it against ``requirements.txt`` (check 18), which is how a
train/serve version skew gets caught before it reaches a container.

Cross-validation lives in ``src/cross_validate.py``, deliberately separate — CV
is a graded deliverable in its own right, not a side effect of fitting
(GUARDRAILS.md, and the reason Assignment 1 lost the mark).

Run:
    python -m src.train --model transfer
    python -m src.train --model baseline --epochs 15
    python -m src.train --model transfer --subset 4000   # quick pipeline check
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import (
    CLASS_INDICES,
    CLASS_NAMES,
    IMG_SIZE,
    RANDOM_STATE,
    ROOT,
    warn_if_synthetic,
)
from src.model import ARCHITECTURES, build_model, compile_model
from src.preprocess import BATCH_SIZE, make_dataset, read_manifest

FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.h5"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
METRICS_PATH = ROOT / "reports" / "metrics.json"

EXPERIMENT_NAME = "cats-dogs-classification"
REGISTERED_MODEL_NAME = "catdog-classifier"

PALETTE = {"train": "#30638E", "val": "#D1495B", "accent": "#EDAE49", "ink": "#003D5B"}


# --------------------------------------------------------------- evaluation

def evaluate(model, ds, y_true: np.ndarray, threshold: float = 0.5) -> tuple[dict, np.ndarray]:
    """Standard binary metrics on a dataset. Returns (metrics, probabilities)."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    proba = model.predict(ds, verbose=0).ravel()
    y_pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }, proba


# --------------------------------------------------------------- figures

def plot_history(history: dict, arch: str, save_dir: Path = FIG_DIR) -> list[Path]:
    """Loss and accuracy curves, one figure each (M1 asks for both by name)."""
    out = []
    for metric, fname, ylabel in (
        ("loss", "loss_curves.png", "binary cross-entropy"),
        ("accuracy", "accuracy_curves.png", "accuracy"),
    ):
        if metric not in history:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.2))
        epochs = range(1, len(history[metric]) + 1)
        ax.plot(epochs, history[metric], color=PALETTE["train"], linewidth=2, label="train")
        vkey = f"val_{metric}"
        if vkey in history:
            ax.plot(epochs, history[vkey], color=PALETTE["val"], linewidth=2,
                    linestyle="--", label="validation")
            best = int(np.argmin(history["val_loss"])) + 1
            ax.axvline(best, color=PALETTE["ink"], linestyle=":", linewidth=1,
                       label=f"best val_loss (epoch {best})")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.capitalize()} per epoch — {arch}")
        ax.legend(fontsize=9)
        fig.tight_layout()
        p = save_dir / fname
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        out.append(p)
    return out


def plot_confusion_matrix(y_true, y_pred, arch: str, save_dir: Path = FIG_DIR) -> Path:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"Confusion matrix (test) — {arch}")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    p = save_dir / "confusion_matrix.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_roc(y_true, proba, arch: str, save_dir: Path = FIG_DIR) -> Path:
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=PALETTE["train"], linewidth=2,
            label=f"{arch} (AUC = {roc_auc_score(y_true, proba):.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve (test split)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    p = save_dir / "roc_curve.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------- packaging

def package_versions() -> dict[str, str]:
    """Versions that determine model output — audit check 18 diffs these
    against requirements.txt to catch a train/serve skew."""
    import importlib.metadata as md

    out = {"python": platform.python_version()}
    for pkg in ("tensorflow", "keras", "numpy", "pillow", "h5py", "scikit-learn"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out


def save_artifacts(model, arch: str, params: dict, metrics: dict,
                   epoch_times: list[float], history: dict) -> None:
    """Write models/model.h5 and the metadata sidecar."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)   # .h5 is legacy in Keras 3 but round-trips exactly (ADR-002)

    metadata = {
        "model_family": arch,
        "framework": f"tensorflow/keras {package_versions().get('tensorflow', '?')}",
        "artifact_format": "h5",
        "package_versions": package_versions(),
        "random_state": RANDOM_STATE,
        "input_shape": [IMG_SIZE[1], IMG_SIZE[0], 3],
        "preprocessing": "RGB, bilinear resize to 224x224, scaled to [0,1]",
        "class_names": CLASS_NAMES,
        "class_indices": CLASS_INDICES,
        "decision_threshold": 0.5,
        "hyperparameters": params,
        "metrics": metrics,
        "training": {
            "epochs_run": len(epoch_times),
            "mean_epoch_seconds": round(float(np.mean(epoch_times)), 1) if epoch_times else None,
            "total_train_seconds": round(float(np.sum(epoch_times)), 1) if epoch_times else None,
            "best_val_loss": round(float(min(history.get("val_loss", [float("nan")]))), 4),
        },
        "selected_by": "single train/val/test fit; see reports/cv_results.csv for "
                       "the cross-validated comparison that drives model choice (ADR-005)",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")


# --------------------------------------------------------------- training

class EpochTimer:
    """Records per-epoch wall-clock — this is what sizes the CV budget on Day 4."""

    def __init__(self):
        self.times: list[float] = []
        self._t0 = 0.0

    def as_callback(self):
        import keras

        outer = self

        class _CB(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                outer._t0 = time.perf_counter()

            def on_epoch_end(self, epoch, logs=None):
                dt = time.perf_counter() - outer._t0
                outer.times.append(dt)
                print(f"[train]   epoch {epoch + 1} wall-clock: {dt:.1f}s")

        return _CB()


def train(arch: str = "transfer", epochs: int = 10, lr: float = 1e-3,
          batch_size: int = BATCH_SIZE, subset: int | None = None,
          weights: str | None = "imagenet", use_mlflow: bool = True) -> dict:
    import keras

    warn_if_synthetic()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(RANDOM_STATE)

    tr_p, tr_y = read_manifest("train")
    va_p, va_y = read_manifest("val")
    te_p, te_y = read_manifest("test")

    if subset:
        # Stratified head of each split, for a fast pipeline check.
        def take(paths, labels, n):
            idx = list(range(len(paths)))
            keep = [i for i in idx if labels[i] == 0][: n // 2] + \
                   [i for i in idx if labels[i] == 1][: n // 2]
            keep.sort()
            return [paths[i] for i in keep], [labels[i] for i in keep]
        tr_p, tr_y = take(tr_p, tr_y, subset)
        va_p, va_y = take(va_p, va_y, max(2, subset // 8))
        te_p, te_y = take(te_p, te_y, max(2, subset // 8))
        print(f"[train] SUBSET MODE: train={len(tr_p)} val={len(va_p)} test={len(te_p)}")

    train_ds = make_dataset(paths=tr_p, labels=tr_y, batch_size=batch_size,
                            augment=True, shuffle=True)
    val_ds = make_dataset(paths=va_p, labels=va_y, batch_size=batch_size,
                          augment=False, shuffle=False)
    test_ds = make_dataset(paths=te_p, labels=te_y, batch_size=batch_size,
                           augment=False, shuffle=False)

    kw = {"weights": weights} if arch == "transfer" else {}
    model = compile_model(build_model(arch, **kw), learning_rate=lr)
    print(f"[train] {arch}: {model.count_params():,} params "
          f"({sum(int(w.numpy().size) for w in model.trainable_weights):,} trainable)")

    timer = EpochTimer()
    callbacks = [
        timer.as_callback(),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=2, min_lr=1e-6, verbose=1),
    ]

    print(f"[train] fitting for up to {epochs} epochs "
          f"(train={len(tr_p):,}, val={len(va_p):,})")
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     callbacks=callbacks, verbose=2)
    history = {k: [float(x) for x in v] for k, v in hist.history.items()}

    y_test = np.asarray(te_y)
    test_metrics, proba = evaluate(model, test_ds, y_test)
    y_val = np.asarray(va_y)
    val_metrics, _ = evaluate(model, val_ds, y_val)

    print("\n[train] === test metrics ===")
    for k, v in test_metrics.items():
        print(f"[train]   {k:10s} {v:.4f}")

    plot_history(history, arch)
    plot_confusion_matrix(y_test, (proba >= 0.5).astype(int), arch)
    plot_roc(y_test, proba, arch)

    params = {
        "architecture": arch, "epochs_requested": epochs, "learning_rate": lr,
        "batch_size": batch_size, "optimizer": "adam", "loss": "binary_crossentropy",
        "augmentation": "flip/rotation(0.1)/zoom(0.1)/contrast(0.1), clipped to [0,1]",
        "pretrained_weights": weights if arch == "transfer" else None,
        "subset": subset,
    }
    metrics = {"test": test_metrics, "val": val_metrics}
    save_artifacts(model, arch, params, metrics, timer.times, history)

    mean_epoch = float(np.mean(timer.times)) if timer.times else 0.0
    print(f"\n[train] saved {MODEL_PATH.relative_to(ROOT)} "
          f"({MODEL_PATH.stat().st_size / 1e6:.1f}MB) + {METADATA_PATH.name}")
    print(f"[train] mean epoch: {mean_epoch:.1f}s over {len(timer.times)} epochs")
    print(f"[train] => 5-fold CV on this architecture would cost roughly "
          f"{5 * mean_epoch * len(timer.times) / 60:.0f} min at these settings")

    if use_mlflow:
        _log_to_mlflow(arch, params, metrics, history, timer.times)

    return {"metrics": metrics, "history": history, "epoch_times": timer.times}


def _log_to_mlflow(arch, params, metrics, history, epoch_times) -> None:
    """Log the run. Isolated so a tracking failure can't lose a trained model."""
    try:
        import mlflow

        mlflow.set_tracking_uri(f"file://{(ROOT / 'mlruns').resolve()}")
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"train-{arch}"):
            mlflow.set_tag("kind", "final-fit")
            mlflow.set_tag("model_family", arch)
            mlflow.log_params(params)
            for split, m in metrics.items():
                mlflow.log_metrics({f"{split}_{k}": v for k, v in m.items()})
            for epoch, vals in enumerate(zip(*history.values(), strict=True)):
                mlflow.log_metrics(dict(zip(history.keys(), vals, strict=True)), step=epoch)
            if epoch_times:
                mlflow.log_metric("mean_epoch_seconds", float(np.mean(epoch_times)))
            for fname in ("loss_curves.png", "accuracy_curves.png",
                          "confusion_matrix.png", "roc_curve.png"):
                p = FIG_DIR / fname
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path="figures")
            if METADATA_PATH.exists():
                mlflow.log_artifact(str(METADATA_PATH))
        print(f"[train] logged to MLflow experiment '{EXPERIMENT_NAME}'")
    except Exception as exc:
        print(f"[train] WARNING: MLflow logging failed ({exc}); model artifact is safe")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a Cats vs Dogs classifier")
    ap.add_argument("--model", default="transfer", choices=ARCHITECTURES)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--subset", type=int, help="stratified subset size, for a fast check")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="build the transfer model without ImageNet weights")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()

    train(arch=a.model, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size,
          subset=a.subset, weights=None if a.no_pretrained else "imagenet",
          use_mlflow=not a.no_mlflow)
    print("[train] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
