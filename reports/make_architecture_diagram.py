"""Generate the end-to-end architecture diagram for the README.

Drawn with matplotlib rather than a diagramming tool so the figure is a
reproducible build artifact: it regenerates from source, cannot drift from the
repo, and needs no binary editor. Same approach used on Assignment 1.

    python reports/make_architecture_diagram.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "figures" / "architecture_diagram.png"

# One colour per rubric module, so a grader can map the picture to the marks.
C = {
    "M1": "#30638E",   # data / model / tracking
    "M2": "#00798C",   # packaging / serving
    "M3": "#EDAE49",   # CI
    "M4": "#D1495B",   # CD / deploy
    "M5": "#8F2D56",   # monitoring
    "ink": "#003D5B",
    "muted": "#7A8B99",
}

# (x, y, w, h, label, sublabel, module)
BOXES = [
    (0.5, 8.6, 2.5, 0.72, "Kaggle dataset", "24,998 images", "M1"),
    (0.5, 7.5, 2.5, 0.72, "src/data.py", "audit: 1 corrupt", "M1"),
    (0.5, 6.4, 2.5, 0.72, "DVC", "data/raw.dvc, 848MB", "M1"),
    (0.5, 5.3, 2.5, 0.72, "src/preprocess.py", "224x224, 80/10/10", "M1"),

    (3.6, 6.4, 2.4, 0.72, "src/train.py", "full split, 10 ep", "M1"),
    (3.6, 5.3, 2.4, 0.72, "src/cross_validate.py", "5-fold stratified", "M1"),
    (3.6, 4.2, 2.4, 0.72, "MLflow", "runs + nested folds", "M1"),

    (6.6, 5.3, 2.6, 0.9, "models/model.h5", "+ metadata sidecar", "M2"),
    (6.6, 4.0, 2.6, 0.72, "api/main.py", "FastAPI, 6 routes", "M2"),
    (6.6, 2.9, 2.6, 0.72, "Dockerfile", "428MB, non-root", "M2"),

    (3.6, 1.6, 2.4, 0.72, "GitHub Actions CI", "6 gated jobs", "M3"),
    (6.6, 1.6, 2.6, 0.72, "GHCR", "SHA + latest, 2 arch", "M3"),

    (9.9, 1.6, 2.4, 0.72, "Argo CD", "auto-sync, selfHeal", "M4"),
    (9.9, 2.9, 2.4, 0.72, "Minikube", "2 replicas, probes", "M4"),
    (9.9, 4.0, 2.4, 0.72, "smoke gate", "fail -> rollback", "M4"),

    (9.9, 5.3, 2.4, 0.9, "Prometheus + Grafana", "4 panels, JSON logs", "M5"),
    (9.9, 6.6, 2.4, 0.72, "replay_batch.py", "live vs offline", "M5"),
]

# (from_label, to_label, style)
ARROWS = [
    ("Kaggle dataset", "src/data.py", "-|>"),
    ("src/data.py", "DVC", "-|>"),
    ("DVC", "src/preprocess.py", "-|>"),
    ("src/preprocess.py", "src/cross_validate.py", "-|>"),
    ("src/preprocess.py", "src/train.py", "-|>"),
    ("src/cross_validate.py", "MLflow", "-|>"),
    ("src/train.py", "MLflow", "-|>"),
    ("src/train.py", "models/model.h5", "-|>"),
    ("src/cross_validate.py", "models/model.h5", "-|>"),
    ("models/model.h5", "api/main.py", "-|>"),
    ("api/main.py", "Dockerfile", "-|>"),
    ("Dockerfile", "GitHub Actions CI", "-|>"),
    ("GitHub Actions CI", "GHCR", "-|>"),
    ("GHCR", "Argo CD", "-|>"),
    ("Argo CD", "Minikube", "-|>"),
    ("Minikube", "smoke gate", "-|>"),
    ("Minikube", "Prometheus + Grafana", "-|>"),
    ("Minikube", "replay_batch.py", "-|>"),
]


def _centre(box):
    x, y, w, h = box[:4]
    return x + w / 2, y + h / 2


def main() -> Path:
    plt.rcParams.update({"font.size": 8.5, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(14.5, 10))
    ax.set_xlim(0, 13)
    ax.set_ylim(0.6, 10.4)
    ax.axis("off")

    by_label = {b[4]: b for b in BOXES}

    for x, y, w, h, label, sub, mod in BOXES:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.045,rounding_size=0.09",
            facecolor=C[mod], edgecolor="white", linewidth=1.6, alpha=0.93, zorder=2))
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                color="white", fontweight="bold", fontsize=9.2, zorder=3)
        ax.text(x + w / 2, y + h * 0.24, sub, ha="center", va="center",
                color="white", fontsize=7.4, alpha=0.93, zorder=3)

    for src, dst, style in ARROWS:
        x1, y1 = _centre(by_label[src])
        x2, y2 = _centre(by_label[dst])
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=1,
                    arrowprops=dict(arrowstyle=style, color=C["muted"],
                                    linewidth=1.3, shrinkA=26, shrinkB=26,
                                    connectionstyle="arc3,rad=0.06"))

    # The train->serve seam, called out because it is the architectural claim that
    # matters: model.h5 is the ONLY thing crossing from training into serving.
    ax.annotate("the only train -> serve seam",
                xy=(7.9, 5.28), xytext=(6.2, 8.9),
                ha="center", fontsize=8.6, color=C["ink"], fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=C["ink"], linewidth=1.4,
                                linestyle=":", shrinkB=6))
    ax.text(6.2, 8.55,
            "src/preprocess.py::load_image is shared by training and the API,\n"
            "so train and serve cannot drift apart",
            ha="center", fontsize=7.6, color=C["ink"], style="italic")

    ax.text(0.4, 10.05, "Cats vs Dogs — end-to-end MLOps pipeline",
            fontsize=15, fontweight="bold", color=C["ink"])
    ax.text(0.4, 9.72,
            "BITS Pilani AIMLCZG523 Assignment 2  ·  boxes coloured by rubric module",
            fontsize=9, color=C["muted"])

    legend = [
        mpatches.Patch(facecolor=C["M1"], label="M1  data, model, CV, tracking"),
        mpatches.Patch(facecolor=C["M2"], label="M2  packaging, serving"),
        mpatches.Patch(facecolor=C["M3"], label="M3  CI, registry"),
        mpatches.Patch(facecolor=C["M4"], label="M4  CD, deployment"),
        mpatches.Patch(facecolor=C["M5"], label="M5  monitoring, tracking"),
    ]
    ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.01, 0.0),
              ncol=5, frameon=False, fontsize=8.4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print(f"[diagram] wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024}KB)")
    return OUT


if __name__ == "__main__":
    main()
