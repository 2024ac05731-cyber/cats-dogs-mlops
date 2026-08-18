"""Post-deployment model performance tracking (M5).

The spec asks to "collect a small batch of real or simulated requests and true
labels" against the deployed model. This script does that end to end: it sends a
held-out labelled batch to the **deployed** endpoint over HTTP, compares each
response against the known label, and writes a report.

The point is not to re-measure accuracy — training already did that offline. The
point is that **offline and live accuracy should agree**, and if they don't, the
gap localises a real class of production bug: a train/serve preprocessing skew.
Training resizes and scales images one way; if the API does it even slightly
differently, offline metrics stay excellent while live predictions quietly
degrade. Comparing the two is the only way to see it.

The batch is drawn from the **test split**, which the model never trained,
validated, or cross-validated on — so these numbers are honest, not memorised.

    python scripts/replay_batch.py --url http://127.0.0.1 --n 100
    python scripts/replay_batch.py --url http://localhost:8000 --n 40   # local container
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_MD = ROOT / "reports" / "post_deployment_report.md"
REPORT_CSV = ROOT / "reports" / "post_deployment_predictions.csv"
FIG_DIR = ROOT / "reports" / "figures"
BATCH_DIR = ROOT / "data" / "monitoring" / "labelled_batch"

BOUNDARY = "----catdogreplayboundary"


def _post(url: str, path: Path, timeout: int = 60):
    """POST one image; return (status, parsed_json_or_None, latency_ms)."""
    data = path.read_bytes()
    body = b"".join([
        f"--{BOUNDARY}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
        b"Content-Type: image/jpeg\r\n\r\n", data,
        f"\r\n--{BOUNDARY}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode())
            return r.status, payload, (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        return e.code, None, (time.perf_counter() - t0) * 1000
    except Exception:                                  # noqa: BLE001 - reported, not raised
        return 0, None, (time.perf_counter() - t0) * 1000


def build_batch(n: int, seed: int = 42) -> list[tuple[Path, int]]:
    """Sample a stratified labelled batch from the TEST split.

    Copies the chosen files into data/monitoring/labelled_batch/ with a labels.csv
    so the batch is a reproducible artifact rather than something regenerated
    differently on each run.
    """
    import random

    from src.preprocess import read_manifest

    paths, labels = read_manifest("test")
    by_class: dict[int, list[str]] = {0: [], 1: []}
    for p, y in zip(paths, labels, strict=True):
        by_class[y].append(p)

    rng = random.Random(seed)
    per = max(1, n // 2)
    chosen: list[tuple[Path, int]] = []
    for y, files in by_class.items():
        for p in rng.sample(files, min(per, len(files))):
            chosen.append((Path(p), y))
    rng.shuffle(chosen)

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    with (BATCH_DIR / "labels.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "true_label", "true_class", "source_path"])
        for p, y in chosen:
            w.writerow([p.name, y, ["cat", "dog"][y], str(p)])
    print(f"[replay] batch of {len(chosen)} images "
          f"(labels.csv -> {(BATCH_DIR / 'labels.csv').relative_to(ROOT)})")
    return chosen


def plot_confusion(cm: list[list[int]], save_dir: Path = FIG_DIR) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks([0, 1], ["cat", "dog"])
    ax.set_yticks([0, 1], ["cat", "dog"])
    ax.set_xlabel("predicted (from the DEPLOYED service)")
    ax.set_ylabel("true label")
    ax.set_title("Post-deployment confusion matrix")
    hi = max(max(r) for r in cm) / 2 or 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i][j] > hi else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    p = save_dir / "post_deploy_confusion_matrix.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_latency(latencies: list[float], save_dir: Path = FIG_DIR) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(latencies, bins=min(30, max(5, len(latencies) // 3)),
            color="#30638E", edgecolor="white")
    for q, style, label in ((50, "--", "p50"), (95, ":", "p95")):
        v = statistics.quantiles(latencies, n=100)[q - 1] if len(latencies) > 2 else latencies[0]
        ax.axvline(v, linestyle=style, color="#D1495B", linewidth=1.5,
                   label=f"{label} = {v:.0f} ms")
    ax.set_xlabel("end-to-end request latency (ms, includes network)")
    ax.set_ylabel("requests")
    ax.set_title("Post-deployment latency distribution")
    ax.legend()
    fig.tight_layout()
    p = save_dir / "post_deploy_latency.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1", help="deployed service base URL")
    ap.add_argument("--n", type=int, default=100, help="batch size (stratified)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    base = a.url.rstrip("/")

    print(f"[replay] target: {base}")
    batch = build_batch(a.n, a.seed)

    rows, latencies = [], []
    errors = 0
    for i, (path, y_true) in enumerate(batch, 1):
        status, payload, ms = _post(f"{base}/predict", path)
        if status != 200 or not payload:
            errors += 1
            rows.append({"filename": path.name, "true_label": y_true,
                         "predicted_label": "", "confidence": "",
                         "latency_ms": round(ms, 1), "status": status})
            print(f"[replay] {i:>4}/{len(batch)} {path.name:16s} HTTP {status}")
            continue
        latencies.append(ms)
        rows.append({
            "filename": path.name,
            "true_label": y_true,
            "predicted_label": payload["prediction"],
            "confidence": payload["confidence"],
            "latency_ms": round(ms, 1),
            "status": status,
        })
        if i % 20 == 0 or i == len(batch):
            print(f"[replay] {i:>4}/{len(batch)} sent...")

    scored = [r for r in rows if r["predicted_label"] != ""]
    if not scored:
        print("[replay] FAIL: no successful predictions — is the service reachable?")
        return 1

    y_true = [r["true_label"] for r in scored]
    y_pred = [int(r["predicted_label"]) for r in scored]

    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 0)

    acc = (tp + tn) / len(scored)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    cm = [[tn, fp], [fn, tp]]

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Offline metrics for the comparison that is the whole point of this exercise.
    meta_path = ROOT / "models" / "model_metadata.json"
    offline = {}
    if meta_path.exists():
        offline = json.loads(meta_path.read_text()).get("metrics", {}).get("test", {})

    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = (statistics.quantiles(latencies, n=100)[94]
           if len(latencies) > 2 else (latencies[0] if latencies else 0.0))

    print("\n[replay] === live vs offline ===")
    print(f"[replay]   {'metric':10s} {'live':>9s} {'offline':>9s} {'delta':>9s}")
    deltas = {}
    for name, live in (("accuracy", acc), ("precision", prec), ("recall", rec), ("f1", f1)):
        off = offline.get(name)
        d = (live - off) if isinstance(off, (int, float)) else None
        deltas[name] = d
        off_s = f"{off:.4f}" if isinstance(off, (int, float)) else "n/a"
        d_s = f"{d:+.4f}" if d is not None else "n/a"
        print(f"[replay]   {name:10s} {live:9.4f} {off_s:>9s} {d_s:>9s}")
    print(f"[replay]   latency p50={p50:.0f}ms p95={p95:.0f}ms  errors={errors}")

    figs = []
    if not a.no_figures:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        figs = [plot_confusion(cm), plot_latency(latencies)]
        for f in figs:
            print(f"[replay] wrote {f.relative_to(ROOT)}")

    worst = max((abs(d) for d in deltas.values() if d is not None), default=0.0)
    verdict = (
        "No meaningful train/serve skew: live metrics track the offline test "
        "metrics closely, so the deployed preprocessing matches training."
        if worst < 0.05 else
        f"**Investigate:** the largest live-vs-offline gap is {worst:.4f}. On the "
        "same held-out data this usually means the API preprocesses differently "
        "from training — check `src.preprocess.load_image` is the only resize/scale "
        "path in both."
    )

    def _row(name: str, live: float) -> str:
        off = offline.get(name)
        off_s = f"{off:.4f}" if isinstance(off, (int, float)) else "n/a"
        d = deltas.get(name)
        d_s = f"{d:+.4f}" if d is not None else "n/a"
        return f"| {name.capitalize()} | {live:.4f} | {off_s} | {d_s} |"

    metrics_table = "\n".join([
        _row("accuracy", acc), _row("precision", prec),
        _row("recall", rec), _row("f1", f1),
    ])

    REPORT_MD.write_text(f"""# Post-deployment performance report (M5)

Generated by `python scripts/replay_batch.py --url {base} --n {a.n}`.

A labelled batch was sent to the **deployed service over HTTP** and each response
compared against its known label. Images come from the **test split**, which the
model never trained, validated, or cross-validated on.

## Live results ({len(scored)} scored requests, {errors} error(s))

| Metric | Live (deployed) | Offline (test split) | Delta |
|---|---|---|---|
{metrics_table}

## Confusion matrix (from deployed responses)

|  | predicted cat | predicted dog |
|---|---|---|
| **true cat** | {tn} | {fp} |
| **true dog** | {fn} | {tp} |

![Post-deployment confusion matrix](figures/post_deploy_confusion_matrix.png)

## Latency

End-to-end, measured client-side, so it includes network and serialisation — not
just model inference.

| | ms |
|---|---|
| p50 | {p50:.0f} |
| p95 | {p95:.0f} |
| min | {min(latencies) if latencies else 0:.0f} |
| max | {max(latencies) if latencies else 0:.0f} |

![Latency distribution](figures/post_deploy_latency.png)

## Interpretation

{verdict}

Why this comparison is worth making at all: training already measured accuracy
offline, so re-measuring it proves little on its own. What it *does* prove is that
the deployed path agrees with the training path. A train/serve preprocessing skew
leaves offline metrics untouched while quietly degrading live predictions, and
comparing the two on identical data is the only way to see it.

Per-request detail: [`post_deployment_predictions.csv`](post_deployment_predictions.csv).
""")
    print(f"[replay] wrote {REPORT_MD.relative_to(ROOT)}")
    print(f"[replay] wrote {REPORT_CSV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
