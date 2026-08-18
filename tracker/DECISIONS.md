# Architecture & Design Decisions Log

ADR-style log of choices made during Assignment 2. Two reasons to keep this:

1. **Future-you needs to remember why** — when reviewing a choice two weeks later.
2. **Academic integrity** — explicit, reasoned decisions show original thinking. If two students
   submit similar architectures, the one that documented WHY chose those tools is much harder to
   flag as copied.

---

## How to write an ADR

Keep entries short (10–30 lines). Capture the choice BEFORE you implement it.

```
### ADR-NNN: <Decision Title>

**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Superseded by ADR-XXX
**Day in plan:** Day N

**Context:** What problem are we solving? What constraints?

**Options considered:**
1. Option A — pros / cons
2. Option B — pros / cons
3. Option C — pros / cons

**Decision:** Chose Option X.

**Rationale:** Why this option won.

**Consequences:** What this means for later steps. Things to watch out for.
```

---

## Decisions

### ADR-000: Optimise this submission against the Assignment 1 feedback

**Date:** 2026-08-15
**Status:** Accepted
**Day in plan:** Day 0 (planning)

**Context:** Assignment 1 scored below target with the grader comment *"No cross-validation, thin
CI/CD."* The A1 repo did in fact run 5-fold `GridSearchCV` inside `src/train.py`, and its CI did run
lint + pytest + a docker build. So one comment is about **visibility** and the other about **depth**.

**Options considered:**
1. **Treat both as first-class deliverables with their own artifacts and evidence** — costs a
   dedicated CV day and three CD days, and shapes the whole plan.
2. **Do the same work as A1 and describe it better in the README** — cheap, but "thin CI/CD" is a
   fair description of a pipeline with no registry push and no automated deploy. Wouldn't fix it.
3. **Go maximal on both** — every CI feature imaginable. Risks running out of time on M1/M5.

**Decision:** Option 1.

**Rationale:** The two comments have different root causes and need different fixes. CV was
*present but invisible*, so the fix is artifacts a grader trips over: a per-fold CSV, two figures,
nested MLflow runs, and a README section. CI/CD was *genuinely thin*, so the fix is real depth:
registry publishing, automated GitOps deploy, a post-deploy smoke gate, and rollback.

**Consequences:**
- Day 6 is dedicated entirely to cross-validation; Days 12–17 are dedicated to CI/CD.
- Items in `TASKS.md` are tagged `[GAP-CV]` and `[GAP-CICD]`; those may not be marked done on the
  strength of code existing alone — artifact + screenshot + README mention are required.
- Deliberate cost: less time for EDA depth than A1 had. Acceptable, since M1 here weights
  versioning and tracking over exploratory analysis, and no report.pdf is required this time.

---

### ADR-001: 21-day timeline with weekly module gates

**Date:** 2026-08-15
**Status:** Accepted
**Day in plan:** Day 0

**Context:** Five modules worth 10 marks each, solo, with two genuinely new tools relative to A1
(DVC and Argo CD) plus a switch from sklearn to TensorFlow.

**Options considered:**
1. **7 days** — only if the deadline forces it. Highest risk that M4 stays thin, which is exactly
   the mark already lost once.
2. **14 days** — feasible because the A1 scaffolding is reusable, but little slack for Kaggle auth,
   DVC, and Argo CD debugging.
3. **21 days, one week per module group** — matches the cadence that worked on A1.

**Decision:** 21 days: Week 1 → M1, Week 2 → M2 + M3, Week 3 → M4 + M5.

**Rationale:** Same cadence as A1, which delivered. The new-tool risk sits in Week 3 (Argo CD), and
a week-long block gives room to debug it without eating the monitoring module.

**Consequences:**
- Checkpoint gates on Days 7, 14, 21 — each closes a module group before moving on.
- Buffer is folded into Days 7 and 14 rather than standalone idle days, because Week 3 has no slack.
- If the real deadline is shorter, compress Week 1 (the most familiar ground), never Week 3.

---

### ADR-002: TensorFlow / Keras, with split requirements files

**Date:** 2026-08-15
**Status:** Accepted
**Day in plan:** Day 1

**Context:** The spec allows any framework and names `.pkl`/`.pt`/`.h5` as acceptable artifacts.
Need to choose one and pin it for reproducibility across training (macOS, Apple Silicon) and
serving (linux/amd64 container).

**Options considered:**
1. **TensorFlow / Keras** — `model.fit()` and Keras preprocessing layers cover augmentation cleanly;
   `.h5` is exactly the format the spec names. Cost: large wheel, so a serving image likely well
   over 1GB, and the k-fold loop needs care (see ADR-003).
2. **PyTorch** — smaller CPU wheel and a smaller image; explicit training loop makes per-fold CV
   trivial. Cost: more boilerplate for augmentation and training.
3. **sklearn on flattened pixels** — the spec permits it as a baseline. Rejected: throws away the
   spatial structure the use case is about, and reads as avoiding the work.

**Decision:** TensorFlow / Keras, artifact `models/model.h5`.

**Rationale:** User preference, and the `tf.data` + Keras-augmentation-layer path is the most direct
route to the spec's "224x224 RGB with data augmentation" requirement. Image size is not constrained
by this assignment.

**Consequences:**
- **RESOLVED on Day 1 by measurement, not assumption.** Findings:
  - macOS arm64 resolves `tensorflow` to **2.21.0**, but **`tensorflow-cpu` stops at 2.20.0** on
    PyPI (verified: `pip download tensorflow-cpu==2.21.0 --platform manylinux2014_x86_64` fails;
    available versions end at 2.20.0). Left alone, this guarantees a train/serve skew.
  - **Both sides are therefore pinned to `2.20.0`** — `tensorflow==2.20.0` in `requirements.txt`
    (macOS arm64 wheel confirmed available) and `tensorflow-cpu==2.20.0` in
    `requirements-serve.txt`. Audit check 14 enforces the alignment; check 15 asserts the
    `tensorflow-cpu` version actually exists.
  - TF 2.20.0 ships **Keras 3.15.1**, which marks `.h5` legacy and prints a warning. Probed the
    round-trip explicitly: `.h5` and `.keras` both save, load, and reproduce predictions
    **identically** (`np.allclose` at 1e-6). Since the spec names `.h5` and Guardrail Rule 4 says
    match the rubric's vocabulary, **`.h5` stays the artifact format**; the warning is suppressed in
    `pytest.ini` with a comment pointing here.
  - `mlflow==3.15.1` constrains pandas to 2.x — pip silently downgraded 3.0.5 → **2.3.3**. Pinned
    explicitly so it's a decision rather than a resolver accident.
  - Pillow is **no longer a TensorFlow dependency**; it has to be requested explicitly.
- Expect a serving image over 1GB. Document the number rather than fighting it. If it becomes a
  problem, the fallback is exporting to TFLite or SavedModel and serving without full TF — noted
  here so it isn't re-derived later.
- TF's memory appetite affects k8s resource limits (Day 8) and startup time affects probe
  `initialDelaySeconds`.

---

### ADR-003: Hand-rolled stratified k-fold CV over two architectures

**Date:** 2026-08-15
**Status:** Accepted
**Day in plan:** Day 6

**Context:** Cross-validation is the mark lost in A1 and must be unambiguous here. Keras has no
native CV, and the dataset is ~25k images, so CV cost is a real constraint on CPU.

**Options considered:**
1. **`scikeras`/`KerasClassifier` + `cross_val_score`** — least code. Rejected: the wrapper is
   brittle with `tf.data` inputs and Keras preprocessing layers, obscures per-fold results behind a
   single aggregate score, and would make the *one thing that must not break* the fragile part.
2. **Hand-rolled loop over `StratifiedKFold` splits** — explicit, per-fold metrics are naturally
   available, easy to log as nested MLflow runs. Costs ~40 lines.
3. **A single train/val/test split only** — what A1 effectively presented. Rejected outright.

**Decision:** Hand-rolled `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` loop in
`src/cross_validate.py`, run for **both** a from-scratch baseline CNN and a MobileNetV2 transfer
model.

**Rationale:** Explicit control is what makes CV *visible*: per-fold rows in a CSV, per-fold nested
MLflow runs, and a mean ± std figure all fall out of the loop for free. Two architectures turn CV
into a genuine model-selection instrument rather than a formality, so the production model can be
chosen by CV mean (ADR-005) instead of by one lucky test split.

**Consequences:**
- The model must be **rebuilt and recompiled inside each fold** — reusing a compiled model leaks
  weights and optimizer state across folds and silently invalidates the whole exercise. Assert
  freshness rather than trusting it.
- CV runs over the pooled **train + val** files; the test split stays untouched so it remains a
  clean final holdout.
- MobileNetV2 with a frozen base is chosen partly *because* it converges in few epochs, making
  5-fold CV affordable on CPU.
- If wall-clock still forces a reduction (fewer epochs per fold, or a stratified subset), the
  protocol gets documented in the module docstring and the README. A documented reduction is
  defensible; a silent one is the A1 mistake repeated.

---

### ADR-004: DVC with a local-filesystem remote

**Date:** 2026-08-15
**Status:** Accepted
**Day in plan:** Day 3

**Context:** M1 requires DVC (or Git-LFS) for dataset and pre-processed-data versioning. The raw
dataset is ~25k images — far too large to commit to Git.

**Options considered:**
1. **DVC, local-filesystem remote** (`~/dvc-store/`) — works offline, zero cost, zero credentials.
   Limit: the grader cannot `dvc pull` from your machine.
2. **DVC, Google Drive / S3 remote** — the grader could actually pull. Costs OAuth or AWS setup, and
   adds a credential to keep out of the repo.
3. **Git-LFS** — the spec allows it, but it versions blobs without giving pipeline stages, and
   `dvc.yaml`/`dvc repro` is a much stronger artifact for an MLOps rubric.

**Decision:** DVC with a local-filesystem remote, plus a `dvc.yaml` pipeline
(`preprocess` → `train`).

**Rationale:** The graded artifacts are the DVC config, the `.dvc` files, `dvc.yaml`, and `dvc.lock`
— all of which exist identically regardless of remote backend. A local remote removes an entire
class of credential and connectivity failure from the critical path. `dvc.yaml` additionally
demonstrates reproducible pipeline stages, which Git-LFS cannot.

**Consequences:**
- The README must state that the remote is local, and explain how to reproduce from the raw Kaggle
  download instead of `dvc pull`.
- **`models/model.h5` must be committed to Git**, not left DVC-only — otherwise the submitted zip
  has no usable model artifact, which the deliverables explicitly require.
- `.dvc/cache` goes in `.gitignore`.
- Upgrading to a cloud remote later is a one-line `dvc remote add` change if it turns out to matter.

---

### ADR-005: Ship the MobileNetV2 transfer model, selected by cross-validated mean

**Date:** 2026-08-18
**Status:** Accepted
**Day in plan:** Day 4

**Context:** Two architectures needed comparing so that one could be packaged as
`models/model.h5` and served. The point of doing this by cross-validation rather than by a single
train/test fit is that A1 lost a mark for cross-validation being invisible, and a mark is not the
only reason: a single 2,501-row test split is a noisy basis for a decision.

**Evidence — 5-fold stratified CV, 4,000-image random stratified subset of pooled train+val,
8 epochs/fold, `random_state=42`, 31.3 min wall-clock:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **transfer** (MobileNetV2, frozen base) | **0.9840 ± 0.0037** | 0.9798 ± 0.0088 | 0.9885 ± 0.0025 | 0.9841 ± 0.0036 | 0.9984 ± 0.0013 |
| baseline (from-scratch CNN) | 0.6198 ± 0.0373 | 0.7229 ± 0.1120 | 0.4915 ± 0.2429 | 0.5285 ± 0.1484 | 0.7113 ± 0.0474 |

Per-fold rows in `reports/cv_results.csv`; figures in `reports/figures/cv_comparison.png` and
`cv_fold_scores.png`; nested `fold_1`…`fold_5` runs under `cv-transfer` / `cv-baseline` in MLflow.

**Options considered:**
1. **MobileNetV2 transfer, frozen base** — CV accuracy 0.9840, and a standard deviation of 0.0037,
   i.e. an order of magnitude tighter than the baseline's. Only 1,281 trainable parameters, so it fits
   in ~85 s/fold. Cost: 2.26M total parameters travel in the artifact and the image.
2. **From-scratch baseline CNN** — 0.6198 accuracy, and unstable: recall swings from 0.17 to 0.82
   across folds (± 0.2429). Smaller artifact (242k parameters, 3.0 MB) and no pretrained dependency.
3. **Train the baseline much longer** — it was still improving at 8 epochs. Might close some of the
   36-point gap, but at ~290 s/fold it is already 3.5x the transfer model's cost per fold, and no
   plausible amount of training makes a 4-block CNN on 3,200 images competitive with ImageNet
   features.

**Decision:** Ship the **transfer** model.

**Rationale:** The decision is not close on the mean — 0.9840 vs 0.6198 — but the *variance* is the
more interesting argument and the one that justifies having done CV at all. The baseline's recall
standard deviation of 0.2429 means a single train/test split could have reported anywhere from 0.17
to 0.82 for the same architecture; picking a model on one split would have been close to a coin
flip on which architecture "won" on that metric. The transfer model's ± 0.0037 accuracy says its
performance is a property of the architecture, not of the split.

Secondary: the transfer model is also *cheaper* here (85 s/fold vs 290 s), because only the 1,281-parameter
head is trained. Better and faster is an easy call.

**Consequences:**
- `models/model.h5` is the MobileNetV2 transfer model; `src/train.py --model transfer` is the
  packaging command, and it is the `train` stage default in `dvc.yaml`.
- The artifact carries the frozen 2.2M-parameter base, so it is larger than the baseline's 3.0 MB.
  This propagates to the Docker image (M2) and to k8s memory limits (M4).
- The baseline CNN stays in `src/model.py` and in the CV results. It is not dead code: it is the
  control that makes the transfer model's number meaningful, and the CV comparison is the M1
  deliverable.
- Serving depends on the ImageNet weights only at *training* time. `model.h5` is self-contained, so
  the container needs no network access — verified when the image is built on Day 6.
- **Reported test metrics must come from the transfer model.** The `model.h5` currently committed is
  from the Day 3 baseline validation run (accuracy 0.7180); it has to be retrained with
  `--model transfer` before submission, or the README and metadata will understate the result.

---

### ADR-009: Baseline CNN architecture

**Date:** 2026-08-18
**Status:** Accepted
**Day in plan:** Day 3

**Context:** The spec asks for "at least one baseline model (e.g. a simple CNN)". It needs to be a
credible control for the transfer model, not a strawman, but also cheap enough to cross-validate.

**Decision:** Four Conv-BatchNorm-MaxPool blocks (32→64→128→128) → GlobalAveragePooling → Dropout(0.3)
→ Dense(1, sigmoid). 242,369 parameters.

**Rationale:**
- **BatchNorm after every conv.** Inputs are only [0,1]-scaled, so without normalising intermediate
  activations the deeper blocks train poorly at any learning rate worth using.
- **GlobalAveragePooling instead of Flatten.** Flatten on a 14x14x128 feature map gives 25,088
  features and a ~3.2M-parameter head; GAP gives 128 features and a 129-parameter head. On 3,200
  images per fold that single choice is the largest overfitting guard in the model.
- **Widths stop at 128.** Wider layers cost CV wall-clock, which is the binding constraint (ADR-005).

**Consequences:** It is genuinely undertrained at 8 epochs — measured on Day 3, it needs ~11 epochs
before it stops predicting a single class. Its CV numbers therefore represent "a small CNN given a
CV-affordable budget", which is stated explicitly in the README rather than presented as the
architecture's ceiling.

---

### ADR-010: Augmentation policy

**Date:** 2026-08-18
**Status:** Accepted
**Day in plan:** Day 2

**Context:** The spec asks for augmentation "for better generalization". Which transforms are valid
depends on the task, and a label-destroying transform actively harms training.

**Decision:** `RandomFlip("horizontal")`, `RandomRotation(0.10)`, `RandomZoom(0.10)`,
`RandomContrast(0.10)`, then a **clip to [0,1]**. Applied in the `tf.data` pipeline and gated behind
`augment=True`, so it cannot reach val/test.

**Rationale:**
- **Horizontal flip only.** A mirrored pet is still the same species. Vertical flip is rejected:
  upside-down animals do not occur in adoption-listing photos, so it would spend model capacity on
  invariance to inputs that never arrive.
- **Small rotation and zoom** mimic the real variation in user-submitted photos — the EDA sample grid
  shows wide framing and aspect-ratio variation.
- **Mild contrast** covers lighting differences without distorting colour, which matters because
  colour is genuinely informative for this task.
- **The clip is not cosmetic.** Measured during Day 2: `RandomContrast` pushed values to 1.02, and
  `build_transfer_model`'s `Rescaling(2.0, -1.0)` layer assumes a clean [0,1] input when mapping to
  MobileNetV2's [-1,1] range. Without clipping, augmented inputs silently violate that contract.

**Consequences:** Augmentation is verified visually in `reports/figures/augmentation_grid.png`, which
prints the observed output range — that figure is how the 1.02 overflow was caught in the first place.
Applied as a dataset `map`, never as a model layer, so it is not serialized into the served artifact.

---

### ADR-011: No GPU acceleration — tensorflow-metal is incompatible with TF 2.20

**Date:** 2026-08-18
**Status:** Accepted
**Day in plan:** Day 4

**Context:** Training runs on an Apple M4 Pro (14 CPU cores, 20 GPU cores) and TF reports
`GPU devices: []`. Profiling showed model compute at 18.3 ms/image versus data loading at 1.25 ms/image,
so compute is the bottleneck by ~15x and GPU acceleration would genuinely help.

**Options considered:**
1. **Install `tensorflow-metal`** — the Apple Metal plugin, which would use the integrated GPU.
2. **Downgrade TensorFlow to ~2.17** so a compatible `tensorflow-metal` can be used.
3. **Stay on CPU.**

**Decision:** Option 3, stay on CPU.

**Rationale:** `tensorflow-metal 1.2.0` (the newest release, built for ~TF 2.17) **fails to load
under TF 2.20**, tested in an isolated venv:

```
NotFoundError: dlopen(libmetal_plugin.dylib):
  Library not loaded: @rpath/_pywrap_tensorflow_internal.so
```

It declares no TF version constraint, so pip installs it without complaint and TensorFlow then fails
at *import*. Installing it into the project venv would have taken the project offline entirely.

Downgrading to TF 2.17 to regain Metal was considered and rejected on timing: the compute-heavy phase
is finished. CV is complete (31.3 min) and the remaining days are API, Docker, CI, k8s and inference —
none training-bound. The only significant run left is an optional full-data final fit (~34 min). A
downgrade would require re-pinning both requirements files, re-verifying `tensorflow-cpu 2.17` for
linux/amd64, re-checking the `.h5` round-trip, and re-running CV — hours of work and new risk to save
tens of minutes.

**Consequences:**
- All reported timings are CPU timings on an M4 Pro; recorded in `model_metadata.json` and the README
  so nobody misreads them as GPU numbers.
- If a future full-data CV run is wanted, budget ~5.7 h or keep the documented subset.
- `requirements.txt` deliberately does **not** list `tensorflow-metal`. Anyone adding it must
  downgrade TensorFlow first, and the audit's pin-consistency check (14) would flag the resulting skew
  against `requirements-serve.txt`.

---

### ADR-006: Publish images to GHCR

**Date:** 2026-08-15
**Status:** Accepted
**Day in plan:** Day 13

**Context:** M3 requires pushing the image to a container registry. The image also has to be pullable
by the Minikube cluster in M4.

**Options considered:**
1. **GHCR (`ghcr.io`)** — auth via the workflow's built-in `GITHUB_TOKEN`, so no secrets to create
   or rotate. The Packages page sits on the repo, which is convenient evidence.
2. **Docker Hub** — familiar, anonymous pulls by default. Needs two repo secrets and has free-tier
   rate limits.
3. **Local registry** — the spec allows it, but it demonstrates the least and can't be shown as a
   published artifact.

**Decision:** GHCR, tagged with both the **commit SHA** and `latest`.

**Rationale:** Zero credential management is the deciding factor — one less thing to break in a
pipeline that already has many moving parts. Repo-linked packages are easy evidence for M3. SHA tags
are what make the GitOps flow in ADR-007 possible at all: `latest` cannot express "deploy *this*
build".

**Consequences:**
- The workflow needs `permissions: packages: write`.
- Pushes happen only on `main`; PRs build without publishing.
- Minikube must be able to pull: either make the package **public**, or wire an `imagePullSecret`
  from a PAT with `read:packages`. Decide on Day 15 and record the outcome here.
- `k8s/deployment.yaml` references the SHA tag, not `latest`, so a rollout is always deliberate.

---

### ADR-007: Minikube + Argo CD GitOps, with a self-hosted runner for the smoke gate

**Date:** 2026-08-15
**Status:** Accepted
**Day in plan:** Days 15–17

**Context:** M4 requires automated deployment on `main` changes plus a post-deploy smoke test that
fails the pipeline. "Thin CI/CD" was explicit A1 feedback, so this module needs real depth.

**Options considered:**
1. **Minikube + Argo CD (GitOps)** — CI pushes the image and commits the new tag; Argo CD auto-syncs
   and self-heals. Costs about half a day of setup, and Argo's UI is strong demo material.
2. **Minikube + self-hosted runner running `kubectl set image`** — fewer moving parts, still
   automated, but reads as a deploy script rather than continuous deployment.
3. **Docker Compose on the Mac** — quickest, and the spec permits it. Rejected: thinnest possible
   deployment evidence for a 10-mark module, on the exact axis already criticised.

**Decision:** Option 1 — Minikube + Argo CD auto-sync, **plus** a self-hosted-runner CD job that
waits for rollout, smoke-tests, and rolls back on failure.

**Rationale:** Argo CD makes deployment a consequence of a commit rather than of a human, which is
the substantive difference between CI and CD. `prune` and `selfHeal` demonstrate reconciliation,
something a deploy script cannot show. Reuses the Minikube experience from A1, so the cluster itself
isn't new risk.

**Consequences:**
- A **self-hosted runner on the Mac is unavoidable**: GitHub-hosted runners cannot reach a local
  cluster, and the smoke gate must run against the real deployment.
- The CI tag-bump commit must be guarded (`[skip ci]` or a path filter) or it retriggers CI forever.
  Set the guard on the first attempt.
- The commit-back step needs write permission and a configured git identity.
- Argo CD's `selfHeal` will fight manual `kubectl` edits — that's the point, but it means debugging
  by hand-editing live resources won't work; change the manifest in Git instead.
- Rollback is `kubectl rollout undo` in an `if: failure()` step. Note that Argo CD may re-sync the
  bad manifest afterwards; reverting the tag-bump commit is the durable fix. Record what actually
  happens on Day 17.

---

### ADR-008: Structured JSON logging plus Prometheus, with no image data logged

**Date:** 2026-08-15
**Status:** Accepted
**Day in plan:** Days 9, 18

**Context:** M5 requires request/response logging that excludes sensitive data, plus request-count
and latency metrics. The spec allows logs, Prometheus, or simple in-app counters.

**Options considered:**
1. **JSON logs + Prometheus + Grafana** — machine-parseable logs and real dashboards. More setup.
2. **Plain-text logs + in-app counters on an endpoint** — much simpler, and explicitly permitted.
3. **Both** — JSON logs, Prometheus/Grafana, *and* in-app counters surfaced on `GET /`.

**Decision:** Option 3.

**Rationale:** Prometheus and Grafana carry the marks, but the in-app counter costs almost nothing
and makes request count demonstrable in the video without port-forwarding Grafana — useful insurance
given a 5-minute limit. Prometheus scraping is wired via a `ServiceMonitor` committed into `k8s/`,
so Argo CD manages monitoring config too.

**Consequences:**
- Logging is built on Day 9, not Week 3, so M5 becomes assembly rather than invention.
- **Sensitive-data rule:** log image byte size and a short hash, never image contents or filenames
  from user uploads. The spec calls this out explicitly, so the README states it explicitly too.
- A custom counter for predictions-by-class is added — it makes class skew visible, which is the
  monitoring signal that actually matters for this use case.
- `uvicorn --no-access-log` so the middleware's JSON line isn't duplicated by a plain-text one.

---

## Suggested ADRs still to write

Capture these as the decisions come up:

- ADR-009: Baseline CNN architecture — depth, width, regularisation, and why
- ADR-010: Augmentation policy — which transforms, and why those are valid for cat-vs-dog
- ADR-011: Corrupt-image handling — exclude at manifest time vs skip at load time
- ADR-012: Test-fixture strategy — committed sample images vs generated synthetic images
- ADR-013: CI dependency strategy — full `requirements.txt` vs serve + test subset, given TF's size
- ADR-014: k8s resource requests/limits — how the TF memory ceiling was chosen
- ADR-015: Labelled monitoring batch — provenance, and why it's disjoint from training data
