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
- **Wheel split is a real trap:** `tensorflow-cpu` has no macOS arm64 wheel and `tensorflow-metal`
  is macOS-only. So `requirements.txt` carries `tensorflow` (plus optional `tensorflow-metal` for
  local speed) while `requirements-serve.txt` carries `tensorflow-cpu` for the image. Keep the
  **minor version aligned** across both or `model.h5` may fail to load.
- Verify the exact available wheels on Day 1 *before* pinning, and verify `model.h5` loads inside
  the container on Day 10.
- Expect a serving image over 1GB. Document the number rather than fighting it. If it becomes a
  problem, the fallback is exporting to TFLite or SavedModel and serving without full TF — noted
  here so it isn't re-derived later.
- TF's memory appetite affects k8s resource limits (Day 15) and startup time affects probe
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

### ADR-005: Production model selection (placeholder — write on Day 6)

**Date:** _
**Status:** Proposed
**Day in plan:** Day 6

**Context:** Two architectures cross-validated over 5 folds. One must be packaged as
`models/model.h5` and served.

**Decision:** _to be recorded once CV numbers exist._

**Rationale to capture:** which model won on **CV mean** (not the single test split), whether the
std overlaps, and the inference-cost/interpretability trade-off. If the test split and the CV mean
disagree, say so explicitly and follow CV — and explain why. That reasoning is itself the evidence
that cross-validation was actually used for something.

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
