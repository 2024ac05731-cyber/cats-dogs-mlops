# 10-Day Progress Dashboard (compressed)

**Overall:** 3/10 days complete (30%) | **Last updated:** 2026-08-18 (Days 1-3 done and audited; Day 4 CV running)

Status legend: `[ ]` not started | `[/]` in progress | `[x]` done | `[!]` blocked | `[~]` deferred

> **Compressed from the original 21-day plan.** Same scope, same five modules, ~2x the daily load
> (budget **5–6 h/day**, not 2–3 h). Per ADR-001 the compression comes out of Week 1 — the most
> familiar ground — and out of polish. It does **not** come out of Day 4 (cross-validation) or
> Days 7–9 (CI/CD), because those are the marks already lost once. See `GUARDRAILS.md`.

**Module map:** D1–D4 → M1 · D5–D6 → M2 · D7 → M3 · D8–D9 → M4 · D9–D10 → M5

## What was cut to fit 10 days

Deliberate, recorded so nothing is silently dropped (Guardrail Rule 8):

| Cut | Rationale | Risk |
|---|---|---|
| Deep EDA (5+ figures, statistical tests) → 3 essential figures | M1 here grades versioning + tracking, not exploratory depth (unlike A1's Task 1) | Low |
| CV on the full ~25k images → documented stratified subset | Makes 5-fold × 2 models affordable on CPU. Protocol written down, not hidden | Low — Rule 8 satisfied |
| Standalone buffer days (was 7, 14, 21) | No slack left; each day absorbs its own overrun | **High** — the main risk of this plan |
| `report.pdf` | Not a deliverable this time (zip + video only) | None |
| Separate API-hardening day → merged into the Docker day | Logging still lands before Week-2 equivalent, so M5 remains assembly | Low |
| Notebook depth → one thin `01_eda.ipynb` driver | M1 asks for notebooks under Git, not a rich analysis | Low |

**Not cut, and not negotiable:** the full CV artifact set, the multi-job CI, the GHCR push, Argo CD
auto-deploy, the post-deploy smoke gate, rollback, and every red-path demonstration.

---

## Day 1 — Skeleton, Environment, Data Acquisition & Audit
*(was Days 1–2)*
- **Status:** [x]
- **Date completed:** 2026-08-18
- **Time estimate:** ~5 h (actual ~5 h across sessions)
- **Goal:** Repo scaffolded, TF verified, dataset on disk and audited.

**Sub-tasks:**
- [x] Folder structure created (`data/{raw,processed,monitoring}`, `notebooks/`, `src/`, `api/`,
      `tests/fixtures/`, `k8s/`, `argocd/`, `monitoring/`, `scripts/`, `models/`,
      `reports/{figures,screenshots}`, `.github/workflows/`, `tracker/`)
- [x] `git init -b main`, `commit.gpgsign=false` set locally (A1 friction, pre-empted)
- [x] venv created with Python 3.11 (`/opt/homebrew/bin/python3.11`)
- [x] Kaggle token installed — `KGAT_` format at `~/.kaggle/access_token`, mode 600
- [x] **Resolve the TensorFlow wheel question before pinning** — install `tensorflow` on macOS
      arm64, record the exact version, then confirm a matching `tensorflow-cpu` minor version exists
      for linux/amd64 (`requirements-serve.txt`). ADR-002
- [x] Record numpy + Pillow versions TF resolved to — these are the pins that must match across both
      requirements files
- [x] `.gitignore` (Python, `.venv`, `mlruns/`, `data/raw/`, `data/processed/`, `models/*.h5` with a
      `!models/model.h5` exception, OS junk, `.dvc/cache`, **`access_token`, `kaggle.json`,
      `.kaggle/`**)
- [x] `.python-version` (3.11), `ruff.toml`, `pytest.ini`, root `conftest.py`
- [x] Starter `requirements.txt` with everything pinned to what actually resolved
- [x] `README.md` skeleton: stack table, structure, setup
- [x] **Confirm kaggle.com is reachable from this machine** — a managed-network proxy may block it.
      Test with `kaggle datasets files bhavikjikadara/dog-and-cat-classification-dataset`. If
      blocked → one-time browser download + `scripts/download.sh` ingesting a local archive
      (record as ADR-016)
- [x] Accept the dataset terms in a browser once (API returns 403 until you do)
- [x] `scripts/download.sh` — idempotent, `DATA_FORCE=1` to refetch
- [x] `src/data.py` — path constants, `RANDOM_STATE = 42`, `IMG_SIZE = (224, 224)`,
      `CLASS_NAMES = ["cat", "dog"]`, `download_data()`, `inspect()`
- [x] Record the **actual** archive layout (don't assume — this dataset has been republished in
      more than one shape) and the real per-class counts
- [x] **Corrupt-image audit** — walk every file, verify with Pillow, write `data/corrupt_files.txt`
      with a count. This dataset family is known for truncated JPEGs
- [x] GitHub repo `cats-dogs-mlops` created; remote added; first commit pushed
- [x] PAT has the `workflow` scope (A1 friction — needed before Day 7)

**Verification:**
- [x] `import tensorflow` works — 2.20.0 / keras 3.15.1 (pinned to match `tensorflow-cpu`, ADR-002)
- [x] Real counts recorded: 24,998 files, 24,997 usable, 1 corrupt (`Dog/9041.jpg`, truncated)
- [x] Re-running the download is idempotent; `.synthetic` marker prevents fixture/real confusion
- [x] Repo live at github.com/2024ac05731-cyber/cats-dogs-mlops

**Risk:** the Kaggle connectivity check is the gate for the whole plan. Do it first, not last.

---

## Day 2 — DVC, Preprocessing, Splits
*(was Days 3–4)*
- **Status:** [x]
- **Date completed:** 2026-08-18
- **Time estimate:** ~6 h (actual ~4 h)
- **Goal:** Data versioned, 224x224 pipeline, stratified 80/10/10, augmentation working.

**Sub-tasks:**
- [x] `dvc init`; local-filesystem remote (`dvc remote add -d localremote ~/dvc-store/cats-dogs`) — ADR-004
- [x] `dvc add data/raw`; commit `data/raw.dvc`; `dvc push`
- [x] `dvc.yaml` with `preprocess` and `train` stages; declare metrics/plots; commit `dvc.lock`
- [x] `src/preprocess.py::load_image()` — decode, force RGB (kills greyscale/alpha surprises),
      resize 224x224, scale to [0,1]
- [x] `build_split_manifests()` — stratified 80/10/10, seed 42, writes
      `data/processed/{train,val,test}.csv`; **excludes the Day-1 corrupt list at manifest level**
- [x] `make_dataset(manifest, augment=False)` → batched, prefetched `tf.data.Dataset`
- [x] Augmentation as Keras layers (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`),
      **train only**
- [x] 3 essential figures: `class_balance.png`, `sample_grid.png`, `augmentation_grid.png`
- [x] `notebooks/01_eda.ipynb` — thin driver calling the above (M1 wants notebooks under Git)
- [x] `dvc repro preprocess`; commit

**Verification:**
- [x] Batch is `(batch, 224, 224, 3)` float32 in [0,1]
- [x] Class balance within ~1% across all three splits
- [x] **No file in more than one split** — assert it
- [x] Augmentation provably absent from val/test
- [x] `dvc status` clean; no large blobs in `git status`

**Note:** write these assertions so they lift straight into Day 7's `test_preprocess.py`.

---

## Day 3 — Two Architectures + Training
*(was Day 5)*
- **Status:** [x]
- **Date completed:** 2026-08-18
- **Time estimate:** ~5 h (actual ~4 h)
- **Goal:** Both models train; measured epoch time sizes the Day 4 CV budget.

**Sub-tasks:**
- [x] `src/model.py::build_baseline_cnn()` — own architecture (Conv/BN/Pool → GAP → Dense →
      sigmoid), choices documented in the docstring (ADR-009)
- [x] `src/model.py::build_transfer_model()` — MobileNetV2, ImageNet weights, frozen base, custom
      head. Chosen partly *because* it converges fast enough to cross-validate (ADR-003)
- [x] `src/train.py` — `--model {baseline,transfer}`, `--epochs`; trains on train, validates on val,
      evaluates on test
- [x] Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
- [x] Test metrics: accuracy, precision, recall, F1, ROC-AUC
- [x] Figures: `loss_curves.png`, `accuracy_curves.png`, `confusion_matrix.png`, `roc_curve.png`
- [x] Save `models/model.h5` + `models/model_metadata.json` (versions, hyperparameters, input shape,
      class map, metrics)
- [x] **Record wall-clock per epoch for both architectures** — this decides Day 4's protocol

**Verification:**
- [x] Both train without OOM or NaN loss
- [x] Test accuracy clearly beats 50% chance
- [x] `keras.models.load_model('models/model.h5')` reloads and scores identically
- [x] Per-epoch time logged in `DAILY_LOG.md`

**Actual results (2026-08-18):**
- baseline CNN: 242,369 params (241,665 trainable); transfer: 2,259,265 params (1,281 trainable —
  base correctly frozen)
- Validated fit (baseline, 4k subset, 12 epochs): accuracy 0.7180, precision 0.6799, recall 0.8240,
  F1 0.7450, ROC-AUC 0.8129. Saved artifact independently re-verified at 0.7250 on 80 real test images.
- **Per-epoch wall-clock: 40.9 s at 4,000 images (~10 ms/image); ~3.4 min on the full 19,997 split.**

**Decision point → DECIDED:** 5-fold CV on a ~4,000-image random stratified subset (~41 min) rather
than the full pool (~5.7 h). Documented in `src/cross_validate.py`, ADR-005, and the README.

---

## Day 4 — 5-Fold Cross-Validation **`[GAP-CV]` — PROTECTED DAY**
*(was Day 6 — unchanged in scope)*
- **Status:** [/] src/cross_validate.py written, validated, committed; full run in progress
- **Time estimate:** ~6 h including training wall-clock
- **Goal:** Cross-validation a grader cannot miss.

**This day does not get compressed, shortened, or merged.** It is the direct fix for half the A1
feedback. If Day 3 overruns, start the CV runs first thing and finish Day 3's figures while they run.

**Sub-tasks:**
- [ ] `src/cross_validate.py` — `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over
      the pooled **train + val** files; the test split stays untouched
- [ ] Per fold: **rebuild and recompile from scratch** — no weights or optimizer state may cross a
      fold boundary
- [ ] Assert fold-model freshness (compare initial weight hashes) so no-leakage is *provable*
- [ ] Run CV for **both** architectures — makes CV a real selection instrument
- [ ] Log each fold as a **nested MLflow run** (`fold_1`…`fold_5`) under one parent per model
- [ ] `reports/cv_results.csv` — **one row per (model, fold)** = 10 rows, plus mean and std rows
- [ ] `reports/figures/cv_comparison.png` — mean ± std error bars, both models
- [ ] `reports/figures/cv_fold_scores.png` — per-fold variance
- [ ] Select the production model **by CV mean**, not the single test split; write ADR-005
- [ ] Paste the fold table into `README.md` under a literal `## Cross-validation` heading
- [ ] Document the protocol: k, pooled splits, epochs/fold, subset size, total wall-clock
- [ ] Capture `cv_console_output.png` and `cv_nested_runs.png`

**Verification:**
- [ ] 10 fold rows present (2 models × 5 folds)
- [ ] **Fold scores differ from each other** — identical values mean the loop is broken
- [ ] `mean ± std` recomputes by hand from the per-fold rows
- [ ] Test split provably never touched (assert file sets disjoint)
- [ ] README section renders on GitHub

---

## Day 5 — MLflow + FastAPI Service
*(was Days 7–8)*
- **Status:** [ ]
- **Time estimate:** ~6 h
- **Goal:** M1 closed. Inference API serving real predictions.

**Sub-tasks — MLflow (closes M1):**
- [ ] Wire MLflow into `src/train.py` and `src/cross_validate.py` (file store at `./mlruns`,
      experiment `cats-dogs-classification`)
- [ ] Log params: architecture, optimizer, lr, batch size, epochs, augmentation config, img size, seed
- [ ] Log metrics: train/val/test accuracy, precision, recall, F1, ROC-AUC, losses; per-epoch too
- [ ] Log artifacts: **confusion matrix, loss curves, accuracy curves**, ROC, `cv_results.csv`, both
      CV figures, augmentation grid
- [ ] `mlflow.tensorflow.log_model` + register as `catdog-classifier`
- [ ] `scripts/mlflow_ui.sh`; capture 5 MLflow screenshots
- [ ] **M1 gate:** no `[ ]` left in the M1 section of `TASKS.md`

**Sub-tasks — API (starts M2):**
- [ ] `src/predict.py` — `load_model()`, `load_metadata()`, `predict_image(bytes)`. **Reuses
      `src/preprocess.py::load_image`** so train and serve share one code path
- [ ] `api/main.py` — FastAPI, model loaded once in `lifespan`
- [ ] `GET /health`, `POST /predict` (multipart → label + probabilities),
      `POST /predict/base64`, `GET /` (metadata + CV mean + test accuracy)
- [ ] Verify against a known cat and a known dog; check `/docs`

**Verification:**
- [ ] Parent + 5 nested fold runs visible per architecture; artifacts render inline
- [ ] Registered model has a version
- [ ] Cat predicts cat, dog predicts dog; probabilities sum to ~1

---

## Day 6 — API Hardening, Logging, Docker
*(was Days 9–10)*
- **Status:** [ ]
- **Time estimate:** ~6 h
- **Goal:** M2 closed. Container serves predictions with no host dependency.

**Sub-tasks — hardening + logging (feeds M5):**
- [ ] Validation: content-type allowlist, max upload size, decode failure → clean 400/422
- [ ] 503 if the model failed to load; catch-all handler → clean JSON on 500
- [ ] Structured JSON logging (`python-json-logger`), one object per request: method, path, status,
      latency_ms, predicted label, confidence, image byte size
- [ ] **Log image size and a short hash — never image contents.** README states this (spec asks) — ADR-008
- [ ] `/metrics` via `prometheus-fastapi-instrumentator`; custom counter for predictions-by-class;
      in-app request counter on `GET /`

**Sub-tasks — Docker:**
- [ ] `requirements-serve.txt` — inference-only subset with `tensorflow-cpu`
- [ ] **Verify TF/numpy/Pillow pins match `requirements.txt`** — skew breaks `model.h5` loading
- [ ] `Dockerfile`: `python:3.11-slim`, deps layer first, copy only `src/`, `api/`, `models/model.h5`,
      metadata; non-root user; `HEALTHCHECK` via stdlib urllib; `CMD uvicorn ... --no-access-log`
- [ ] `.dockerignore` excluding `data/`, `mlruns/`, `notebooks/`, `reports/`, `tests/`, `tracker/`,
      `.venv/`, `.git/`
- [ ] Build; run; verify `/predict` via **curl** and **Swagger**; record image size
- [ ] Screenshots: swagger, predict cat/dog, 422, JSON logs, build, images+ps, predict from
      container, container logs, non-root

**Verification:**
- [ ] Oversized / wrong-type / corrupt inputs each return a clean 4xx
- [ ] Logs are one valid JSON object per line (pipe through `jq`); no image bytes anywhere
- [ ] Container prediction matches the local API for the same image
- [ ] **M2 gate:** no `[ ]` left in the M2 section of `TASKS.md`

---

## Day 7 — Tests + Full CI + Registry **`[GAP-CICD]`**
*(was Days 11–14)*
- **Status:** [ ]
- **Time estimate:** ~7 h — the longest day
- **Goal:** M3 closed. Multi-job gated pipeline publishing to GHCR.

**Sub-tasks — tests:**
- [ ] `tests/fixtures/` — tiny `cat_sample.jpg`, `dog_sample.jpg`, `corrupt.jpg`, a greyscale/alpha oddity
- [ ] `tests/conftest.py` — model, TestClient, temp-manifest fixtures
- [ ] `tests/test_preprocess.py` — **the required preprocessing test**: resize shape; greyscale and
      RGBA → 3 channels; range [0,1]; 80/10/10 ratios; stratification; splits disjoint; corrupt
      excluded; augmentation train-only
- [ ] `tests/test_model.py` — **the required model/inference test**: output shape; class-index map
      matches metadata; probabilities sum to 1; deterministic given a seed
- [ ] `tests/test_api.py` — health 200; predict on fixture; wrong content-type 422; missing file 422
- [ ] **Tests must pass without the dataset present** (fixtures only — CI has no `data/`)
- [ ] `pytest -v` green; coverage recorded; `ruff check .` clean

**Sub-tasks — CI:**
- [ ] `.github/workflows/ci.yml` on push + PR to `main`; concurrency group; least-privilege `permissions`
- [ ] Job `lint` (ruff) · job `test` (pytest, junit XML, coverage, artifacts, `$GITHUB_STEP_SUMMARY`)
- [ ] Job `build` — `needs: [lint, test]`, buildx + GHA layer cache
- [ ] GHCR login via `GITHUB_TOKEN` + `packages: write`; `metadata-action` tagging **commit SHA**
      and `latest`; `push: github.event_name != 'pull_request'` — ADR-006
- [ ] Job `image-smoke` — run the image, `/health`, then `/predict` with a fixture
- [ ] Job `security` — Trivy scan
- [ ] **Prove the gates:** break a test → red, `build` **skipped** → screenshot → revert. Break lint
      → `lint` fails first → screenshot → revert
- [ ] Branch protection on `main` requiring lint/test/build; screenshot a blocked PR
- [ ] CI badge in `README.md`
- [ ] Pull the published image fresh and verify it serves

**Verification:**
- [ ] Both tags on the GHCR package page
- [ ] A PR builds but does **not** push
- [ ] Red-run screenshot exists showing `build` skipped
- [ ] **M3 gate:** no `[ ]` left in the M3 section of `TASKS.md`

**Note:** installing full TF in CI is slow. Cache pip hard; if the test job drags, install
`requirements-serve.txt` + test deps instead of the whole training stack, and comment why.

---

## Day 8 — Minikube, Manifests, Argo CD GitOps **`[GAP-CICD]`**
*(was Days 15–16)*
- **Status:** [ ]
- **Time estimate:** ~7 h
- **Goal:** Deployment happens because of a commit, not a human.

**Sub-tasks — cluster + manifests:**
- [ ] `minikube start --driver=docker` with 4GB+ memory (TF is hungry)
- [ ] GHCR access: make the package **public**, or wire an `imagePullSecret` from a PAT with
      `read:packages`. Record the outcome in ADR-006
- [ ] `k8s/deployment.yaml` — 2 replicas, **image by SHA tag**, `imagePullPolicy: Always`,
      `/health` readiness + liveness probes with generous `initialDelaySeconds` (TF starts slow),
      resource requests/limits sized for TF
- [ ] `k8s/service.yaml` — LoadBalancer, 80 → 8000
- [ ] `kubectl apply -f k8s/`; `minikube tunnel`; curl `/health` and `/predict` through the cluster
- [ ] Scale to 4 and back to 2; delete a pod to show self-heal

**Sub-tasks — Argo CD:**
- [ ] Install Argo CD into the `argocd` namespace; port-forward the UI; get the admin password
- [ ] `argocd/application.yaml` — repo + `path: k8s`, target `main`, `syncPolicy.automated` with
      `prune: true` and `selfHeal: true`
- [ ] Apply; confirm it adopts the Deployment and reports Synced / Healthy
- [ ] CI job on `main` that rewrites the image tag in `k8s/deployment.yaml` to the new SHA and
      commits it back — **guard against retriggering CI** (`[skip ci]` or path filter) on the first
      attempt, not after ten runs
- [ ] Watch Argo CD detect the commit and roll out automatically
- [ ] Prove self-heal: `kubectl edit` a live resource → Argo reverts it
- [ ] Screenshots: deployment state, `describe pod` **proving a GHCR pull**, curl through cluster,
      JSON pod logs, scale, self-heal, Argo app + sync history + self-heal revert, tag-bump commit
- [ ] Write ADR-007

**Verification:**
- [ ] Both pods `Running 1/1`, probes passing
- [ ] `describe pod` events prove the image was **pulled from GHCR**, not side-loaded
- [ ] Argo CD sync event ties to the tag-bump commit
- [ ] The tag-bump commit does **not** cause an infinite CI loop

---

## Day 9 — Smoke Gate, Rollback, End-to-End, Monitoring **`[GAP-CICD]`**
*(was Days 17–18)*
- **Status:** [ ]
- **Time estimate:** ~7 h
- **Goal:** M4 closed. Monitoring live.

**Sub-tasks — smoke gate + rollback (closes M4):**
- [ ] `scripts/smoke_test_deployed.py --url <base>` — `/health` **and** `/predict` with a bundled
      fixture; asserts status, response shape, valid label, probabilities ≈ 1; retries with backoff;
      **exits non-zero on failure**
- [ ] Register a **self-hosted runner** on the Mac (GitHub-hosted runners cannot reach a local
      cluster) — ADR-007
- [ ] CD job: `kubectl rollout status --timeout=180s`, then the smoke test
- [ ] Smoke failure → job fails → `kubectl rollout undo` in `if: failure()` → verify the old version
      serves again. Note whether Argo re-syncs the bad manifest, and record the durable fix
- [ ] **Prove the gate:** deploy a deliberately broken image (`/predict` → 500), watch the pipeline
      go red and roll back, then revert
- [ ] **Full end-to-end rehearsal, timed:** edit code on `main` → CI green → GHCR push → tag bump →
      Argo sync → rollout → smoke passes → new prediction. This is the video's spine
- [ ] Screenshots: smoke pass, smoke fail red, rollback, end-to-end
- [ ] **M4 gate:** no `[ ]` left in the M4 section of `TASKS.md`

**Sub-tasks — monitoring (starts M5):**
- [ ] Helm-install `kube-prometheus-stack` with `serviceMonitorSelectorNilUsesHelmValues=false`
- [ ] `k8s/servicemonitor.yaml` — committed into `k8s/` so Argo CD manages it too
- [ ] Confirm the Prometheus target is UP for all replicas
- [ ] Grafana dashboard, 4 panels: request rate, p95 latency, error rate, **predictions by class**
- [ ] PromQL documented in `monitoring/README.md`; generate load so panels move
- [ ] Export → `monitoring/grafana_dashboard.json`
- [ ] Screenshots: Prometheus targets, full dashboard under load, `/metrics` raw, in-app counter

**Verification:**
- [ ] Smoke test provably exits non-zero on a broken deployment
- [ ] Pipeline goes red and rollback restores service
- [ ] All 4 Grafana panels show data; exported JSON re-imports cleanly

---

## Day 10 — Post-Deployment Tracking, README, Video, Submit
*(was Days 19–21)*
- **Status:** [ ]
- **Time estimate:** ~7 h
- **Goal:** M5 closed. Both deliverables shipped.

**Sub-tasks — post-deployment tracking (closes M5):**
- [ ] `data/monitoring/labelled_batch/` — ~100 labelled images held out from **test**, never trained
      on, plus `labels.csv`
- [ ] `scripts/replay_batch.py --url <base> --n 100` — POSTs each to the **deployed** endpoint,
      records predicted label, confidence, latency vs the true label
- [ ] Live accuracy / precision / recall / F1; `post_deploy_confusion_matrix.png`;
      `post_deploy_latency.png`
- [ ] `reports/post_deployment_report.md` — offline vs live comparison table, gap commented on
- [ ] Screenshot Grafana **during** the replay, tying monitoring to the labelled batch

**Sub-tasks — submission:**
- [ ] `reports/make_architecture_diagram.py` → `architecture_diagram.png` (data → DVC → train/CV →
      MLflow → model → API → Docker → GHCR → Argo CD → Minikube → Prometheus/Grafana)
- [ ] `README.md` final: stack table, **`## Cross-validation` fold table**, results, diagram, setup,
      quickstart, DVC, CI/CD, deployment, monitoring, post-deploy results, CI badge
- [ ] **`## How this maps to the rubric`** — M1–M5, each with the paths that satisfy it (Guardrail
      Rule 1)
- [ ] **Run the full `GUARDRAILS.md` Part 3 pre-submission audit** — all 40 rows
- [ ] `tracker/video_script.md`; rehearse once **timed**, unrecorded
- [ ] Record the demo, **under 5:00**: spine = code change → deployed prediction, with beats for CV,
      MLflow, tests, CI/CD, monitoring, post-deploy
- [ ] Video link into `README.md`
- [ ] Hygiene: strip `Co-Authored-By` trailers; `git status` + `dvc status` clean; tag `v1.0`
- [ ] **Confirm `models/model.h5` + metadata are committed to Git** — the grader cannot `dvc pull`
      from a local remote
- [ ] Build the zip; **verify by extracting to `/tmp` and running `pytest` + loading the model**
- [ ] Final `TASKS.md` self-score; final `DAILY_LOG.md` entry; submit

**Verification:**
- [ ] Live accuracy within a sensible margin of offline test accuracy (a real gap means a
      train/serve preprocessing skew — worth finding)
- [ ] Every `GUARDRAILS.md` audit row `[x]`
- [ ] Video under 5:00, covers all five modules
- [ ] Extracted zip is self-contained

---

## Gates

Each gate closes a module group. **Do not carry an open gate forward** — that is how a compressed
plan silently loses a module.

| Gate | Day | Requirement |
|---|---|---|
| M1 | 5 | DVC + model + **CV artifacts** + MLflow; CV visible in README |
| M2 | 6 | API + pinned deps + container serving predictions |
| M3 | 7 | Tests + multi-job CI + **GHCR push**; red run proven |
| M4 | 9 | Manifests + **Argo CD auto-deploy** + smoke gate + rollback |
| M5 | 10 | Logging + metrics + dashboard + post-deploy tracking |

## Risk register (compressed plan)

| Risk | Likelihood | Mitigation |
|---|---|---|
| **No buffer days left** | Certain | Days 7–10 are 7 h. If a day overruns, cut polish (extra figures, Trivy, notebook depth) — never a `[GAP-*]` item |
| Kaggle blocked by managed-network proxy | Medium | **Test on Day 1 before anything else.** Fallback: browser download + local-archive ingest (ADR-016) |
| CV wall-clock too slow on CPU | High | Transfer learning + documented stratified subset. Protocol decided Day 3 from measured epoch time |
| Corrupt JPEGs crash training mid-epoch | High | Day 1 audit; exclude at manifest level on Day 2 |
| TF wheel differs macOS vs linux image | High | Resolve Day 1 before pinning; verify `model.h5` loads in the container on Day 6 |
| GitHub-hosted runners can't reach local Minikube | Certain | Self-hosted runner on the Mac (Day 9) |
| Tag-bump commit retriggers CI in a loop | Medium | Guard on the first attempt (Day 8) |
| Pods OOMKilled by TF memory | Medium | Generous limits; read `kubectl describe` before debugging code |
| Argo CD setup eats Day 8 | Medium | Fallback in ADR-007 option 2: self-hosted runner running `kubectl set image`. Still automated, still passes M4 — decide by midday Day 8 |
| 5-min video too short for 5 modules | High | Stage everything up-front, pre-stage the code change, rehearse timed on Day 10 |
