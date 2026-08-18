# Assignment Tasks — Grading Tracker

**Total marks: 50** | **Earned (self-assessed): _ / 50**

**Last synced:** 2026-08-18 (Days 1-3 complete + audited; Day 4 CV run in progress). Only independently verified items are ticked — see `tracker/DAILY_LOG.md` for the evidence behind each.

Maps each of the 5 graded modules to deliverables, file paths, and the days they're worked on.
Update the **Status** and **Self-score** columns weekly.

Status legend: `[ ]` not started | `[/]` in progress | `[x]` done | `[~]` partial

> **Standing rule from A1 feedback ("No cross-validation, thin CI/CD"):** items tagged
> **`[GAP-CV]`** and **`[GAP-CICD]`** are the ones that cost marks last time. They are not done
> until a grader could find them in under a minute. Do not mark them `[x]` on the strength of the
> code existing — only when the artifact, the screenshot, and the README mention all exist.

---

## M1 — Model Development & Experiment Tracking — 10 marks
**Status:** `[/]` | **Self-score:** _ / 10 | **Days:** 1-7

### 1. Data & Code Versioning
- [x] Git for source versioning — repo `cats-dogs-mlops`, meaningful commit history, `main` branch
- [x] DVC initialised (`dvc init`), remote configured (local filesystem — ADR-004)
- [x] Raw dataset tracked (`data/raw.dvc`), **not** committed as blobs
- [x] Pre-processed data tracked (`data/processed.dvc` or a `dvc.yaml` stage output)
- [x] `dvc.yaml` pipeline with reproducible stages (download → preprocess → train)
- [ ] `dvc repro` runs clean on a fresh clone + `dvc pull`
- [x] `.dvc/config`, `dvc.lock`, and `*.dvc` files committed

### 2. Model Building
- [x] Pre-processing to 224x224 RGB (`src/preprocess.py`)
- [x] Stratified 80/10/10 train/val/test split, seed fixed at 42, split manifests written
- [x] Data augmentation (flip / rotation / zoom / contrast) applied to train only
- [x] Baseline CNN implemented (`src/model.py::build_baseline_cnn()`)
- [x] Second model for comparison — MobileNetV2 transfer learning (`build_transfer_model()`)
- [x] Trained model saved in a standard serialized format (`models/model.h5`)
- [x] `models/model_metadata.json` sidecar: versions, hyperparameters, input shape, class map, metrics

### 3. Experiment Tracking
- [ ] MLflow integrated (`mlflow.set_experiment` + `mlflow.start_run`)
- [ ] Parameters logged (architecture, optimizer, lr, batch size, epochs, augmentation config)
- [ ] Metrics logged (accuracy, precision, recall, F1, ROC-AUC, loss — train/val/test)
- [ ] Artifacts logged: **confusion matrix**, **loss curves**, **accuracy curves**, ROC curve
- [ ] Model logged via `mlflow.tensorflow.log_model` + registered as `catdog-classifier`
- [ ] All runs visible in the UI, 5+ screenshots captured

### 4. Cross-validation **`[GAP-CV]`** — the mark lost in A1
- [x] `src/cross_validate.py` — hand-rolled stratified k-fold (k=5), model rebuilt **and
      recompiled** per fold so no weight leaks across folds (ADR-003)
- [ ] Per-fold metrics printed AND logged as separate nested MLflow runs (`fold_1` … `fold_5`)
- [ ] `reports/cv_results.csv` — one row per fold per model, plus mean and std rows
- [ ] `reports/figures/cv_comparison.png` — mean ± std bar chart with error bars, both models
- [ ] `reports/figures/cv_fold_scores.png` — per-fold scatter/line showing fold-to-fold variance
- [ ] Final model choice justified **by CV mean**, not by a single test split (write it in the ADR)
- [ ] CV results table pasted into `README.md` under a `## Cross-validation` heading
- [ ] CV protocol documented: k, what was held out, epochs per fold, wall-clock cost

### Files
`src/data.py`, `src/preprocess.py`, `src/model.py`, `src/train.py`, `src/cross_validate.py`,
`dvc.yaml`, `dvc.lock`, `data/*.dvc`, `models/model.h5`, `models/model_metadata.json`,
`reports/cv_results.csv`, `reports/figures/*.png`, `notebooks/01_eda.ipynb`

---

## M2 — Model Packaging & Containerization — 10 marks
**Status:** `[ ]` | **Self-score:** _ / 10 | **Days:** 8-10

### 1. Inference Service
- [ ] FastAPI app (`api/main.py`), model loaded once at startup via `lifespan`
- [ ] `GET /health` — health check, returns 200 + status (used by k8s probes)
- [ ] `POST /predict` — accepts an image, returns **class label + class probabilities**
- [ ] Accepts `multipart/form-data` file upload
- [ ] Also accepts base64 JSON (`POST /predict/base64`) so curl/Postman demos are easy
- [ ] `GET /` — service metadata (name, version, which model, its CV/test scores)
- [ ] Input validation: content-type check, max file size, corrupt-image → clean 422/400
- [ ] Graceful failures: 503 if the model is missing, clean 500 on inference error
- [ ] Swagger UI at `/docs` renders both endpoints with examples

### 2. Environment Specification
- [ ] `requirements.txt` — full stack, **every** key ML library version-pinned
- [ ] `requirements-serve.txt` — inference-only subset
- [ ] TensorFlow / NumPy / Pillow pins verified identical across both files (ADR-002)
- [ ] `.python-version` pinning 3.11
- [ ] Clean-room check: fresh venv from `requirements.txt` → retrain → metrics reproduce

### 3. Containerization
- [ ] `Dockerfile` — slim base, serving deps only, non-root user, `HEALTHCHECK`
- [ ] `.dockerignore` keeping data/, mlruns/, notebooks/, tests/ out of the build context
- [ ] Image builds locally
- [ ] Container runs locally (`docker run -p 8000:8000`), `docker ps` shows healthy
- [ ] Prediction verified from the container via **curl** and via **Swagger/Postman**
- [ ] Image size recorded (TF makes this large — note the number, and the TFLite fallback if needed)

### Files
`api/main.py`, `api/__init__.py`, `Dockerfile`, `.dockerignore`, `requirements.txt`,
`requirements-serve.txt`, `.python-version`, `src/predict.py`

---

## M3 — CI Pipeline for Build, Test & Image Creation — 10 marks
**Status:** `[/]` | **Self-score:** _ / 10 | **Days:** 11-14

### 1. Automated Testing
- [x] Unit tests for a **data pre-processing function** — `tests/test_preprocess.py`
      (resize to 224x224, channel order, normalization range, split ratios + stratification,
      corrupt-image handling, augmentation only on train)
- [ ] Unit tests for a **model utility / inference function** — `tests/test_model.py`
      (build returns expected output shape, class-index map, `predict()` returns probabilities
      summing to 1, deterministic given a seed)
- [ ] API tests — `tests/test_api.py` via `TestClient` (health 200, predict on a fixture image,
      bad content-type 422, missing file 422)
- [x] Test fixtures committed: `tests/fixtures/cat_sample.jpg`, `dog_sample.jpg`, `corrupt.jpg`
- [x] `pytest -v` green locally; `pytest.ini` + root `conftest.py` in place
- [ ] Coverage measured and reported (`pytest --cov=src --cov=api`)

### 2. CI Setup — GitHub Actions **`[GAP-CICD]`**
- [ ] `.github/workflows/ci.yml` triggering on **push** and **pull_request** to `main`
- [ ] Job `lint` — ruff
- [ ] Job `test` — checkout, setup-python 3.11 with pip cache, install deps, pytest, junit XML
- [ ] Coverage + junit uploaded as workflow artifacts
- [ ] Job `build` — `docker/build-push-action` with buildx + layer cache, needs `test`
- [ ] Job `security` — Trivy image scan (extra credit against "thin")
- [ ] Job `image-smoke` — run the built image, hit `/health` and `/predict` inside CI
- [ ] Jobs are **gated**: a failing test must prevent build and push
- [ ] Pipeline failure demonstrated: break a test → red run → build skipped → revert
- [ ] CI status badge in `README.md`
- [ ] Branch protection on `main` requiring the checks to pass

### 3. Artifact Publishing **`[GAP-CICD]`**
- [ ] Image pushed to **GHCR** — `ghcr.io/<user>/catdog-api` (ADR-006)
- [ ] Auth via built-in `GITHUB_TOKEN` with `packages: write` permission
- [ ] Tagged by **commit SHA** (immutable) **and** `latest`
- [ ] Push only on `main`, not on PRs (build-only on PRs)
- [ ] Package visible on the GitHub Packages page, pull verified from a clean machine/context
- [ ] Multi-tag / metadata via `docker/metadata-action`

### Files
`tests/*.py`, `tests/fixtures/*`, `pytest.ini`, `conftest.py`, `ruff.toml`,
`.github/workflows/ci.yml`

---

## M4 — CD Pipeline & Deployment — 10 marks
**Status:** `[ ]` | **Self-score:** _ / 10 | **Days:** 15-17

### 1. Deployment Target
- [ ] Minikube cluster running (`minikube start --driver=docker`)
- [ ] `k8s/deployment.yaml` — 2 replicas, `/health` readiness + liveness probes, resource
      requests/limits, image referenced by SHA tag
- [ ] `k8s/service.yaml` — Service exposing the pods (LoadBalancer via `minikube tunnel`)
- [ ] GHCR pull secret wired (`imagePullSecrets`) or the package made public — decide and record
- [ ] `kubectl apply -f k8s/` brings up 2 healthy pods; prediction works through the cluster

### 2. CD / GitOps Flow **`[GAP-CICD]`**
- [ ] Argo CD installed into the cluster (`argocd` namespace) (ADR-007)
- [ ] `argocd/application.yaml` — Application pointing at the repo's `k8s/` path
- [ ] Auto-sync enabled with `prune: true` and `selfHeal: true`
- [ ] CI job bumps the image tag in `k8s/deployment.yaml` and commits it on `main` (GitOps trigger)
- [ ] Argo CD picks up the commit and rolls out the new image automatically — no manual kubectl
- [ ] Argo CD UI shows Synced / Healthy; sync history visible
- [ ] Self-heal proven: delete a pod / hand-edit a live manifest → Argo restores it
- [ ] End-to-end proven once: **code change on main → green CI → image pushed → auto-deployed →
      new prediction served.** This is the exact sentence the video has to demonstrate.

### 3. Smoke Tests / Health Check **`[GAP-CICD]`**
- [ ] `scripts/smoke_test_deployed.py` — calls `/health` **and** one `/predict` against the
      deployed URL, asserts the response shape, exits non-zero on failure
- [ ] CD job waits for rollout (`kubectl rollout status --timeout`) before smoke-testing
- [ ] Smoke test failure **fails the pipeline** (verified deliberately)
- [ ] Automatic rollback on smoke failure (`kubectl rollout undo`) — verified
- [ ] Deployment screenshots: pods, service, Argo UI, scale up/down, self-heal, rollback

### Files
`k8s/deployment.yaml`, `k8s/service.yaml`, `k8s/servicemonitor.yaml`,
`argocd/application.yaml`, `.github/workflows/cd.yml`, `scripts/smoke_test_deployed.py`

---

## M5 — Monitoring, Logs & Final Submission — 10 marks
**Status:** `[ ]` | **Self-score:** _ / 10 | **Days:** 18-21

### 1. Basic Monitoring & Logging
- [ ] Structured JSON request/response logging middleware (`python-json-logger`)
- [ ] Each log line carries method, path, status, latency_ms, predicted class, confidence
- [ ] **No sensitive data logged** — log image size/hash, never raw image bytes. State this in the
      README (the spec asks for it explicitly)
- [ ] Request **count** tracked — Prometheus counter, plus an in-app counter surfaced on `/`
- [ ] Request **latency** tracked — Prometheus histogram, p50/p95
- [ ] `/metrics` endpoint via `prometheus-fastapi-instrumentator`
- [ ] Custom metric: predictions by predicted class (cat vs dog), to make skew visible
- [ ] Prometheus scraping the service (`k8s/servicemonitor.yaml`, target UP)
- [ ] Grafana dashboard: request rate, p95 latency, error rate, class distribution
- [ ] Dashboard exported to `monitoring/grafana_dashboard.json`
- [ ] `monitoring/README.md` with the install steps and the PromQL used

### 2. Model Performance Tracking (Post-Deployment)
- [ ] `data/monitoring/labelled_batch/` — a held-out labelled batch (~100 images, never trained on)
- [ ] `scripts/replay_batch.py` — sends each image to the **deployed** endpoint, collects the
      predicted class, compares against the true label
- [ ] Live accuracy / precision / recall / F1 computed from the responses
- [ ] Live confusion matrix → `reports/figures/post_deploy_confusion_matrix.png`
- [ ] Comparison table: offline test metrics vs live deployed metrics, with the gap commented on
- [ ] Results written to `reports/post_deployment_report.md` (or logged as an MLflow run)
- [ ] Latency distribution from the replay recorded

### 3. Final Submission
- [ ] `README.md` complete: stack table, **CV results table**, architecture diagram, setup,
      quickstart, deployment, monitoring, CI badge
- [ ] `reports/figures/architecture_diagram.png` (generated by a script, as in A1)
- [ ] `EVIDENCE.md` fully ticked
- [ ] `tracker/` up to date — PROGRESS, TASKS, DECISIONS, DAILY_LOG all current
- [ ] **Deliverable 1:** zip with all source, DVC config, CI/CD config, Docker, k8s manifests,
      and trained model artifacts
- [ ] **Deliverable 2:** screen recording **under 5 minutes**, code change → deployed prediction
- [ ] Repo hygiene: `Co-Authored-By` trailers stripped, `v1.0` tag, `git status` and `dvc status`
      both clean

### Files
`api/main.py` (logging + metrics), `monitoring/grafana_dashboard.json`, `monitoring/README.md`,
`scripts/replay_batch.py`, `reports/post_deployment_report.md`, `README.md`,
`tracker/video_script.md`

---

## Cross-cutting: what "full marks" needs

Checked at the end of each week, not just at the end.

- [ ] Every module's artifacts exist **and** are referenced from `README.md`
- [ ] `[GAP-CV]` items all `[x]` — CV is visible in code, CSV, figure, MLflow, README, and video
- [ ] `[GAP-CICD]` items all `[x]` — registry push, auto-deploy, smoke gate, rollback, red run
- [ ] Fresh-clone reproducibility: `git clone` → `pip install -r requirements.txt` → `dvc pull` →
      `dvc repro` → `pytest` all succeed
- [ ] The container serves correctly in isolation (no host filesystem dependency)
- [ ] The pipeline fails loudly and legibly on a broken test and on a failed smoke test
- [ ] Video is under 5:00 and covers all 5 modules

## Academic Integrity Self-Check

- [ ] Architecture and tool decisions documented with rationale in `DECISIONS.md`
- [ ] EDA / figures personalized (own palette, own captions)
- [ ] Model architecture and augmentation choices reasoned, not copied from a tutorial
- [ ] No code copied verbatim from public tutorials
- [ ] Original write-up in README and post-deployment report
