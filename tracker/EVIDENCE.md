# Evidence / Artifacts Checklist

Tracks every screenshot, figure, and artifact needed for the demo video and the submission zip.

Update each row as you capture it. Rows tagged **`[GAP-CV]`** or **`[GAP-CICD]`** are the ones that
cost marks in Assignment 1 — see `GUARDRAILS.md`. Those are non-negotiable.

Status: `[ ]` not captured | `[x]` captured | `[~]` covered by another shot

---

## M1 — Data, DVC, Model, CV, MLflow

### EDA + preprocessing figures (Day 4)

| File | Status | What it shows |
|------|--------|---------------|
| `reports/figures/class_balance.png` | [x] | Cat vs dog image counts, with the split balance |
| `reports/figures/image_dimensions.png` | [ ] | Distribution of source image sizes before the 224x224 resize |
| `reports/figures/sample_grid.png` | [x] | Sample images per class, own palette and captions |
| `reports/figures/augmentation_grid.png` | [x] | One image through flip/rotate/zoom/contrast — proves augmentation runs |
| `data/corrupt_files.txt` | [x] | The corrupt/truncated JPEGs found in the audit, with a count |

### DVC (Day 3)

| File | Status | What it shows |
|------|--------|---------------|
| `reports/screenshots/dvc/dvc_status_clean.png` | [ ] | `dvc status` clean + `git status` with no large blobs staged |
| `reports/screenshots/dvc/dvc_dag.png` | [ ] | `dvc dag` — preprocess → train stage graph |
| `reports/screenshots/dvc/dvc_repro.png` | [ ] | `dvc repro` running the tracked pipeline |
| `reports/screenshots/dvc/fresh_clone_pull.png` | [ ] | Fresh clone + `dvc pull` restoring the data |

### Cross-validation — **`[GAP-CV]`** (Day 6)

| File | Status | What it shows |
|------|--------|---------------|
| `reports/cv_results.csv` | [x] | **One row per (model, fold)** — 10 rows for 2 models × 5 folds, plus mean and std rows |
| `reports/figures/cv_comparison.png` | [x] | Mean CV score ± std error bars, both architectures side by side |
| `reports/figures/cv_fold_scores.png` | [x] | Per-fold scores showing fold-to-fold variance |
| `reports/screenshots/mlflow/cv_nested_runs.png` | [ ] | MLflow: parent run per model with 5 nested `fold_N` runs visible |
| `reports/screenshots/cv/cv_console_output.png` | [ ] | Terminal output of `python -m src.cross_validate` — per-fold lines as they complete |
| `reports/screenshots/cv/readme_cv_section.png` | [ ] | The `## Cross-validation` section rendered on GitHub |

### Model training (Day 5)

| File | Status | What it shows |
|------|--------|---------------|
| `reports/figures/loss_curves.png` | [x] | Train vs val loss per epoch |
| `reports/figures/accuracy_curves.png` | [x] | Train vs val accuracy per epoch |
| `reports/figures/confusion_matrix.png` | [x] | Test-set confusion matrix for the selected model |
| `reports/figures/roc_curve.png` | [x] | Test-set ROC with AUC |
| `models/model.h5` | [x] | The packaged model — **committed to Git**, not DVC-only |
| `models/model_metadata.json` | [x] | Versions, hyperparameters, input shape, class map, CV + test metrics |

### MLflow (Day 7)

| File | Status | What it shows |
|------|--------|---------------|
| `reports/screenshots/mlflow/01_runs_list.png` | [ ] | Experiment run list — both architectures plus nested folds |
| `reports/screenshots/mlflow/02_run_details.png` | [ ] | Params + metrics table for the selected model, with the run description populated |
| `reports/screenshots/mlflow/03_artifacts.png` | [ ] | Artifacts tab with confusion matrix / loss curves rendered inline |
| `reports/screenshots/mlflow/04_metric_curves.png` | [ ] | Per-epoch metric charts |
| `reports/screenshots/mlflow/05_registered_model.png` | [ ] | `catdog-classifier` registered with a version |

---

## M2 — API + Container

| File | Status | What it shows |
|------|--------|---------------|
| `reports/screenshots/api/swagger_docs.png` | [ ] | `/docs` showing `/health`, `/predict`, `/predict/base64`, `/` |
| `reports/screenshots/api/predict_cat.png` | [ ] | `/predict` with a cat image → label + probabilities |
| `reports/screenshots/api/predict_dog.png` | [ ] | `/predict` with a dog image → label + probabilities |
| `reports/screenshots/api/invalid_input_422.png` | [ ] | Wrong content-type or corrupt image → clean 422, no stack trace |
| `reports/screenshots/api/structured_logs.png` | [ ] | JSON log lines with method/path/status/latency_ms/label/confidence — **and no image bytes** |
| `reports/screenshots/docker/build_success.png` | [ ] | `docker build` completing, with the image tag |
| `reports/screenshots/docker/images_and_ps.png` | [ ] | `docker images` (size recorded) + `docker ps` showing healthy |
| `reports/screenshots/docker/predict_from_container.png` | [ ] | curl to the container → correct prediction |
| `reports/screenshots/docker/container_logs.png` | [ ] | Container logs: model loaded, uvicorn up, request lines |
| `reports/screenshots/docker/nonroot_user.png` | [ ] | `docker exec ... whoami` → not root |

---

## M3 — CI + Registry — **`[GAP-CICD]`**

| File | Status | What it shows |
|------|--------|---------------|
| `reports/screenshots/cicd/workflow_run_success.png` | [ ] | Green run on `main` with **all** jobs: lint, test, build, image-smoke, security |
| `reports/screenshots/cicd/lint_step.png` | [ ] | ruff step passing |
| `reports/screenshots/cicd/test_step_logs.png` | [ ] | pytest output with the pass count and coverage |
| `reports/screenshots/cicd/job_summary.png` | [ ] | The workflow job summary with test/coverage numbers |
| `reports/screenshots/cicd/test_artifacts.png` | [ ] | junit XML + coverage downloadable from the run page |
| `reports/screenshots/cicd/red_run_build_skipped.png` | [ ] | **Broken test → red run → `build` skipped.** Rule 6 evidence |
| `reports/screenshots/cicd/red_run_lint.png` | [ ] | Broken lint → `lint` fails first |
| `reports/screenshots/cicd/branch_protection.png` | [ ] | Branch protection settings requiring the checks |
| `reports/screenshots/cicd/pr_blocked.png` | [ ] | A PR blocked from merging by a failing check |
| `reports/screenshots/cicd/badge_in_readme.png` | [ ] | README with the green CI badge |
| `reports/screenshots/registry/ghcr_package_page.png` | [ ] | GHCR package page showing **SHA tag and `latest`** |
| `reports/screenshots/registry/push_step_logs.png` | [ ] | The push step logs with the digest |
| `reports/screenshots/registry/pr_builds_no_push.png` | [ ] | A PR run that builds but does not publish |
| `reports/screenshots/registry/fresh_pull_run.png` | [ ] | `docker pull` from GHCR in a clean context → container serves a prediction |
| `reports/screenshots/cicd/trivy_summary.png` | [ ] | Trivy scan results |

---

## M4 — Deployment + CD — **`[GAP-CICD]`**

| File | Status | What it shows |
|------|--------|---------------|
| `reports/screenshots/k8s/deployment_state.png` | [ ] | `kubectl get deploy,pods,svc -o wide` — 2/2 pods, Service with an external IP |
| `reports/screenshots/k8s/describe_pod_pulled.png` | [ ] | `describe pod` — probes configured **and events proving the image was pulled from GHCR**. Rule 7 evidence |
| `reports/screenshots/k8s/curl_through_cluster.png` | [ ] | `/predict` through the Service → correct prediction |
| `reports/screenshots/k8s/pod_logs_json.png` | [ ] | `kubectl logs` with the structured JSON request lines |
| `reports/screenshots/k8s/scale_4_replicas.png` | [ ] | Scaled to 4 replicas |
| `reports/screenshots/k8s/pod_self_heal.png` | [ ] | Deleted pod recreated automatically |
| `reports/screenshots/argocd/app_synced_healthy.png` | [ ] | Argo CD app view: Synced + Healthy, resource tree |
| `reports/screenshots/argocd/sync_history.png` | [ ] | Sync history tied to the tag-bump commit |
| `reports/screenshots/argocd/selfheal_revert.png` | [ ] | Hand-edited live resource reverted by Argo CD |
| `reports/screenshots/cd/tag_bump_commit.png` | [ ] | The CI commit rewriting the image tag in `k8s/deployment.yaml` |
| `reports/screenshots/cd/rollout_triggered.png` | [ ] | Rollout caused by that commit — no manual kubectl |
| `reports/screenshots/cd/smoke_test_pass.png` | [ ] | Post-deploy smoke test passing in the CD job (`/health` + `/predict`) |
| `reports/screenshots/cd/smoke_test_fail_red.png` | [ ] | **Broken deployment → smoke test fails → pipeline red.** Rule 6 evidence |
| `reports/screenshots/cd/rollback.png` | [ ] | `kubectl rollout undo` firing on failure, old version serving again |
| `reports/screenshots/cd/end_to_end.png` | [ ] | The full chain in one view: commit → CI green → GHCR tag → Argo sync → new prediction |

---

## M5 — Monitoring + Post-Deployment

| File | Status | What it shows |
|------|--------|---------------|
| `reports/screenshots/monitoring/prometheus_targets.png` | [ ] | Prometheus Targets — the service UP for all replicas |
| `reports/screenshots/monitoring/grafana_dashboard_full.png` | [ ] | Dashboard under load: request rate, p95 latency, error rate, predictions by class |
| `reports/screenshots/monitoring/metrics_endpoint.png` | [ ] | Raw `/metrics` output showing the request counter and latency histogram |
| `reports/screenshots/monitoring/inapp_counter.png` | [ ] | `GET /` showing the in-app request count |
| `monitoring/grafana_dashboard.json` | [ ] | Exported dashboard, re-imports cleanly |
| `monitoring/README.md` | [ ] | Install steps + the PromQL used for each panel |
| `data/monitoring/labelled_batch/labels.csv` | [ ] | ~100 labelled images held out from test, never trained on |
| `reports/screenshots/monitoring/replay_output.png` | [ ] | `replay_batch.py` output: live accuracy/precision/recall/F1 vs offline |
| `reports/figures/post_deploy_confusion_matrix.png` | [ ] | Confusion matrix built from **deployed** responses |
| `reports/figures/post_deploy_latency.png` | [ ] | Latency distribution from the replay |
| `reports/post_deployment_report.md` | [ ] | Offline vs live comparison table with the gap commented on |
| `reports/screenshots/monitoring/grafana_during_replay.png` | [ ] | Dashboard moving during the replay — ties monitoring to the labelled batch |

---

## Final artifacts

| File | Status | Notes |
|------|--------|-------|
| `README.md` | [ ] | Stack table, **`## Cross-validation` section**, results, diagram, setup, quickstart, DVC, CI/CD, deployment, monitoring, CI badge, **`## How this maps to the rubric`** |
| `reports/figures/architecture_diagram.png` | [ ] | Data → DVC → train/CV → MLflow → model → API → Docker → GHCR → Argo CD → Minikube → Prometheus/Grafana |
| `requirements.txt` / `requirements-serve.txt` | [ ] | Fully pinned; TF/NumPy/Pillow pins verified identical across both |
| `dvc.yaml`, `dvc.lock`, `*.dvc`, `.dvc/config` | [ ] | Committed |
| `.github/workflows/ci.yml`, `cd.yml` | [ ] | Green on `main`; fail-paths proven |
| `k8s/*.yaml`, `argocd/application.yaml` | [ ] | Committed and managed by Argo CD |
| `tracker/` | [ ] | PROGRESS, TASKS, DECISIONS, DAILY_LOG, EVIDENCE, GUARDRAILS all current |
| **Deliverable 1 — submission zip** | [ ] | Source + DVC config + CI/CD config + Docker + manifests + model artifacts. Verified by extracting to `/tmp` and running `pytest` |
| **Deliverable 2 — demo video** | [ ] | **Under 5:00**, code change → deployed prediction, all 5 modules touched |

---

## Capture discipline

- Screenshot the moment it happens — Rule 9. Recreating a self-heal or a red pipeline days later
  costs an hour and sometimes can't be reproduced.
- Consistent OS theme throughout (all light or all dark).
- Crop tight; no unrelated tabs or desktop clutter.
- Two shots of anything important, in case one is unreadable.
- For terminal output, make sure the **command** is visible in the frame, not just the result.
- After each capture, tick the row here the same day.
