# Daily Log

Append entries below. Newest at the bottom. One entry per work session.

The point of this log is twofold: (1) you can pick back up after a break without re-deriving
context, and (2) it gives raw material for the demo video narration and for any write-up.

---

## Template (copy this for each new entry)

```
### YYYY-MM-DD — Day N (PROGRESS.md reference)

**Time spent:** _ h
**Sub-tasks completed:** X / Y
**Status at end of day:** on track / behind / ahead

**What I did:**
-

**What worked:**
-

**What didn't / blockers:**
-

**Decisions made:** (link to DECISIONS.md ADR-NNN if architectural)
-

**Evidence captured:** (tick the matching rows in EVIDENCE.md)
-

**Guardrail check:** (any rule from GUARDRAILS.md at risk today?)
-

**Tomorrow's plan:**
-

**Commits:** <sha>, <sha>
```

---

## Entries

### 2026-08-15 — Day 0 (planning)

**Time spent:** ~1 h
**Sub-tasks completed:** N/A — planning phase
**Status at end of day:** planning complete, Day 1 starts next session

**What I did:**
- Read `Assignment 2.pdf` — 5 modules, 10 marks each, Cats vs Dogs binary image classification.
  Deliverables are a zip plus a sub-5-minute video; **no report.pdf this time** (unlike A1).
- Reviewed the Assignment 1 submission (`Assignment_1/MLOps.zip` → `heart-disease-mlops/`) for
  reusable conventions: `tracker/` system, split requirements files, ADR log, root `conftest.py`,
  `model_metadata.json` sidecar pattern.
- Root-caused the A1 grader comment *"No cross-validation, thin CI/CD"* and wrote it up in
  `GUARDRAILS.md`. Conclusion: CV was **present but invisible** (a `cv=5` argument to
  `GridSearchCV`, no per-fold artifacts, buried in a 24-page PDF), while CI/CD was **genuinely
  thin** (no registry push, no automated deploy, no post-deploy verification, no rollback).
- Seeded the tracker: `PROGRESS.md` (21 days), `TASKS.md` (5 modules with `[GAP-CV]` /
  `[GAP-CICD]` tags), `DECISIONS.md` (ADR-000 to ADR-008), `EVIDENCE.md`, `GUARDRAILS.md`.
- Confirmed the dataset: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

**What worked:**
- The A1 tracker system transfers almost unchanged; only the module mapping needed rewriting.
- Splitting the A1 feedback into "invisible" vs "genuinely missing" produced two different fixes
  instead of one vague resolution to "do better".

**What didn't / blockers:**
- None yet. Two unknowns to resolve on Day 1: the exact TensorFlow wheel situation across
  macOS arm64 vs linux/amd64, and whether the Kaggle API token is already set up.

**Decisions made:**
- ADR-000: optimise explicitly against the A1 feedback; CV and CI/CD get dedicated days
- ADR-001: 21-day timeline, one week per module group
- ADR-002: TensorFlow / Keras, `.h5`, with split requirements files
- ADR-003: hand-rolled stratified k-fold (not `scikeras`), two architectures
- ADR-004: DVC with a local-filesystem remote
- ADR-006: GHCR for image publishing, SHA-tagged
- ADR-007: Minikube + Argo CD GitOps, self-hosted runner for the smoke gate
- ADR-008: JSON logs + Prometheus/Grafana + in-app counters, no image data logged

**Evidence captured:**
- None yet — `EVIDENCE.md` seeded with every row the five modules will need.

**Guardrail check:**
- Rule 4 (name things what the rubric calls them) applied up front: the CV module is
  `src/cross_validate.py`, its output is `reports/cv_results.csv`, and the README heading will be
  `## Cross-validation`.

**Tomorrow's plan:**
- Day 1: repo skeleton, GitHub repo, Python 3.11 venv, resolve the TF wheel question before pinning,
  `.gitignore` / `ruff.toml` / `pytest.ini` / `conftest.py`, README skeleton, first commit.
- Clear the two A1 git frictions immediately: local `commit.gpgsign=false`, and a PAT with the
  `workflow` scope.

**Commits:** _

---

### 2026-08-15 — Day 1 (partial: environment + scaffold)

**Time spent:** ~2 h
**Sub-tasks completed:** 11 / 20 (data acquisition half still open)
**Status at end of day:** on track for the compressed 10-day plan

**What I did:**
- Compressed the 21-day plan to **10 days** (user deadline). Compression came out of Week 1 and
  polish per ADR-001; Day 4 (cross-validation) and Days 7–9 (CI/CD) are explicitly protected because
  they are the A1 mark-loss axes. Cuts are recorded in a table at the top of `PROGRESS.md` so nothing
  is silently dropped (Guardrail Rule 8).
- Scaffolded the repo: full directory tree, `git init -b main`, local `commit.gpgsign=false`
  (pre-empting the A1 friction), `.gitignore`, `.python-version`, `ruff.toml`, `pytest.ini`, root
  `conftest.py`, README skeleton.
- Created the Python 3.11 venv and installed the whole stack, then **pinned `requirements.txt` to the
  versions that actually resolved** rather than guessing.
- Installed the Kaggle token — the new `KGAT_` format goes to `~/.kaggle/access_token` (mode 600),
  **not** the older `kaggle.json` username/key pair.
- Wrote `scripts/daily_audit.py`: **100 day-gated checks** enforcing `GUARDRAILS.md`. Stdlib only, so
  it runs before dependencies exist. Emphasis on cross-artifact **consistency**, which is the class of
  problem that produced the A1 feedback.

**What worked:**
- Resolving the TF wheel question on Day 1 rather than Day 6 caught a genuine blocker (below) that
  would otherwise have surfaced as a mysterious `model.h5` load failure inside the container.
- Pinning from measured versions made audit check 16 (installed == pinned) pass immediately.

**What didn't / blockers:**
- **`tensorflow-cpu` maxes out at 2.20.0**, but macOS resolved `tensorflow` to **2.21.0**. That skew
  would have broken the container's model load. Fixed by pinning **both sides to 2.20.0**; verified
  the macOS arm64 wheel for 2.20.0 exists.
- **Keras 3.15.1 marks `.h5` legacy.** Probed it: `.h5` and `.keras` both round-trip with identical
  predictions (`np.allclose` @ 1e-6). Keeping `.h5` — the spec names it, and Guardrail Rule 4 says
  match the rubric's vocabulary. Warning suppressed in `pytest.ini` with a pointer to ADR-002.
- **mlflow 3.15.1 holds pandas at 2.x** — pip silently downgraded 3.0.5 → 2.3.3. Pinned explicitly.
- Pillow is no longer a TF dependency; had to request it explicitly.
- **Could not verify Kaggle auth from the agent shell** — it has no egress to kaggle.com (github.com
  fails identically, only the PyPI mirror is reachable). This is an agent-environment limit, not a
  token problem. **Open: the user must run the connectivity check.**

**Decisions made:**
- ADR-002 updated with all four concrete environment findings above (was speculative, now measured).

**Evidence captured:**
- None yet (Day 1 produces no screenshots). `EVIDENCE.md` seeded.

**Guardrail check:**
- Rule 8 honoured: every 10-day compression cut is written down in `PROGRESS.md`, not hidden.
- Rule 4 honoured: `.h5` kept over `.keras` specifically because the rubric names `.h5`.
- Audit at end of day: **20 pass · 1 fail · 1 manual · 78 not-yet**. The single failure is check 4
  (git remote) — blocked on creating the GitHub repo, which needs the user's account.

**Tomorrow's plan:**
- Close out Day 1: create the GitHub repo + remote (PAT needs `workflow` scope), **verify Kaggle
  connectivity**, then `scripts/download.sh`, `src/data.py`, and the corrupt-image audit.
- Then Day 2: DVC init + remote + `dvc.yaml`, preprocessing to 224x224, stratified 80/10/10
  manifests, augmentation, 3 EDA figures.

**Commits:** `1437337`

---

### 2026-08-15 — Day 2 (data pipeline, preprocessing, EDA)

**Time spent:** ~3 h
**Sub-tasks completed:** Day 1 code half + all of Day 2 except the notebook execution pass
**Status at end of day:** ahead on code, blocked on two external items

**What I did:**
- `src/data.py` — canonical constants defined once; `find_image_root()` locates class dirs by name
  at any depth (the archive has been republished in more than one shape, so hard-coding
  `PetImages/Cat` would have been fragile); two-pass corrupt audit.
- `scripts/download.sh` — dual path: Kaggle API, **or** a local archive argument, because kaggle.com
  reachability on this network is still unconfirmed. Error messages map 403/401/proxy to their
  actual causes.
- `src/preprocess.py` — the shared train/serve seam. `load_image()` forces RGB and raises
  `ValueError` so the API can return a clean 422. Stratified 80/10/10 manifests exclude corrupt files
  at build time.
- `src/model.py` — baseline CNN (242k params) and MobileNetV2 transfer (frozen base, 1,281
  trainable). Both uncompiled by design so CV can recompile per fold (ADR-003).
- `scripts/make_fixtures.py` — synthetic dataset + 7 test fixtures including 5 deliberate edge cases
  (truncated, zero-byte, greyscale, RGBA, not-an-image). This is what lets the pipeline be verified
  before the real download, and what lets CI run without `data/`.
- `src/eda.py` + `notebooks/01_eda.ipynb` — 3 figures, logic in the module, notebook as thin driver.
- DVC initialised, local remote configured, `preprocess` stage reproducible (`dvc status` clean).

**What worked:**
- The synthetic-dataset approach paid off immediately: 122 candidates → 120 usable → 2 corrupt
  caught, splits 80.0/9.2/10.8%, disjoint, batch `(4,224,224,3)` float32. The whole M1 data path is
  verified without a single real image.
- Verifying rather than assuming caught **three real bugs** (below).

**What didn't / blockers:**
- **Augmentation broke the [0,1] contract.** `RandomContrast`/`RandomZoom` produced max 1.02, but
  `build_transfer_model`'s Rescaling layer assumes clean [0,1] when mapping to MobileNetV2's [-1,1].
  Added a clip layer; range is now exactly [0.171, 1.000], visible in `augmentation_grid.png`.
- **DVC ran Apple's system Python 2.7.** `cmd: python -m ...` resolves against PATH, and an
  unactivated shell resolves `python` to 2.7, which dies with a SyntaxError on modern type hints —
  *and DVC had already deleted the stage outputs before the failed re-run*. Documented prominently in
  the README quickstart; the audit's dvc check now prepends `.venv/bin` to PATH.
- **ruff 0.16's default rule set is broader than earlier versions**, so an unchanged codebase could
  start failing CI purely because the linter moved. `ruff.toml` now pins an explicit `select` list.
- **Two of my own audit checks were wrong**, not the code: #46 compared class balance to a flat 2%
  tolerance when an 11-row val split can only express 9.1% steps (now `max(2%, 100/n_smallest)`); #38
  demanded a `train` DVC stage on day 2, which would have meant writing a manifest referencing a
  non-existent script (now day 3).
- **MobileNetV2 ImageNet weights cannot be downloaded here** — same proxy that blocks kaggle.com.
  Added an actionable error plus a `weights=None` escape hatch; architecture verified that way.
- **Notebook cell outputs are not populated** — this machine's security policy blocks Jupyter kernel
  socket binds, so `nbconvert --execute` fails even outside the sandbox. Every cell's logic was
  verified by running it as plain Python. **Open: one execute pass needed on an unrestricted machine.**

**Decisions made:**
- ADR-009 (baseline CNN design) and ADR-010 (augmentation policy) are captured in the module
  docstrings; still to be written up formally in `DECISIONS.md`.

**Evidence captured:**
- `class_balance.png`, `sample_grid.png`, `augmentation_grid.png`. Note these are from the
  **synthetic** fixture and must be regenerated from the real dataset before submission.

**Guardrail check:**
- Rule 6 (demonstrate, don't assert) applied to my own tooling: every bug above was found by running
  the thing, not by reading it.
- Audit at end of day: **40 pass · 1 fail · 57 not-yet**. The single failure is check 4 (git remote),
  blocked on repo creation.

**Tomorrow's plan (Day 3):**
- Blocked-on-user first: create the GitHub repo, verify Kaggle connectivity, then real download +
  audit, and regenerate the EDA figures from real data.
- Then `src/train.py`, both architectures trained, training figures, `models/model.h5` +
  metadata, and **measure per-epoch wall-clock to size the Day 4 CV protocol**.

**Commits:** `a286432`, `3635ef4`, `210b949`, `daf9261`

---

### 2026-08-18 — Day 3 (real data, training pipeline, GitHub remote)

**Time spent:** ~4 h
**Sub-tasks completed:** all of Day 3
**Status at end of day:** on track; M1 needs only cross-validation (Day 4)

**What I did:**
- Ingested the real Kaggle dataset: 24,998 files, 858MB, `PetImages/{Cat,Dog}`.
- Wrote and validated `src/train.py` end-to-end on real data.
- Wrote `tests/test_preprocess.py` — 21 tests, M3's required pre-processing test.
- Regenerated the EDA figures from real data and visually confirmed `sample_grid.png` shows real
  cats and dogs under correct labels.
- Added the `train` stage to `dvc.yaml`; `dvc status` clean.
- Created and pushed to `github.com/2024ac05731-cyber/cats-dogs-mlops` (10 commits, 4.89 MiB).
  Confirmed 0 files tracked under `data/raw/` or `mlruns/`, so the 858MB dataset stayed local.

**PER-EPOCH WALL-CLOCK (audit check 62 — this sizes the Day 4 CV protocol):**

| Config | Per epoch | 10 epochs | 5-fold x 2 architectures |
|---|---|---|---|
| 4,000-image subset | **40.9 s** | ~7 min | **~41 min** |
| 2,000-image subset | 20.4 s | ~3.5 min | ~20 min |
| Full 19,997 train | **~3.4 min** | ~34 min | **~5.7 h** |

Baseline CNN measured at ~10 ms/image/epoch on this CPU. **Decision: cross-validate on a
documented stratified subset (~4,000 images), not the full split.** ~41 min is affordable inside a
10-day plan; 5.7 h is not. The protocol (k=5, subset size, epochs/fold, wall-clock) goes in the
module docstring, ADR-005, and the README — a documented reduction is defensible, a silent one is
the A1 mistake (Guardrail Rule 8).

**Validated training run** (baseline CNN, 4k subset, 12 epochs): accuracy 0.7180, precision 0.6799,
recall 0.8240, F1 0.7450, ROC-AUC 0.8129. Both classes predicted; MLflow run logged.

**What worked:**
- Reconciling counts instead of trusting them found two real data bugs (below). "0 corrupt" looked
  like good news and was actually a broken detector.
- Running the pipeline on a small subset first surfaced the MLflow and degeneracy problems in ~1 min
  each, rather than after a 34-minute full run.

**What didn't / blockers — five bugs, all found by running things:**
1. **Truncated JPEG passed the audit.** Pillow does not raise on truncation — it warns and pads the
   missing scanlines with grey. `except Exception` never fired, so `Dog/9041.jpg` would have entered
   training as partially-grey garbage. Both `audit_images` and `load_image` now escalate that
   specific warning to an error. Result: 24,997 usable, 1 corrupt.
2. **Manifests were one image short (24,996 vs 24,997).** The corrupt-exclusion filter matched bare
   filenames, and PetImages numbers files per class — so the healthy `Cat/9041.jpg` was discarded
   alongside the truncated `Dog/9041.jpg`. Now matches repo-relative paths. Counts reconcile exactly.
3. **MLflow silently lost the entire first run.** MLflow 3.x *raises* on the filesystem backend
   unless `MLFLOW_ALLOW_FILE_STORE=true`, and my handler reported it as a passing warning — M1's
   tracking marks vanishing quietly, the exact "present but invisible" pattern from A1. A1 solved
   this same problem and I should have carried it over. Handler is now a loud banner.
4. **First model was degenerate, not undertrained.** accuracy 0.5000 with precision and recall of
   *exactly* 0.0 = every prediction one class, i.e. the class prior on a balanced set. Reporting it
   as "weak" would have been wrong in kind. Added `check_not_degenerate()`, which refuses to endorse
   such metrics and writes the warning into `model_metadata.json`.
5. **My own audit check 47 had bug #2.** It matched corrupt files by basename and false-positived on
   the healthy `Cat/9041.jpg`. Fixed to compare full paths. Worth noting the audit and the code
   shared my wrong assumption — independent checks would have caught it sooner.

**Decisions made:**
- CV protocol: 5-fold on a ~4,000-image stratified subset, from measured timing. Formalise in
  ADR-005 on Day 4.
- Repo is **public** (created via web UI). Convenient for Day 8 — a public GHCR package needs no
  `imagePullSecret`, resolving ADR-006 the simple way. Flagged the plagiarism-exposure tradeoff to
  the user; awaiting their call on private-repo-plus-public-package.
- Noted: `--subset` takes a stratified *head*, not a random sample. Fine for a pipeline check, but
  CV must sample randomly under `RANDOM_STATE` or fold composition is biased.

**Evidence captured:**
- `class_balance.png`, `sample_grid.png`, `augmentation_grid.png` (real data), plus `loss_curves`,
  `accuracy_curves`, `confusion_matrix`, `roc_curve` from the validated run.
- Still to capture: MLflow UI screenshots (Day 4, alongside the CV runs).

**Guardrail check:**
- Rule 3 (per-unit before aggregate) is what caught bug #2 — the totals only disagreed by 1.
- Rule 8 honoured: the CV subset decision is written down with the numbers behind it.
- Audit at end of day: **50 pass · 1 fail (this uncommitted log) · 47 not-yet**.

**Tomorrow's plan (Day 4 — PROTECTED, the A1 mark-loss axis):**
- `src/cross_validate.py`: hand-rolled `StratifiedKFold(5)`, model rebuilt **and recompiled** per
  fold, freshness asserted, random subset sampling under `RANDOM_STATE`.
- Both architectures. Needs the MobileNetV2 ImageNet weights, which download on first use — this
  machine's proxy blocked it, so confirm it works or fall back to `--no-pretrained`.
- `reports/cv_results.csv` with 10 fold rows plus mean/std, both CV figures, nested MLflow runs,
  README `## Cross-validation` section, ADR-005.

**Commits:** see `git log` (10 pushed to origin/main)

---

### 2026-08-18 — Day 4 (cross-validation — the PROTECTED day)

**Time spent:** ~3 h (plus 31 min unattended CV wall-clock)
**Sub-tasks completed:** all of Day 4
**Status at end of day:** M1 complete bar MLflow screenshots; 4/10 days done

**What I did — the A1 mark-loss fix, delivered:**
- `src/cross_validate.py`: `StratifiedKFold(5, shuffle=True, random_state=42)` over pooled train+val,
  both architectures, model rebuilt **and recompiled** per fold.
- **Result: transfer 0.9840 ± 0.0037 accuracy vs baseline 0.6198 ± 0.0373.** 10 fits, 31.3 min.
- Artifacts: `reports/cv_results.csv` (10 fold rows + 4 summary rows), `cv_comparison.png`,
  `cv_fold_scores.png`, 2 parent + 10 nested `fold_N` MLflow runs.
- README `## Cross-validation` section with the full per-fold table, protocol table, and both figures.
- ADR-005 (selection by CV mean), ADR-009 (baseline architecture), ADR-010 (augmentation policy),
  ADR-011 (no GPU — see below).
- `cross_validate` DVC stage added; `dvc status` clean; DAG now shows 3 stages + `data/raw.dvc`.
- `tests/test_model.py` — M3's required model/inference test. Suite now **39 passing**.

**Guardrail evidence — all twelve `[GAP-CV]` audit checks PASS.** Per-fold rows exist, fold scores
differ, means recompute by hand (0.619750 / 0.984000 exactly), test split provably untouched, README
heading present with a table, ADR records selection by CV mean.

**What worked:**
- Verifying fold isolation rather than assuming it. `verify_fold_isolation()` hashes every weight
  tensor before/after each fit and checks three properties. Weight leakage between folds *inflates*
  scores, so it would never announce itself — the only defence is a positive check.
- Running a 400-image/1-epoch rehearsal before the 31-minute run. Caught two output bugs cheaply.
- The variance turned out to be the real argument, not the mean. Baseline recall std of ±0.2429 means
  a single split could have reported 0.17 or 0.82 for the same architecture. That is the concrete
  case for why CV matters, and it went into ADR-005 and the README.

**What didn't / blockers:**
- **GPU is unavailable.** Profiled first: compute 18.3 ms/img vs data loading 1.25 ms/img, so GPU
  *would* help ~15x. But `tensorflow-metal 1.2.0` (newest, built for ~TF 2.17) **fails to load under
  TF 2.20** with a dlopen error, tested in an isolated venv. It declares no TF version pin, so pip
  installs it happily and TensorFlow then dies at import — installing into the project venv would
  have taken the project offline. Downgrading TF to regain Metal was rejected on timing: the
  compute-heavy phase is over. ADR-011.
- Also corrected a misconception worth recording: the M4 Pro has **one GPU with 20 cores**, not 20
  GPUs.
- **Fourth audit bug of the same shape.** Check 69 pooled both architectures into one mean AND counted
  the `std` summary rows as fold data, yielding 0.6716 and a false failure on a correct CSV. Pattern
  now unmistakable: my checks keep encoding the same assumptions as the code they check (basename
  matching twice, dvc.yaml-as-dataset-versioning, now row-tag handling). **The audit is a useful
  ratchet but not an independent witness** — worth stating plainly rather than trusting it as one.
- `models/model.h5` is still the Day 3 **baseline** validation artifact (accuracy 0.7180). Must be
  retrained with `--model transfer` before submission or the shipped model contradicts ADR-005.
  Recorded in ADR-005 consequences.

**Decisions made:** ADR-005, ADR-009, ADR-010, ADR-011 (all Accepted).

**Evidence captured:** `cv_comparison.png`, `cv_fold_scores.png`, `cv_results.csv`. Still to capture:
MLflow UI screenshots (the nested fold runs are the shot that matters).

**Guardrail check:**
- Rule 2 (three artifacts) satisfied for CV: code + CSV + figures + README section + MLflow runs.
- Rule 3 (per-unit before aggregate) is what check 69 enforces, and what caught my own arithmetic.
- Rule 8: the 4,000-image subset is documented in the module docstring, ADR-005 and the README.
- Audit at day 4: **62 pass · 1 fail (uncommitted work) · 35 not-yet**.

**Tomorrow (Day 5):** retrain the transfer model as the shipped artifact, capture MLflow screenshots
to close M1, then `src/predict.py` + `api/main.py` (FastAPI /health, /predict, /predict/base64, /).

**Commits:** see `git log`

---

### 2026-08-18 — Days 5-9 (API, container, CI/CD, monitoring)

**Time spent:** ~7 h
**Status at end of day:** 9/10 days of code complete; remaining work needs a cluster

**What I did:**
- **M2 complete and container-verified.** `src/predict.py`, `api/main.py` (6 routes),
  `tests/test_api.py`. Image 428MB content; healthy in 4s; runs as `appuser`; known cat ->
  `cat` @0.999999, dog -> `dog` @0.999874, corrupt -> 422.
- **Shipped the transfer model** on full data: test accuracy **0.9924**, F1 0.9924, ROC-AUC 0.9997.
  Re-verified independently at 0.9917 on 120 real test images.
- **M3 CI**: six gated jobs (lint, test, build, image-smoke, security, publish). `publish` is gated
  on `image-smoke`, not `build` — nothing reaches GHCR until the image has served a *correct*
  prediction. Multi-arch (amd64+arm64) with a manifest assertion.
- **M4**: k8s Deployment/Service/ServiceMonitor, Argo CD Application with prune+selfHeal,
  `cd.yml` (gitops-bump -> verify -> rollback), `scripts/smoke_test_deployed.py`.
- **M5**: `monitoring/README.md`, an authored `grafana_dashboard.json` (4 panels), and
  `scripts/replay_batch.py` validated live: **live accuracy 1.0000 vs offline 0.9924**,
  latency p50 45ms / p95 64ms, 0 errors over 60 requests.
- Suite now **64 tests**; audit at day 9: **92 pass, 1 fail (this log), 4 not-yet**.

**Bugs found and fixed — five, all by running things rather than reading them:**
1. **`python-multipart` missing entirely.** FastAPI raises at import without it, so `/predict` would
   have failed at container startup. Found by the API tests before the image was ever built.
2. **Docker build failed on arm64.** `tensorflow-cpu` is published for linux/amd64 ONLY, and
   `docker build` on Apple Silicon defaults to linux/arm64. My Day 1 check verified
   `manylinux2014_x86_64` — the right question, the wrong architecture. Fixed with
   `platform_machine` markers; both branches pin 2.20.0 so the trained-model version is unchanged.
3. **The audit's pin parser did not understand PEP 508 markers.** Left alone, the TF line would have
   stopped parsing and check 14 would have "passed" while comparing nothing. A check that silently
   stops checking is worse than one that fails.
4. **CI would have gone red on missing imports.** Static analysis before pushing found `matplotlib`
   module-level in `src/cross_validate.py` (which `test_model.py` imports) and `scikit-learn` needed
   at runtime by `test_preprocess.py`. matplotlib is now imported lazily inside the plot helpers —
   reaching `weights_fingerprint()` should not require a plotting stack.
5. **Fixtures were synthetic colour blobs**, so a pet-trained model classified both as "dog". The
   strongest assertion available was "these two probabilities differ", which would NOT catch an
   inverted class mapping. Replaced with real 160x160 test-split crops; tests now assert the correct
   *label*.

**A measurement error worth recording:** verifying the smoke gate, `$?` after a pipe returned
`tail`'s status, so both the pass and fail paths reported 0 and the gate looked broken. Re-measured
properly: exits 1 on failure, 0 on success. Nearly reported a working gate as broken.

**Decisions made:**
- ADR-011 (no GPU: tensorflow-metal fails to dlopen under TF 2.20).
- Multi-arch publish chosen over arm64-only: CI runners are amd64, the cluster is arm64, and "the
  image my cluster runs is the image CI published" is the claim M4 is graded on. `build` stays
  single-arch because `load: true` cannot load a multi-platform build.
- `k8s/deployment.yaml` placeholder is a deliberately INVALID tag, not `:latest`. With `:latest` the
  manifest would deploy *something* even if the GitOps bump never ran — working but untraceable,
  the exact failure Rule 7 exists to prevent. An unresolvable tag fails loudly instead.

**Guardrail check:**
- Rule 6 (demonstrate, don't assert) drove the smoke-gate exit-code verification and the
  `imagetools inspect` manifest assertion.
- **Audit-quality note, fifth instance:** my checks keep sharing the assumptions of the code they
  check (basename matching twice, dvc.yaml-as-dataset-versioning, CV row tags, PEP 508 markers). The
  audit is a ratchet, not an independent witness. Stated plainly here because I have been quoting its
  pass counts as evidence.

**Blocked on infrastructure (not code):**
- First CI run is in flight; result unknown at time of writing.
- `verify` job needs a **self-hosted runner** labelled `minikube`, or it queues forever — that is the
  smoke gate and rollback, 10 marks of M4.
- Minikube + Argo CD + kube-prometheus-stack need installing to capture the M4/M5 screenshots.

**Tomorrow (Day 10):** cluster bring-up, Argo CD sync, prove the red-CI and rollback paths, capture
all screenshots, architecture diagram, README rubric map, video, zip.

**Commits:** see `git log` (Days 5-9)
