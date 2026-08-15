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
