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
