# Guardrails — How Marks Get Lost, and the Rules That Stop It

Assignment 1 came back with: **"No cross-validation, thin CI/CD."**

This file exists because that feedback was *avoidable*. It is the pre-submission audit. Nothing
gets submitted until every rule here passes.

---

## Part 1 — Root cause: why A1 lost those marks

### "No cross-validation" — it existed, but nothing pointed at it

A1's `src/train.py` ran `GridSearchCV(pipe, grids[name], cv=5, scoring="roc_auc")` and logged
`cv_auc_mean` / `cv_auc_std` to MLflow. So cross-validation ran. Why did a grader conclude it
didn't?

| Root cause | What actually happened in A1 |
|---|---|
| **CV was a side effect of tuning, not a deliverable** | `cv=5` was an argument to `GridSearchCV`. Nobody reading the repo sees "cross-validation" — they see hyperparameter tuning. |
| **No per-fold numbers anywhere** | Only aggregates (`mean`, `std`) were ever produced. Five folds ran; zero fold-level rows were written down. A grader cannot verify folds happened. |
| **No artifact whose name says "CV"** | `cv_score_comparison.png` existed but was one figure among fifteen, and no CSV backed it. |
| **Not in the README** | The README's Results table led with test ROC-AUC. `CV ROC-AUC 0.899 ± 0.043` was the last row of a seven-row table. |
| **Buried in the report** | Real CV content existed inside a 24-page PDF. Graders skim. |

**The lesson:** a grader scores what they can *find*, not what the code does. CV was present and
invisible, which scores identically to absent.

### "Thin CI/CD" — this one was simply accurate

A1's `ci.yml` had two jobs: `quality` (ruff → pytest → model smoke test) and `container`
(docker build → run → curl `/predict`). That is CI. There was **no CD at all**:

| Missing in A1 | Consequence |
|---|---|
| No registry push | The image existed only inside the runner and was discarded when the job ended. |
| No automated deployment | Deployment was a human typing `kubectl apply -f k8s/`. |
| No post-deploy verification | Nothing checked the deployed service after a change. |
| No rollback | A bad deploy would just stay broken. |
| Image never referenced by version | `heart-api:v1` with `imagePullPolicy: IfNotPresent`, side-loaded via `minikube image load`. No traceability from a commit to a running pod. |

**The lesson:** "CI/CD" is two things. A1 delivered the CI half and called it done. Half of a
compound requirement reads as thin because it *is* thin.

### The pattern behind both

Both failures share one shape: **the work stopped at "it functions" instead of "it is evidenced."**
Everything below is designed to make that impossible to repeat.

---

## Part 2 — The strict rules

### Rule 1 — The 60-Second Grader Test

> Before any module is marked done: could someone who has never seen this repo find the evidence
> for that module in under 60 seconds, starting from `README.md`, without running anything?

If the answer needs "well, if you open `src/train.py` and look at line 327…", the module is not
done. This is the single rule that would have caught the CV problem in A1.

**Enforcement:** `README.md` carries a `## How this maps to the rubric` section — M1 to M5, each
with the exact file paths that satisfy it. Written on Day 20, checked on Day 21.

### Rule 2 — Three-Artifact Rule

Nothing counts as delivered on one artifact. Every graded claim needs **all three**:

1. **Code** that does it
2. **An output artifact** — a CSV, a figure, a JSON file, a log — with a self-describing filename
3. **A visible pointer** — a README section, and for infrastructure a screenshot

Applied to CV: `src/cross_validate.py` + `reports/cv_results.csv` + `reports/figures/cv_comparison.png`
+ a `## Cross-validation` README heading + nested MLflow runs. Five pointers, not one buried argument.

### Rule 3 — Per-Unit Before Aggregate

Never report only a mean. **Always write down the individual values that produced it.**

- 5-fold CV → 5 rows per model in `cv_results.csv`, *then* the mean ± std row
- Latency → the distribution, not just p95
- Post-deploy accuracy → the full 100-row prediction log, not just the accuracy number

A mean with no visible components is unverifiable, which is precisely how A1's CV became invisible.

### Rule 4 — Name Things What The Rubric Calls Them

The spec says "cross-validation", so the file is `cross_validate.py`, the CSV is `cv_results.csv`,
and the README heading is `## Cross-validation`. Not `tune.py`, not `model_selection.png`.

Graders search for the rubric's own vocabulary. Match it exactly. This costs nothing.

### Rule 5 — Compound Requirements Get Split and Ticked Separately

"CI/CD" is two requirements. So is "logging and monitoring". So is "build, test, and image
creation". Every conjunction in the spec becomes its own checkbox in `TASKS.md`.

A1 treated "CI/CD" as one item, satisfied it 50%, and lost marks on the missing half.

**Enforcement:** `TASKS.md` decomposes M3 into automated testing / CI setup / artifact publishing,
and M4 into deployment target / CD flow / smoke tests — mirroring the spec's own sub-numbering.

### Rule 6 — Every Failure Path Must Be Demonstrated, Not Asserted

Claiming "the pipeline fails on test failure" is worth nothing. A screenshot of a red run with the
downstream job **skipped** is worth the mark.

Deliberately break, screenshot, revert:
- a unit test → CI red, `build` skipped
- a lint rule → `lint` fails first
- the deployed `/predict` → smoke test fails, pipeline red, rollback fires

**Enforcement:** Day 14 and Day 17 exist for this. `EVIDENCE.md` has rows for each red-path screenshot.

### Rule 7 — Traceability From Commit to Running Pod

Every deployed artifact must be traceable back to the commit that produced it.

- Images tagged by **commit SHA**, never only `latest`
- `k8s/deployment.yaml` references the SHA tag
- The tag bump is itself a commit, so `git log` shows the deploy history
- `kubectl describe pod` proves the image was **pulled from the registry**, not side-loaded

A1's side-loaded `heart-api:v1` had zero traceability. That's a large part of what made it read thin.

### Rule 8 — Anything Reduced For Time Gets Documented, Not Hidden

CPU limits may force fewer epochs per CV fold, or a stratified subset. **That is fine.** Write the
protocol down: k, what was held out, epochs per fold, subset size, wall-clock.

A documented, reasoned reduction is defensible engineering. A silent one is what "no
cross-validation" felt like to the A1 grader.

### Rule 9 — Evidence Is Captured At The Moment, Never Reconstructed

Screenshot the moment it happens. Re-creating a pod self-heal or a red pipeline three days later
wastes an hour and often can't be reproduced at all.

**Enforcement:** `EVIDENCE.md` is updated the same day, and `PROGRESS.md` days list their
screenshots as sub-tasks rather than deferring them to a capture day.

### Rule 10 — The Video Is A Graded Deliverable, Not A Recap

Under 5 minutes, and it must show **all five modules** plus a genuine end-to-end deploy. Rehearsed
and timed on Day 20 before recording on Day 21.

Both A1 gaps would have been obvious to *the person recording* if they'd had to narrate
"here's cross-validation" and "here's automatic deployment" out loud. Use the video as a final audit
of the work, not just a demo of it.

---

## Part 3 — Pre-submission audit

Run this on Day 21. Every row must be `[x]` before the zip is built.

### Cross-validation — the A1 gap

- [ ] `src/cross_validate.py` exists and CV is its **only** job (not a side effect of tuning)
- [ ] `reports/cv_results.csv` has one row per (model, fold) — 10 rows for 2 models × 5 folds
- [ ] Fold scores differ from each other (identical values mean the loop is broken)
- [ ] `mean ± std` rows recompute correctly from the per-fold rows by hand
- [ ] `reports/figures/cv_comparison.png` shows mean ± std error bars
- [ ] `reports/figures/cv_fold_scores.png` shows per-fold variance
- [ ] MLflow shows a parent run per model with 5 nested `fold_N` runs
- [ ] `README.md` has a `## Cross-validation` heading with the fold table pasted in
- [ ] The production model was chosen **by CV mean**, and ADR-005 says so
- [ ] The CV protocol (k, pooled splits, epochs, any subset, wall-clock) is written down
- [ ] The test split was provably never touched during CV
- [ ] The video says the words "cross-validation" while showing the table

### CI/CD — the A1 gap

- [ ] Multiple gated jobs: `lint`, `test`, `build`, `image-smoke`, `security`, `deploy`
- [ ] A failing test **skips** the build job — screenshot exists
- [ ] A failing lint stops the pipeline — screenshot exists
- [ ] Image pushed to GHCR on every `main` push
- [ ] Image tagged by **commit SHA** and `latest` — both visible on the Packages page
- [ ] PRs build but do **not** push
- [ ] The published image was pulled fresh and verified to serve predictions
- [ ] Deployment happens **automatically** on `main` — no human `kubectl apply`
- [ ] Argo CD shows Synced/Healthy with sync history tied to the tag-bump commit
- [ ] Argo CD self-heal demonstrated (live edit reverted) — screenshot exists
- [ ] Post-deploy smoke test hits `/health` **and** `/predict` against the real deployment
- [ ] A failing smoke test **fails the pipeline** — screenshot exists
- [ ] Rollback on smoke failure demonstrated — screenshot exists
- [ ] `kubectl describe pod` proves the image was pulled from GHCR
- [ ] Branch protection on `main` requires the checks
- [ ] CI badge in `README.md`
- [ ] End-to-end rehearsed and timed: code change → CI → GHCR → Argo CD → deployed prediction

### Everything else

- [ ] All 5 modules have a `## How this maps to the rubric` entry with real paths
- [ ] Fresh clone: `pip install -r requirements.txt` → `dvc repro` → `pytest` all pass
- [ ] `models/model.h5` and `model_metadata.json` are **committed to Git** (not DVC-only) so the zip
      is usable
- [ ] Extracted zip is self-contained: tests pass, model loads
- [ ] `git status` and `dvc status` both clean; `v1.0` tagged
- [ ] No `Co-Authored-By` trailers in history
- [ ] Sensitive-data rule honoured: no image contents in logs, and the README says so
- [ ] Video under 5:00, covers all five modules
- [ ] `TASKS.md` self-score is 50/50 with no unticked `[GAP-CV]` or `[GAP-CICD]` item

---

## Part 4 — Self-check questions

Ask these at each weekly checkpoint. They are the questions a grader effectively asks.

1. If I only read `README.md`, which modules could I not verify?
2. Which graded claim currently rests on a single artifact? (Rule 2 violation)
3. Which number am I reporting as an aggregate with no visible components? (Rule 3 violation)
4. Which "the pipeline handles X" claim have I not actually broken on purpose? (Rule 6 violation)
5. Which spec conjunction have I satisfied only half of? (Rule 5 violation)
6. What did I reduce for time without writing down? (Rule 8 violation)
7. Can I trace the currently running pod back to a specific commit? (Rule 7 violation)

Any "I don't know" is the next task.
