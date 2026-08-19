# Demo video shot list (Day 10)

**Hard limit: under 5:00.** The spec asks for "the complete MLOps workflow from code change to
deployed model prediction", so that chain is the spine — everything else is a supporting beat.

Budget below totals **4:40**, leaving 20 s of slack. If a beat overruns, cut from §7 (monitoring)
or §3 (MLflow), never from §2 (cross-validation) or §5–6 (CI/CD): those are the two axes Assignment 1
lost marks on.

---

## Before you hit record

Everything must already be up. Nothing in a 5-minute video should be waiting for TensorFlow to boot.

```bash
# 1. Cluster + Argo CD + monitoring, all running
minikube start --driver=docker --memory=6144 --cpus=4
kubectl get pods                       # catdog-api 2/2 Running

# 2. minikube tunnel — separate terminal, leave it running
minikube tunnel

# 3. Port-forwards — one terminal each, leave running
kubectl port-forward svc/argocd-server -n argocd 8080:443
kubectl port-forward svc/monitoring-grafana 3000:80
bash scripts/mlflow_ui.sh              # http://localhost:5000

# 4. Pre-warm every browser tab so nothing loads on camera:
#    - GitHub repo README (scrolled to the CV table)
#    - GitHub Actions, last green CI run expanded
#    - GHCR package page showing the SHA tag
#    - Argo CD UI, logged in, catdog-api app open
#    - Grafana, the "Cats vs Dogs API" dashboard
#    - MLflow, the cats-dogs-classification experiment
#    - http://127.0.0.1/docs (Swagger through the cluster)

# 5. Pre-stage the code change so §5 is one keystroke, not typing on camera.
#    A comment-only edit to api/main.py is enough — it changes the image SHA
#    without any risk of breaking the build mid-demo.
```

Rehearse once, timed, without recording. Then record.

---

## Running order

### 1. Repo tour — 0:25
- GitHub repo front page: title, **green CI badge**, the Results table (0.9924 test accuracy).
- Scroll to **"How this maps to the rubric"** and say: *"every graded item, with the path that
  satisfies it."* That section exists so a grader doesn't have to hunt.
- Show the folder layout briefly — `src/`, `api/`, `k8s/`, `argocd/`, `tracker/`.

### 2. Cross-validation — 0:50 ⚠️ **PROTECTED BEAT**

This is the mark lost last time. Say the words *"five-fold stratified cross-validation"* out loud.

- README **`## Cross-validation`** section: the **per-fold table** — 10 rows, not just a mean.
- `reports/figures/cv_fold_scores.png` — point at the two lines: transfer flat at ~0.98,
  baseline scattered 0.57–0.68.
- Say the key sentence: *"the baseline's recall varied by ±0.24 across folds, so a single train/test
  split could have reported anywhere from 0.17 to 0.82 — the transfer model's ±0.0037 says its
  performance is a property of the architecture, not the split. That's why the production model was
  chosen by CV mean."*
- MLflow: the two parent runs each with **5 nested `fold_N` runs**.

### 3. Model + tracking — 0:35
- `models/model_metadata.json` — pinned versions, class map, **and the embedded CV mean/std**.
- MLflow run detail: params, metrics, artifacts (confusion matrix, loss curves).
- One line on DVC: `dvc dag` — 3 stages, and `data/raw.dvc` tracking 24,998 files.

### 4. The API — 0:30
- Swagger at `http://127.0.0.1/docs` (**through the cluster**, not localhost — say so).
- `POST /predict` with a cat image → `{"label":"cat","probabilities":{...}}`.
- Then a corrupt image → **422**, not a 500. Say: *"a bad image is the client's fault, and the
  service says so cleanly."*

### 5. Code change → CI — 0:50 ⚠️ **PROTECTED BEAT**
- Make the pre-staged edit, commit, push. **This is the "code change" the spec asks for.**
- GitHub Actions: the run starting. Walk the **six jobs** and the gating:
  `lint → test → build → image-smoke → publish`.
- Say: *"publish is gated on the smoke test, not just the build — nothing reaches the registry until
  the image has served a correct prediction."*
- Show the **red run** from the broken-test proof: `build` **skipped**, not merely failed alongside.
- GHCR package page: **SHA tag and `latest`, two architectures**.

### 6. CD → deployed prediction — 0:50 ⚠️ **PROTECTED BEAT**
- The **bot's commit**: `deploy: catdog-api:<sha> [skip ci]` by `github-actions[bot]`.
- `git log --oneline -- k8s/deployment.yaml` — *"this reads as a deployment history. Nobody ran
  kubectl."*
- **Argo CD UI**: Synced / Healthy, sync history tied to that commit.
- `kubectl get pods` → new pods; `kubectl describe pod | grep -i pull` → **pulled from GHCR**.
- The smoke gate passing in the CD job.
- **curl the new prediction through the cluster.** This closes the spec's chain.

### 7. Monitoring + post-deployment — 0:30
- Grafana, 4 panels under load. Point at **predictions by class**: *"a sudden skew is how a broken
  model shows up before anyone recomputes accuracy."*
- `kubectl logs` — JSON lines. Say: *"size and a hash, never the image itself."*
- `reports/post_deployment_report.md`: **live 1.0000 vs offline 0.9924** — *"they agree, so there's no
  train/serve preprocessing skew."*

### 8. Close — 0:10
- Back to the README. Mention `tracker/` — 12 ADRs and a 100-point audit.

---

## Timing

| § | Beat | Budget |
|---|---|---|
| 1 | Repo tour | 0:25 |
| 2 | **Cross-validation** | 0:50 |
| 3 | Model + tracking | 0:35 |
| 4 | API | 0:30 |
| 5 | **Code change → CI** | 0:50 |
| 6 | **CD → deployed prediction** | 0:50 |
| 7 | Monitoring | 0:30 |
| 8 | Close | 0:10 |
| | **Total** | **4:40** |

## Rules while recording

- **Don't wait for anything on camera.** If a page is loading, you didn't pre-warm it.
- **Say the rubric's words**: "cross-validation", "container registry", "automated deployment",
  "smoke test", "rollback". Graders listen for them.
- One browser window, one terminal. No desktop clutter, consistent light/dark theme.
- Zoom the terminal font before starting — 5-minute video, small text is unreadable.
- If a live step fails on camera, keep going and narrate it. A recovered failure looks more honest
  than a re-shot success, and the rollback path is itself a graded feature.

## After recording

1. Check the duration is **under 5:00**. Re-record rather than submit 5:10.
2. Upload; paste the link into the README **Demo video** placeholder.
3. `python scripts/daily_audit.py --day 10` — must be clean.
4. Run the full `GUARDRAILS.md` Part 3 pre-submission audit.
5. Build the zip; verify by extracting to `/tmp` and running `pytest`.
