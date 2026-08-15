# Cats vs Dogs MLOps

End-to-end MLOps pipeline for binary image classification (cats vs dogs), framed
as the intake classifier for a pet adoption platform: CNN training with
cross-validation, MLflow experiment tracking, DVC data versioning, a FastAPI
inference service, Docker packaging, GitHub Actions CI publishing to GHCR, Argo
CD continuous deployment onto Kubernetes, and Prometheus/Grafana monitoring.

Built for AIMLCZG523 Assignment 2 at BITS Pilani.

**Repository:** _add the GitHub URL once the remote is created_

**Demo video:** _add the link here (Day 10)_

## Stack

| Layer               | Tool                                  |
|---------------------|---------------------------------------|
| Modeling            | TensorFlow / Keras (CNN)              |
| Data versioning     | DVC                                   |
| Experiment tracking | MLflow                                |
| API                 | FastAPI                               |
| Container           | Docker                                |
| Registry            | GitHub Container Registry (GHCR)      |
| CI                  | GitHub Actions                        |
| CD                  | Argo CD (GitOps)                      |
| Orchestration       | Kubernetes (Minikube)                 |
| Monitoring          | Prometheus + Grafana                  |

## Dataset

[Cats and Dogs classification dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
from Kaggle. Pre-processed to 224x224 RGB and split 80/10/10 into
train/validation/test, stratified by class, with augmentation applied to the
training split only.

## Cross-validation

_Populated on Day 4 by `python -m src.cross_validate`. Per-fold results live in
`reports/cv_results.csv`; figures in `reports/figures/cv_comparison.png` and
`cv_fold_scores.png`. The production model is selected by CV mean, not by a
single test split — see ADR-005 in `tracker/DECISIONS.md`._

| Model | Fold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|------|----------|-----------|--------|----|---------|
| _pending_ | | | | | | |

## Results

_Populated on Day 4-5._

## Architecture

_Diagram generated on Day 10 into `reports/figures/architecture_diagram.png`._

## Project structure

```
cats-dogs-mlops/
├── data/              # raw + processed (DVC-tracked) + monitoring batch
├── notebooks/         # thin EDA driver
├── src/               # data, preprocess, model, train, cross_validate, predict
├── api/               # FastAPI service
├── tests/             # pytest + committed fixtures
├── k8s/               # Deployment, Service, ServiceMonitor
├── argocd/            # Argo CD Application
├── monitoring/        # Grafana dashboard + PromQL notes
├── scripts/           # download, mlflow_ui, smoke tests, replay, daily_audit
├── models/            # model.h5 + model_metadata.json
├── reports/           # figures, cv_results.csv, screenshots
├── tracker/           # progress, tasks, decisions, evidence, guardrails
├── .github/workflows/ # ci.yml, cd.yml
├── Dockerfile
├── requirements.txt / requirements-serve.txt
└── README.md
```

## Setup

Requires Python 3.11 and a Kaggle API token.

```bash
git clone <repo-url>
cd cats-dogs-mlops

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Kaggle credentials — this project uses the newer `KGAT_` access-token format:

```bash
mkdir -p ~/.kaggle
echo "<your-token>" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Accept the dataset terms once in a browser, or the API returns 403.

## Quickstart

```bash
# 1. Data (writes data/raw/; idempotent, DATA_FORCE=1 to refetch)
bash scripts/download.sh
python -m src.data                 # counts, class balance, corrupt-image audit

# 2. Preprocess: 224x224 RGB, stratified 80/10/10 manifests
dvc repro preprocess

# 3. Train and cross-validate
python -m src.train --model transfer
python -m src.cross_validate       # -> reports/cv_results.csv + CV figures
bash scripts/mlflow_ui.sh          # http://localhost:5000

# 4. Quality gate (the same commands CI runs)
ruff check .
pytest -v
python scripts/smoke_test.py

# 5. Serve
uvicorn api.main:app --reload      # http://127.0.0.1:8000/docs
curl -F "file=@tests/fixtures/cat_sample.jpg" http://localhost:8000/predict

# 6. Container
docker build -t catdog-api:dev .
docker run -p 8000:8000 catdog-api:dev
```

## Daily audit

This project is graded against a rubric, so a 100-point audit enforces the
artifact and consistency rules in `tracker/GUARDRAILS.md`:

```bash
python scripts/daily_audit.py --day 3            # end of day 3
python scripts/daily_audit.py --day 3 --verbose  # include not-yet-active checks
python scripts/daily_audit.py --day 10 --only CONSISTENCY
```

It fails non-zero on any active violation, and calls out failures on the two
axes that lost marks on Assignment 1 (cross-validation visibility and CI/CD
depth).

## CI/CD

_Populated on Days 7-9._

## Deployment

_Populated on Days 8-9._

## Monitoring

_Populated on Day 9._

## How this maps to the rubric

_Populated on Day 10 — every module mapped to the file paths that satisfy it._

## Author

maheshkumar Ganesan, BITS Pilani (AIMLCZG523).

## License

Coursework. Please don't redistribute.
