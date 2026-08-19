#!/usr/bin/env bash
# Build and VERIFY the submission zip (Deliverable 1).
#
# The spec asks for "a zip file containing all source code, configuration files
# (DVC, CI/CD, Docker, deployment manifests), and trained model artifacts".
#
# Two things this does that a plain `zip -r` does not:
#
#   1. Excludes what must not ship — the 848MB dataset, the DVC cache, mlruns, the
#      venv — while explicitly INCLUDING models/model.h5, which is the artifact the
#      grader actually needs and which .gitignore would otherwise hide.
#
#   2. Verifies the result by extracting it to a temp dir and running the test
#      suite against the extraction. A zip nobody unpacked is not a deliverable;
#      it is a hope. This is the step that catches "works in my repo" errors like a
#      missing fixture or an un-committed model.
#
# Usage:
#   bash scripts/make_submission.sh
#   bash scripts/make_submission.sh --skip-verify     # faster, not recommended
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
ROLL="2024AC05731"
NAME="Assignment_2_${ROLL}"
OUT="${ROOT}/${NAME}.zip"
SKIP_VERIFY="${1:-}"

echo "[submission] building ${NAME}.zip"

# --- pre-flight: refuse to package a repo that is not submission-ready --------
FAILED=0

if [[ -n "$(git status --porcelain)" ]]; then
  echo "[submission] ERROR: uncommitted changes — commit first so the zip matches git" >&2
  git status --short | sed 's/^/    /' >&2
  FAILED=1
fi

if [[ ! -f models/model.h5 ]]; then
  echo "[submission] ERROR: models/model.h5 missing. The spec requires trained" >&2
  echo "             model artifacts. Run: python -m src.train --model transfer" >&2
  FAILED=1
fi

if ! git ls-files --error-unmatch models/model.h5 >/dev/null 2>&1; then
  echo "[submission] ERROR: models/model.h5 is not tracked by git." >&2
  echo "             The DVC remote is local (ADR-004), so the grader cannot" >&2
  echo "             'dvc pull' — the artifact must be in the repo." >&2
  FAILED=1
fi

if grep -q "PLACEHOLDER" k8s/deployment.yaml 2>/dev/null; then
  echo "[submission] WARNING: k8s/deployment.yaml still has the placeholder image tag." >&2
  echo "             CD has not yet deployed. Not fatal, but M4 evidence is incomplete." >&2
fi

if [[ "$FAILED" != "0" ]]; then
  echo "[submission] aborted" >&2
  exit 1
fi

# --- build -------------------------------------------------------------------
rm -f "$OUT"

# Explicit include list rather than exclusions: it is easier to audit what a
# grader receives than to guess what a wildcard swept up.
zip -q -r "$OUT" \
  src/ api/ tests/ scripts/ notebooks/ \
  k8s/ argocd/ monitoring/ \
  .github/ \
  models/model.h5 models/model_metadata.json \
  reports/ tracker/ \
  data/processed/ data/corrupt_files.txt data/monitoring/ \
  dvc.yaml dvc.lock data/raw.dvc .dvc/config \
  Dockerfile .dockerignore \
  requirements.txt requirements-serve.txt \
  pytest.ini conftest.py ruff.toml .python-version \
  README.md .gitignore \
  -x '*__pycache__*' '*.pyc' '*.DS_Store' \
     '*.pytest_cache*' '*.ruff_cache*' '*.ipynb_checkpoints*'

SIZE=$(du -h "$OUT" | cut -f1)
COUNT=$(unzip -l "$OUT" | tail -1 | awk '{print $2}')
echo "[submission] wrote $(basename "$OUT")  ${SIZE}, ${COUNT} files"

# --- prove nothing forbidden got in -----------------------------------------
echo "[submission] checking for accidental inclusions..."
LEAKED=0
for pat in 'data/raw/PetImages' '.venv/' 'mlruns/' '.dvc/cache' 'access_token' 'kaggle.json'; do
  if unzip -l "$OUT" | grep -q -- "$pat"; then
    echo "[submission]   ERROR: '$pat' is in the zip" >&2
    LEAKED=1
  else
    echo "[submission]   ok: no $pat"
  fi
done
[[ "$LEAKED" == "0" ]] || { echo "[submission] aborted: forbidden content" >&2; exit 1; }

# --- verify by extracting and running the suite ------------------------------
if [[ "$SKIP_VERIFY" == "--skip-verify" ]]; then
  echo "[submission] SKIPPING verification (--skip-verify). Not recommended."
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "[submission] verifying: extracting to $TMP"
unzip -q "$OUT" -d "$TMP"

echo "[submission] does the model load from the extraction?"
"${ROOT}/.venv/bin/python" - "$TMP" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
import keras  # noqa: E402

m = keras.models.load_model(root / "models" / "model.h5")
meta = json.loads((root / "models" / "model_metadata.json").read_text())
print(f"[submission]   model loads: {meta['model_family']}, "
      f"input {list(m.input_shape[1:])}, test acc {meta['metrics']['test']['accuracy']}")
assert list(m.input_shape[1:]) == meta["input_shape"]
PY

echo "[submission] running the test suite against the extraction..."
( cd "$TMP" && "${ROOT}/.venv/bin/python" -m pytest -q --no-header 2>&1 | tail -3 )

echo ""
echo "[submission] VERIFIED — $(basename "$OUT") ($SIZE) is self-contained"
echo "[submission] remaining manual steps:"
echo "  1. record the demo video (< 5:00) — see tracker/video_script.md"
echo "  2. add the video link to README.md, commit, push"
echo "  3. git tag -a v1.0 -m 'Assignment 2 submission' && git push --tags"
echo "  4. submit ${NAME}.zip + the video"
