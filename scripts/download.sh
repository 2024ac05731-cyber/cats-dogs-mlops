#!/usr/bin/env bash
# Fetch the Cats vs Dogs dataset into data/raw/ (M1 — data acquisition).
#
# Two paths, because Kaggle API access is not guaranteed on a managed network:
#
#   1. Kaggle API (default) — needs ~/.kaggle/access_token (the newer KGAT_
#      format) or the KAGGLE_API_TOKEN env var, and the dataset terms accepted
#      once in a browser.
#
#   2. Local archive fallback — if the API is unreachable, download the zip by
#      hand from the dataset page and point this script at it:
#
#          bash scripts/download.sh /path/to/archive.zip
#
# Idempotent: re-running re-uses the existing data. DATA_FORCE=1 to refetch.
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="bhavikjikadara/dog-and-cat-classification-dataset"
RAW_DIR="data/raw"
LOCAL_ARCHIVE="${1:-}"

# Already populated with REAL data? Bail out unless forced.
#
# The .synthetic marker distinguishes scripts/make_fixtures.py output from the
# real dataset. Without that check this guard counted the 122 synthetic smoke
# images as "already downloaded" and skipped the real fetch — and a training run
# on synthetic data would have reported meaningless accuracy.
if [[ -f "$RAW_DIR/.synthetic" ]]; then
  echo "[download] data/raw currently holds SYNTHETIC smoke-test images"
  echo "[download] removing them and fetching the real dataset"
  rm -rf "${RAW_DIR:?}"/*
  rm -f "$RAW_DIR/.synthetic"
elif [[ -d "$RAW_DIR" ]] && [[ -n "$(find "$RAW_DIR" -type f -name '*.jpg' -print -quit 2>/dev/null)" ]]; then
  if [[ "${DATA_FORCE:-0}" != "1" ]]; then
    count=$(find "$RAW_DIR" -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) | wc -l | tr -d ' ')
    echo "[download] re-using existing $RAW_DIR ($count image files)"
    echo "[download] pass DATA_FORCE=1 to refetch"
    exit 0
  fi
  echo "[download] DATA_FORCE=1 set — refetching"
fi

mkdir -p "$RAW_DIR"

if [[ -n "$LOCAL_ARCHIVE" ]]; then
  # ---- Path 2: local archive ----
  if [[ ! -f "$LOCAL_ARCHIVE" ]]; then
    echo "[download] ERROR: archive not found: $LOCAL_ARCHIVE" >&2
    exit 1
  fi
  echo "[download] extracting local archive: $LOCAL_ARCHIVE"
  unzip -q -o "$LOCAL_ARCHIVE" -d "$RAW_DIR"
else
  # ---- Path 1: Kaggle API ----
  if ! command -v kaggle >/dev/null 2>&1; then
    echo "[download] ERROR: kaggle CLI not found. Activate the venv, or:" >&2
    echo "           pip install -r requirements.txt" >&2
    exit 1
  fi
  if [[ ! -f "$HOME/.kaggle/access_token" ]] && [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
    echo "[download] ERROR: no Kaggle credentials found." >&2
    echo "           Expected ~/.kaggle/access_token or \$KAGGLE_API_TOKEN." >&2
    echo "           See README 'Setup'." >&2
    exit 1
  fi

  echo "[download] fetching $DATASET via the Kaggle API..."
  if ! kaggle datasets download -d "$DATASET" -p "$RAW_DIR" --unzip; then
    echo "" >&2
    echo "[download] Kaggle download FAILED. Common causes:" >&2
    echo "  403 -> dataset terms not accepted. Open the dataset page in a browser once." >&2
    echo "  401 -> bad token. Check ~/.kaggle/access_token contents and permissions (600)." >&2
    echo "  proxy/tunnel error -> kaggle.com blocked on this network. Use the fallback:" >&2
    echo "        download the zip in a browser, then: bash scripts/download.sh <archive.zip>" >&2
    exit 1
  fi
fi

# Strip macOS archive cruft that would otherwise be discovered as images.
find "$RAW_DIR" -name '__MACOSX' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$RAW_DIR" -name '._*' -type f -delete 2>/dev/null || true
find "$RAW_DIR" -name '.DS_Store' -type f -delete 2>/dev/null || true

count=$(find "$RAW_DIR" -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) | wc -l | tr -d ' ')
echo "[download] done — $count image files under $RAW_DIR"
echo "[download] next: python -m src.data    # counts, balance, corrupt-image audit"
