# syntax=docker/dockerfile:1

# Cats vs Dogs inference service (M2).
#
# Slim base + serving-only deps + non-root user. The image is large (TensorFlow
# alone is several hundred MB) — that cost is accepted and documented in ADR-002
# rather than worked around, since this assignment sets no size limit. The
# fallback if it ever matters is exporting to TFLite and dropping full TF.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Dependencies first, in their own layer, so this cache survives every code
# change. Rebuilds after a src/ edit then take seconds instead of re-pulling
# TensorFlow.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Only what serving needs at runtime. Note src/ is copied in full rather than
# cherry-picked: api.main imports src.predict, which imports src.preprocess and
# src.data for the shared preprocessing path and the canonical constants.
COPY src/ ./src/
COPY api/ ./api/
COPY models/model.h5 models/model_metadata.json ./models/

# Run unprivileged. Kubernetes can enforce this too, but an image that only works
# as root is a liability regardless of who runs it.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# python:3.11-slim ships no curl, so probe with stdlib urllib instead of adding a
# package purely for the healthcheck.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

# --no-access-log because the app's own middleware already emits one structured
# JSON line per request; uvicorn's plain-text access log would duplicate every
# entry and break log parsing (ADR-008).
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
