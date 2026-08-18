"""FastAPI inference service for the Cats vs Dogs classifier (M2).

Loads ``models/model.h5`` once at startup and serves predictions. Endpoints:

    GET  /health          liveness/readiness probe (used by the k8s manifests)
    POST /predict         multipart image upload -> label + class probabilities
    POST /predict/base64  JSON body with a base64 image, for easy curl/Postman
    GET  /                service metadata: which model, its CV and test scores
    GET  /metrics         Prometheus exposition (added by the instrumentator)

M5's logging and metrics are wired in here rather than bolted on later, so the
monitoring module becomes assembly instead of invention:

  * one structured JSON log line per request (method, path, status, latency,
    predicted label, confidence);
  * **image contents are never logged** — only byte size and a short SHA-256
    prefix. The spec asks for request logging "excluding sensitive data", and a
    user-submitted photo is exactly that (ADR-008);
  * Prometheus counters/histograms via ``prometheus-fastapi-instrumentator``,
    plus a custom predictions-by-class counter so class skew is visible;
  * an in-process counter surfaced on ``GET /``, so request count is
    demonstrable in the demo video without port-forwarding Grafana.

Run locally:
    uvicorn api.main:app --reload    # http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from pythonjsonlogger.json import JsonFormatter

from src.data import CLASS_NAMES
from src.predict import load_metadata, load_model, model_summary, predict_image

# Reject before decoding: a 224x224 JPEG is a few tens of KB, so anything above
# this is either a mistake or an attempt to exhaust memory.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp",
                         "application/octet-stream"}

# --- structured JSON logging (ADR-008) ---------------------------------------
logger = logging.getLogger("catdog_api")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# --- custom Prometheus metric (M5) -------------------------------------------
# Request count and latency come free from the instrumentator; predictions BY
# CLASS is the signal specific to this use case, because a sudden skew toward one
# class is how a broken model or shifted input distribution shows up.
PREDICTIONS = Counter(
    "catdog_predictions_total",
    "Predictions served, labelled by predicted class",
    ["predicted_class"],
)

STATE: dict = {}
COUNTERS = {"requests": 0, "predictions": 0, "errors": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once. A failure here is logged, not fatal.

    Serving 503s from /predict while /health still answers is more useful than
    refusing to start: the pod becomes inspectable, and `kubectl logs` shows why.
    """
    try:
        STATE["model"] = load_model()
        STATE["metadata"] = load_metadata()
        logger.info("startup: model loaded", extra={
            "model_family": STATE["metadata"].get("model_family"),
            "tensorflow": STATE["metadata"].get("package_versions", {}).get("tensorflow"),
        })
    except Exception as exc:
        STATE["model"] = None
        STATE["load_error"] = str(exc)
        logger.error("startup: model FAILED to load", extra={"error": str(exc)})
    yield
    STATE.clear()


app = FastAPI(
    title="Cats vs Dogs Classifier",
    description=(
        "Binary image classification for a pet adoption platform. "
        "Upload an image to `/predict` and get the predicted class with "
        "probabilities for both classes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# --- middleware ---------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One JSON line per request: method, path, status, latency."""
    start = time.perf_counter()
    COUNTERS["requests"] += 1
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if response.status_code >= 400:
        COUNTERS["errors"] += 1
    logger.info("request", extra={
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": round(elapsed_ms, 2),
    })
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Last resort: clean JSON instead of a traceback leaking to the client."""
    logger.exception("unhandled error", extra={
        "method": request.method, "path": request.url.path,
    })
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


# --- schemas ------------------------------------------------------------------

class PredictionResponse(BaseModel):
    prediction: int = Field(description="0 = cat, 1 = dog")
    label: str = Field(description="predicted class name")
    probabilities: dict[str, float] = Field(description="probability per class, sums to 1")
    confidence: float = Field(description="probability of the predicted class")
    threshold: float = Field(description="decision threshold applied")

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 1, "label": "dog",
                "probabilities": {"cat": 0.0161, "dog": 0.9839},
                "confidence": 0.9839, "threshold": 0.5,
            }
        }
    }


class Base64Request(BaseModel):
    image_base64: str = Field(description="base64-encoded image bytes")

    model_config = {
        "json_schema_extra": {
            "example": {"image_base64": "iVBORw0KGgoAAAANSUhEUg..."}
        }
    }


# --- helpers ------------------------------------------------------------------

def _require_model():
    model = STATE.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"model not loaded: {STATE.get('load_error', 'unknown error')}",
        )
    return model


def _score(raw: bytes, source: str) -> PredictionResponse:
    """Validate size, score, log safely, and update the metrics."""
    if not raw:
        raise HTTPException(status_code=422, detail="empty request body")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"image too large: {len(raw)} bytes > {MAX_UPLOAD_BYTES} limit",
        )

    model = _require_model()
    try:
        result = predict_image(raw, model)
    except ValueError as exc:
        # Undecodable / truncated image: the client's problem, not a server fault.
        logger.warning("undecodable image", extra={
            "source": source, "bytes": len(raw), "error": str(exc),
        })
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    COUNTERS["predictions"] += 1
    PREDICTIONS.labels(predicted_class=result["label"]).inc()

    # Size and a short digest only — never the image itself (ADR-008).
    logger.info("prediction", extra={
        "source": source,
        "bytes": len(raw),
        "image_sha256_prefix": hashlib.sha256(raw).hexdigest()[:12],
        "predicted_class": result["label"],
        "confidence": result["confidence"],
    })
    return PredictionResponse(**result)


# --- routes -------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness/readiness probe. Answers 200 even if the model failed to load.

    Deliberate: if this 503'd on a load failure, Kubernetes would restart-loop the
    pod forever and hide the cause. `model_loaded` carries the real state, and
    /predict is the endpoint that refuses.
    """
    return {"status": "ok", "model_loaded": STATE.get("model") is not None}


@app.get("/", tags=["ops"])
def root() -> dict:
    """Service metadata, including which model is serving and its scores."""
    return {
        "service": app.title,
        "version": app.version,
        "model": model_summary(),
        "counters": dict(COUNTERS),
        "endpoints": ["/health", "/predict", "/predict/base64", "/metrics", "/docs"],
        "docs": "/docs",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict_endpoint(
    file: Annotated[UploadFile, File(description="image file")],
):
    """Score an uploaded image and return the label with both class probabilities."""
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported content type {file.content_type!r}; "
                   f"expected one of {sorted(ALLOWED_CONTENT_TYPES)}",
        )
    return _score(await file.read(), source="multipart")


@app.post("/predict/base64", response_model=PredictionResponse, tags=["inference"])
def predict_base64_endpoint(payload: Base64Request):
    """Same as /predict but takes base64 JSON, which curl and Postman handle easily."""
    data = payload.image_base64
    if "," in data[:64] and data.lstrip().startswith("data:"):
        data = data.split(",", 1)[1]        # tolerate a data: URL prefix
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid base64: {exc}") from exc
    return _score(raw, source="base64")


@app.get("/classes", tags=["ops"])
def classes() -> dict:
    """The class-index mapping, so a client need not hard-code it."""
    return {"classes": CLASS_NAMES, "class_indices": {c: i for i, c in enumerate(CLASS_NAMES)}}
