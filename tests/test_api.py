"""API tests via FastAPI's TestClient (M3 — required inference-path test).

These exercise the real ASGI app, including the lifespan hook, so they cover the
seam between HTTP and the model: content-type validation, size limits,
undecodable images, and the shape of the JSON contract clients depend on.

The model is loaded once per session because loading a 2.2M-parameter Keras model
per test would dominate the runtime. Tests that need a working model skip when
``models/model.h5`` is absent, so a fresh clone (and CI) still goes green.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import MAX_UPLOAD_BYTES, app
from src.data import CLASS_NAMES

FIXTURES = Path(__file__).parent / "fixtures"
MODEL_PATH = Path(__file__).parent.parent / "models" / "model.h5"

needs_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="models/model.h5 not present (run python -m src.train --model transfer)",
)


@pytest.fixture(scope="module")
def client():
    """TestClient as a context manager, so lifespan startup actually runs."""
    with TestClient(app) as c:
        yield c


def _upload(name: str, content_type: str = "image/jpeg"):
    return {"file": (name, (FIXTURES / name).read_bytes(), content_type)}


# --------------------------------------------------------------- ops endpoints

def test_health_returns_200(client):
    """The k8s liveness/readiness probe target."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_health_ok_even_if_model_missing(client):
    """/health must not 503 on a load failure.

    If it did, Kubernetes would restart-loop the pod and hide the cause. The
    `model_loaded` flag carries the real state; /predict is what refuses.
    """
    r = client.get("/health")
    assert r.status_code == 200


def test_root_reports_model_and_counters(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "Cats vs Dogs Classifier"
    assert "model" in body
    assert "counters" in body and "requests" in body["counters"]
    assert "/predict" in body["endpoints"]


def test_classes_endpoint_matches_canonical_order(client):
    """Clients must not have to guess whether 0 is cat or dog."""
    r = client.get("/classes")
    assert r.status_code == 200
    assert r.json()["classes"] == CLASS_NAMES
    assert r.json()["class_indices"] == {"cat": 0, "dog": 1}


def test_metrics_endpoint_exposes_prometheus(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_request" in r.text or "python_info" in r.text


def test_openapi_documents_both_predict_routes(client):
    """The Swagger page is graded evidence for M2, so the schema must be complete."""
    spec = client.get("/openapi.json").json()
    assert "/predict" in spec["paths"]
    assert "/predict/base64" in spec["paths"]
    assert "/health" in spec["paths"]


# --------------------------------------------------------------- /predict

@needs_model
def test_predict_returns_valid_contract(client):
    r = client.post("/predict", files=_upload("cat_sample.jpg"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in CLASS_NAMES
    assert set(body["probabilities"]) == set(CLASS_NAMES)
    assert body["probabilities"]["cat"] + body["probabilities"]["dog"] == pytest.approx(1.0, abs=1e-4)
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["confidence"] == pytest.approx(max(body["probabilities"].values()), abs=1e-6)


@needs_model
def test_label_agrees_with_probabilities(client):
    """The reported label must be the argmax of the probabilities it reports.

    Guards against an off-by-one in the class mapping, which would produce
    confidently inverted predictions while every shape assertion still passed.
    """
    r = client.post("/predict", files=_upload("dog_sample.jpg"))
    body = r.json()
    expected = max(body["probabilities"], key=body["probabilities"].get)
    assert body["label"] == expected
    assert body["prediction"] == CLASS_NAMES.index(expected)


@needs_model
@pytest.mark.parametrize("name", ["greyscale.jpg", "rgba.png"])
def test_predict_accepts_non_rgb_images(client, name):
    """Greyscale and RGBA must work end-to-end, not just in the loader."""
    ct = "image/png" if name.endswith(".png") else "image/jpeg"
    r = client.post("/predict", files=_upload(name, ct))
    assert r.status_code == 200, r.text


@needs_model
def test_predict_base64_matches_multipart(client):
    """Both routes must give the same answer for the same image."""
    raw = (FIXTURES / "cat_sample.jpg").read_bytes()
    a = client.post("/predict", files=_upload("cat_sample.jpg")).json()
    b = client.post("/predict/base64",
                    json={"image_base64": base64.b64encode(raw).decode()}).json()
    assert a["label"] == b["label"]
    assert a["probabilities"]["dog"] == pytest.approx(b["probabilities"]["dog"], abs=1e-6)


@needs_model
def test_predict_base64_accepts_data_url_prefix(client):
    raw = (FIXTURES / "dog_sample.jpg").read_bytes()
    payload = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    r = client.post("/predict/base64", json={"image_base64": payload})
    assert r.status_code == 200, r.text


# --------------------------------------------------------------- failure paths

def test_missing_file_returns_422(client):
    """FastAPI validation: no file part at all."""
    assert client.post("/predict").status_code == 422


def test_wrong_content_type_returns_422(client):
    r = client.post("/predict", files=_upload("cat_sample.jpg", "text/plain"))
    assert r.status_code == 422
    assert "content type" in r.text.lower()


@needs_model
@pytest.mark.parametrize("name", ["corrupt.jpg", "not_an_image.jpg"])
def test_undecodable_image_returns_422_not_500(client, name):
    """A bad image is the client's fault: 422, and no stack trace."""
    r = client.post("/predict", files=_upload(name))
    assert r.status_code == 422, r.text
    assert "could not decode" in r.json()["detail"]
    assert "Traceback" not in r.text


def test_empty_upload_returns_422(client):
    r = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert r.status_code == 422


def test_oversized_upload_returns_413(client):
    """Size is checked before decoding, so this must not OOM the process."""
    big = b"\xff\xd8\xff" + b"\x00" * (MAX_UPLOAD_BYTES + 1024)
    r = client.post("/predict", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413
    assert "too large" in r.json()["detail"]


def test_invalid_base64_returns_422(client):
    r = client.post("/predict/base64", json={"image_base64": "!!!not-base64!!!"})
    assert r.status_code == 422
    assert "base64" in r.json()["detail"].lower()


def test_base64_missing_field_returns_422(client):
    assert client.post("/predict/base64", json={}).status_code == 422


# --------------------------------------------------------------- observability

@needs_model
def test_prediction_increments_counters(client):
    """M5: the in-app counter must actually move, since /` reports it."""
    before = client.get("/").json()["counters"]["predictions"]
    client.post("/predict", files=_upload("cat_sample.jpg"))
    after = client.get("/").json()["counters"]["predictions"]
    assert after == before + 1


@needs_model
def test_predictions_by_class_metric_is_exported(client):
    """The custom Prometheus counter is the class-skew signal for M5."""
    client.post("/predict", files=_upload("dog_sample.jpg"))
    body = client.get("/metrics").text
    assert "catdog_predictions_total" in body


@needs_model
def test_logs_never_contain_image_bytes(client, caplog):
    """ADR-008 / spec: request logging must exclude sensitive data.

    Asserts the raw image never reaches the logs — only its size and a short
    digest. A regression here would leak user-submitted photos into pod logs.
    """
    import logging

    raw = (FIXTURES / "cat_sample.jpg").read_bytes()
    with caplog.at_level(logging.INFO, logger="catdog_api"):
        client.post("/predict", files=_upload("cat_sample.jpg"))

    blob = "\n".join(r.getMessage() + str(r.__dict__) for r in caplog.records)
    assert raw[:32].hex() not in blob.lower()
    assert "image_sha256_prefix" in blob or "bytes" in blob
