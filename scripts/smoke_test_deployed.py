"""Post-deploy smoke test — the M4 gate that fails the pipeline (GAP-CICD).

Assignment 1 had no post-deploy verification at all: a bad image would simply
stay deployed. This script is the gate. It runs against the **deployed** service,
not a local process, and **exits non-zero** on any failure so the CD job goes red
and the rollback step fires.

Checks, in order of how much they prove:

  1. ``/health`` answers 200 and reports ``model_loaded: true``
  2. ``/predict`` classifies a known cat as "cat" and a known dog as "dog"
  3. probabilities are well-formed (both classes present, summing to 1)
  4. a corrupt image is rejected with 422, not a 500
  5. ``/metrics`` exposes the custom prediction counter

Check 2 is the one that matters. A container can start, pass /health, and still
mispredict everything — asserting only HTTP 200 would let a broken model through
the gate, which would make the gate decorative.

    python scripts/smoke_test_deployed.py --url http://127.0.0.1
    python scripts/smoke_test_deployed.py --url http://127.0.0.1 --timeout 180
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

# Kept to stdlib on purpose: this runs on a CD runner that may not have the
# project's dependencies installed, and `requests` would be one more thing to
# install before the gate can even report.
BOUNDARY = "----catdogsmoketestboundary"


def _multipart(field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    body = b"".join([
        f"--{BOUNDARY}\r\n".encode(),
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: image/jpeg\r\n\r\n",
        data,
        f"\r\n--{BOUNDARY}--\r\n".encode(),
    ])
    return body, f"multipart/form-data; boundary={BOUNDARY}"


def _get(url: str, timeout: int = 15) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def _post_image(url: str, path: Path, timeout: int = 60) -> tuple[int, str]:
    body, ctype = _multipart("file", path.name, path.read_bytes())
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def wait_for_health(base: str, timeout: int) -> bool:
    """Poll /health until ready. Rollouts and TF startup are both slow."""
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            code, body = _get(f"{base}/health", timeout=5)
            if code == 200:
                print(f"[smoke] /health OK after {attempt} attempt(s): {body.strip()}")
                return True
        except Exception as exc:                      # noqa: BLE001 - retry anything
            if attempt % 10 == 1:
                print(f"[smoke]   not up yet ({type(exc).__name__}), retrying...")
        time.sleep(2)
    print(f"[smoke] FAIL: /health did not return 200 within {timeout}s")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1",
                    help="base URL of the deployed service")
    ap.add_argument("--timeout", type=int, default=180,
                    help="seconds to wait for /health before failing")
    a = ap.parse_args()
    base = a.url.rstrip("/")

    print(f"[smoke] target: {base}")
    failures: list[str] = []

    # --- 1. health -----------------------------------------------------------
    if not wait_for_health(base, a.timeout):
        print("\n[smoke] RESULT: FAIL (service never became healthy)")
        return 1

    code, body = _get(f"{base}/health")
    try:
        health = json.loads(body)
    except json.JSONDecodeError:
        health = {}
    if not health.get("model_loaded"):
        failures.append(f"/health reports model_loaded={health.get('model_loaded')!r} "
                        "— the service is up but has no model")
    else:
        print("[smoke] model_loaded = true")

    # --- 2/3. predictions must be CORRECT, not merely 200 --------------------
    for filename, expected in (("cat_sample.jpg", "cat"), ("dog_sample.jpg", "dog")):
        path = FIXTURES / filename
        if not path.exists():
            failures.append(f"fixture missing: {path}")
            continue
        code, body = _post_image(f"{base}/predict", path)
        if code != 200:
            failures.append(f"/predict {filename}: HTTP {code} — {body[:160]}")
            continue
        try:
            r = json.loads(body)
        except json.JSONDecodeError:
            failures.append(f"/predict {filename}: non-JSON response {body[:160]}")
            continue

        if r.get("label") != expected:
            failures.append(
                f"/predict {filename}: expected label {expected!r}, got {r.get('label')!r} "
                f"(probabilities {r.get('probabilities')})"
            )
            continue

        probs = r.get("probabilities") or {}
        if set(probs) != {"cat", "dog"}:
            failures.append(f"/predict {filename}: probabilities keys {sorted(probs)}")
        elif abs(sum(probs.values()) - 1.0) > 1e-3:
            failures.append(f"/predict {filename}: probabilities sum to {sum(probs.values())}")
        else:
            print(f"[smoke] /predict {filename:18s} -> {r['label']:4s} "
                  f"confidence {r.get('confidence')}")

    # --- 4. a corrupt image must 4xx, not 500 --------------------------------
    corrupt = FIXTURES / "corrupt.jpg"
    if corrupt.exists():
        code, _ = _post_image(f"{base}/predict", corrupt)
        if code != 422:
            failures.append(f"/predict corrupt.jpg: expected 422, got {code}")
        else:
            print("[smoke] corrupt image correctly rejected with 422")

    # --- 5. metrics ----------------------------------------------------------
    code, body = _get(f"{base}/metrics")
    if code != 200:
        failures.append(f"/metrics: HTTP {code}")
    elif "catdog_predictions_total" not in body:
        failures.append("/metrics: custom counter catdog_predictions_total absent")
    else:
        print("[smoke] /metrics exposes catdog_predictions_total")

    # --- verdict -------------------------------------------------------------
    print()
    if failures:
        print(f"[smoke] RESULT: FAIL ({len(failures)} problem(s))")
        for f in failures:
            print(f"[smoke]   - {f}")
        print("[smoke] the CD job should now roll back")
        return 1

    print("[smoke] RESULT: PASS — deployment verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
