# Monitoring — Prometheus + Grafana (M5)

The API exposes Prometheus metrics at `/metrics` via
`prometheus-fastapi-instrumentator`, plus one custom counter. Monitoring itself is
the community **kube-prometheus-stack** Helm chart (Prometheus + Grafana + the
Prometheus Operator) installed into the Minikube cluster.

Three layers, deliberately (ADR-008):

| Layer | What it gives | Why it exists |
|---|---|---|
| Structured JSON logs | one object per request: method, path, status, `latency_ms`, predicted class, confidence | machine-parseable in `kubectl logs`; the spec asks for request/response logging |
| Prometheus + Grafana | request rate, latency histogram, error rate, predictions by class | the graded monitoring deliverable |
| In-app counters on `GET /` | `requests`, `predictions`, `errors` | demonstrable in the demo video without port-forwarding Grafana — useful insurance given a 5-minute limit |

**No image data is ever logged.** Each request logs its byte size and a 12-character
SHA-256 prefix, never the image itself. The spec requires logging "excluding
sensitive data", and a user-submitted photo is exactly that. `tests/test_api.py`
asserts this mechanically rather than trusting the code review.

---

## 1. Install the stack

```bash
brew install helm     # if not already installed

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# The selector flag matters: by default Prometheus only picks up ServiceMonitors
# carrying the Helm release's own labels, so ours would be silently ignored and
# the target would never appear.
helm install monitoring prometheus-community/kube-prometheus-stack \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

kubectl get pods -w      # wait for prometheus + grafana to be Running
```

This installs the Prometheus Operator CRDs, which is why `k8s/servicemonitor.yaml`
can only apply *after* this step. Until then Argo CD reports the Application as
degraded — expected, not a failure (see the `SkipDryRunOnMissingResource` sync
option in `argocd/application.yaml`).

## 2. Point Prometheus at the API

`k8s/servicemonitor.yaml` is committed into `k8s/`, so Argo CD applies it along
with the Deployment and Service — monitoring config is managed the same way as
the workload, which is the usual reason a dashboard stops working silently.

If Argo CD has already synced, it is applied. Otherwise:

```bash
kubectl apply -f k8s/servicemonitor.yaml
```

Confirm the target is UP:

```bash
kubectl port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090
# http://localhost:9090 -> Status > Targets
# expect: serviceMonitor/default/catdog-api/0   2/2 up   (one per replica)
```

If the target is missing entirely, the cause is almost always the selector flag in
step 1 rather than anything in the ServiceMonitor.

## 3. Grafana dashboard

```bash
kubectl port-forward svc/monitoring-grafana 3000:80
# http://localhost:3000   user: admin
kubectl get secret monitoring-grafana \
  -o jsonpath="{.data.admin-password}" | base64 -d; echo
```

Import `monitoring/grafana_dashboard.json`, or build four panels by hand
(**Add panel → PromQL**):

| Panel | Query |
|---|---|
| Request rate (req/s) | `sum(rate(http_requests_total{handler="/predict"}[1m]))` |
| p95 latency (s) | `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{handler="/predict"}[5m])) by (le))` |
| Error rate (5xx/s) | `sum(rate(http_requests_total{status=~"5.."}[5m]))` |
| Predictions by class | `sum(rate(catdog_predictions_total[5m])) by (predicted_class)` |

The fourth panel is the one specific to this use case. Request rate and latency are
generic service health; **predictions by class** is where a broken model or a
shifted input distribution actually shows up — a sudden skew to one class is
visible there long before accuracy is recomputed.

## 4. Generate load so the panels move

With `minikube tunnel` running in another terminal:

```bash
for i in $(seq 1 200); do
  curl -s -F "file=@tests/fixtures/cat_sample.jpg" http://127.0.0.1/predict > /dev/null
  curl -s -F "file=@tests/fixtures/dog_sample.jpg" http://127.0.0.1/predict > /dev/null
done
```

Or use the M5 replay script, which additionally scores accuracy against known
labels:

```bash
python scripts/replay_batch.py --url http://127.0.0.1 --n 100
```

Export the finished dashboard (**Share → Export → Save to file**) to
`monitoring/grafana_dashboard.json` so it is reproducible rather than living only
in one browser session.

## 5. Structured logs

```bash
kubectl logs -l app=catdog-api --tail=20
```

Each line is a single JSON object:

```json
{"timestamp": "2026-08-18 14:02:11,431", "level": "INFO", "logger": "catdog_api",
 "message": "prediction", "source": "multipart", "bytes": 4584,
 "image_sha256_prefix": "a3f19c02b7d1", "predicted_class": "cat",
 "confidence": 0.999999}
```

Pipe through `jq` to confirm it parses:

```bash
kubectl logs -l app=catdog-api --tail=50 | jq -c 'select(.message=="prediction")'
```

`uvicorn` runs with `--no-access-log` so its plain-text log does not duplicate
every entry and break parsing.

## Screenshots to capture (Day 9 — `tracker/EVIDENCE.md`)

- `prometheus_targets.png` — Targets page, `catdog-api` 2/2 UP
- `grafana_dashboard_full.png` — all four panels under load
- `metrics_endpoint.png` — raw `/metrics` showing `catdog_predictions_total`
- `inapp_counter.png` — `GET /` with non-zero counters
- `pod_logs_json.png` — `kubectl logs` showing JSON lines and **no image bytes**
