# Phase 9 + 10 — inference, monitoring and cost controls

## Status and Azure outcome

The local inference and monitoring path is implemented and tested with the registered
`epm-cmapss-rul-xgboost:1` model. The single authenticated Managed Online Endpoint
attempt used one `Standard_D2s_v3` instance. The Activity Log now provides the
service-side response: endpoint write `epm-rul-smoke-f93d5264` was accepted with
HTTP 201 at `2026-09-27T05:10:09Z`, then reached terminal failure at
`2026-09-27T05:10:25Z`:

```text
ResourceOperationFailure: The resource operation completed with terminal provisioning state 'Failed'.
SubscriptionNotRegistered: Resource provider [N/A] isn't registered with Subscription [N/A].
```

The failed write correlation ID is
`1431c605-584b-410a-badd-2d8db2f843b0`. The event does **not** contain the resource
provider namespace (`[N/A]`) or additional details. Read-only checks show
`Microsoft.MachineLearningServices`, `Microsoft.ContainerRegistry`,
`Microsoft.Network`, `Microsoft.Storage` and `Microsoft.Compute` are registered,
but that does not establish which other provider, if any, Azure requires. The endpoint
was deleted successfully at `2026-09-27T05:10:59Z`; the current inventory is empty.
No deployment or Azure inference request completed.

Because the actual error omits the provider name, no provider was guessed or registered
and no retry was made. The exact namespace must come from Azure Support or a more
detailed service-side diagnostic before a safe deployment retry. The ignored receipt
`.azure\endpoint-smoke.json` retains cleanup evidence. Transient provisioning may
have incurred a small charge; an exact amount is not available and billing can be
delayed. The estimate was approximately **USD 0.096/hour** for one active
`Standard_D2s_v3` instance, excluding ancillary charges.

Azure Monitor/Application Insights endpoint monitoring is therefore **not live**.
The attempted deployment was configured for token authentication and the workspace's
existing Application Insights, but Azure did not provide a verifiable serving
deployment. No monitoring service or Azure resource was added.

## Scoring contract

The serving code lives in `src\epm_platform\serving` and loads only the registered
native XGBoost JSON after checking the model and artifact-manifest SHA-256 values.
Predictions use the frozen 36-tree range and the Phase 4 zero floor. Output is
**uncapped RUL in cycles**, not a failure probability or a safety decision.

Requests use this closed JSON shape:

```json
{
  "instances": [
    {
      "subset": "FD001",
      "unit_id": 1,
      "observations": [
        {
          "cycle": 1,
          "setting_1": 0.1,
          "setting_2": 0.2,
          "setting_3": 100.0,
          "sensor_01": 500.0,
          "sensor_02": 600.0,
          "sensor_03": 1400.0,
          "sensor_04": 1200.0,
          "sensor_05": 10.0,
          "sensor_06": 15.0,
          "sensor_07": 400.0,
          "sensor_08": 2200.0,
          "sensor_09": 8500.0,
          "sensor_10": 1.0,
          "sensor_11": 45.0,
          "sensor_12": 300.0,
          "sensor_13": 2300.0,
          "sensor_14": 8000.0,
          "sensor_15": 9.0,
          "sensor_16": 0.02,
          "sensor_17": 350.0,
          "sensor_18": 2200.0,
          "sensor_19": 100.0,
          "sensor_20": 25.0,
          "sensor_21": 15.0
        }
      ]
    }
  ]
}
```

Each instance contains exactly the latest `min(cycle, 10)` ordered, contiguous
observations for that engine, ending at the reported cycle. The client must preserve
engine boundaries and send the real cycle numbers. The scorer derives the 35 approved
features using only that trailing history; it rejects missing/extra fields, nulls,
non-finite/out-of-range values, gaps, duplicate JSON properties, invalid unit IDs and
unknown subsets. Requests contain 1–100 engines. No raw values or engine identifiers
are written to inference monitoring logs.

## Local VS Code inference API

The dependency set already includes the pinned CPU XGBoost runtime; no new package,
GPU, container, or service is required. After Phase 8 has downloaded and verified the
model asset:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_endpoint_smoke_request.py `
  --output .\.azure\endpoint-smoke-request.json --sample-size 20
.\.venv\Scripts\python.exe -m epm_platform.serving.local_api `
  --model-dir .\.azure\registered-model-verification\epm-cmapss-rul-xgboost `
  --host 127.0.0.1 --port 8000
```

The local HTTP service binds only to loopback. In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Uri http://127.0.0.1:8000/score -Method Post `
  -ContentType application/json -InFile .\.azure\endpoint-smoke-request.json
```

`GET /health` is a local process check. `POST /score` returns one nonnegative
`rul_cycles` prediction per engine; malformed requests receive 4xx responses. This
local HTTP server is a developer test surface, **not** a production web server.
Azure ML's `init()`/`run()` entrypoint calls the same scoring service and was packaged
in the one attempted endpoint deployment.

The smoke payload is generated from the verified Phase 2 **training** observations;
it does not use test labels or retrain/change either model.

## Monitoring behavior and limits

The monitoring reference `src\epm_platform\serving\monitoring-reference.json` is
reproducibly generated by `scripts\build_serving_reference.py`. It is pinned to the
registered model and ML-ready manifest. It summarizes all 35 features from the
128,967-row Phase 4 training partition (mean, standard deviation and 1st/99th
percentiles) and predictions from the 142 validation endpoints.

Every local or Azure scoring call produces structured log events for:

- request success/rejection and inference latency;
- strict input-schema, finite-value and contiguous-history validation;
- batch RUL mean/min/max and zero-floor count;
- per-feature fraction outside the training 1st/99th percentile range, and output
  fraction outside the validation-prediction range.

Range-based drift is assessed only for batches of at least **20 engines**. A feature
or prediction-range alert is raised when at least **10%** of that batch is outside its
reference range. Single-engine requests still receive schema, range, prediction and
latency checks, but distribution drift is marked unassessable rather than inferred
from one observation. Logs contain aggregate values and feature names only—never
payloads, engine IDs, secrets, or raw sensors.

This is a practical warning signal, not a formal drift test: it does not detect
distribution changes that remain inside the reference percentile bounds, handle
seasonality, or establish model performance without delayed ground truth. The
validation predictions reflect the retrospective 50–80%-life censoring policy, so
the output reference is biased and must not be treated as a production SLO or
operating limit. No automated alerts, continuous dashboard, label join, retraining,
or drift-triggered action is enabled.

For local inspection, keep the process output in the ignored `.azure` directory:

```powershell
.\.venv\Scripts\python.exe -m epm_platform.serving.local_api `
  --model-dir .\.azure\registered-model-verification\epm-cmapss-rul-xgboost `
  --host 127.0.0.1 --port 8000 2>&1 |
  Tee-Object .\.azure\inference-monitoring.jsonl
```

Do not commit generated requests, model files, receipts, or telemetry.

## Azure lifecycle, health and cleanup

`scripts\deploy_xgboost_endpoint.py` records the one attempted Azure flow. It uses
the registered model, Microsoft curated `sklearn-1.5:54` environment, one fixed
`Standard_D2s_v3` replica, `aml_token` endpoint authentication and the already-linked
Application Insights component. It creates a uniquely named endpoint, smoke-tests
20 representative engines, compares predictions to local inference, and deletes the
entire endpoint in `finally`. It refuses to repeat when a receipt already exists.
Because the Azure attempt was blocked, do **not** rerun this script as an automatic
retry. The endpoint inventory was verified empty after cleanup.

If a separately authorized future deployment is created, inspect Azure ML endpoint
metrics (request count, latency, failures) and deployment logs while the endpoint is
active. Keep authentication token-based, limit replicas/SKU, validate with a bounded
smoke test, and delete the endpoint after the approved test window. Emergency cleanup:

```powershell
az ml online-endpoint list --resource-group rg-epm-dev-eastus `
  --workspace-name mlw-epm-dev-eastus --output table
az ml online-endpoint delete --name <temporary-endpoint-name> --yes `
  --resource-group rg-epm-dev-eastus --workspace-name mlw-epm-dev-eastus
az ml online-endpoint list --resource-group rg-epm-dev-eastus `
  --workspace-name mlw-epm-dev-eastus --output table
```

Use the same selected Azure CLI subscription for every command. Verify the deleted
endpoint is absent; do not assume a submitted delete request has completed. Endpoint
delete removes its deployments, but does not delete the registered model. The
cost-conscious fallback that is currently validated is the loopback local API and
its structured monitoring logs.

## Validation and boundaries

Focused checks cover exact feature parity with the causal training pipeline, malformed
and non-finite inputs, drift summaries, lineage-pinned reference stats, loopback-only
HTTP behavior, health, error responses, and real local inference over 20 verified
engine histories. The current Azure status is intentionally reported as **not
deployed**; no successful cloud prediction or cloud metrics readback is claimed.
No model retraining, endpoint persistence, GPU or new Azure service was added in
Phases 9 + 10. The retraining workflow and CI were implemented separately in Phases
11 + 12; neither is scheduled or automatically connected to this local monitor.
