# Phase 11 + 12 — Retraining and CI/CD

## Current scope and cost boundary

The workflow is implemented but **no retraining job, model registration, endpoint
deployment or Azure resource change was run for these phases**. The existing managed
CPU cluster remains configured to scale from zero to one node; no compute was started.
Automated scheduled runs and Azure CI credentials are intentionally not configured.

The orchestration reuses the existing Azure ML v2 baseline command job, CPU
configuration, curated `sklearn-1.5:54` environment, ML-ready asset and XGBoost model
name. It does not introduce an Azure service or change the training recipe. A remote
job requires the explicit `--approve-costs` flag; the existing job limit is 60 minutes
on one `Standard_D2s_v3` node. A local plan and all CI checks are free of Azure compute
usage.

## Trigger policy

The reviewed policy is `config/retraining.json`. A run is warranted when either:

- The existing inference monitor marks a batch drift-assessable (at least 20
  requests) and reports one or more feature alerts or a prediction-range alert. This
  reuses the serving monitor's current 10% out-of-reference threshold.
- At least 20 labeled production outcomes are available and **either** RMSE or mean
  asymmetric NASA score is at least 10% worse than the registered incumbent's
  benchmark.

The performance summary is a strict JSON object with `sample_size`, `rmse`, and
`mean_nasa_score`. It must be computed from labeled outcomes using the same endpoint
and scoring definitions; unlabeled drift is never treated as measured model
performance. The benchmark dataset, feature-manifest SHA-256, and incumbent version
are pinned in policy to prevent accidental cross-dataset comparisons.

Use the offline planner to evaluate one or both signal files:

```powershell
.\.venv\Scripts\python.exe -m epm_platform.retraining plan `
  --drift-summary .\.azure\monitor-summary.json `
  --performance-summary .\.azure\labeled-performance.json
```

The drift summary is the JSON object from the `epm_inference_monitoring` log event,
not the raw request payload. The planner validates the input shape and thresholds; it
does not authenticate to Azure or submit a job.

## Training, evaluation and conditional registration

After reviewing an eligible trigger and the cost impact, run the explicit Azure path:

```powershell
.\.venv\Scripts\python.exe -m epm_platform.retraining run `
  --drift-summary .\.azure\monitor-summary.json `
  --approve-costs
```

Supply `--performance-summary` instead or as well when labeled evidence exists.
Without a signal the command is skipped; without `--approve-costs` it refuses remote
training. Before submission it verifies the newest registry model and records its
version, hash and metrics. It submits the existing baseline command job once, polls
the named job, downloads the output with the baseline's checksum/ETag verification,
and checks model/data provenance before comparing metrics. An unresolved submission
or training receipt blocks another submission.

Promotion requires the candidate's same pinned 707-engine test population to have:

1. At least a **1% reduction in test RMSE** relative to the current registry
   incumbent; and
2. No increase in the mean asymmetric NASA score.

The gate verifies dataset manifest, uncapped-RUL target, engine count, and model
checksum. Rejected candidates remain job artifacts and do not create a new model
version. Accepted candidates are registered as a new version of
`epm-cmapss-rul-xgboost`, with the source job, asset/version, incumbent, test metrics
and hashes in tags; the registry result is retrieved and checked before success is
reported. No automatic deployment follows registration.

The current C-MAPSS test split is reused to preserve exact comparability with the
registered baseline. Repeated promotions on a fixed public test set can overfit the
selection process; before applying this workflow to newly collected field data,
establish a rolling, time-based labeled evaluation window and update the acceptance
policy through review. This project currently has no live labeled-performance
feed, so that trigger is ready for supplied evidence but not wired to an Azure
schedule or production telemetry store.

If a process stops while a job is active or its submission result is uncertain,
inspect `.azure\retraining-state.json` and the Azure ML job before doing anything
else. Resume checks only the recorded job and **never submits or retries**:

```powershell
.\.venv\Scripts\python.exe -m epm_platform.retraining resume
```

The resumable receipt and downloaded outputs are ignored local artifacts; protect
them like other workspace metadata and never commit `.azure`.

## GitHub CI and promotion

`.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual
dispatch. It uses read-only repository permissions and no Azure credentials. CI
installs the locked Python 3.12 dependencies and runs:

- All unit and integration tests, including local HTTP scoring, inference contract,
  monitoring, a synthetic native-XGBoost API smoke test, data/config validation, and
  trigger/promotion gates.
- Synthetic PyTorch training and MLflow tracking tests in their isolated CPU-only
  runtime; CI does not train on the real dataset.
- Ruff linting.
- Bicep compilation and the existing infrastructure security/configuration contract
  checks.

CI does **not** submit Azure ML jobs, deploy an endpoint, register a model, change
resource providers, or call Azure control/data planes. Model promotion remains a
separate, explicit operator action after the acceptance gate. Configure no GitHub
secrets for the current workflow. If remote execution is introduced later, use
short-lived federated Entra credentials with minimum workspace permissions and a
separately reviewed protected environment; never add client secrets to repository
variables or workflow YAML.
