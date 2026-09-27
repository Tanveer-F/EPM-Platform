# Bounded PyTorch training and basic Azure ML tracking

## Approved scope

This adapter submits **one CPU command job**, not an endpoint, model registration, sweep,
new compute cluster, data publication or tracking server. It uses existing `cpu-dev`
(`Standard_D2s_v3`, dedicated, private node, system-assigned identity, minimum 0,
maximum 1, 300-second idle scale-down) and refuses compute-policy drift.
The process uses two CPU threads, seed 42, at most 100 epochs and a hard command-job
timeout of 3,600 seconds, including runtime bootstrap. The detailed learning and
checkpoint contract is in [pytorch-design.md](pytorch-design.md).

Only this previously approved Phase 3/4 input is accepted:

- Asset: `epm-cmapss-ml-ready:d-f4ueae6u7g4c5islgehonqveey`.
- Manifest SHA-256:
  `2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd`.
- Existing asset/source identities, full hashes, paths and tags are checked using the
  unchanged baseline publication helpers. The training wrapper checks the mounted
  manifest bytes against the full approved digest, before opening a tracking run.
- `config\baseline-reference.json` is required and must identify this same input;
  core training additionally validates the comparison reference before fitting.

The job is named `epm-pytorch-` followed by 12 lowercase hexadecimal characters,
under experiment `epm-pytorch-rul`. It uses the pinned, already proven public environment
`azureml://registries/azureml/environments/sklearn-1.5/versions/54`.
This is a public environment **reference**, not a registry mutation. No workspace
or private environment is registered.

## Keep orchestration and training environments separate

The root Python 3.12 `.venv` remains the existing Azure orchestration environment.
Do not install Torch/MLflow into it or weaken its Blob SDK constraint.
Use `.venv-torch` for offline model tests and tracking-compatible training dependencies.
The remote LF-only bootstrap installs the reviewed Linux runtime requirements into its
isolated job environment, then invokes:

```text
python -m epm_platform.deep_learning.tracking --data <mounted-input> --config config/pytorch.json --baseline-reference config/baseline-reference.json --output <mounted-output>
```

The runtime uses `mlflow-skinny==3.15.0` with `azureml-mlflow==1.62.0.post6`.
The Azure connector constrains `azure-storage-blob<=12.27.1`, conflicting with the root
orchestration requirement `azure-storage-blob>=12.30`. Separate manifests/locks and
virtual environments are intentional. `xgboost-cpu` remains a training-runtime dependency
because the PyTorch core reuses the existing offline baseline partition/metric helpers.
No GPU runtime is required.

## Reviewed code upload only

`deep_learning.azure` builds a content-addressed snapshot under
`.azure\pytorch-code\<content-sha256>`. It copies an explicit allowlist rather than
uploading the working tree:

- The existing baseline stage's offline package, data and feature helper modules,
  including `baseline\training.py`.
- `deep_learning\__init__.py`, `training.py`, and `tracking.py`.
- `config\cmapss-source.json`, `features.json`, `pytorch.json`, and
  `baseline-reference.json`.
- `scripts\run-pytorch.sh` becomes `config\run-pytorch.sh`.
- `environments\pytorch\requirements-linux.txt` becomes
  `config\runtime-requirements.txt`.

`deep_learning\azure.py`, `.env`, credentials, local data, `.azure` receipts, virtual
environments and unrelated files are not uploaded. Source and staged links/junctions,
traversal, unexpected staged entries, changed content, oversized files/bundles and
CRLF shell scripts are rejected. The code digest identifies the copied bytes, not a
mutable source directory. Only generic existing safe helpers are reused; no baseline
module globals, baseline job names or baseline output assumptions are patched.

## Explicit submission and uncertain-response recovery

Use the root environment and existing Azure configuration/identity. These are operational
commands, not commands executed by unit tests:

```powershell
$env:PYTHONPATH = 'src'
& '.\.venv\Scripts\python.exe' -m epm_platform.deep_learning.azure submit --approve-costs
```

The asset, digest and environment defaults are exactly the approved values. Explicit
`--asset-version`, `--manifest-sha256` and `--environment-version` arguments are accepted
only if equal to those values. There is no `publish` operation.

The adapter writes `.azure\pytorch-job.json` with the generated job name and
`SubmissionPending` **before** calling `jobs.create_or_update` once. A lost response
records `SubmissionUnknown` when possible; a hard interruption can leave
`SubmissionPending`. Both require inspecting that saved name rather than resubmitting.
There is no automatic retry, cancellation, credential fallback or duplicate-job recovery
by creating another job. Issuing another explicit `submit` is a new billable job and can
replace the current receipt; retain the original receipt when investigating uncertainty.

```powershell
$receipt = Get-Content '.\.azure\pytorch-job.json' -Raw | ConvertFrom-Json
& '.\.venv\Scripts\python.exe' -m epm_platform.deep_learning.azure status --job-name $receipt.job_name
```

Status returns only the requested job identity, an allowlisted status (or `Unknown`)
and fixed failure/cancellation codes. Service messages, portal URLs, SAS tokens and
credential-bearing exceptions are not printed. The CLI prints one sanitized JSON result.

The command submitted is fixed:

```text
bash config/run-pytorch.sh --data ${{inputs.ml_ready}} --config config/pytorch.json --baseline-reference config/baseline-reference.json --output ${{outputs.pytorch}}
```

Input is the existing exact asset in `download` mode. Output `pytorch` is `uri_folder`,
`upload` mode, at the existing workspace Blob datastore path `pytorch/<job-name>/`.
`ManagedIdentityConfiguration()` is explicit, with one compute instance. Environment
variables are `PYTHONPATH=./src`, `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`,
`PYTHONHASHSEED=42`, and false values for `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING`,
`MLFLOW_ENABLE_ASYNC_LOGGING`, and `MLFLOW_ENABLE_TELEMETRY`.
No raw input URI, tracking URI or secret is injected by the adapter.

## Basic MLflow run artifacts, not registered models

Azure injects the tracking URI, existing run identity and connector credentials.
The wrapper requires an effective `azureml://` tracking URI; a file backend, local
server or missing/mismatched URI fails closed. It neither prints nor overrides that URI.
It resumes `MLFLOW_RUN_ID` explicitly when supplied. Without an active run it opens one
`mlflow.start_run(log_system_metrics=False)` context, never an experiment or nested run.
An already active matching run is respected and remains owned by its caller. A conflicting
active and injected run identity is rejected.

The core has no MLflow import; the adapter imports MLflow lazily. Logging is explicit:

- Approved model configuration and nested provenance are flattened to parameters.
  Asset/version/digest and baseline-reference hash are logged alongside core feature,
  configuration and source hashes. Source and feature provenance also receive tags.
- Each one-based epoch logs `train_loss`, `train_rmse`, and `val_rmse` at that step.
- Final overall and per-subset numeric metrics have stable keys such as
  `validation_rmse`, `test_rmse`, `test_fd001_rmse`, and `test_fd004_mean_nasa_score`.
  Undefined R² explanations remain in the JSON report instead of becoming invalid metrics.
- `best_epoch`, `epochs_run` and `best_validation_rmse` are recorded after completion.
- The complete verified output directory is uploaded exactly once with
  `mlflow.log_artifacts(output_dir, artifact_path='pytorch')`. It includes the checkpoint,
  preprocessing, model specification, predictions, metrics, comparison and reports.

No `mlflow.pytorch.log_model`, MLflow 3.x LoggedModel creation, registry client, autologging,
system-metrics monitor, experiment creation or server process is used. Parameter, tag
and metric calls explicitly request synchronous logging; artifact calls are synchronous.
Failures propagate and fail the job. An owned run context finishes as `FINISHED` on
success or `FAILED` on an exception. The wrapper does not prematurely assert finished
status while still inside the context. An artifact-upload failure retains completed local
files for diagnosis but does not claim successful tracking.

The sanitized success summary includes run ID and best epoch, not a tracking URL.
Any later remote `get_run(job-name)` / artifact verification record belongs outside the
immutable output manifest, for example under local `.azure`; this wrapper adds no
tracking receipt to the eleven-file bundle.

## Integrity-checked download

```powershell
& '.\.venv\Scripts\python.exe' -m epm_platform.deep_learning.azure download `
  --job-name $receipt.job_name --destination '.\artifacts\pytorch-run'
```

Download accepts only `Completed` jobs and the exact `pytorch` output URI for that name.
It validates the existing keyless default workspace datastore, storage account,
container and private-access policy, then streams directly through the Blob SDK.
It does not trust an Azure ML SDK download that may return without copying files.

The remote prefix must contain exactly these eleven files:

```text
model.pt                         model-spec.json
preprocessing.json               metrics.json
predictions_validation.parquet   predictions_test.parquet
run-metadata.json                training-history.json
evaluation.md                    comparison.json
artifact-manifest.json
```

Each object is bounded to 64 MiB; total output to 256 MiB; the manifest to 1 MiB.
The manifest must match the PyTorch schema, exact feature/source provenance, ten unique
safe content names and model SHA-256. Every download verifies declared size, SHA-256 and
conditional ETag before/after streaming; a final prefix listing rechecks inventory and
ETags. Unexpected names, traversal, duplicate entries, short/long streams, extra objects,
checksum drift or a concurrent remote change fail without publishing a completed bundle.

The destination must be new or empty. Downloads use exclusive staging within it, publish
content without overwrite and publish `artifact-manifest.json` last. Caught failures
remove only this invocation's files. The downloader never deserializes `model.pt` or
imports Torch; safe loading is a separate trusted-model operation.

## Offline validation

```powershell
$env:PYTHONPATH = 'src'
& '.\.venv\Scripts\python.exe' -m pytest --import-mode=importlib tests\deep_learning\test_azure.py tests\deep_learning\test_tracking.py -q
& '.\.venv-torch\Scripts\python.exe' -m pytest tests\deep_learning\test_tracking.py -q
& '.\.venv\Scripts\python.exe' -m ruff check src\epm_platform\deep_learning\azure.py src\epm_platform\deep_learning\tracking.py tests\deep_learning\test_azure.py tests\deep_learning\test_tracking.py
```

Azure tests mock SDK operations. Tracking unit tests use a strict fake MLflow surface,
not a real run or server. The optional installed-API signature check uses
`pytest.importorskip('mlflow')`, so the unchanged root environment need not install it.
Tests cover budgets, identity, exact selectors, allowlisted staging, recovery receipts,
manifest-last integrity download, absence of registry/experiment calls, synchronous
parameters/metrics/artifacts and fail-closed tracking. They perform no cloud calls.
