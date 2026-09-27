# Phase 5/6 CPU runtime and scope

## Approved model and comparison

The compact PyTorch MLP uses the **same 35 causal features, uncapped RUL targets, engine weights, 142 validation endpoints and 707 test endpoints** as the Phase 4 XGBoost benchmark. It is a feed-forward neural model over engineered time-series features, not a recurrent sequence model. No dataset, split or feature pipeline is redesigned.

Model: 35 → 64 → 32 → 1, ReLU, dropout 0.1. Train-only feature/target standardization is necessary for neural optimization and saved with the checkpoint. Constant training features use scale 1 and remain present. Training uses AdamW, weighted MSE, batch 512, up to 100 epochs, patience 12, two CPU threads and seed 42. The validation checkpoint is restored before final test evaluation. No sweep or refit on validation/test.

The reference in `config\baseline-reference.json` was copied from the verified Phase 4 output; it is not a newly fitted comparator. Its job name and full ML-ready manifest hash are checked. Results must be compared on the original cycles scale, with the same nonnegative prediction floor and RMSE, MAE, R², bias and asymmetric NASA score. Improvements are not assumed in advance.

## Why the runtime is isolated

The official `azureml-mlflow==1.62.0.post6` connector requires `azure-storage-blob<=12.27.1`, while existing orchestration uses a validated newer Blob SDK. Therefore:

- `.venv` remains the Phase 1–4 Azure orchestration environment.
- `.venv-torch` is the Python 3.12 PyTorch/MLflow test and tracking-query environment.
- Remote execution uses the same pinned Microsoft curated image `sklearn-1.5:54` as Phase 4. Its conda creates an isolated Python 3.12.10 runtime during the command job.
- No custom environment image, ACR, GPU or other service is provisioned.

Runtime pins include **PyTorch 2.14.0+cpu**, **MLflow skinny 3.15.0**, the Azure MLflow connector, NumPy 2.5.3, PyArrow 25.0.1 and CPU-only XGBoost 3.4.1. XGBoost is present only because the existing baseline module supplies the shared metric/loading helpers; the PyTorch job does not fit another tree model. The skinny MLflow package avoids installing a full tracking server/model-serving stack. Some third-party transitive packages are part of the connector, not additional deployed services.

PyTorch is acquired through a SHA-256-pinned **official CPU wheel URL**. Installing an unqualified Linux `torch` package can pull CUDA packages; that is deliberately avoided. `requirements-windows.txt` and `requirements-linux.txt` use platform-appropriate CPU wheels and otherwise matching numerical/tracking pins. The remote bootstrap asserts `torch.version.cuda is None` and runs `pip check` before training.

```powershell
py -3.12 -m venv .venv-torch
.\.venv-torch\Scripts\python.exe -m pip install -r environments\pytorch\requirements-windows.txt -r environments\pytorch\requirements-test.txt
$env:PYTHONPATH = 'src'
.\.venv-torch\Scripts\python.exe -m pip check
.\.venv-torch\Scripts\python.exe -m pytest tests\deep_learning\test_training.py tests\deep_learning\test_tracking.py -q --disable-warnings
.\.venv\Scripts\python.exe -m pytest tests\deep_learning\test_azure.py -q --disable-warnings
```

Tests requiring Azure ML SDK orchestration are run in `.venv`; numerical/tracking tests run in `.venv-torch`. Missing optional-runtime tests may skip in the other environment, but must pass in their intended environment. Do not install the project dependency set into `.venv-torch` and undo this separation. No machine-wide Python packages or execution-policy changes are required.

## Tracking scope

MLflow tracking attaches to the Azure ML command run and records parameters, epoch/final metrics, provenance and saved artifacts. The checkpoint and preprocessing files are **run artifacts**, not a registered/promoted model. No autologging, model registry API, endpoint, server, system-metrics monitoring or scheduled retraining is added.

Azure's MLflow backend limits parameter values to 500 characters. Longer structured provenance is represented by a SHA-256 parameter reference; complete values remain in the model/run artifact manifests. This does not truncate provenance or change model configuration. A fingerprint-key collision fails closed.

Tracking is explicit and synchronous: a tracking failure cannot silently become a successful tracked run. Local unit tests use fakes; the actual Azure run and its stored MLflow metrics/artifact list must be independently inspected for completion. Credentials and tracking URLs remain runtime configuration, not source constants or saved secrets.

## Cost and execution boundary

The existing `cpu-dev` remains min=0/max=1 and uses a 3,600-second execution limit. Dependency setup counts against that execution time; allocation/preparation and idle shutdown can incur additional charges. The prior public VM estimate is ~$0.096/hour while allocated, plus existing storage/telemetry and operations. This is not a guaranteed budget cap. Validate actual zero allocated/target nodes after execution.

## Remote commands and verification

Run the Azure adapter in the unchanged project environment via these wrappers:

```powershell
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Invoke-PyTorch.ps1 -Action Submit -ApproveCosts
$job = Get-Content .\.azure\pytorch-job.json -Raw | ConvertFrom-Json
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Wait-Baseline.ps1 -JobName $job.job_name
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Invoke-PyTorch.ps1 -Action Download -JobName $job.job_name -Destination "artifacts\pytorch\$($job.job_name)"
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Verify-PyTorchTracking.ps1 -JobName $job.job_name -ArtifactsPath "artifacts\pytorch\$($job.job_name)"
```

`Wait-Baseline.ps1` is the shared read-only Azure command-job waiter despite its historical name. Every explicit submit creates a new billable run; use the saved receipt to inspect an uncertain submission rather than resubmitting blindly. Download requires a completed job and an empty destination, validates all eleven output objects and their manifest hashes, and never loads a pickle object.

MLflow verification reads back the finished run, configuration parameters, all 30 epoch metrics for the verified run, final metrics and every artifact's actual bytes. Azure's connector does not expose the newer MLflow 3 logged-model search route, which run-based artifact listing may probe. The verifier therefore uses the **public URI-based artifact APIs** against the run's artifact URI; it does not use model registry APIs or assume a successful log call proves persistence.

Actual results and the comparison are recorded in [pytorch-results.md](pytorch-results.md). Registry/promotion, endpoint, monitoring and retraining/CI code were implemented in their later authorized phases; the final project status and operational limits are summarized in [the final architecture](architecture.md). No GPU infrastructure was added.
