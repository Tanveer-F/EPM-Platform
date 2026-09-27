# Phase 5 + 6 — PyTorch results and baseline comparison

## Completion

**Phase 5 + 6 checkpoint, complete and verified 2026-09-26.** PyTorch trained successfully in an Azure ML v2 command job on the existing CPU cluster. Basic MLflow parameters, metric history and artifacts were independently read back. The checkpoint was safely reloaded and all saved predictions reproduced. Compute returned to **zero allocated and zero target nodes**. At this checkpoint, later phases had not started; see [final lifecycle status](architecture.md) for the completed project state.

Successful Azure/MLflow run: `epm-pytorch-d97b75fd1f65`.

## Exact benchmark contract

- ML-ready asset: `azureml:epm-cmapss-ml-ready:d-f4ueae6u7g4c5islgehonqveey`.
- Manifest SHA-256: `2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd`.
- Same 35 causal features and original engine-disjoint partitions as Phase 4: **128,967 training rows / 567 engines**, **142 validation endpoints**, **707 test endpoints**.
- Uncapped RUL regression, inverse preprocessing to cycles, predictions floored at zero. No sequence reconstruction, new split, feature-selection change or new data asset.
- Training-only, engine-weighted population feature/target standardization; constant scales replaced by one without dropping columns. Preprocessing state saved with the model.
- Metrics use the unchanged Phase 4 regression implementation, with one equal-weight endpoint per evaluated engine. NASA score penalizes optimistic RUL errors more heavily.
- Frozen comparator: XGBoost job `epm-baseline-de82ea3141be`, using the verified original metrics in `config\baseline-reference.json`. No baseline refit.

## Model and execution

MLP **35 → 64 → 32 → 1**, ReLU and dropout 0.1. AdamW learning rate 0.001, weight decay 0.0001, engine-weighted MSE, batch size 512, gradient norm cap 5, seed 42, two CPU threads. Maximum 100 epochs with patience 12; validation RMSE alone selects the checkpoint.

Actual run: **30 epochs; selected epoch 18**. Selected state was restored and saved before the single final test evaluation. No sweeps, tuning based on test results, or validation/test refit.

Actual runtime: Linux x86_64, **Python 3.12.10, PyTorch 2.14.0+cpu, NumPy 2.5.3, PyArrow 25.0.1**. Deterministic algorithms enabled, no worker subprocesses and no CUDA runtime. The pinned Microsoft `sklearn-1.5:54` image bootstraps the isolated Python runtime inside the one-node, 3,600-second command job. No ACR, GPU, or other new service was added.

Job code hash: `41320bfd44f1193a9881f3b3aa612048d452fe66f62c365b08df7151f23cab61`.

## Main comparison

Errors and bias are in cycles. Lower is better for RMSE, MAE and NASA scores; higher is better for R². Bias is prediction minus actual; proximity to zero is preferable.

| Metric | XGBoost baseline | PyTorch MLP | Assessment |
|---|---:|---:|---|
| Validation RMSE | 31.361 | 32.819 | Worse |
| Validation MAE | 25.519 | 23.892 | Better |
| Validation R² | -0.534 | -0.680 | Worse |
| Test RMSE | **30.807** | 31.932 | Worse |
| Test MAE | 25.481 | **23.842** | Better |
| Test R² | **0.636** | 0.609 | Worse |
| Test bias | +11.394 | +7.593 | Closer to zero |
| Test NASA score sum | **51,555.979** | 316,736.285 | Substantially worse |
| Test NASA mean | **72.922** | 448.000 | Substantially worse |

**Conclusion:** the MLP improves typical absolute error and mean bias but does **not** improve the overall benchmark. XGBoost remains the stronger reference on RMSE, R² and the safety-sensitive asymmetric score. No model promotion decision or registry action was taken.

### PyTorch test results by subset

| Subset | Engines | RMSE | MAE | R² | Bias | NASA score sum |
|---|---:|---:|---:|---:|---:|---:|
| FD001 | 100 | 23.965 | 17.742 | 0.667 | +3.341 | 3,223.691 |
| FD002 | 259 | 29.514 | 22.168 | 0.699 | +3.158 | 15,541.162 |
| FD003 | 100 | 33.596 | 24.205 | 0.341 | +15.914 | 166,002.479 |
| FD004 | 248 | 36.200 | 27.904 | 0.559 | +10.583 | 131,968.952 |

Large optimistic errors, particularly in FD003/FD004, dominate the exponential NASA score despite improved overall MAE. Negative validation R² remains a concern. Validation uses retrospective 50–80% lifetime cuts, whose target distribution differs from official test endpoints. Neither result establishes field performance or calibrated failure probability. The model is a neural regressor over engineered temporal features, not a learned sequence encoder.

## MLflow verification

Independent readback confirmed:

- Run status **FINISHED**.
- **31 parameters**, **76 metric keys**, and **30 validation-epoch history entries**.
- Validation/test RMSE, MAE and R² agree with the output metrics.
- **All 11 MLflow artifacts** were downloaded through public URI-based artifact APIs and compared byte-for-byte with the verified command-job outputs.
- Checkpoint/preprocessing/spec files are **run artifacts**, not a registered MLflow/Azure model. At this Phase 5 + 6 checkpoint, model-registry and online-endpoint inventories were empty; later registry and deployment status is summarized in the [final lifecycle status](architecture.md).

The first attempt, `epm-pytorch-c591cbd651a7`, stopped **before fitting** because a 641-character provenance parameter exceeded Azure MLflow's 500-character limit. A diagnostic replay on that failed run confirmed the limit without training. The adapter now logs a SHA-256 reference for long structured values; complete provenance remains in artifacts. The retry used the unchanged model, data and optimizer configuration.

MLflow 3 run-based artifact listing also probes a logged-model search API unsupported by Azure's connector. Verification uses supported public **artifact-URI** operations instead. No service, model registry workflow or compatibility downgrade was introduced.

## Saved model and reproducibility evidence

Eleven outputs are preserved locally under ignored `artifacts\pytorch\epm-pytorch-d97b75fd1f65` and as Azure job/MLflow artifacts:

`model.pt`, `model-spec.json`, `preprocessing.json`, `metrics.json`, validation/test prediction Parquet files, `run-metadata.json`, `training-history.json`, `evaluation.md`, `comparison.json`, and `artifact-manifest.json`.

- Model SHA-256: `0783a3536b043caa48769df88c35a4554287a8c2f958aeda4cd98de1f420ab3f`.
- Artifact manifest SHA-256: `99dfe9752809e50b789332c96ccb36ae7e1fc27a88d620fb0732c074297d41af`.
- Safe reload uses `torch.load(weights_only=True, map_location='cpu')` after inventory/hash checks; no Python model object is pickled.
- All **849 validation/test predictions** match the reloaded checkpoint. Maximum Linux-to-Windows difference: **0.00000914 cycles**, within the explicit floating-point tolerance. This was a serialization/integrity check, not another training or model-selection pass.
- Source data, baseline data asset, splits and earlier dependency environment remain unchanged.

Sanitized receipts: `.azure\pytorch-job.json`, `pytorch-mlflow-verification.json`, `pytorch-reload-verification.json`, and `pytorch-compute-final.json`.

## Tests, costs and boundaries

PyTorch training/tracking tests run in `.venv-torch`; Azure orchestration and earlier-phase regression tests run in the original `.venv`. Both environments must be checked because the Azure MLflow connector pins an older Blob SDK. Final counts are recorded with the completed command results; the Windows real-symlink test remains privilege-dependent, while simulated rejection is tested.

Final validation: **94 isolated PyTorch/MLflow tests passed**, **660 root project regression tests passed**, and **48 infrastructure/script checks passed**; Ruff and both dependency checks were clean. The root run has three expected skips: the separate PyTorch module, the installed-MLflow compatibility case (both exercised successfully in `.venv-torch`), and the Windows real-symlink privilege test. Counts from the two environments overlap and are not presented as a unique-test total.

Earlier-phase runtime dependencies were not downgraded. CPU-only wheel hashes are pinned; the successful job validated the Linux lock with `pip check`. No infrastructure resources were added; existing Checkov exceptions remain unchanged.

Compute is **allocated=0, target=0, min=0, max=1** after completion. The prior VM estimate is ~$0.096/hour while allocated; setup, failed-attempt time, idle shutdown, storage and MLflow artifact operations may incur charges. No exact invoice total is asserted.

Reproduction: [pytorch-runtime.md](pytorch-runtime.md), [model design](pytorch-design.md), and [Azure/tracking adapter](pytorch-azure.md).

**Phase 5 + 6 boundary:** no endpoint, registry/promotion, monitoring, retraining, CI/CD, hyperparameter sweep or GPU work was implemented as part of this checkpoint. Subsequent phase outcomes are documented in the [final lifecycle status](architecture.md).
