# Phase 7 + 8 — selected model and Azure ML registry

## Status

**Complete, verified 2026-09-26.** The existing XGBoost baseline was selected and registered without retraining. The versioned Azure ML model asset is `epm-cmapss-rul-xgboost:1` (`custom_model`). Its source job, model metadata and downloaded artifacts were verified against the completed Phase 4 job output. The later Phase 9 Azure endpoint attempt did not produce a retained endpoint; no new infrastructure or compute allocation remains.

## Selection

The comparison uses the same frozen engine-disjoint data, target and evaluation contract documented in [the baseline report](baseline-results.md) and [the PyTorch comparison](pytorch-results.md).

| Test metric | XGBoost | PyTorch MLP |
|---|---:|---:|
| RMSE (cycles) | **30.807** | 31.932 |
| MAE (cycles) | 25.481 | **23.842** |
| R² | **0.636** | 0.609 |
| NASA score sum (lower is better) | **51,555.979** | 316,736.285 |

XGBoost was selected because it has lower test RMSE and a substantially lower asymmetric NASA score, an important penalty for optimistic remaining-life estimates. The MLP's better MAE does not outweigh those measures. These are benchmark metrics, not evidence of calibrated failure probability or field performance.

## Registration and lineage

| Field | Verified value |
|---|---|
| Azure ML model | `epm-cmapss-rul-xgboost:1` |
| Asset type | `custom_model` |
| Azure ML training job | `epm-baseline-de82ea3141be` |
| Source MLflow run / experiment | `epm-baseline-de82ea3141be` / `epm-baseline-rul` |
| Metadata-only MLflow run | `6213db0e-626d-4733-8a6f-e289e07440db` |
| ML-ready asset | `epm-cmapss-ml-ready:d-f4ueae6u7g4c5islgehonqveey` |
| ML-ready manifest SHA-256 | `2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd` |
| Native XGBoost model SHA-256 | `e58cb0a9285c364856361ede3c10de16facc7c4f2a48b1ae643515db39d5d0fe` |
| Artifact manifest SHA-256 | `c280467cc28509fb04e63ad6ba1c26c86b8e16a09636b8ec1ce9e952c2641e09` |

The original completed baseline run was `FINISHED` but had no recorded parameters or metrics. To avoid retraining or modifying that immutable source run, a clearly tagged metadata-backfill run was added to its existing experiment using only checksum-verified job outputs. The backfill records **100 parameters, 71 metrics and all 8 output artifacts**, including dataset/code/environment provenance and per-subset validation/test results. It explicitly records that no training occurred. Registration tags link both the original training job/run and this supplementary metadata run.

Azure ML resolved the job output reference to the workspace datastore path and records `job_name=epm-baseline-de82ea3141be`. `scripts\Register-XGBoostModel.ps1` checks the source job, selected metrics, existing version inventory, registry metadata and source URI; downloads version 1 and verifies the artifact manifest and every artifact's size and SHA-256 before writing a local sanitized receipt.

## Reproduction and verification

After initializing the configured environment and selecting the intended Azure subscription, rerun the idempotent verification/registration script:

```powershell
powershell -NoProfile -ExecutionPolicy RemoteSigned -File .\scripts\Register-XGBoostModel.ps1
```

It will refuse to overwrite unexpected model versions and will stop on inconsistent lineage or artifact hashes. The local verification receipt is `.azure\registered-xgboost-model.json`; downloaded verification files are under `.azure\registered-model-verification`. These generated evidence files are ignored and must not be committed.

Final live checks: model version 1 is retrievable; its source-job lineage and registered metrics/tags match the reviewed artifacts; **8/8** job outputs were downloaded and verified; `cpu-dev` has **0 nodes** with min=0/max=1; the online-endpoint inventory is empty.

## Format and boundaries

The registered asset contains the validated native XGBoost JSON bundle. It is not a packaged MLflow pyfunc or endpoint-ready scoring interface. The Phase 9 serving wrapper loads this artifact and makes an uncapped RUL regression prediction, not a calibrated failure probability. The asset remains in development stage; no alias, production stage or retained endpoint was created. The original run was not edited, and neither model was retrained during registry selection. The separately implemented Phase 11 promotion workflow is documented in [retraining and CI/CD](retraining-cicd.md); it has not been executed.
