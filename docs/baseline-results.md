# Phase 3 + 4 — classical RUL baseline evidence

## Status

**Phases 3 + 4 checkpoint complete**, verified 2026-09-26. Remote training completed; all eight outputs were downloaded with checksum/ETag verification; all 849 saved validation/test predictions match the reloaded model; CPU allocated and target nodes are both zero. At this checkpoint, deep learning, model registry/promotion, endpoint, monitoring, retraining and CI/CD had not yet been implemented. See [final lifecycle status](architecture.md) for later phase outcomes.

## Dataset and leakage boundary

Source: `azureml:epm-cmapss-curated:d-xjsfezqpozevgso26sct6tpodm`.

ML-ready: `azureml:epm-cmapss-ml-ready:d-f4ueae6u7g4c5islgehonqveey`.

Feature manifest SHA-256: `2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd`.

- One pooled model, FD001–FD004; report each subset independently.
- **35 features**: cycle, 3 current operating settings, 21 current sensors, trailing-10 means/slopes for sensors 03/04/11, trailing operating-setting means and history count.
- Engine-hash split, seed 42: **567 training engines / 128,967 rows**, **142 validation engines / 142 censored endpoints**, **707 test engines / 707 original endpoints**.
- Every engine stays in one partition. Rolling statistics use only current/past observations within that engine. IDs, full lifetime and target are not predictors.
- Uncapped RUL: retrospective training/validation labels use the full known run-to-failure record; test labels are the original supplied endpoint RUL. Full lifetime is used for offline labels/censoring only.
- Validation endpoints are deterministically censored at 50–80% of held-out engine lifetime. This simulates partial histories but biases the evaluation distribution; it is not a claimed live sampling policy.
- Training weights equalize total contribution per engine. Validation/test metrics give one equal contribution per evaluated engine.
- Two independent feature builds produced identical bytes/version. All seven cloud objects were checksum-verified before registration.

Detailed definitions: [features.md](features.md). Core model/evaluation contract: [baseline-design.md](baseline-design.md).

## Frozen configuration and remote run

Successful command job: **`epm-baseline-de82ea3141be`**, existing `cpu-dev`, `Standard_D2s_v3`, one node, two CPU threads, 3,600-second job limit. No sweep.

Pinned curated environment: `azureml://registries/azureml/environments/sklearn-1.5/versions/54`. Its container supplies conda; `scripts/run-baseline.sh` creates an isolated **Linux Python 3.12.10** runtime with **XGBoost CPU 3.4.1, NumPy 2.5.3, SciPy 1.18.1, PyArrow 25.0.1**. The actual runtime versions are recorded in `run-metadata.json`. Setup/package download occurs inside the bounded CPU job; this avoids a workspace image build at the cost of startup time/network dependency.

XGBoost: squared-error objective, CPU histogram trees, depth 6, learning rate 0.05, min child weight 5, L2=5, max bins 256, subsample/column sample=1, seed 42. Maximum 600 rounds, patience 50, validation-only RMSE early stopping. There were **86 attempted rounds; best zero-based iteration 35; 36 trees saved**. No refit on validation; the model is frozen before a single final test evaluation. Predictions are floored at zero with no upper cap.

Model SHA-256: `e58cb0a9285c364856361ede3c10de16facc7c4f2a48b1ae643515db39d5d0fe`.

Job code bundle hash: `fca4f48ea125278a9b6089701970b6d5a150a4921bfaff4a896202b366bea2c5`.

## Actual results

Units for RMSE, MAE and bias are cycles. Bias is predicted minus actual; positive is optimistic/late-maintenance risk.

| Partition / subset | Engines | RMSE | MAE | R² | Bias | asymmetric RUL score sum |
|---|---:|---:|---:|---:|---:|---:|
| Validation overall | 142 | 31.361 | 25.519 | -0.534 | +15.888 | 17,885.171 |
| Test overall | 707 | **30.807** | **25.481** | **0.636** | **+11.394** | **51,555.979** |
| Test FD001 | 100 | 24.056 | 20.593 | 0.665 | +12.334 | 1,571.893 |
| Test FD002 | 259 | 30.887 | 25.573 | 0.670 | +8.154 | 13,565.077 |
| Test FD003 | 100 | 31.648 | 25.514 | 0.415 | +21.540 | 15,921.532 |
| Test FD004 | 248 | 32.743 | 27.344 | 0.639 | +10.307 | 20,497.476 |

asymmetric RUL score uses `expm1(-error/13)` for underestimation and `expm1(error/10)` for overestimation, summed over engines; lower is better and values depend on evaluation population. Overall test mean asymmetric RUL score: **72.922**. No precision/recall/F1/ROC-AUC is reported because the approved task is regression, not a thresholded failure classifier.

### Observations and limitations

- This is a reproducible **benchmark**, not a claim of production-quality failure prediction. Negative validation R² means it underperforms a constant validation-target mean under that diagnostic metric. The censored validation target distribution differs from the official test distribution; test and validation R² are not directly comparable.
- Positive bias in every subset, particularly FD003, is operationally concerning. The asymmetric score makes optimistic remaining-life errors expensive. No parameter/feature changes were made after seeing test results.
- Pooled operating/fault regimes and uncapped early-life RUL remain difficult. Only three sensors receive trend features; no operating-regime normalization, tuned feature selection, hyperparameter sweep or probabilistic calibration was added.
- Exact validation TreeSHAP contributions and training gain are stored as global feature rankings, not causal explanations. No extra SHAP library or test-based feature selection is used.
- Future PyTorch comparison must use the same frozen asset, split, target, weighting and endpoint evaluation. Changing any of those is a new benchmark, not a directly comparable model improvement.

## Cloud integration and changes from initial proposal

Basic ACR was initially approved and created, but Azure rejected attachment because the existing managed network requires **Premium**. After explicit user selection of the cost-saving alternative, the job uses the pinned public curated environment. The empty, unattached Basic registry was then **deleted with explicit approval**. No Premium registry was created; final IaC contains no registry resource and does not re-create it.

Only scoped runtime grants remain: compute identity has Blob Data Reader on the curated container; compute and the authorized publisher have Blob Data Contributor on the existing job/artifact containers. Input and artifact write scopes are separate. No storage keys were enabled and no raw-data write role was granted to compute.

The first submitted job, `epm-baseline-61eb0b4f6cf4`, failed **before training** because the Linux bootstrap had Windows CRLF line endings. LF enforcement and a staging guard corrected it; the frozen model/data configuration was unchanged. The second job completed. Both runs remain visible for audit.

An Azure ML command job is grouped under `epm-baseline-rul` in Studio. That platform job grouping is necessary for job execution; no MLflow tracking/autologging SDK, experiment-management pipeline or model-registry workflow was implemented.

## Outputs and cost

Eight outputs: native `model.json`, `metrics.json`, validation/test prediction Parquet files, `evaluation.md`, `feature-importance.json`, `run-metadata.json`, and hashed `artifact-manifest.json`. They remain under the job's unique existing-storage output prefix and locally under ignored `artifacts\baseline\epm-baseline-de82ea3141be`. At Phase 4 completion there was no registered/promoted model. The selected baseline was later registered as `epm-cmapss-rul-xgboost:1` during Phases 7 + 8; see [model selection and registry evidence](model-registry.md).

The public CPU estimate remains ~$0.096/hour while allocated; bootstrap, queue transitions and idle shutdown affect billable duration. Blob operations/retained artifacts and existing workspace dependencies may cost extra. The brief deleted Basic registry may incur prorated charges; no exact billing total is asserted.

## Final verification

- Integrated Python suite: **579 passed, 1 Windows symlink-privilege skip**, followed by **98 passing Azure-orchestration tests** including two added SDK provisioning-enum/Linux LF bootstrap regressions. All prior phase tests remained green.
- Ruff and dependency consistency checks passed. Foundation/script contracts: 46 passed. Training template compiles and contains exactly five container-scoped role assignments, with read-only curated input access.
- Checkov: 9 passed, 0 failed, 6 previously documented development exceptions. Checkov cannot parse the training template's existing-resource identity references; that file is explicitly excluded from the Bicep parser and validated through compiled ARM scope/type assertions. This limitation is not presented as full scanner coverage.
- Artifact manifest SHA-256: `c280467cc28509fb04e63ad6ba1c26c86b8e16a09636b8ec1ce9e952c2641e09`.
- Download uses direct authenticated Blob SDK retrieval because the SDK named-output convenience download returned no files. The corrected path validates the exact job output prefix, all eight files, ETags, lengths and manifest hashes before reporting success.
- Reloaded native model has 36 trees; its hash matches provenance, and all **142 validation + 707 test** predictions match the saved outputs within 1e-6. This was a serialization/integrity check, not another fit or model-selection pass.
- At Phase 4 completion, compute was **allocated=0, target=0, min=0, max=1** and the registry inventory was empty. Subsequent Phases 7 + 8 registered the selected model without changing compute; no endpoint was created.

Reproduction commands: [baseline-azure.md](baseline-azure.md). Stop before any next phase unless explicitly instructed.
