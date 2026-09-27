# PyTorch RUL experiment: frozen features, CPU MLP

## Scope and immutable inputs

This phase adds a tabular regressor; it does not reconstruct sequences, change features,
change splits, cap RUL, refit on validation, or modify the Phase 4 XGBoost baseline.
The live experiment consumes existing asset
`epm-cmapss-ml-ready:d-f4ueae6u7g4c5islgehonqveey`, manifest SHA-256
`2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd`.
Its local location is
`data\ml-ready\cmapss\sha256-2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd`.
Training has 128,967 rows from 567 engines; validation has 142 censored endpoints;
test has 707 original final-observation endpoints. Core training accepts any bundle that
passes the unchanged `features.pipeline.verify_features` contract, allowing small synthetic
tests; the live wrapper pins the exact asset and requires a baseline reference.
Verification checks content hashes, schema, split semantics and provenance before fitting.
This verification may inspect test bytes; test arrays are not passed to the learner or
selection loop and are loaded for prediction only after the selected checkpoint is saved.

The explicit ordered 35 predictors come directly from `features.pipeline.FEATURE_COLUMNS`:
`cycle`, `setting_1..3`, `sensor_01..21`, mean10/slope10 for sensors 03/04/11,
`setting_1_mean10..setting_3_mean10`, and `history_count`.
`subset`, `unit_id`, `split`, `rul`, and `sample_weight` are never predictors.
Cycle is intentionally a predictor. No numerical-column inference or feature dropping occurs.

## Approved model and training

`config\pytorch.json` is a closed recipe: MLP 35 → 64 → 32 → 1, ReLU and dropout 0.1
after each hidden layer, linear output, AdamW learning rate 0.001, weight decay 0.0001,
batch size 512, maximum 100 epochs, patience 12, gradient norm clip 5.0.
Only maximum epochs and patience may be reduced for synthetic tests. There are no sweeps.
Epoch numbers are one-based. Python, NumPy and Torch seeds are 42; deterministic Torch
algorithms, CPU with two threads, a seeded shuffle generator, and zero loader workers are
used. Reproducibility is tested within a fixed runtime; cross-version/hardware bitwise
identity is not promised. The core intentionally sets process-wide training seeds and
Torch thread/determinism settings; run separate jobs for independent concurrent training.

Both feature and target preprocessing fit **only training rows**, using float64
engine-weighted population mean and variance:

- `mean = sum(weight * value) / sum(weight)`.
- `variance = sum(weight * (value - mean)^2) / sum(weight)`.
- `scale = sqrt(variance)`; variance ≤ `1e-12` gets scale 1 and is recorded.
- All 35 columns remain, including constant/nearly constant columns.
- Validation/test only transform with saved training statistics; no fitting or adaptation.
- Standardized features and targets become float32 tensors.

Training sample weights are divided by their **training-global mean**, giving mean-one
weights and equal total influence per engine. Each shuffled row appears once per epoch;
there is no weighted sampler. Each batch loss is
`mean(normalized_weight * (prediction - standardized_target)^2)`.
Weights are applied exactly once, without per-batch sum-of-weights renormalization.
`train_loss` is the row-count-weighted mean of these online dropout-enabled batch losses.
`train_rmse` is the engine-weighted, dropout-disabled epoch-end RMSE in original cycles.
`val_rmse` is the unweighted endpoint RMSE after inverse transformation and the zero floor.

Validation RMSE in actual RUL cycles selects the checkpoint, with strict improvement and
patience counting consecutive non-improving epochs. Ties keep the earlier model.
Each best state is detached, cloned independently onto CPU and restored at loop completion.
The selected `state_dict` is saved before test loading/prediction. No training resumes,
no validation refit occurs, and test is evaluated exactly once. Loss, gradient, parameter
and prediction nonfiniteness fail explicitly. Inference disables dropout and converts
network outputs to float64 before target inverse scaling and `max(prediction, 0)`.
No upper target/prediction cap is applied.

## Evaluation and baseline comparison

Evaluation reuses `baseline.training.regression_metrics` and its partition/prediction
helpers without changes: RMSE, MAE, R² (null with an explicit reason when undefined), signed
bias, NASA sum and NASA mean. Every evaluated engine contributes one equally weighted
endpoint overall and within FD001–FD004, for both validation and test.
Error is prediction minus true uncapped RUL. NASA uses `expm1(-error / 13)` for negative
errors and `expm1(error / 10)` otherwise: overpredicting remaining life is more costly.
Overflow fails, never clips. No SHAP/global explanation was requested for this MLP.
These retrospective validation cuts and offline test metrics do not establish field
performance or calibrated failure probability.

The optional core `baseline_reference_path` contains schema version 1, `job_name`,
`ml_ready_manifest_sha256`, `target: uncapped_rul`, the exact embedded Phase 4 `metrics`
object, and its original file's `metrics_sha256`. The live reference identifies
`epm-baseline-de82ea3141be`. Before fitting, the core checks matching top-level and embedded
feature manifest hashes, target, prediction floor, feature order, row/engine counts,
all evaluated subset counts, and finite metric values. Test counts come from the already
verified feature summary, not a test prediction. Mismatches fail rather than produce an
unfair comparison. With no reference, `comparison.json` records `status: not_provided`.

Comparison records original baseline values, MLP values and signed `MLP - baseline`
deltas for every metric, split and subset. Lower RMSE/MAE/NASA scores are better; higher
R² is better; signed bias is judged by **absolute distance from zero**, not by becoming
more negative. Undefined comparisons remain null. Baseline files are never modified.
The original metrics-file hash is informational because JSON reserialization cannot
recreate its original bytes. Both the complete reference-file SHA-256 and canonical
embedded-metrics SHA-256 are recorded; this is integrity provenance, not a signature or
claim that unauthenticated reference contents are trusted.

## Public APIs and tracking boundary

```python
from pathlib import Path
from epm_platform.deep_learning.training import train_pytorch, load_model, predict_saved

metrics = train_pytorch(
    data_root=Path("verified-feature-bundle"),
    config_path=Path(r"config\pytorch.json"),
    output_dir=Path("new-output-directory"),
    tracker=None,
    baseline_reference_path=Path(r"config\baseline-reference.json"),
)
loaded = load_model(Path("new-output-directory"))
# features: PyArrow table containing all named predictors, or N x 35 matrix in saved order
predictions = loaded.predict(features)
# Equivalent checked reload:
predictions = predict_saved(features, Path("new-output-directory"))
```

The core imports no Azure or MLflow client and uses no autologging. The optional tracker
protocol is explicit:

1. `log_parameters(dict)` once after input/reference verification, before training.
2. `log_epoch(metrics: dict, step: int)` once per epoch with `train_loss`, `train_rmse`,
   `val_rmse`, and one-based `step`.
3. `log_final(dict)` once with flattened numeric keys such as `test.overall.rmse` and
   `validation.FD001.nasa_score`; null/string R² explanations stay in metrics JSON.
4. The **outer wrapper**, not the core, calls `log_artifacts(output_dir: Path)` exactly
   once after `train_pytorch` returns. Publication and staging cleanup are complete then,
   and all eleven final files are present. The core never calls this method.

A core logging exception raises `TrainingError`; no successful tracking is claimed.
Parameter, epoch or final-metric failures discard staging. The wrapper must propagate any
artifact-upload failure without claiming successful tracking; the completed local bundle
remains available. There is no tracking receipt inside the output folder. Adapter
authentication, MLflow/Azure runtime configuration and live job submission belong to the
outer wrapper.

## Eleven-file artifact contract and safe reload

Output must be a new or empty directory, including a pre-created managed output mount.
Existing files are never overwritten. Private staging is inside the output filesystem;
files are copied with exclusive creation, and `artifact-manifest.json` is published last
as the completion marker. Caught publication failures remove only this run's published
files; a hard process crash can leave incomplete files without a valid completion marker.
Readers must require and verify the manifest, not infer success from directory existence.

Exactly these files are produced:

1. `model.pt`: `torch.save(model.state_dict())`, never a pickled model object.
2. `model-spec.json`: architecture, exact feature order, tensor dtype, state keys/shapes.
3. `preprocessing.json`: feature/target means/scales, constant-column list, fit partition,
   variance threshold, statistic/tensor dtypes, training counts and weight semantics.
4. `metrics.json`: both evaluation partitions, overall/per-subset metrics, model/provenance.
5. `predictions_validation.parquet`: identity fields, actual, prediction and signed error.
6. `predictions_test.parquet`: the same schema for the single final test evaluation.
7. `run-metadata.json`: selected epoch, epochs attempted, config, seed, runtime versions,
   feature/source/config hashes, train/evaluation counts and test-freezing assertions.
8. `training-history.json`: ordered epoch records and selected-best flags.
9. `evaluation.md`: readable metric table, evaluation conventions and limitations.
10. `comparison.json`: exact-dataset baseline deltas or explicit no-reference status.
11. `artifact-manifest.json`: schema version 1, artifact type, model SHA-256, provenance,
    completion marker and ten non-self file entries with `path`, `size_bytes`, `sha256`.

`load_model(root)` validates the exact flat inventory and hashes, rejects linked files,
validates the fixed architecture/preprocessing, then calls
`torch.load(..., weights_only=True, map_location="cpu")`. It checks exact state keys,
shapes, float32 dtype and finiteness before strict loading into the known MLP and eval
mode. No `safetensors` dependency or unsafe full-model deserialization is used. Hashes
protect against accidental/tampered bytes relative to the manifest; they are not digital
signatures. Only obtain model bundles from trusted provenance.

## Local validation only

Use the isolated Python 3.12 `.venv-torch`, not the unchanged root `.venv`.
The approved runtime includes Torch 2.14.0+cpu, NumPy 2.5.3, PyArrow 25.0.1 and
xgboost-cpu 3.4.1 (the latter only to reuse existing baseline metric/partition helpers).
No local real-dataset training is part of implementation validation.

```powershell
$env:PYTHONPATH = 'src'
& '.\.venv-torch\Scripts\python.exe' -m pytest tests\deep_learning\test_training.py -q
& '.\.venv-torch\Scripts\python.exe' -m epm_platform.deep_learning.training --help
```

The CLI accepts `--data`, `--config`, `--output`, and optional local
`--baseline-reference`. Cloud tracking is a separate wrapper. Tests build small genuine
feature bundles and patch only the synthetic source-provenance seam; real feature
verification remains active. The module skips when Torch is absent so previous-phase
root-environment tests do not acquire a Torch dependency. Tests cover architecture,
weighted training-only statistics, held-out/future perturbations, constant columns,
determinism, best-state restoration, frozen-before-test ordering, safe reload, metric
identity, nonnegative predictions, schema/config rejection, tracking calls/failures,
comparison fairness and no-overwrite behavior without actual cloud runs.
