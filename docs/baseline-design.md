# Pooled C-MAPSS RUL baseline

## Scope and reviewed recipe

`epm_platform.baseline.training` is cloud-independent Python. It has no Azure imports,
registry operations, MLflow, scikit-learn, Torch, sweeps, or endpoints. Native
`xgboost.train` and `DMatrix` perform the fit; NumPy implements the metrics and
PyArrow reads and writes Parquet. The expected runtime is Python 3.12,
`xgboost-cpu==3.4.1`, NumPy, SciPy and PyArrow 25.0.1.

The approved recipe is `config\baseline.json`; the dataset recipe is
`config\features.json`. All four subsets (FD001–FD004) are pooled. The regression
target is **uncapped remaining useful life in cycles**. Predictions are floored at
zero, with no upper cap. The fixed allowlist has 35 predictors: current cycle,
three settings, 21 sensors, trailing-10 means and OLS slopes for sensors 03/04/11,
three trailing setting means and history count. The verified manifest supplies
column order. Unit ID, subset, split, RUL and sample weights are never predictors.
Negative slopes/settings are legitimate; every predictor must be finite, and RUL
must be nonnegative and integer-valued.

## Isolation and model selection

- Training uses all observed rows for the deterministic 80% training-engine split.
  Stored sample weights give each training engine equal total influence.
- The disjoint held-out 20% contributes one causally computed snapshot at a
  deterministic 50–80% life prefix. Validation engines are equally weighted.
- The original 707 test engines contribute their last observed snapshots with the
  supplied test RUL. Test IDs are independent of the original training IDs; engine
  identity within either source partition is `(subset, unit_id)`.
- The model is trained **once**, with validation-only RMSE early stopping: at most
  600 rounds, patience 50, CPU histogram algorithm, two threads and seed 42. All
  remaining parameters are fixed to the reviewed config. A smaller round/patience
  budget is supported for synthetic smoke tests, not parameter search.
- Early stopping uses raw model-output validation RMSE. Reported metrics apply
  the approved zero floor. No refit on validation occurs. No test features or
  targets are passed to fitting, selection or explanations. Artifact integrity
  verification necessarily checks test bytes before training; that is not model
  selection. Test arrays are loaded for prediction only after the model is frozen.

The early-stopped booster can retain patience-tail trees. Training explicitly
predicts over `[0, best_iteration + 1)` and saves a **sliced booster** containing
exactly that many trees, so native reload and the in-memory predictor agree. The
best iteration is zero-based. Native JSON preserves the feature names and
attributes recording the zero floor and selected iteration; consumers must still
apply the zero floor themselves. XGBoost itself does not apply that postprocessing.

## API and command line

```python
from pathlib import Path
from epm_platform.baseline.training import train_baseline

metrics = train_baseline(
    data_root=Path("data") / "mounted-ml-ready-input",
    config_path=Path("config") / "baseline.json",
    output_dir=Path("artifacts") / "baseline-run",
)
```

```powershell
python -m epm_platform.baseline.training --data data\mounted-ml-ready-input --config config\baseline.json --output artifacts\baseline-run
```

The input directory name is irrelevant. `features.pipeline.verify_features`
validates the ML-ready manifest, fingerprints, schemas, causal metadata and splits
before fitting. Additional training guards validate the exact predictor allowlist,
finite numeric data, target/ID domains, engine balancing, engine-disjoint
training/validation partitions and one snapshot per validation/test engine.
Unrecognized config keys, duplicate JSON keys, GPU objectives/devices, altered
parameters, invalid budgets and output-name injection are rejected. CLI errors are
sanitized and nonzero; they do not echo filesystem paths or provider exceptions.

Output must be new or **empty**; an empty mounted output directory is supported.
All work is staged under that directory, then complete files are published with
exclusive creation and `artifact-manifest.json` last. Publication is not a single
filesystem transaction on mounted outputs: consumers must require the manifest as
the completion marker. Existing artifacts are never overwritten. Recoverable
failures remove staged/new artifacts; an interrupted process may leave an
incomplete staging directory, which must be handled explicitly before retrying.

## Evaluation

Validation and test each include overall and per-subset engine-equal metrics:

- RMSE and MAE in RUL cycles.
- R², or JSON `null` with an explicit reason for constant actual RUL or fewer than
  two engines.
- Bias, with `error = prediction - actual`.
- NASA score **sum**, and mean score per engine:
  `expm1(-error / 13)` for negative error, otherwise `expm1(error / 10)`.

Overpredicting life is more severely penalized because it implies late
maintenance/failure risk. Exponential overflow fails explicitly rather than
silently clipping or writing nonfinite JSON. No classification precision, recall
or AUC is reported without an approved failure threshold, and no calibrated
failure probability is claimed. No additional model is trained as a comparator.

`feature-importance.json` reports training-split gain and **exact native TreeSHAP**
mean absolute feature contributions from validation snapshots only. It uses
`pred_contribs=True, approx_contribs=False`, with no `shap` dependency. Contributions
explain raw pre-floor outputs; the bias contribution is reported separately.
Rankings are global associations, not causal effects or calibrated failure risk.

## Persistent outputs

| File | Contents |
|---|---|
| `model.json` | Native XGBoost JSON, selected trees only; no pickle |
| `metrics.json` | Overall/per-subset metrics, counts, reviewed config, selected iteration, provenance and runtime |
| `predictions_validation.parquet` | Subset, engine ID, cycle, split, actual, prediction and signed error |
| `predictions_test.parquet` | Same schema, one row per original test engine |
| `evaluation.md` | Human-readable metrics, assumptions and limitations |
| `run-metadata.json` | Selection history, tree count, seed, package/platform versions and source/input/config fingerprints |
| `feature-importance.json` | Training gain and validation-only exact TreeSHAP rankings |
| `artifact-manifest.json` | Relative filenames, byte sizes and SHA-256 for every preceding output; published last |

The artifact manifest cannot hash itself. It hashes all other outputs and serves
as the completion marker. Source asset identity and manifest hashes, ML-ready
manifest/files, feature-config hash and baseline-config hashes link the run to its
inputs. Metadata records platform type/architecture and package versions, not the
hostname, absolute input paths, credentials or environment variables. Test
prediction is performed once after serialization and no further fitting occurs.

## Focused validation

```powershell
.venv\Scripts\python.exe -m pytest tests\baseline\test_training.py -q
.venv\Scripts\python.exe -m ruff check src\epm_platform\baseline\training.py tests\baseline\test_training.py
```

Tests use small synthetic 35-feature CPU datasets, not full local C-MAPSS training.
They cover known metrics and NASA asymmetry, undefined R² and overflow, strict
config validation, grouping/leakage checks, test-data independence, deterministic
training, artifact hashes and native-model reload/selected-tree consistency.
