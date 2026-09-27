"""Cloud-independent, validation-selected native XGBoost RUL baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
# This is an explicit predictor allowlist, never numerical-column inference.
FEATURE_COLUMNS = (
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *(f"sensor_{number:02d}" for number in range(1, 22)),
    *(
        name
        for sensor in ("sensor_03", "sensor_04", "sensor_11")
        for name in (f"{sensor}_mean10", f"{sensor}_slope10")
    ),
    *(f"setting_{number}_mean10" for number in range(1, 4)),
    "history_count",
)
APPROVED_PARAMETERS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "device": "cpu",
    "nthread": 2,
    "seed": 42,
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "lambda": 5.0,
    "gamma": 0.0,
    "max_bin": 256,
}
_FIXED_CONFIG = {
    "schema_version": 1,
    "model_name": "xgboost-rul-baseline",
    "target": "uncapped_rul",
    "prediction_floor": 0.0,
    "explainability": "validation_tree_shap_and_training_gain",
}
_METADATA_COLUMNS = {"subset", "unit_id", "cycle", "split", "rul", "sample_weight"}


class TrainingError(ValueError):
    """An error whose message is safe to show without paths or credentials."""


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate configuration key")
        result[key] = value
    return result


def load_config(path: Path) -> dict:
    """Accept only the reviewed CPU recipe; reduced round budgets support smoke tests."""
    try:
        config = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
        expected = set(_FIXED_CONFIG) | {"parameters", "num_boost_round", "early_stopping_rounds"}
        if type(config) is not dict or set(config) != expected:
            raise ValueError
        for key, expected_value in _FIXED_CONFIG.items():
            value = config[key]
            if isinstance(value, bool) or value != expected_value:
                raise ValueError
        if type(config["schema_version"]) is not int:
            raise ValueError
        parameters = config["parameters"]
        if type(parameters) is not dict or set(parameters) != set(APPROVED_PARAMETERS):
            raise ValueError
        for key, expected_value in APPROVED_PARAMETERS.items():
            value = parameters[key]
            if isinstance(value, bool) or value != expected_value:
                raise ValueError
            if type(expected_value) is int and type(value) is not int:
                raise ValueError
        rounds, patience = config["num_boost_round"], config["early_stopping_rounds"]
        if type(rounds) is not int or not 1 <= rounds <= 600:
            raise ValueError
        if type(patience) is not int or not 1 <= patience <= min(rounds, 50):
            raise ValueError
        _canonical_json(config)
        return config
    except (OSError, ValueError, TypeError, KeyError):
        raise TrainingError(
            "Invalid baseline configuration; use the approved CPU recipe."
        ) from None


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    """Unweighted engine-equal metrics; error is prediction minus true uncapped RUL."""
    try:
        actual = np.asarray(actual, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise TrainingError("Metric inputs must be finite numeric arrays.") from None
    if (
        actual.ndim != 1
        or prediction.ndim != 1
        or actual.shape != prediction.shape
        or actual.size == 0
        or not np.isfinite(actual).all()
        or not np.isfinite(prediction).all()
        or (actual < 0).any()
    ):
        raise TrainingError("Metric inputs must be aligned, nonempty finite arrays with valid RUL.")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            error = prediction - actual
            squared_error = np.square(error)
            sse = float(squared_error.sum())
            total = float(np.square(actual - actual.mean()).sum())
            if actual.size < 2:
                r2, reason = None, "fewer_than_two_observations"
            elif total == 0:
                r2, reason = None, "constant_actual_rul"
            else:
                r2, reason = 1.0 - sse / total, None
            exponent = np.where(error < 0, -error / 13.0, error / 10.0)
            nasa = np.expm1(exponent)
            result = {
                "count": int(actual.size),
                "rmse": float(np.sqrt(squared_error.mean())),
                "mae": float(np.abs(error).mean()),
                "r2": r2,
                "r2_undefined_reason": reason,
                "bias": float(error.mean()),
                "nasa_score": float(nasa.sum()),
                "mean_nasa_score": float(nasa.mean()),
            }
        if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
            raise FloatingPointError
        return result
    except (FloatingPointError, OverflowError):
        raise TrainingError("Metric overflow; NASA exponential scores are not clipped.") from None


@dataclass
class Partition:
    table: pa.Table
    features: np.ndarray
    target: np.ndarray
    weights: np.ndarray
    engines: list[tuple[str, int]]


def _validate_manifest(manifest: dict) -> tuple[str, ...]:
    try:
        columns = manifest["feature_columns"]
        if (
            type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
            or manifest["dataset"] != "nasa-cmapss-ml-ready"
            or manifest["target"] != "rul"
            or not isinstance(columns, list)
            or len(columns) != 35
            or len(set(columns)) != 35
            or set(columns) != set(FEATURE_COLUMNS)
            or set(columns) & (_METADATA_COLUMNS - {"cycle"})
        ):
            raise ValueError
        return tuple(columns)
    except (KeyError, TypeError, ValueError):
        raise TrainingError(
            "Feature manifest does not match the approved 35-predictor contract."
        ) from None


def _numeric_column(table: pa.Table, name: str) -> np.ndarray:
    column = table[name]
    if column.null_count or not (
        pa.types.is_integer(column.type) or pa.types.is_floating(column.type)
    ):
        raise TrainingError("Feature data contains missing or nonnumeric values.")
    values = column.to_numpy().astype(np.float64)
    if not np.isfinite(values).all():
        raise TrainingError("Feature data contains nonfinite values.")
    return values


def _load_partition(root: Path, split: str, columns: tuple[str, ...], manifest: dict) -> Partition:
    table = pq.ParquetFile(root / f"{split}.parquet").read()
    if (
        set(table.column_names) != set(columns) | _METADATA_COLUMNS
        or len(table.column_names) != len(set(table.column_names))
        or table.num_rows == 0
        or table.num_rows != manifest["row_counts"][split]
    ):
        raise TrainingError("Feature partition schema or row counts are invalid.")
    subsets, splits = table["subset"].to_pylist(), table["split"].to_pylist()
    if set(subsets) != set(SUBSETS) or set(splits) != {split}:
        raise TrainingError("Every partition must contain all four subsets and its declared split.")
    units, cycles, target = (_numeric_column(table, name) for name in ("unit_id", "cycle", "rul"))
    for values, minimum in ((units, 1), (cycles, 1), (target, 0)):
        if (values < minimum).any() or not np.allclose(values, np.rint(values), rtol=0, atol=1e-7):
            raise TrainingError(
                "Engine IDs, cycles and RUL must be valid integer-valued quantities."
            )
    weights = _numeric_column(table, "sample_weight")
    if (weights <= 0).any():
        raise TrainingError("Sample weights must be positive.")
    features = np.column_stack([_numeric_column(table, name) for name in columns])
    if np.abs(features).max() > np.finfo(np.float32).max:
        raise TrainingError("Features exceed the supported numeric range.")
    engines = list(zip(subsets, (int(value) for value in units), strict=True))
    unique_engines = set(engines)
    if len(unique_engines) != manifest["engine_counts"][split]:
        raise TrainingError("Feature partition engine counts are invalid.")
    if len(set(zip(engines, cycles, strict=True))) != len(engines):
        raise TrainingError("Duplicate engine cycle snapshots are not permitted.")
    if split != "train":
        if len(unique_engines) != len(engines) or not np.allclose(weights, 1, rtol=0, atol=1e-7):
            raise TrainingError(
                "Validation and test must have one equally weighted snapshot per engine."
            )
    else:
        totals: dict[tuple[str, int], float] = {}
        for engine, weight in zip(engines, weights, strict=True):
            totals[engine] = totals.get(engine, 0.0) + float(weight)
        totals_array = np.array(list(totals.values()))
        if not np.allclose(totals_array, totals_array[0], rtol=1e-5, atol=1e-7):
            raise TrainingError("Training sample weights must balance engines.")
    return Partition(table, features, target, weights, engines)


def _validate_disjoint(train: Partition, validation: Partition) -> None:
    if set(train.engines) & set(validation.engines):
        raise TrainingError("Training and validation engines must be disjoint within each subset.")


def _matrix(
    partition: Partition, columns: tuple[str, ...], *, training: bool = False
) -> xgb.DMatrix:
    return xgb.DMatrix(
        partition.features,
        label=partition.target,
        weight=partition.weights if training else None,
        feature_names=list(columns),
        nthread=2,
    )


def predict_rul(model: xgb.Booster, matrix: xgb.DMatrix, best_iteration: int) -> np.ndarray:
    """Apply the deployed model's explicit tree range and reviewed zero floor."""
    prediction = np.asarray(
        model.predict(matrix, iteration_range=(0, best_iteration + 1)), dtype=np.float64
    )
    if not np.isfinite(prediction).all():
        raise TrainingError("The model produced nonfinite RUL predictions.")
    return np.maximum(prediction, 0.0)


def _partition_metrics(partition: Partition, prediction: np.ndarray) -> dict:
    subsets = np.array(partition.table["subset"].to_pylist())
    return {
        "overall": regression_metrics(partition.target, prediction),
        "per_subset": {
            subset: regression_metrics(
                partition.target[subsets == subset], prediction[subsets == subset]
            )
            for subset in SUBSETS
        },
    }


def _predictions(partition: Partition, prediction: np.ndarray, path: Path) -> None:
    table = partition.table.select(["subset", "unit_id", "cycle", "split"])
    table = table.append_column("actual", pa.array(partition.target, type=pa.float64()))
    table = table.append_column("prediction", pa.array(prediction, type=pa.float64()))
    table = table.append_column("error", pa.array(prediction - partition.target, type=pa.float64()))
    pq.write_table(table, path, compression="zstd")


def _explanations(model: xgb.Booster, matrix: xgb.DMatrix, columns: tuple[str, ...]) -> dict:
    contributions = model.predict(
        matrix,
        pred_contribs=True,
        approx_contribs=False,
        iteration_range=(0, model.num_boosted_rounds()),
    )
    if (
        contributions.shape != (matrix.num_row(), len(columns) + 1)
        or not np.isfinite(contributions).all()
    ):
        raise TrainingError("Validation TreeSHAP contributions are invalid.")
    mean_abs = np.abs(contributions[:, :-1].astype(np.float64)).mean(axis=0)
    gains = model.get_score(importance_type="gain")
    entries = [
        {
            "feature": column,
            "training_gain": float(gains.get(column, 0.0)),
            "validation_mean_abs_tree_shap": float(mean_abs[index]),
        }
        for index, column in enumerate(columns)
    ]
    return {
        "method": "exact native XGBoost TreeSHAP, validation snapshots only",
        "scope": "global association, not causality or calibrated failure probability",
        "prediction_space": "raw uncapped model output, before the zero prediction floor",
        "validation_count": matrix.num_row(),
        "mean_bias_contribution": float(contributions[:, -1].astype(np.float64).mean()),
        "features": entries,
        "training_gain_ranking": sorted(
            entries, key=lambda item: (-item["training_gain"], item["feature"])
        ),
        "validation_tree_shap_ranking": sorted(
            entries, key=lambda item: (-item["validation_mean_abs_tree_shap"], item["feature"])
        ),
    }


def _runtime() -> dict:
    packages = {"numpy": np.__version__, "pyarrow": pa.__version__, "xgboost": xgb.__version__}
    for package in ("xgboost-cpu", "scipy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json(value))


def _report(metrics: dict) -> str:
    lines = [
        "# Pooled C-MAPSS XGBoost RUL baseline",
        "",
        "Target: uncapped remaining useful life in cycles. Predictions are floored at zero; "
        "there is no upper cap.",
        "",
        "Training uses all rows of the training-engine partition with engine-balanced weights. "
        "Held-out validation engines contribute one deterministic 50–80% life prefix snapshot. "
        "Original test engines contribute their final supplied snapshot and supplied RUL labels.",
        "",
        "The CPU histogram model is trained once. Early stopping sees validation RMSE only "
        "(raw model outputs); no validation refit occurs. The best tree range is serialized "
        "before test prediction. Test data never selects parameters, features or iterations.",
        "",
        f"Best iteration (zero-based): {metrics['model']['best_iteration']}. "
        f"Saved trees: {metrics['model']['tree_count']}.",
        "",
        "| Partition | Subset | Engines | RMSE | MAE | R² | Bias | NASA sum | NASA mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        for subset, result in [
            ("overall", metrics[split]["overall"]),
            *metrics[split]["per_subset"].items(),
        ]:
            r2 = "undefined" if result["r2"] is None else f"{result['r2']:.6f}"
            lines.append(
                f"| {split} | {subset} | {result['count']} | {result['rmse']:.6f} | "
                f"{result['mae']:.6f} | {r2} | {result['bias']:.6f} | "
                f"{result['nasa_score']:.6f} | {result['mean_nasa_score']:.6f} |"
            )
    lines.extend(
        [
            "",
            "All reported metrics use one equally weighted snapshot per evaluated engine. "
            "Error and bias use prediction minus actual. NASA score sums expm1(-error/13) "
            "for negative errors and expm1(error/10) otherwise. Overpredicting life is penalized "
            "more severely because it implies late maintenance. "
            "Overflow fails explicitly, never clips.",
            "",
            "R² is null with a reason in metrics.json for constant targets "
            "or fewer than two engines. No classification precision, recall or AUC is "
            "reported: no failure threshold was selected. "
            "This regression output is not a calibrated failure probability.",
            "",
            "feature-importance.json includes training split gain and exact TreeSHAP "
            "mean absolute contributions from validation only, in the raw pre-floor "
            "prediction space. These global rankings are associations, not causal "
            "explanations. Test is used only for final evaluation.",
            "",
            "model.json is native XGBoost JSON with early-stopping tail trees removed. "
            "Reload it with xgboost.Booster.load_model, retain the recorded feature order, "
            "predict over the recorded best iteration + 1 trees and floor results at zero. "
            "No model registry or MLflow is used.",
            "",
        ]
    )
    return "\n".join(lines)


def _train(data_root: Path, config_path: Path, stage: Path) -> dict:
    config = load_config(config_path)
    # Lazy import keeps importing this module independent of feature-pipeline/cloud tooling.
    from epm_platform.features.pipeline import verify_features

    manifest = verify_features(data_root)
    columns = _validate_manifest(manifest)
    train = _load_partition(data_root, "train", columns, manifest)
    validation = _load_partition(data_root, "validation", columns, manifest)
    _validate_disjoint(train, validation)
    training_matrix = _matrix(train, columns, training=True)
    validation_matrix = _matrix(validation, columns)
    history: dict = {}
    booster = xgb.train(
        config["parameters"],
        training_matrix,
        num_boost_round=config["num_boost_round"],
        evals=[(validation_matrix, "validation")],
        early_stopping_rounds=config["early_stopping_rounds"],
        evals_result=history,
        verbose_eval=False,
    )
    best_iteration = int(booster.best_iteration)
    attempted_rounds = booster.num_boosted_rounds()
    frozen = booster[: best_iteration + 1]
    frozen.set_attr(
        best_iteration=str(best_iteration),
        best_score=str(booster.best_score),
        prediction_floor="0.0",
        target="uncapped_rul",
    )
    frozen.save_model(stage / "model.json")
    model_sha256 = _sha256(stage / "model.json")
    validation_prediction = predict_rul(frozen, validation_matrix, best_iteration)
    validation_metrics = _partition_metrics(validation, validation_prediction)
    importance = _explanations(frozen, validation_matrix, columns)

    # Integrity verification above may read test bytes; fitting and selection never see test arrays.
    test = _load_partition(data_root, "test", columns, manifest)
    test_prediction = predict_rul(frozen, _matrix(test, columns), best_iteration)
    test_metrics = _partition_metrics(test, test_prediction)
    if _sha256(stage / "model.json") != model_sha256:
        raise TrainingError("The frozen model changed during final evaluation.")
    provenance = {
        "ml_ready_manifest_sha256": _sha256(data_root / "manifest.json"),
        "ml_ready_files": manifest["files"],
        "feature_config_sha256": manifest["config_sha256"],
        "source": manifest["config"]["source"],
        "baseline_config_sha256": hashlib.sha256(_canonical_json(config)).hexdigest(),
        "baseline_config_file_sha256": _sha256(config_path),
    }
    model_info = {
        "configuration": config,
        "feature_columns": list(columns),
        "best_iteration": best_iteration,
        "tree_count": frozen.num_boosted_rounds(),
        "attempted_boosting_rounds": attempted_rounds,
        "best_validation_rmse_raw": float(booster.best_score),
        "prediction_iteration_range": [0, best_iteration + 1],
        "sha256": model_sha256,
    }
    metrics = {
        "schema_version": 1,
        "model": model_info,
        "row_counts": {
            split: manifest["row_counts"][split] for split in ("train", "validation", "test")
        },
        "engine_counts": {
            split: manifest["engine_counts"][split] for split in ("train", "validation", "test")
        },
        "seed": 42,
        "provenance": provenance,
        "runtime": _runtime(),
        "validation": validation_metrics,
        "test": test_metrics,
    }
    _write_json(stage / "metrics.json", metrics)
    _write_json(stage / "feature-importance.json", importance)
    _write_json(
        stage / "run-metadata.json",
        {
            "schema_version": 1,
            "model": model_info,
            "seed": 42,
            "provenance": provenance,
            "runtime": metrics["runtime"],
            "row_counts": metrics["row_counts"],
            "engine_counts": metrics["engine_counts"],
            "validation_rmse_history": history["validation"]["rmse"],
            "selection": "validation_only_early_stopping_no_refit",
            "test_evaluation_count": 1,
            "model_frozen_before_test": True,
        },
    )
    _predictions(validation, validation_prediction, stage / "predictions_validation.parquet")
    _predictions(test, test_prediction, stage / "predictions_test.parquet")
    (stage / "evaluation.md").write_text(_report(metrics), encoding="utf-8")
    _write_json(
        stage / "artifact-manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "xgboost-rul-baseline",
            "files": [
                {"path": path.name, "sha256": _sha256(path), "size_bytes": path.stat().st_size}
                for path in sorted(stage.iterdir())
            ],
            "provenance": provenance,
            "completion_marker": "artifact-manifest.json",
        },
    )
    return metrics


def train_baseline(data_root: Path, config_path: Path, output_dir: Path) -> dict:
    """Train once into a new or empty output directory, publishing the manifest last."""
    stage = None
    published: list[Path] = []
    try:
        data_root, config_path, output_dir = Path(data_root), Path(config_path), Path(output_dir)
        # Refuse aliasing the input or any nonempty output, including existing completed runs.
        if output_dir.is_symlink() or (
            output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir()))
        ):
            raise TrainingError(
                "Output directory must be new or empty; existing artifacts are never overwritten."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        # Staging under an existing mount avoids cross-filesystem renames on managed compute.
        stage = output_dir / f".staging-{uuid.uuid4().hex}"
        stage.mkdir()
        metrics = _train(data_root, config_path, stage)
        if set(output_dir.iterdir()) != {stage}:
            raise TrainingError(
                "Output directory changed during training; refusing to overwrite files."
            )
        files = sorted(
            stage.iterdir(), key=lambda path: (path.name == "artifact-manifest.json", path.name)
        )
        for path in files:
            destination = output_dir / path.name
            # Exclusive creation prevents silently replacing a concurrent writer's artifact.
            with path.open("rb") as source, destination.open("xb") as target:
                published.append(destination)
                shutil.copyfileobj(source, target)
        return metrics
    except Exception as error:
        for path in reversed(published):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        if isinstance(error, TrainingError):
            raise
        raise TrainingError(
            "Baseline training failed; verify the input artifacts, approved configuration, "
            "runtime dependencies and output permissions."
        ) from None
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(
            2, "Error: invalid command-line arguments; use --help for the supported options.\n"
        )


def main(argv: list[str] | None = None) -> int:
    parser = _SafeArgumentParser(description="Train the approved pooled C-MAPSS CPU baseline.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        train_baseline(arguments.data, arguments.config, arguments.output)
    except TrainingError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print("Baseline completed: native model, evaluation and verified artifact inventory written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
