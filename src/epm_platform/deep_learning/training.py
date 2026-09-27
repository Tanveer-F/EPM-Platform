"""Cloud-independent, validation-selected CPU PyTorch regression on frozen features."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow as pa
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from epm_platform.baseline import training as baseline
from epm_platform.features.pipeline import FEATURE_COLUMNS, verify_features

TrainingError = baseline.TrainingError
regression_metrics = baseline.regression_metrics
SUBSETS = baseline.SUBSETS
_FILES = frozenset(
    {
        "model.pt",
        "model-spec.json",
        "preprocessing.json",
        "metrics.json",
        "predictions_validation.parquet",
        "predictions_test.parquet",
        "run-metadata.json",
        "training-history.json",
        "evaluation.md",
        "comparison.json",
        "artifact-manifest.json",
    }
)
_FIXED_CONFIG = {
    "schema_version": 1,
    "model_name": "pytorch-mlp-rul",
    "target": "uncapped_rul",
    "input_features": 35,
    "hidden_sizes": [64, 32],
    "activation": "relu",
    "dropout": 0.1,
    "seed": 42,
    "device": "cpu",
    "threads": 2,
    "batch_size": 512,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "loss": "engine_weighted_mse",
    "preprocessing": "train_only_feature_and_target_standardization",
    "prediction_floor": 0.0,
    "selection": "validation_rmse",
    "gradient_clip_norm": 5.0,
}
_VARIANCE_EPSILON = 1e-12
_METRIC_NAMES = ("rmse", "mae", "r2", "bias", "nasa_score", "mean_nasa_score")


class Tracker(Protocol):
    """Optional explicit adapter; failures propagate, and no autologging is used."""

    def log_parameters(self, parameters: dict) -> None: ...
    def log_epoch(self, metrics: dict, step: int) -> None: ...
    def log_final(self, metrics: dict) -> None: ...
    def log_artifacts(self, output_dir: Path) -> None: ...


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=baseline._unique_keys)


def load_config(path: Path) -> dict:
    """Closed approved recipe; only epoch and patience budgets may be reduced."""
    try:
        config = _read_json(Path(path))
        if type(config) is not dict or set(config) != set(_FIXED_CONFIG) | {
            "max_epochs",
            "patience",
        }:
            raise ValueError
        for key, expected in _FIXED_CONFIG.items():
            value = config[key]
            if isinstance(value, bool) or value != expected:
                raise ValueError
            if type(expected) is int and type(value) is not int:
                raise ValueError
        if any(type(size) is not int for size in config["hidden_sizes"]):
            raise ValueError
        epochs, patience = config["max_epochs"], config["patience"]
        if type(epochs) is not int or not 1 <= epochs <= 100:
            raise ValueError
        if type(patience) is not int or not 1 <= patience <= min(epochs, 12):
            raise ValueError
        baseline._canonical_json(config)
        return config
    except (OSError, ValueError, TypeError, KeyError):
        raise TrainingError("Invalid PyTorch configuration; use the approved CPU recipe.") from None


class MLP(nn.Module):
    """35 predictors -> 64 -> 32 -> one standardized RUL prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(35, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _model_spec(model: MLP) -> dict:
    return {
        "schema_version": 1,
        "model_name": "pytorch-mlp-rul",
        "target": "uncapped_rul",
        "feature_columns": list(FEATURE_COLUMNS),
        "architecture": {
            "input_features": 35,
            "hidden_sizes": [64, 32],
            "output_features": 1,
            "activation": "relu",
            "dropout": 0.1,
            "dropout_after_each_hidden_layer": True,
            "output_activation": "identity",
        },
        "state_dict_keys": list(model.state_dict()),
        "state_dict_shapes": {key: list(value.shape) for key, value in model.state_dict().items()},
        "dtype": "float32",
        "serialization": "torch.save(state_dict); "
        "torch.load(weights_only=True, map_location='cpu')",
        "prediction_floor": 0.0,
    }


def _numeric(values: object, dimensions: int) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != dimensions or array.size == 0 or not np.isfinite(array).all():
            raise ValueError
        return array
    except (TypeError, ValueError, OverflowError):
        raise TrainingError("Preprocessing requires nonempty finite numeric arrays.") from None


def fit_preprocessing(train: baseline.Partition) -> dict:
    """Fit weighted population moments exclusively on the supplied training partition."""
    if set(train.table["split"].to_pylist()) != {"train"}:
        raise TrainingError("Preprocessing may only fit the training partition.")
    features, target, weights = (
        _numeric(train.features, 2),
        _numeric(train.target, 1),
        _numeric(train.weights, 1),
    )
    if features.shape != (len(target), 35) or weights.shape != target.shape:
        raise TrainingError(
            "Training preprocessing arrays are not aligned with the feature schema."
        )
    if (weights <= 0).any() or (target < 0).any():
        raise TrainingError("Training weights must be positive and uncapped RUL nonnegative.")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mean = np.average(features, axis=0, weights=weights)
        variance = np.average(np.square(features - mean), axis=0, weights=weights)
        target_mean = float(np.average(target, weights=weights))
        target_variance = float(np.average(np.square(target - target_mean), weights=weights))
        zero = variance <= _VARIANCE_EPSILON
        scale = np.where(zero, 1.0, np.sqrt(variance))
        target_scale = 1.0 if target_variance <= _VARIANCE_EPSILON else math.sqrt(target_variance)
    result = {
        "schema_version": 1,
        "fit_partition": "train",
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "target_mean": target_mean,
        "target_scale": target_scale,
        "zero_variance_columns": [
            name for name, flag in zip(FEATURE_COLUMNS, zero, strict=True) if flag
        ],
        "target_zero_variance": target_variance <= _VARIANCE_EPSILON,
        "variance_epsilon": _VARIANCE_EPSILON,
        "statistics_dtype": "float64",
        "tensor_dtype": "float32",
        "sample_weight_semantics": "engine-balanced weighted population mean and variance; "
        "training loss uses sample_weight / training_sample_weight_mean exactly once",
        "training_sample_weight_mean": float(weights.mean()),
        "training_rows": int(len(target)),
        "training_engines": len(set(train.engines)),
    }
    baseline._canonical_json(result)
    return result


def _features_tensor(features: object, preprocessing: dict) -> torch.Tensor:
    if isinstance(features, pa.Table):
        if not set(FEATURE_COLUMNS).issubset(features.column_names):
            raise TrainingError("Prediction data is missing required feature columns.")
        features = np.column_stack(
            [baseline._numeric_column(features, name) for name in FEATURE_COLUMNS]
        )
    array = _numeric(features, 2)
    if array.shape[1] != 35:
        raise TrainingError("Prediction data must have the recorded 35 feature columns in order.")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        array = (array - preprocessing["feature_mean"]) / preprocessing["feature_scale"]
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise TrainingError("Standardized features exceed the finite float32 range.")
    return torch.from_numpy(array)


def weighted_mse(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
    """Mean of weighted errors, not a second weighted sampler or batch weight renormalization."""
    if prediction.ndim != 1 or prediction.shape != target.shape or target.shape != weights.shape:
        raise TrainingError("Weighted MSE requires aligned one-dimensional tensors.")
    return torch.mean(weights * torch.square(prediction - target))


def _predict(model: MLP, features: torch.Tensor, preprocessing: dict) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        standardized = torch.cat([model(batch) for batch in features.split(512)]).numpy()
    with np.errstate(over="raise", invalid="raise"):
        prediction = standardized.astype(np.float64) * preprocessing["target_scale"]
        prediction += preprocessing["target_mean"]
    if not np.isfinite(prediction).all():
        raise TrainingError("The model produced nonfinite RUL predictions.")
    return np.maximum(prediction, 0.0)


def _validation_rmse(model: MLP, features: torch.Tensor, preprocessing: dict, target) -> float:
    return regression_metrics(target, _predict(model, features, preprocessing))["rmse"]


def _track(tracker: Tracker | None, method: str, *args, **kwargs) -> None:
    if tracker is not None:
        try:
            getattr(tracker, method)(*args, **kwargs)
        except Exception as error:
            raise TrainingError(
                f"Tracking failed in {method}; tracking is not successful."
            ) from error


def _fit(model, train, validation, preprocessing, config, tracker):
    x = _features_tensor(train.features, preprocessing)
    validation_x = _features_tensor(validation.features, preprocessing)
    y = torch.from_numpy(
        ((train.target - preprocessing["target_mean"]) / preprocessing["target_scale"]).astype(
            np.float32
        )
    )
    weights = torch.from_numpy((train.weights / train.weights.mean()).astype(np.float32))
    if not torch.isfinite(y).all() or not torch.isfinite(weights).all():
        raise TrainingError("Training targets or weights exceed the finite float32 range.")
    loader = DataLoader(
        TensorDataset(x, y, weights),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    best_state, best_epoch, best_score, stale, history = None, 0, math.inf, 0, []
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        loss_sum = 0.0
        for batch_x, batch_y, batch_weights in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_mse(model(batch_x), batch_y, batch_weights)
            if not torch.isfinite(loss):
                raise TrainingError("Nonfinite training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise TrainingError("Nonfinite model parameters.")
            loss_sum += loss.item() * len(batch_y)
        train_prediction = _predict(model, x, preprocessing)
        train_rmse = float(
            np.sqrt(np.average(np.square(train_prediction - train.target), weights=train.weights))
        )
        score = _validation_rmse(model, validation_x, preprocessing, validation.target)
        epoch_metrics = {
            "train_loss": loss_sum / len(y),
            "train_rmse": train_rmse,
            "val_rmse": score,
        }
        if not all(math.isfinite(value) for value in epoch_metrics.values()):
            raise TrainingError("Nonfinite epoch metrics.")
        improved = score < best_score
        if improved:
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_epoch, best_score, stale = epoch, score, 0
        else:
            stale += 1
        history.append({"epoch": epoch, **epoch_metrics, "selected_best": improved})
        _track(tracker, "log_epoch", epoch_metrics, step=epoch)
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return best_epoch, best_score, history


def _reference(path: Path | None, data_root: Path, manifest: dict) -> dict | None:
    if path is None:
        return None
    try:
        reference = _read_json(Path(path))
        if (
            set(reference)
            != {
                "schema_version",
                "job_name",
                "ml_ready_manifest_sha256",
                "target",
                "metrics",
                "metrics_sha256",
            }
            or type(reference["schema_version"]) is not int
            or reference["schema_version"] != 1
        ):
            raise ValueError
        digest = baseline._sha256(data_root / "manifest.json")
        metrics = reference["metrics"]
        if (
            reference["target"] != "uncapped_rul"
            or not isinstance(reference["job_name"], str)
            or not reference["job_name"]
            or not isinstance(reference["metrics_sha256"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", reference["metrics_sha256"])
            or reference["ml_ready_manifest_sha256"] != digest
            or type(metrics["schema_version"]) is not int
            or metrics["schema_version"] != 1
            or metrics["provenance"]["ml_ready_manifest_sha256"] != digest
            or metrics["provenance"]["feature_config_sha256"] != manifest["config_sha256"]
            or metrics["provenance"]["source"] != manifest["config"]["source"]
            or metrics["provenance"]["ml_ready_files"] != manifest["files"]
            or metrics["model"]["configuration"]["model_name"] != "xgboost-rul-baseline"
            or metrics["model"]["configuration"]["target"] != "uncapped_rul"
            or metrics["model"]["configuration"]["prediction_floor"] != 0.0
            or metrics["model"]["feature_columns"] != list(FEATURE_COLUMNS)
            or metrics["row_counts"] != manifest["row_counts"]
            or metrics["engine_counts"] != manifest["engine_counts"]
        ):
            raise ValueError
        summary = _read_json(data_root / "feature-summary.json")
        for split in ("validation", "test"):
            if set(metrics[split]["per_subset"]) != set(SUBSETS):
                raise ValueError
            for subset in ("overall", *SUBSETS):
                record = (
                    metrics[split]["overall"]
                    if subset == "overall"
                    else (metrics[split]["per_subset"][subset])
                )
                counts = manifest if subset == "overall" else summary["by_subset"][subset]
                if (
                    type(record["count"]) is not int
                    or record["count"] != counts["engine_counts"][split]
                ):
                    raise ValueError
                for name in _METRIC_NAMES:
                    value = record[name]
                    if (
                        name == "r2"
                        and value is None
                        and record["r2_undefined_reason"]
                        in {"constant_actual_rul", "fewer_than_two_observations"}
                    ):
                        continue
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError
                    if name not in {"bias", "r2"} and value < 0:
                        raise ValueError
        baseline._canonical_json(reference)
        return {
            **reference,
            "reference_file_sha256": baseline._sha256(Path(path)),
            "embedded_metrics_canonical_sha256": hashlib.sha256(
                baseline._canonical_json(metrics)
            ).hexdigest(),
        }
    except (OSError, ValueError, TypeError, KeyError):
        raise TrainingError(
            "Baseline reference does not match the exact dataset, target and counts."
        ) from None


def _comparison(metrics: dict, reference: dict | None) -> dict:
    if reference is None:
        return {"schema_version": 1, "status": "not_provided"}
    result = {
        "schema_version": 1,
        "status": "compared",
        "baseline_job_name": reference["job_name"],
        "ml_ready_manifest_sha256": reference["ml_ready_manifest_sha256"],
        "target": "uncapped_rul",
        "baseline_metrics_sha256": reference["metrics_sha256"],
        "baseline_metrics_hash_policy": "original file hash is informational; embedded metrics "
        "are recorded with a canonical content hash, not claimed to authenticate original bytes",
        "baseline_reference_file_sha256": reference["reference_file_sha256"],
        "embedded_metrics_canonical_sha256": reference["embedded_metrics_canonical_sha256"],
    }
    for split in ("validation", "test"):
        records = {}
        for subset in ("overall", *SUBSETS):
            current = (
                metrics[split]["overall"]
                if subset == "overall"
                else metrics[split]["per_subset"][subset]
            )
            previous = (
                reference["metrics"][split]["overall"]
                if subset == "overall"
                else (reference["metrics"][split]["per_subset"][subset])
            )
            compared = {"count": current["count"]}
            for name in _METRIC_NAMES:
                new, old = current[name], previous[name]
                direction = (
                    "higher" if name == "r2" else "lower_absolute" if name == "bias" else "lower"
                )
                delta = None if new is None or old is None else new - old
                improved = (
                    None
                    if delta is None
                    else (
                        new > old
                        if name == "r2"
                        else abs(new) < abs(old)
                        if name == "bias"
                        else new < old
                    )
                )
                compared[name] = {
                    "baseline": old,
                    "pytorch": new,
                    "delta_pytorch_minus_baseline": delta,
                    "better": direction,
                    "improved": improved,
                }
            records[subset] = compared
        result[split] = {"overall": records.pop("overall"), "per_subset": records}
    return result


def _flatten_metrics(metrics: dict) -> dict:
    result = {}
    for split in ("validation", "test"):
        groups = {"overall": metrics[split]["overall"], **metrics[split]["per_subset"]}
        for subset, values in groups.items():
            for name, value in values.items():
                if type(value) in (int, float):
                    result[f"{split}.{subset}.{name}"] = value
    return result


def _report(metrics: dict) -> str:
    lines = [
        "# Pooled C-MAPSS PyTorch MLP RUL evaluation",
        "",
        "Uncapped RUL cycles; float64 inverse standardization then a zero floor, no upper cap.",
        "Training-only engine-weighted population feature and target standardization. "
        "The 35 existing predictors, engine splits and endpoint policy are unchanged.",
        "Validation RMSE after the floor selects the best epoch; no refit. The selected CPU "
        "state_dict is serialized before the single final test evaluation.",
        f"Best epoch (one-based): {metrics['model']['best_epoch']}; "
        f"epochs run: {metrics['model']['epochs_run']}.",
        "",
        "| Partition | Subset | Engines | RMSE | MAE | R² | Bias | NASA sum | NASA mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        for subset, values in {
            "overall": metrics[split]["overall"],
            **metrics[split]["per_subset"],
        }.items():
            cells = [
                "undefined" if values[name] is None else f"{values[name]:.6f}"
                for name in _METRIC_NAMES
            ]
            lines.append(f"| {split} | {subset} | {values['count']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Metrics use baseline.regression_metrics unchanged: one equal-weight endpoint per "
        "engine; error = prediction - actual. NASA sums expm1(-error/13) for underprediction "
        "and expm1(error/10) otherwise, with no overflow clipping. Undefined R² has a JSON reason.",
        "Validation endpoints are retrospective 50–80% life cuts, so these offline metrics do "
        "not establish field performance. RUL is not a calibrated failure probability.",
        "comparison.json reports exact-dataset baseline deltas; signed bias is judged by "
        "distance from zero, R² higher is better, errors and NASA scores lower are better.",
        "No SHAP or global explanations were requested for this MLP. No model registry or "
        "implicit tracking is used. Optional explicit tracker errors fail the caller.",
        "Reload through load_model/predict_saved, which verify the inventory and use "
        "torch.load(weights_only=True, map_location='cpu'); no Python model object is pickled.",
        "",
    ]
    return "\n".join(lines)


def _train(data_root, config_path, stage, tracker, baseline_reference_path):
    config = load_config(config_path)
    config_hash = baseline._sha256(config_path)
    manifest = verify_features(data_root)
    columns = baseline._validate_manifest(manifest)
    if columns != FEATURE_COLUMNS:
        raise TrainingError("Feature order does not match the approved feature pipeline.")
    reference = _reference(baseline_reference_path, data_root, manifest)
    train = baseline._load_partition(data_root, "train", columns, manifest)
    validation = baseline._load_partition(data_root, "validation", columns, manifest)
    baseline._validate_disjoint(train, validation)
    preprocessing = fit_preprocessing(train)
    provenance = {
        "ml_ready_manifest_sha256": baseline._sha256(data_root / "manifest.json"),
        "ml_ready_files": manifest["files"],
        "feature_config_sha256": manifest["config_sha256"],
        "source": manifest["config"]["source"],
        "pytorch_config_sha256": hashlib.sha256(baseline._canonical_json(config)).hexdigest(),
        "pytorch_config_file_sha256": config_hash,
    }
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    model = MLP().to(device="cpu", dtype=torch.float32)
    _track(tracker, "log_parameters", {**config, **provenance})
    best_epoch, best_score, history = _fit(model, train, validation, preprocessing, config, tracker)
    # No test arrays reach fitting or selection; verification above may inspect their bytes.
    torch.save(model.state_dict(), stage / "model.pt")
    model_hash = baseline._sha256(stage / "model.pt")
    baseline._write_json(stage / "model-spec.json", _model_spec(model))
    baseline._write_json(stage / "preprocessing.json", preprocessing)
    validation_prediction = _predict(
        model, _features_tensor(validation.features, preprocessing), preprocessing
    )
    test = baseline._load_partition(data_root, "test", columns, manifest)
    test_prediction = _predict(model, _features_tensor(test.features, preprocessing), preprocessing)
    if (
        baseline._sha256(stage / "model.pt") != model_hash
        or baseline._sha256(config_path) != config_hash
    ):
        raise TrainingError("The selected model or configuration changed during evaluation.")
    runtime = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "xgboost": baseline.xgb.__version__,
        },
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "num_workers": 0,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    model_info = {
        "configuration": config,
        "feature_columns": list(columns),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_rmse": best_score,
        "sha256": model_hash,
    }
    metrics = {
        "schema_version": 1,
        "model": model_info,
        "seed": config["seed"],
        "row_counts": manifest["row_counts"],
        "engine_counts": manifest["engine_counts"],
        "provenance": provenance,
        "runtime": runtime,
        "validation": baseline._partition_metrics(validation, validation_prediction),
        "test": baseline._partition_metrics(test, test_prediction),
    }
    baseline._write_json(stage / "metrics.json", metrics)
    baseline._write_json(stage / "comparison.json", _comparison(metrics, reference))
    baseline._write_json(stage / "training-history.json", {"schema_version": 1, "epochs": history})
    baseline._write_json(
        stage / "run-metadata.json",
        {
            "schema_version": 1,
            "model": model_info,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "seed": config["seed"],
            "config": config,
            "runtime": runtime,
            "provenance": provenance,
            "row_counts": manifest["row_counts"],
            "engine_counts": manifest["engine_counts"],
            "selection": "validation_rmse_after_zero_floor",
            "model_frozen_before_test": True,
            "test_evaluation_count": 1,
            "refit": False,
        },
    )
    baseline._predictions(
        validation, validation_prediction, stage / "predictions_validation.parquet"
    )
    baseline._predictions(test, test_prediction, stage / "predictions_test.parquet")
    (stage / "evaluation.md").write_text(_report(metrics), encoding="utf-8")
    _track(tracker, "log_final", _flatten_metrics(metrics))
    baseline._write_json(
        stage / "artifact-manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "pytorch-mlp-rul",
            "model_sha256": model_hash,
            "files": [
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": baseline._sha256(path),
                }
                for path in sorted(stage.iterdir())
            ],
            "provenance": provenance,
            "completion_marker": "artifact-manifest.json",
        },
    )
    return metrics


def train_pytorch(
    data_root: Path,
    config_path: Path,
    output_dir: Path,
    *,
    tracker: Tracker | None = None,
    baseline_reference_path: Path | None = None,
) -> dict:
    """Train once and publish manifest last without overwrite.

    The outer wrapper owns log_artifacts and calls it once after this function returns.
    Core failures clean staging/partial publication and do not leave a completion marker.
    """
    stage, published = None, []
    try:
        data_root, config_path, output_dir = Path(data_root), Path(config_path), Path(output_dir)
        if output_dir.is_symlink() or (
            output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir()))
        ):
            raise TrainingError(
                "Output directory must be new or empty; existing artifacts are never overwritten."
            )
        if (
            output_dir.resolve() == data_root.resolve()
            or data_root.resolve() in output_dir.resolve().parents
        ):
            raise TrainingError("Output cannot be inside the immutable feature bundle.")
        output_dir.mkdir(parents=True, exist_ok=True)
        stage = output_dir / f".staging-{uuid.uuid4().hex}"
        stage.mkdir()
        metrics = _train(data_root, config_path, stage, tracker, baseline_reference_path)
        if (
            set(output_dir.iterdir()) != {stage}
            or {path.name for path in stage.iterdir()} != _FILES
        ):
            raise TrainingError("Output inventory changed; refusing to overwrite artifacts.")
        for path in sorted(
            stage.iterdir(), key=lambda item: (item.name == "artifact-manifest.json", item.name)
        ):
            destination = output_dir / path.name
            with path.open("rb") as source, destination.open("xb") as target:
                published.append(destination)
                shutil.copyfileobj(source, target)
    except Exception as error:
        for path in reversed(published):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        if isinstance(error, TrainingError):
            raise
        raise TrainingError(
            "PyTorch training failed; verify feature integrity, configuration and runtime."
        ) from error
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
    return metrics


@dataclass
class LoadedModel:
    model: MLP
    preprocessing: dict

    def predict(self, features: object) -> np.ndarray:
        """Accept a feature table or an N x 35 matrix in the recorded feature order."""
        return _predict(
            self.model, _features_tensor(features, self.preprocessing), self.preprocessing
        )


def load_model(root: Path) -> LoadedModel:
    """Verify a completed inventory and strictly load CPU tensor weights, never a model pickle."""
    try:
        root = Path(root)
        if root.is_symlink() or {path.name for path in root.iterdir()} != _FILES:
            raise ValueError
        if any(path.is_symlink() or not path.is_file() for path in root.iterdir()):
            raise ValueError
        manifest = _read_json(root / "artifact-manifest.json")
        entries = manifest["files"]
        if (
            manifest["schema_version"] != 1
            or manifest["artifact_type"] != "pytorch-mlp-rul"
            or manifest["completion_marker"] != "artifact-manifest.json"
            or len(entries) != len(_FILES) - 1
            or {entry["path"] for entry in entries} != _FILES - {"artifact-manifest.json"}
        ):
            raise ValueError
        for entry in entries:
            path = root / entry["path"]
            if (
                path.stat().st_size != entry["size_bytes"]
                or baseline._sha256(path) != entry["sha256"]
            ):
                raise ValueError
        if manifest["model_sha256"] != baseline._sha256(root / "model.pt"):
            raise ValueError
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            model = MLP().to(device="cpu", dtype=torch.float32)
        if _read_json(root / "model-spec.json") != _model_spec(model):
            raise ValueError
        preprocessing = _read_json(root / "preprocessing.json")
        if (
            preprocessing["schema_version"] != 1
            or preprocessing["fit_partition"] != "train"
            or preprocessing["feature_columns"] != list(FEATURE_COLUMNS)
            or _numeric(preprocessing["feature_mean"], 1).shape != (35,)
            or _numeric(preprocessing["feature_scale"], 1).shape != (35,)
            or (np.asarray(preprocessing["feature_scale"]) <= 0).any()
            or not math.isfinite(preprocessing["target_mean"])
            or not math.isfinite(preprocessing["target_scale"])
            or preprocessing["target_scale"] <= 0
        ):
            raise ValueError
        state = torch.load(root / "model.pt", weights_only=True, map_location="cpu")
        expected = model.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError
        for key, tensor in state.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != expected[key].shape
                or tensor.dtype != torch.float32
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError
        model.load_state_dict(state, strict=True)
        model.eval()
        return LoadedModel(model, preprocessing)
    except Exception as error:
        raise TrainingError(
            "Saved PyTorch bundle failed integrity or safe-loading validation."
        ) from error


def predict_saved(datafeatures: object, outputroot: Path) -> np.ndarray:
    return load_model(outputroot).predict(datafeatures)


def main(argv: list[str] | None = None) -> int:
    parser = baseline._SafeArgumentParser(description="Train the approved CPU PyTorch RUL MLP.")
    for name in ("data", "config", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--baseline-reference", type=Path)
    arguments = parser.parse_args(argv)
    try:
        train_pytorch(
            arguments.data,
            arguments.config,
            arguments.output,
            baseline_reference_path=arguments.baseline_reference,
        )
    except TrainingError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print("PyTorch completed: selected state_dict, evaluation and artifact inventory written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
