"""Validated C-MAPSS request transformation and native XGBoost inference."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from epm_platform.serving.monitoring import load_reference, summarize_batch
from epm_platform.serving.schema import (
    ARTIFACT_MANIFEST_SHA256,
    BEST_ITERATION,
    FEATURE_COLUMNS,
    MAX_INSTANCES,
    MAX_INT32,
    MAX_OBSERVATIONS_PER_INSTANCE,
    MODEL_SHA256,
    OBSERVATION_COLUMNS,
    SAVED_TREE_COUNT,
    SENSOR_COLUMNS,
    SETTING_COLUMNS,
    SUBSETS,
)

_LOGGER = logging.getLogger("epm.inference")


class InferenceInputError(ValueError):
    """A safe, caller-correctable request validation failure."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class _DuplicateKeyError(ValueError):
    pass


class InferenceExecutionError(RuntimeError):
    """A sanitized model execution failure for the local HTTP boundary."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _parse_payload(raw_data: str | bytes | dict) -> dict:
    if isinstance(raw_data, bytes):
        try:
            raw_data = raw_data.decode("utf-8")
        except UnicodeDecodeError:
            raise InferenceInputError("invalid_utf8") from None
    if isinstance(raw_data, str):
        try:
            raw_data = json.loads(raw_data, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, _DuplicateKeyError):
            raise InferenceInputError("invalid_json") from None
    if not isinstance(raw_data, dict) or set(raw_data) != {"instances"}:
        raise InferenceInputError("invalid_request_shape")
    instances = raw_data["instances"]
    if not isinstance(instances, list) or not 1 <= len(instances) <= MAX_INSTANCES:
        raise InferenceInputError("invalid_instance_count")
    return raw_data


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InferenceInputError("nonfinite_or_nonnumeric_sensor_value")
    try:
        result = float(value)
    except OverflowError:
        raise InferenceInputError("sensor_value_out_of_supported_range") from None
    if not math.isfinite(result):
        raise InferenceInputError("nonfinite_or_nonnumeric_sensor_value")
    if abs(result) > np.finfo(np.float32).max:
        raise InferenceInputError("sensor_value_out_of_supported_range")
    return result


def _features_for_instance(instance: object) -> tuple[dict, list[float]]:
    if not isinstance(instance, dict) or set(instance) != {"subset", "unit_id", "observations"}:
        raise InferenceInputError("invalid_instance_schema")
    subset, unit_id, observations = (
        instance["subset"],
        instance["unit_id"],
        instance["observations"],
    )
    if not isinstance(subset, str) or subset not in SUBSETS:
        raise InferenceInputError("invalid_subset")
    if isinstance(unit_id, bool) or not isinstance(unit_id, int) or not 1 <= unit_id <= MAX_INT32:
        raise InferenceInputError("invalid_unit_id")
    if (
        not isinstance(observations, list)
        or not 1 <= len(observations) <= MAX_OBSERVATIONS_PER_INSTANCE
    ):
        raise InferenceInputError("invalid_observation_count")
    normalized = []
    for row in observations:
        if not isinstance(row, dict) or set(row) != set(OBSERVATION_COLUMNS):
            raise InferenceInputError("invalid_observation_schema")
        cycle = row["cycle"]
        if isinstance(cycle, bool) or not isinstance(cycle, int) or not 1 <= cycle <= MAX_INT32:
            raise InferenceInputError("invalid_cycle")
        normalized.append(
            {
                "cycle": cycle,
                **{name: _number(row[name]) for name in (*SETTING_COLUMNS, *SENSOR_COLUMNS)},
            }
        )

    cycles = [row["cycle"] for row in normalized]
    current_cycle = cycles[-1]
    required_count = min(current_cycle, MAX_OBSERVATIONS_PER_INSTANCE)
    expected_cycles = list(range(current_cycle - required_count + 1, current_cycle + 1))
    if len(normalized) != required_count or cycles != expected_cycles:
        raise InferenceInputError("history_must_be_contiguous_trailing_window")

    current = normalized[-1]
    features: dict[str, float] = {
        "cycle": current_cycle,
        **{name: current[name] for name in (*SETTING_COLUMNS, *SENSOR_COLUMNS)},
        "history_count": required_count,
    }
    for sensor in ("sensor_03", "sensor_04", "sensor_11"):
        values = [row[sensor] for row in normalized]
        features[f"{sensor}_mean10"] = math.fsum(values) / len(values)
        x_mean = math.fsum(cycles) / len(cycles)
        y_mean = features[f"{sensor}_mean10"]
        denominator = math.fsum((cycle - x_mean) ** 2 for cycle in cycles)
        features[f"{sensor}_slope10"] = (
            math.fsum(
                (cycle - x_mean) * (row[sensor] - y_mean)
                for cycle, row in zip(cycles, normalized, strict=True)
            )
            / denominator
            if denominator
            else 0.0
        )
    for setting in SETTING_COLUMNS:
        features[f"{setting}_mean10"] = math.fsum(row[setting] for row in normalized) / len(
            normalized
        )
    vector = [float(features[name]) for name in FEATURE_COLUMNS]
    if not all(math.isfinite(value) and abs(value) <= np.finfo(np.float32).max for value in vector):
        raise InferenceInputError("derived_features_nonfinite")
    return {"subset": subset, "unit_id": unit_id, "cycle": current_cycle}, vector


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_file(model_root: Path) -> Path:
    candidates = list(model_root.rglob("model.json"))
    if len(candidates) != 1:
        raise ValueError("Registered model bundle must contain exactly one model.json.")
    path = candidates[0]
    manifest_path = path.parent / "artifact-manifest.json"
    if (
        _sha256(path) != MODEL_SHA256
        or not manifest_path.is_file()
        or _sha256(manifest_path) != ARTIFACT_MANIFEST_SHA256
    ):
        raise ValueError("Registered model bundle checksums do not match the approved asset.")
    return path


class XGBoostPredictor:
    def __init__(self, model_path: Path):
        model_file = _model_file(model_path)
        self.model = xgb.Booster()
        self.model.load_model(str(model_file))
        if self.model.num_boosted_rounds() != SAVED_TREE_COUNT:
            raise ValueError("Registered XGBoost model has an unexpected tree count.")

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = xgb.DMatrix(features, feature_names=list(FEATURE_COLUMNS), nthread=2)
        raw_predictions = np.asarray(
            self.model.predict(matrix, iteration_range=(0, BEST_ITERATION + 1)),
            dtype=np.float64,
        )
        if raw_predictions.shape != (features.shape[0],) or not np.isfinite(raw_predictions).all():
            raise RuntimeError("The registered model returned invalid predictions.")
        return np.maximum(raw_predictions, 0.0)


class InferenceService:
    def __init__(self, predictor, reference: dict):
        self.predictor = predictor
        self.reference = reference

    @classmethod
    def load(cls, model_root: Path, reference_path: Path) -> InferenceService:
        return cls(XGBoostPredictor(model_root), load_reference(reference_path))

    def score(self, raw_data: str | bytes | dict) -> dict:
        started = time.perf_counter()
        try:
            payload = _parse_payload(raw_data)
            records = [_features_for_instance(item) for item in payload["instances"]]
            metadata = [record[0] for record in records]
            features = np.asarray([record[1] for record in records], dtype=np.float64)
            try:
                predictions = self.predictor.predict(features)
            except (xgb.core.XGBoostError, RuntimeError):
                _LOGGER.exception(
                    "epm_inference_request status=model_error instance_count=%s",
                    len(records),
                )
                raise InferenceExecutionError("model_execution_failed") from None
            summarize_batch(features, predictions, self.reference)
            result = {
                "predictions": [
                    {**item, "rul_cycles": float(prediction)}
                    for item, prediction in zip(metadata, predictions, strict=True)
                ]
            }
            _LOGGER.info(
                "epm_inference_request %s",
                json.dumps(
                    {
                        "status": "success",
                        "instance_count": len(records),
                        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    },
                    sort_keys=True,
                ),
            )
            return result
        except InferenceInputError as error:
            _LOGGER.warning(
                "epm_inference_request %s",
                json.dumps(
                    {
                        "status": "rejected",
                        "error_code": error.code,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    },
                    sort_keys=True,
                ),
            )
            raise
