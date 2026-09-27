"""Low-cardinality, payload-free request and prediction drift summaries."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

from epm_platform.serving.schema import (
    DRIFT_ALERT_RATE,
    DRIFT_MINIMUM_BATCH,
    FEATURE_COLUMNS,
    MODEL_SHA256,
    TRAINING_MANIFEST_SHA256,
)

_LOGGER = logging.getLogger("epm.inference")


def load_reference(path: Path) -> dict:
    try:
        reference = json.loads(path.read_text(encoding="utf-8"))
        if (
            type(reference["schema_version"]) is not int
            or reference["schema_version"] != 1
            or reference["model_sha256"] != MODEL_SHA256
            or reference["training_manifest_sha256"] != TRAINING_MANIFEST_SHA256
            or tuple(reference["feature_columns"]) != FEATURE_COLUMNS
            or set(reference["features"]) != set(FEATURE_COLUMNS)
            or set(reference["prediction"]) != {"mean", "std", "p01", "p99"}
        ):
            raise ValueError
        for record in [*reference["features"].values(), reference["prediction"]]:
            if (
                set(record) != {"mean", "std", "p01", "p99"}
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in record.values()
                )
                or record["std"] < 0
                or record["p01"] > record["p99"]
            ):
                raise ValueError
        return reference
    except (OSError, ValueError, TypeError, KeyError):
        raise ValueError("The model monitoring reference is invalid or mismatched.") from None


def summarize_batch(features: np.ndarray, predictions: np.ndarray, reference: dict) -> dict:
    if (
        features.ndim != 2
        or features.shape[1] != len(FEATURE_COLUMNS)
        or features.shape[0] == 0
        or predictions.shape != (features.shape[0],)
        or not np.isfinite(features).all()
        or not np.isfinite(predictions).all()
    ):
        raise ValueError("Monitoring requires aligned, finite feature and prediction batches.")

    rates = {}
    for index, name in enumerate(FEATURE_COLUMNS):
        bounds = reference["features"][name]
        outside = (features[:, index] < bounds["p01"]) | (features[:, index] > bounds["p99"])
        rate = float(outside.mean())
        if rate:
            rates[name] = rate

    prediction_bounds = reference["prediction"]
    prediction_outside = (predictions < prediction_bounds["p01"]) | (
        predictions > prediction_bounds["p99"]
    )
    assessable = features.shape[0] >= DRIFT_MINIMUM_BATCH
    summary = {
        "sample_size": int(features.shape[0]),
        "drift_assessable": assessable,
        "feature_out_of_reference_rates": rates if assessable else {},
        "feature_alerts": (
            sorted(name for name, rate in rates.items() if rate >= DRIFT_ALERT_RATE)
            if assessable
            else []
        ),
        "prediction_mean_cycles": float(predictions.mean()),
        "prediction_min_cycles": float(predictions.min()),
        "prediction_max_cycles": float(predictions.max()),
        "prediction_out_of_reference_rate": (
            float(prediction_outside.mean()) if assessable else None
        ),
        "prediction_alert": bool(assessable and prediction_outside.mean() >= DRIFT_ALERT_RATE),
        "zero_floor_count": int(np.count_nonzero(predictions == 0)),
    }
    _LOGGER.info("epm_inference_monitoring %s", json.dumps(summary, sort_keys=True))
    return summary
