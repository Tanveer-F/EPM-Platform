"""Build checksum-verified feature and prediction ranges for inference monitoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from epm_platform.serving.schema import (
    ARTIFACT_MANIFEST_SHA256,
    FEATURE_COLUMNS,
    MODEL_SHA256,
    TRAINING_MANIFEST_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = ROOT / "data" / "ml-ready" / "cmapss" / f"sha256-{TRAINING_MANIFEST_SHA256}"
BASELINE_ROOT = ROOT / "artifacts" / "baseline" / "epm-baseline-de82ea3141be"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: np.ndarray) -> dict:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Monitoring reference source values must be nonempty and finite.")
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p01": float(np.quantile(values, 0.01)),
        "p99": float(np.quantile(values, 0.99)),
    }


def build(output: Path) -> dict:
    from epm_platform.features.pipeline import verify_features

    verify_features(FEATURE_ROOT)
    if _sha256(FEATURE_ROOT / "manifest.json") != TRAINING_MANIFEST_SHA256:
        raise ValueError("ML-ready source manifest differs from the approved training asset.")

    artifact_manifest_path = BASELINE_ROOT / "artifact-manifest.json"
    if _sha256(artifact_manifest_path) != ARTIFACT_MANIFEST_SHA256:
        raise ValueError("Baseline artifact manifest differs from the approved registered model.")
    artifacts = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    files = {item["path"]: item for item in artifacts["files"]}
    prediction_path = BASELINE_ROOT / "predictions_validation.parquet"
    prediction_entry = files.get("predictions_validation.parquet")
    if (
        artifacts.get("artifact_type") != "xgboost-rul-baseline"
        or prediction_entry is None
        or _sha256(prediction_path) != prediction_entry["sha256"]
        or (BASELINE_ROOT / "model.json").is_file() is False
        or _sha256(BASELINE_ROOT / "model.json") != MODEL_SHA256
    ):
        raise ValueError("Baseline validation predictions or model failed checksum checks.")

    training = pq.ParquetFile(FEATURE_ROOT / "train.parquet").read(columns=list(FEATURE_COLUMNS))
    feature_stats = {
        name: _distribution(np.asarray(training[name].to_numpy(), dtype=np.float64))
        for name in FEATURE_COLUMNS
    }
    validation_predictions = (
        pq.ParquetFile(prediction_path)
        .read(columns=["prediction"])["prediction"]
        .to_numpy()
        .astype(np.float64)
    )
    reference = {
        "schema_version": 1,
        "model_sha256": MODEL_SHA256,
        "training_manifest_sha256": TRAINING_MANIFEST_SHA256,
        "training_row_count": training.num_rows,
        "validation_prediction_row_count": len(validation_predictions),
        "feature_columns": list(FEATURE_COLUMNS),
        "features": feature_stats,
        "prediction": _distribution(validation_predictions),
    }
    content = (json.dumps(reference, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    return {
        "output": str(output),
        "training_rows": training.num_rows,
        "validation_predictions": len(validation_predictions),
        "model_sha256": MODEL_SHA256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src" / "epm_platform" / "serving" / "monitoring-reference.json",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
