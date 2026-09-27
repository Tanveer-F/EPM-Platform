"""Explicit synchronous MLflow run tracking, with no registry or model-flavor operations."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import os
import re
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from urllib.parse import urlsplit

ASSET_NAME = "epm-cmapss-ml-ready"
ASSET_VERSION = "d-f4ueae6u7g4c5islgehonqveey"
MANIFEST_SHA256 = "2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd"
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


class TrackingError(ValueError):
    """Safe public diagnostic; connector errors and credentials are never exposed."""


def _load_mlflow():
    import mlflow

    return mlflow


def _load_train():
    from epm_platform.deep_learning.training import train_pytorch

    return train_pytorch


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TrackingError("Tracking metadata contains duplicate keys.")
        result[key] = value
    return result


def _read_json(path: Path) -> dict:
    return json.loads(path.read_bytes(), object_pairs_hook=_unique_keys)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _no_links(path: Path) -> None:
    if ".." in path.parts or any(
        item.is_symlink() or item.is_junction() for item in (path, *path.absolute().parents)
    ):
        raise TrackingError("Tracking artifacts cannot contain links or parent traversal.")


def _flatten(parameters: dict, prefix: str = "") -> dict:
    result = {}
    for key, value in parameters.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise TrackingError("Tracking parameter keys must be explicit identifiers.")
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, name))
        else:
            result[name] = (
                json.dumps(value, sort_keys=True, allow_nan=False)
                if isinstance(value, (list, tuple))
                else value
            )
    return result


def _numeric_metrics(metrics: dict, *, final: bool) -> dict:
    result = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
            raise TrackingError("Tracking metric keys must be explicit identifiers.")
        name = key.lower().replace(".overall.", ".").replace(".", "_") if final else key
        if name in result or type(value) not in (int, float) or not math.isfinite(value):
            raise TrackingError("Tracking metrics must have unique keys and finite values.")
        result[name] = value
    return result


def _verify_artifacts(root: Path) -> None:
    _no_links(root)
    if not root.is_dir() or {path.name for path in root.iterdir()} != _FILES:
        raise TrackingError("Tracking requires the exact completed eleven-file artifact bundle.")
    for path in root.iterdir():
        _no_links(path)
        if not path.is_file():
            raise TrackingError("Tracking artifacts must be regular files.")
    manifest = _read_json(root / "artifact-manifest.json")
    entries = manifest.get("files")
    if (
        manifest.get("artifact_type") != "pytorch-mlp-rul"
        or manifest.get("schema_version") != 1
        or manifest.get("completion_marker") != "artifact-manifest.json"
        or not isinstance(entries, list)
        or len(entries) != len(_FILES) - 1
    ):
        raise TrackingError("Tracking artifact manifest is invalid.")
    found = set()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "size_bytes", "sha256"}
            or not isinstance(entry["path"], str)
            or entry["path"] not in _FILES - {"artifact-manifest.json"}
            or entry["path"] in found
            or type(entry["size_bytes"]) is not int
        ):
            raise TrackingError("Tracking artifact manifest inventory is invalid.")
        path = root / entry["path"]
        if path.stat().st_size != entry["size_bytes"] or _sha256(path) != entry["sha256"]:
            raise TrackingError("Tracking artifact content differs from its manifest.")
        found.add(entry["path"])
    if manifest.get("model_sha256") != _sha256(root / "model.pt"):
        raise TrackingError("Tracking model fingerprint differs from its manifest.")


class MLflowTracker:
    """The training protocol adapter. Every call requires an already active run."""

    def __init__(self, mlflow_module):
        self.mlflow = mlflow_module
        self.artifacts_logged = False

    def _call(self, name: str, *args, **kwargs):
        try:
            if self.mlflow.active_run() is None:
                raise TrackingError("Tracking requires an explicit active MLflow run.")
            return getattr(self.mlflow, name)(*args, **kwargs)
        except Exception:
            raise TrackingError(
                "Synchronous MLflow tracking failed; the job is not successful."
            ) from None

    def log_parameters(self, parameters: dict) -> None:
        flat = _flatten(parameters)
        bounded = {}
        for key, value in flat.items():
            if len(str(value)) > 500:
                hash_key = key + "_sha256"
                if hash_key in flat or hash_key in bounded:
                    raise TrackingError(
                        "Long-parameter fingerprint key conflicts with configuration."
                    )
                bounded[hash_key] = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
            else:
                bounded[key] = value
        self._call("log_params", bounded, synchronous=True)
        source = parameters.get("source", {})
        tags = {
            "framework": "pytorch",
            "tracking_policy": "explicit-run-artifacts-only",
            "ml_ready_asset_name": ASSET_NAME,
            "ml_ready_asset_version": ASSET_VERSION,
            "ml_ready_manifest_sha256": MANIFEST_SHA256,
        }
        for key in ("feature_config_sha256", "pytorch_config_sha256", "pytorch_config_file_sha256"):
            if key in parameters:
                tags[key] = parameters[key]
        for key in ("asset_name", "asset_version", "manifest_sha256"):
            if key in source:
                tags["source_" + key] = source[key]
        self._call("set_tags", tags, synchronous=True)

    def log_epoch(self, metrics: dict, step: int) -> None:
        if type(step) is not int or not 1 <= step <= 100:
            raise TrackingError("Epoch steps must be within the approved one-based budget.")
        self._call(
            "log_metrics", _numeric_metrics(metrics, final=False), step=step, synchronous=True
        )

    def log_final(self, metrics: dict) -> None:
        self._call("log_metrics", _numeric_metrics(metrics, final=True), synchronous=True)

    def log_artifacts(self, output_dir: Path) -> None:
        if self.artifacts_logged:
            raise TrackingError("The completed artifact bundle must be logged exactly once.")
        root = Path(output_dir)
        _verify_artifacts(root)
        model = _read_json(root / "metrics.json")["model"]
        selected = {key: model[key] for key in ("best_epoch", "epochs_run", "best_validation_rmse")}
        self.log_final(selected)
        self._call("log_artifacts", str(root), artifact_path="pytorch")
        self.artifacts_logged = True


def _remote_tracking_uri(mlflow) -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "azureml"
        or not parsed.netloc
        or not parsed.path
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or mlflow.get_tracking_uri() != uri
    ):
        raise TrackingError(
            "An injected Azure ML tracking URI is required; local fallback is disabled."
        )


def _verify_live_inputs(data: Path, config: Path, reference: Path) -> dict:
    for path in (data, config, reference):
        _no_links(path)
    manifest = data / "manifest.json"
    _no_links(manifest)
    if _sha256(manifest) != MANIFEST_SHA256:
        raise TrackingError("Remote tracking requires the exact approved feature manifest.")
    if _read_json(reference).get("ml_ready_manifest_sha256") != MANIFEST_SHA256:
        raise TrackingError("Remote tracking requires the matching baseline reference.")
    return {
        "ml_ready_asset_name": ASSET_NAME,
        "ml_ready_asset_version": ASSET_VERSION,
        "ml_ready_manifest_sha256": MANIFEST_SHA256,
        "baseline_reference_sha256": _sha256(reference),
    }


def train_with_tracking(
    data_root: Path,
    config_path: Path,
    output_dir: Path,
    *,
    baseline_reference_path: Path,
) -> dict:
    """Join Azure's run or open one explicit run; propagate failures without raw diagnostics."""
    for name in (
        "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING",
        "MLFLOW_ENABLE_ASYNC_LOGGING",
        "MLFLOW_ENABLE_TELEMETRY",
    ):
        os.environ[name] = "false"
    try:
        parameters = _verify_live_inputs(
            Path(data_root), Path(config_path), Path(baseline_reference_path)
        )
        mlflow = _load_mlflow()
        _remote_tracking_uri(mlflow)
        active = mlflow.active_run()
        run_id = os.environ.get("MLFLOW_RUN_ID")
        if run_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id):
            raise TrackingError("The injected MLflow run identity is invalid.")
        if active is not None and run_id is not None and active.info.run_id != run_id:
            raise TrackingError("The active MLflow run differs from the injected Azure run.")
        context = (
            nullcontext(active)
            if active is not None
            else mlflow.start_run(
                **({"run_id": run_id} if run_id else {}), log_system_metrics=False
            )
        )
        with context as run:
            tracker = MLflowTracker(mlflow)
            tracker.log_parameters(parameters)
            metrics = _load_train()(
                Path(data_root),
                Path(config_path),
                Path(output_dir),
                tracker=tracker,
                baseline_reference_path=Path(baseline_reference_path),
            )
            if not tracker.artifacts_logged:
                tracker.log_artifacts(Path(output_dir))
            result = {
                "status": "tracked",
                "run_id": run.info.run_id,
                "best_epoch": metrics["model"]["best_epoch"],
                "artifacts_logged": 11,
            }
        return result
    except Exception:
        raise TrackingError(
            "PyTorch tracking failed; verify approved inputs, Azure identity and isolated runtime."
        ) from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-reference", type=Path, required=True)
    args = parser.parse_args(argv)
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = train_with_tracking(
                args.data, args.config, args.output, baseline_reference_path=args.baseline_reference
            )
        exit_code = 0
    except Exception:
        result, exit_code = (
            {
                "status": "failed",
                "code": "pytorch_tracking_failed",
                "message": "PyTorch training or tracking failed; no successful run is claimed.",
            },
            1,
        )
    finally:
        logging.disable(previous)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
