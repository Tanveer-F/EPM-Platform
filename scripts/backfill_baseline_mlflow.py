"""Backfill Phase 4's existing MLflow experiment with verified run metadata; never train."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path

from mlflow import MlflowClient, artifacts, start_run
from mlflow.entities import Metric, Param, RunTag

JOB = "epm-baseline-de82ea3141be"
EXPERIMENT = "epm-baseline-rul"
ASSET = "epm-cmapss-ml-ready"
VERSION = "d-f4ueae6u7g4c5islgehonqveey"
DATASET_SHA256 = "2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd"
MODEL_SHA256 = "e58cb0a9285c364856361ede3c10de16facc7c4f2a48b1ae643515db39d5d0fe"
MANIFEST_SHA256 = "c280467cc28509fb04e63ad6ba1c26c86b8e16a09636b8ec1ce9e952c2641e09"
FILE_NAMES = {
    "artifact-manifest.json",
    "evaluation.md",
    "feature-importance.json",
    "metrics.json",
    "model.json",
    "predictions_test.parquet",
    "predictions_validation.parquet",
    "run-metadata.json",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def flatten_params(metadata: dict, metrics_document: dict) -> dict[str, str]:
    model = metadata["model"]
    config = model["configuration"]
    values = {
        "model_type": "xgboost_cpu_tree_ensemble",
        "target": config["target"],
        "objective": config["parameters"]["objective"],
        "tree_method": config["parameters"]["tree_method"],
        "device": config["parameters"]["device"],
        "seed": str(config["parameters"]["seed"]),
        "max_depth": str(config["parameters"]["max_depth"]),
        "learning_rate": str(config["parameters"]["eta"]),
        "min_child_weight": str(config["parameters"]["min_child_weight"]),
        "l2_regularization": str(config["parameters"]["lambda"]),
        "max_bins": str(config["parameters"]["max_bin"]),
        "subsample": str(config["parameters"]["subsample"]),
        "column_subsample": str(config["parameters"]["colsample_bytree"]),
        "max_boosting_rounds": str(config["num_boost_round"]),
        "early_stopping_rounds": str(config["early_stopping_rounds"]),
        "best_iteration_zero_based": str(model["best_iteration"]),
        "saved_tree_count": str(model["tree_count"]),
        "feature_count": str(len(model["feature_columns"])),
        "ml_ready_asset": f"{ASSET}:{VERSION}",
        "ml_ready_manifest_sha256": DATASET_SHA256,
        "source_ml_asset_version": metadata["provenance"]["source"]["asset_version"],
        "training_code_sha256": "fca4f48ea125278a9b6089701970b6d5a150a4921bfaff4a896202b366bea2c5",
        "training_environment": "sklearn-1.5:54/python-3.12.10/xgboost-cpu-3.4.1",
        "python_version": metadata["runtime"]["python"],
        "xgboost_version": metadata["runtime"]["packages"]["xgboost"],
        "numpy_version": metadata["runtime"]["packages"]["numpy"],
        "pyarrow_version": metadata["runtime"]["packages"]["pyarrow"],
        "config_sha256": metadata["provenance"]["baseline_config_sha256"],
        "model_sha256": MODEL_SHA256,
        "artifact_manifest_sha256": MANIFEST_SHA256,
    }
    for split in ("validation", "test"):
        records = {"overall": metrics_document[split]["overall"]}
        records.update(metrics_document[split]["per_subset"])
        for subset, metrics in records.items():
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values[f"{split}_{subset}_{metric}"] = str(value)
    if any(len(value) > 500 for value in values.values()):
        raise ValueError("Prepared metadata exceeds Azure MLflow's parameter limit.")
    return values


def metric_values_for_tracking(metrics_document: dict, best_iteration: int) -> dict[str, float]:
    result = {}
    for split in ("validation", "test"):
        records = {"overall": metrics_document[split]["overall"]}
        records.update(metrics_document[split]["per_subset"])
        for subset, values in records.items():
            for name, value in values.items():
                if (
                    name != "r2_undefined_reason"
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                ):
                    result[f"{split}_{subset}_{name}"] = float(value)
    result["training_best_iteration_zero_based"] = float(best_iteration)
    return result


def run(source: Path, receipt: Path) -> dict:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri.startswith("azureml://"):
        raise ValueError("An Azure ML tracking URI is required.")
    source = source.resolve(strict=True)
    if {item.name for item in source.iterdir()} != FILE_NAMES:
        raise ValueError("The source baseline output inventory is unexpected.")
    model_path = source / "model.json"
    artifact_path = source / "artifact-manifest.json"
    metrics_path = source / "metrics.json"
    run_metadata_path = source / "run-metadata.json"
    if digest(model_path) != MODEL_SHA256 or digest(artifact_path) != MANIFEST_SHA256:
        raise ValueError("The validated baseline model or artifact manifest has changed.")
    output_manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
    if output_manifest.get("artifact_type") != "xgboost-rul-baseline":
        raise ValueError("The output is not the approved XGBoost model bundle.")
    for entry in output_manifest["files"]:
        item = source / entry["path"]
        if (
            item.name not in FILE_NAMES
            or item.stat().st_size != entry["size_bytes"]
            or digest(item) != entry["sha256"]
        ):
            raise ValueError("A baseline output failed checksum validation.")
    metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    if (
        metadata["provenance"]["ml_ready_manifest_sha256"] != DATASET_SHA256
        or metadata["model"]["sha256"] != MODEL_SHA256
        or metadata["selection"] != "validation_only_early_stopping_no_refit"
        or metadata["test_evaluation_count"] != 1
    ):
        raise ValueError("Baseline provenance, selection or evaluation count changed.")

    client = MlflowClient(tracking_uri=uri)
    original = client.get_run(JOB)
    if original.info.status != "FINISHED" or original.info.experiment_id is None:
        raise ValueError("The source Azure ML baseline run is not finished.")
    experiment = client.get_experiment(original.info.experiment_id)
    if experiment is None or experiment.name != EXPERIMENT:
        raise ValueError("The source baseline experiment could not be verified.")
    existing = [
        item
        for item in client.search_runs([experiment.experiment_id], max_results=100)
        if item.data.tags.get("epm.tracking_purpose") == "verified-baseline-metadata-backfill"
        and item.data.tags.get("epm.source_training_job") == JOB
    ]
    metrics_document = json.loads(metrics_path.read_text(encoding="utf-8"))
    params = flatten_params(metadata, metrics_document)
    metric_values = metric_values_for_tracking(
        metrics_document, metadata["model"]["best_iteration"]
    )
    tags = {
        "epm.tracking_purpose": "verified-baseline-metadata-backfill",
        "epm.training_performed": "false",
        "epm.source_training_job": JOB,
        "epm.source_training_run_id": original.info.run_id,
        "epm.source_experiment": EXPERIMENT,
        "epm.model_name": "xgboost-rul-baseline",
        "epm.model_sha256": MODEL_SHA256,
        "epm.metrics_sha256": digest(metrics_path),
        "epm.artifact_manifest_sha256": MANIFEST_SHA256,
        "epm.ml_ready_asset": f"{ASSET}:{VERSION}",
        "epm.ml_ready_manifest_sha256": DATASET_SHA256,
        "epm.source_mlflow_run_id": "epm-pytorch-d97b75fd1f65",
        "epm.source_mlflow_experiment": "epm-pytorch-rul",
    }
    recovered = bool(existing)
    active_run = None
    if recovered:
        if len(existing) != 1 or existing[0].info.status not in {"RUNNING", "FINISHED"}:
            raise ValueError("Conflicting or incomplete metadata runs require manual review.")
        active_run = existing[0]
        run_id = active_run.info.run_id
        for key, value in tags.items():
            actual = active_run.data.tags.get(key)
            if actual is not None and actual != value:
                raise ValueError("The existing metadata run has conflicting lineage.")
            if active_run.info.status == "FINISHED" and actual != value:
                raise ValueError("The finished metadata run is missing expected lineage tags.")
        for key, value in active_run.data.params.items():
            if key not in params or params[key] != value:
                raise ValueError("The existing metadata run has conflicting parameters.")
        for key, value in active_run.data.metrics.items():
            if key not in metric_values or not math.isclose(
                value, metric_values[key], rel_tol=0, abs_tol=1e-10
            ):
                raise ValueError("The existing metadata run has conflicting metrics.")
        if active_run.info.status == "FINISHED" and (
            len(active_run.data.params) != len(params)
            or len(active_run.data.metrics) != len(metric_values)
        ):
            raise ValueError("A finished metadata run is missing expected entries.")
        if active_run.info.status == "RUNNING":
            missing_params = {
                key: value for key, value in params.items() if key not in active_run.data.params
            }
            missing_metrics = {
                key: value
                for key, value in metric_values.items()
                if key not in active_run.data.metrics
            }
            client.log_batch(
                run_id,
                params=[Param(key, value) for key, value in missing_params.items()],
                metrics=[Metric(key, value, 0, 0) for key, value in missing_metrics.items()],
                tags=[
                    RunTag(key, value)
                    for key, value in tags.items()
                    if key not in active_run.data.tags
                ],
            )
        active = None
    else:
        context = start_run(
            experiment_id=experiment.experiment_id,
            run_name="epm-baseline-metadata-backfill",
            log_system_metrics=False,
        )
        active = context.__enter__()
        run_id = active.info.run_id
        client.log_batch(
            run_id,
            params=[Param(key, value) for key, value in params.items()],
            metrics=[Metric(key, value, 0, 0) for key, value in metric_values.items()],
            tags=[RunTag(key, value) for key, value in tags.items()],
        )

    def finalize_backfill():
        current_run = client.get_run(run_id)
        current_artifact_uri = current_run.info.artifact_uri.rstrip("/") + "/training-output"
        existing_artifacts = artifacts.list_artifacts(
            artifact_uri=current_artifact_uri, tracking_uri=uri
        )
        if not existing_artifacts:
            client.log_artifacts(run_id, str(source), artifact_path="training-output")
        elif (
            any(item.is_dir for item in existing_artifacts)
            or {Path(item.path).name for item in existing_artifacts} != FILE_NAMES
            or len(existing_artifacts) != len(FILE_NAMES)
        ):
            raise ValueError("The existing metadata run has a conflicting artifact inventory.")

        completed = client.get_run(run_id)
        if completed.data.params.get("model_sha256") != MODEL_SHA256:
            raise ValueError("MLflow did not persist the selected model fingerprint.")
        if (
            completed.data.metrics.get("test_overall_rmse")
            != metrics_document["test"]["overall"]["rmse"]
        ):
            raise ValueError("MLflow did not persist the verified test RMSE.")
        if completed.data.tags.get("epm.source_training_run_id") != original.info.run_id:
            raise ValueError("MLflow source training lineage was not persisted.")
        artifact_uri = completed.info.artifact_uri.rstrip("/") + "/training-output"
        remote_files = artifacts.list_artifacts(artifact_uri=artifact_uri, tracking_uri=uri)
        if {Path(item.path).name for item in remote_files} != FILE_NAMES or len(
            remote_files
        ) != len(FILE_NAMES):
            raise ValueError("MLflow artifact inventory verification failed.")
        with tempfile.TemporaryDirectory(prefix="epm-mlflow-backfill-") as directory:
            local_model = Path(
                artifacts.download_artifacts(
                    artifact_uri=artifact_uri + "/model.json",
                    dst_path=directory,
                    tracking_uri=uri,
                )
            )
            if digest(local_model) != MODEL_SHA256:
                raise ValueError(
                    "The tracked model artifact does not match the Azure ML job output."
                )
        if completed.info.status == "RUNNING":
            client.set_terminated(run_id, status="FINISHED")

    try:
        finalize_backfill()
        if not recovered:
            context.__exit__(None, None, None)
    except BaseException as error:
        if not recovered:
            context.__exit__(type(error), error, error.__traceback__)
        raise

    result = {
        "status": "recovered_and_verified" if recovered else "created_and_verified",
        "backfill_run_id": run_id,
        "source_training_run_id": original.info.run_id,
        "experiment_name": EXPERIMENT,
        "parameters_verified": len(params),
        "metrics_verified": len(metric_values),
        "artifacts_verified": len(FILE_NAMES),
        "training_performed": False,
        "model_sha256": MODEL_SHA256,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    try:
        result = run(args.source, args.receipt)
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": (
                        "Baseline tracking metadata backfill failed; "
                        "no model registration is claimed."
                    ),
                }
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
