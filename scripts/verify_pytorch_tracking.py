"""Read-only MLflow run/metric/artifact verification in the isolated tracking runtime."""

import argparse
import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path

from mlflow import MlflowClient, artifacts

EXPECTED_FILES = {
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    try:
        uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        if not uri.startswith("azureml://"):
            raise ValueError("Azure tracking URI is required")
        client = MlflowClient(tracking_uri=uri)
        run = client.get_run(args.job_name)
        if run.info.status != "FINISHED" or run.data.tags.get("framework") != "pytorch":
            raise ValueError("Run is not a finished PyTorch run")
        if run.data.tags.get("ml_ready_asset_version") != "d-f4ueae6u7g4c5islgehonqveey":
            raise ValueError("Run data version does not match")
        local_metrics = json.loads((args.artifacts / "metrics.json").read_text())
        for split in ("validation", "test"):
            for key in ("rmse", "mae", "r2"):
                expected = local_metrics[split]["overall"][key]
                value = run.data.metrics.get(f"{split}_{key}")
                if value is None or not math.isclose(value, expected, rel_tol=1e-8, abs_tol=1e-8):
                    raise ValueError("Tracked metric differs from the verified model output")
        epoch_history = client.get_metric_history(args.job_name, "val_rmse")
        if len(epoch_history) != local_metrics["model"]["epochs_run"]:
            raise ValueError("Incomplete tracked epoch history")
        if not run.data.params or "pytorch_config_sha256" not in run.data.params:
            raise ValueError("Tracked parameters/provenance missing")
        # Run-based MLflow 3 listing probes a logged-model API not supported by Azure's connector.
        artifact_uri = run.info.artifact_uri.rstrip("/") + "/pytorch"
        listed = artifacts.list_artifacts(artifact_uri=artifact_uri, tracking_uri=uri)
        if (
            any(entry.is_dir for entry in listed)
            or {Path(entry.path).name for entry in listed} != EXPECTED_FILES
            or len(listed) != len(EXPECTED_FILES)
        ):
            raise ValueError("MLflow artifact inventory mismatch")
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="tracking-verify-", dir=args.receipt.parent
        ) as temp:
            downloaded = Path(
                artifacts.download_artifacts(
                    artifact_uri=artifact_uri, dst_path=temp, tracking_uri=uri
                )
            )
            for name in EXPECTED_FILES:
                source = args.artifacts / name
                copy = downloaded / name
                if (
                    not copy.is_file()
                    or hashlib.sha256(source.read_bytes()).digest()
                    != hashlib.sha256(copy.read_bytes()).digest()
                ):
                    raise ValueError("MLflow artifact bytes differ from job output")
        receipt = {
            "status": "passed",
            "run_id": run.info.run_id,
            "run_status": run.info.status,
            "parameters_verified": len(run.data.params),
            "metrics_verified": len(run.data.metrics),
            "epochs_verified": len(epoch_history),
            "artifacts_verified": len(EXPECTED_FILES),
            "test_rmse": run.data.metrics["test_rmse"],
            "validation_rmse": run.data.metrics["validation_rmse"],
            "model_registration_performed": False,
        }
        args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        print(json.dumps(receipt, sort_keys=True))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": "Tracking verification failed; no cloud writes were performed.",
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
