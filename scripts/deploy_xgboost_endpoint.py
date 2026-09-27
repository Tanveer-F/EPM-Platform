"""Run one authenticated Azure ML endpoint smoke test and always tear it down."""

from __future__ import annotations

import json
import logging
import secrets
import subprocess
import sys
from pathlib import Path

from azure.ai.ml.entities import (
    CodeConfiguration,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    OnlineRequestSettings,
    ProbeSettings,
)
from azure.core.exceptions import ResourceNotFoundError

from epm_platform.client import create_credential, create_ml_client
from epm_platform.config import load_config
from epm_platform.serving.schema import MODEL_SHA256
from epm_platform.serving.scoring import InferenceService

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "epm-cmapss-rul-xgboost"
MODEL_VERSION = "1"
DEPLOYMENT_NAME = "blue"
INSTANCE_TYPE = "Standard_D2s_v3"
ENVIRONMENT = "azureml://registries/azureml/environments/sklearn-1.5/versions/54"
REFERENCE = ROOT / "src" / "epm_platform" / "serving" / "monitoring-reference.json"
VERIFICATION_ROOT = ROOT / ".azure" / "registered-model-verification"
MODEL_ROOT = VERIFICATION_ROOT / MODEL_NAME
RECEIPT = ROOT / ".azure" / "endpoint-smoke.json"


def _local_model_root(client) -> Path:
    if not any(MODEL_ROOT.rglob("model.json")):
        download_root = ROOT / ".azure" / "endpoint-local-model"
        if download_root.exists():
            raise RuntimeError("The existing local model download path must be inspected.")
        download_root.mkdir(parents=True)
        client.models.download(
            name=MODEL_NAME,
            version=MODEL_VERSION,
            download_path=str(download_root),
        )
        candidates = list(download_root.rglob("model.json"))
        if len(candidates) != 1:
            raise RuntimeError("The registered model download inventory was unexpected.")
        return candidates[0].parent
    return MODEL_ROOT


def _verify_model(client) -> dict:
    model = client.models.get(name=MODEL_NAME, version=MODEL_VERSION)
    if (
        model.name != MODEL_NAME
        or str(model.version) != MODEL_VERSION
        or model.type != "custom_model"
        or model.tags.get("baseline_model_sha256") != MODEL_SHA256
        or model.tags.get("source_training_job") != "epm-baseline-de82ea3141be"
    ):
        raise RuntimeError("Registered model version or lineage does not match Phase 8 evidence.")
    return model


def _build_smoke_request(path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_endpoint_smoke_request.py"),
            "--output",
            str(path),
            "--sample-size",
            "20",
        ],
        cwd=ROOT,
        check=True,
    )


def _verify_predictions(expected: dict, actual_text: str) -> dict:
    try:
        actual = json.loads(actual_text)
        predictions = actual["predictions"]
        expected_predictions = expected["predictions"]
        if len(predictions) != len(expected_predictions):
            raise ValueError
        differences = []
        for local, remote in zip(expected_predictions, predictions, strict=True):
            if (
                any(remote.get(key) != local[key] for key in ("subset", "unit_id", "cycle"))
                or isinstance(remote.get("rul_cycles"), bool)
                or not isinstance(remote.get("rul_cycles"), (int, float))
            ):
                raise ValueError
            differences.append(abs(float(remote["rul_cycles"]) - local["rul_cycles"]))
        maximum_difference = max(differences, default=0.0)
        if maximum_difference > 0.05:
            raise ValueError
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise RuntimeError(
            "The endpoint response failed the registered-model comparison."
        ) from None
    return {
        "prediction_count": len(predictions),
        "max_abs_difference_cycles": maximum_difference,
        "mean_prediction_cycles": sum(float(item["rul_cycles"]) for item in predictions)
        / len(predictions),
    }


def run() -> dict:
    if RECEIPT.exists():
        raise RuntimeError("An endpoint smoke receipt already exists; inspect it before rerunning.")
    config = load_config()
    client = create_ml_client(config, create_credential(config))
    model = _verify_model(client)
    local_model_root = _local_model_root(client)
    service = InferenceService.load(local_model_root, REFERENCE)
    request_file = ROOT / ".azure" / "endpoint-smoke-request.json"
    _build_smoke_request(request_file)
    request = request_file.read_bytes()
    expected = service.score(request)

    endpoint_name = f"epm-rul-smoke-{secrets.token_hex(4)}"
    endpoint_attempted = False
    primary_error: Exception | None = None
    cleanup_error: Exception | None = None
    cleanup_verified = False
    result = None
    try:
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Temporary cost-controlled Phase 9 inference validation.",
            auth_mode="aml_token",
            public_network_access="enabled",
            tags={
                "project": "epm",
                "purpose": "phase9-smoke-test",
                "model_version": MODEL_VERSION,
            },
        )
        endpoint_attempted = True
        client.online_endpoints.begin_create_or_update(endpoint).result(timeout=900)

        deployment = ManagedOnlineDeployment(
            name=DEPLOYMENT_NAME,
            endpoint_name=endpoint_name,
            model=model,
            environment=ENVIRONMENT,
            code_configuration=CodeConfiguration(
                code=str(ROOT / "src"),
                scoring_script="epm_platform/serving/score.py",
            ),
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            app_insights_enabled=True,
            request_settings=OnlineRequestSettings(
                max_concurrent_requests_per_instance=2,
                request_timeout_ms=30_000,
                max_queue_wait_ms=1_000,
            ),
            liveness_probe=ProbeSettings(
                initial_delay=10,
                period=10,
                timeout=2,
                success_threshold=1,
                failure_threshold=30,
            ),
            readiness_probe=ProbeSettings(
                initial_delay=10,
                period=10,
                timeout=2,
                success_threshold=1,
                failure_threshold=30,
            ),
            tags={"project": "epm", "phase": "9", "model_sha256": MODEL_SHA256},
        )
        created_deployment = client.online_deployments.begin_create_or_update(deployment).result(
            timeout=1800
        )
        if getattr(created_deployment, "provisioning_state", None) != "Succeeded":
            raise RuntimeError("Managed Online Deployment did not reach Succeeded.")

        endpoint = client.online_endpoints.get(endpoint_name)
        endpoint.traffic = {DEPLOYMENT_NAME: 100}
        endpoint = client.online_endpoints.begin_create_or_update(endpoint).result(timeout=900)
        if endpoint.auth_mode != "aml_token" or endpoint.traffic.get(DEPLOYMENT_NAME) != 100:
            raise RuntimeError(
                "Endpoint authentication or production traffic configuration failed."
            )

        response = client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            request_file=str(request_file),
            deployment_name=DEPLOYMENT_NAME,
        )
        comparison = _verify_predictions(expected, response)
        try:
            logs = client.online_deployments.get_logs(
                name=DEPLOYMENT_NAME,
                endpoint_name=endpoint_name,
                lines=200,
            )
            monitor_log_observed = "epm_inference_monitoring" in str(logs)
        except (ResourceNotFoundError, RuntimeError):
            monitor_log_observed = False
        result = {
            "status": "verified_then_deleted",
            "endpoint_name": endpoint_name,
            "auth_mode": "aml_token",
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "source_training_job": model.tags["source_training_job"],
            "deployment_name": DEPLOYMENT_NAME,
            "instance_type": INSTANCE_TYPE,
            "instance_count": 1,
            "environment": ENVIRONMENT,
            "app_insights_enabled": True,
            "monitor_log_observed": monitor_log_observed,
            **comparison,
        }
    except Exception as error:
        primary_error = error
    finally:
        if endpoint_attempted:
            try:
                try:
                    client.online_endpoints.get(endpoint_name)
                except ResourceNotFoundError:
                    pass
                else:
                    client.online_endpoints.begin_delete(name=endpoint_name).result(timeout=900)
                cleanup_verified = not any(
                    item.name == endpoint_name for item in client.online_endpoints.list()
                )
                if not cleanup_verified:
                    cleanup_error = RuntimeError("endpoint deletion was not confirmed")
            except Exception as error:
                cleanup_error = error

    if primary_error is not None or cleanup_error is not None:
        failure = {
            "status": "failed",
            "endpoint_name": endpoint_name,
            "primary_failure_class": (
                type(primary_error).__name__ if primary_error is not None else None
            ),
            "cleanup_failure_class": (
                type(cleanup_error).__name__ if cleanup_error is not None else None
            ),
            "cleanup_verified": cleanup_verified,
        }
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT.write_text(
            json.dumps(failure, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if cleanup_error is not None:
            raise RuntimeError(
                "URGENT: Azure endpoint cleanup was not verified. Check the named endpoint "
                "immediately; it may still incur compute charges."
            ) from None
        raise RuntimeError(
            "Azure endpoint validation failed; cleanup is verified and no Azure retry was made. "
            f"Failure class: {type(primary_error).__name__}."
        ) from None
    if result is None:
        raise RuntimeError("Endpoint validation produced no verified result.")
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("epm.inference").setLevel(logging.INFO)
    try:
        result = run()
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_class": type(error).__name__,
                    "message": str(error),
                    "local_inference_available": True,
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
