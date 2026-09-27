"""Drift-triggered, acceptance-gated retraining orchestration for the XGBoost baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from uuid import uuid4

from epm_platform.baseline import azure as baseline_azure
from epm_platform.config import AzureConfig, load_config
from epm_platform.data.errors import DataError

_ROOT = Path(__file__).resolve().parents[2]
_POLICY = _ROOT / "config" / "retraining.json"
_JOB_NAME = re.compile(r"epm-baseline-[a-f0-9]{12}")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_ACTIVE_STATES = {
    "submission_pending",
    "submission_unknown",
    "training",
    "evaluation_pending",
    "registration_pending",
}


class RetrainingError(ValueError):
    """A safe retraining workflow input or state error."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise RetrainingError("JSON input contains duplicate properties.")
        result[key] = value
    return result


def _read_json(path: Path) -> dict:
    path = Path(path)
    baseline_azure._no_links(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_048_576:
        raise RetrainingError("JSON input must be a regular file no larger than 1 MiB.")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        raise RetrainingError("JSON input is invalid or unreadable.") from None
    if not isinstance(value, dict):
        raise RetrainingError("JSON input must contain an object.")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetrainingError(f"{name} must be a finite nonnegative number.")
    try:
        result = float(value)
    except OverflowError:
        raise RetrainingError(f"{name} must be a finite nonnegative number.") from None
    if not math.isfinite(result) or result < 0:
        raise RetrainingError(f"{name} must be a finite nonnegative number.")
    return result


def load_policy(path: Path = _POLICY) -> dict:
    policy = _read_json(path)
    expected = {
        "schema_version",
        "model_name",
        "training_asset",
        "incumbent",
        "triggers",
        "promotion",
    }
    if (
        set(policy) != expected
        or type(policy["schema_version"]) is not int
        or policy["schema_version"] != 1
        or policy["model_name"] != "epm-cmapss-rul-xgboost"
    ):
        raise RetrainingError("Retraining policy schema or model name is not supported.")
    asset = policy["training_asset"]
    incumbent = policy["incumbent"]
    triggers = policy["triggers"]
    promotion = policy["promotion"]
    if (
        not isinstance(asset, dict)
        or set(asset) != {"name", "version", "manifest_sha256"}
        or asset["name"] != baseline_azure.ASSET_NAME
        or not isinstance(asset["version"], str)
        or not asset["version"]
        or not isinstance(asset["manifest_sha256"], str)
        or not _SHA256.fullmatch(asset["manifest_sha256"])
        or not isinstance(incumbent, dict)
        or set(incumbent)
        != {
            "version",
            "model_sha256",
            "test_engine_count",
            "test_rmse_cycles",
            "test_mean_nasa_score",
        }
        or not isinstance(incumbent["version"], str)
        or not incumbent["version"].isdecimal()
        or not isinstance(incumbent["model_sha256"], str)
        or not _SHA256.fullmatch(incumbent["model_sha256"])
        or type(incumbent["test_engine_count"]) is not int
        or incumbent["test_engine_count"] < 2
        or not isinstance(triggers, dict)
        or set(triggers)
        != {
            "minimum_drift_batch",
            "minimum_labeled_samples",
            "performance_degradation_ratio",
        }
        or any(
            type(triggers[key]) is not int or triggers[key] < 1
            for key in ("minimum_drift_batch", "minimum_labeled_samples")
        )
        or not isinstance(promotion, dict)
        or set(promotion)
        != {"minimum_rmse_improvement_fraction", "maximum_nasa_score_regression_ratio"}
    ):
        raise RetrainingError("Retraining policy fields are invalid.")
    for key in ("test_rmse_cycles", "test_mean_nasa_score"):
        _number(incumbent[key], key)
    ratio = _number(triggers["performance_degradation_ratio"], "performance ratio")
    improvement = _number(
        promotion["minimum_rmse_improvement_fraction"], "minimum RMSE improvement"
    )
    nasa_ratio = _number(
        promotion["maximum_nasa_score_regression_ratio"], "NASA-score regression ratio"
    )
    if ratio <= 1 or not 0 <= improvement < 1 or nasa_ratio < 1:
        raise RetrainingError("Retraining thresholds are inconsistent.")
    return policy


def evaluate_trigger(
    policy: dict,
    *,
    drift_summary: dict | None = None,
    performance_summary: dict | None = None,
) -> dict:
    """Return whether existing inference drift or labeled-performance evidence warrants a run."""
    if drift_summary is None and performance_summary is None:
        raise RetrainingError("Provide a drift summary, a labeled performance summary, or both.")

    reasons = []
    if drift_summary is not None:
        required = {
            "sample_size",
            "drift_assessable",
            "feature_alerts",
            "prediction_alert",
        }
        if (
            not isinstance(drift_summary, dict)
            or not required.issubset(drift_summary)
            or type(drift_summary["sample_size"]) is not int
            or drift_summary["sample_size"] < 0
            or type(drift_summary["drift_assessable"]) is not bool
            or type(drift_summary["prediction_alert"]) is not bool
            or not isinstance(drift_summary["feature_alerts"], list)
            or any(
                not isinstance(item, str) or not item
                for item in drift_summary["feature_alerts"]
            )
        ):
            raise RetrainingError("Monitoring summary does not match the serving contract.")
        if (
            drift_summary["drift_assessable"]
            and drift_summary["sample_size"] >= policy["triggers"]["minimum_drift_batch"]
            and (drift_summary["feature_alerts"] or drift_summary["prediction_alert"])
        ):
            reasons.append("serving_drift_alert")

    if performance_summary is not None:
        if (
            not isinstance(performance_summary, dict)
            or set(performance_summary) != {"sample_size", "rmse", "mean_nasa_score"}
            or type(performance_summary["sample_size"]) is not int
            or performance_summary["sample_size"] < 0
        ):
            raise RetrainingError(
                "Performance summary must contain sample_size, rmse and mean_nasa_score."
            )
        rmse = _number(performance_summary["rmse"], "performance RMSE")
        nasa = _number(performance_summary["mean_nasa_score"], "performance mean NASA score")
        incumbent = policy["incumbent"]
        if (
            performance_summary["sample_size"] >= policy["triggers"]["minimum_labeled_samples"]
            and (
                rmse
                >= incumbent["test_rmse_cycles"]
                * policy["triggers"]["performance_degradation_ratio"]
                or nasa
                >= incumbent["test_mean_nasa_score"]
                * policy["triggers"]["performance_degradation_ratio"]
            )
        ):
            reasons.append("labeled_performance_regression")

    return {"triggered": bool(reasons), "reasons": reasons}


def evaluate_candidate(
    policy: dict,
    metrics: dict,
    incumbent: dict | None = None,
) -> dict:
    """Apply the frozen engine-disjoint test acceptance gate to a verified training output."""
    incumbent = incumbent or policy["incumbent"]
    try:
        provenance = metrics["provenance"]
        engine_counts = metrics["engine_counts"]
        model = metrics["model"]
        configuration = model["configuration"]
        test = metrics["test"]["overall"]
        data_digest = provenance["ml_ready_manifest_sha256"]
        test_count = engine_counts["test"]
        model_digest = model["sha256"]
    except (KeyError, TypeError):
        raise RetrainingError(
            "Candidate metrics do not match the baseline artifact schema."
        ) from None
    if (
        not isinstance(configuration, dict)
        or not isinstance(test, dict)
        or data_digest != policy["training_asset"]["manifest_sha256"]
        or type(test_count) is not int
        or test_count != incumbent["test_engine_count"]
        or configuration.get("target") != "uncapped_rul"
        or not isinstance(model_digest, str)
        or not _SHA256.fullmatch(model_digest)
    ):
        raise RetrainingError(
            "Candidate data, target, model identity or test population differs from policy."
        )
    rmse = _number(test.get("rmse"), "candidate test RMSE")
    nasa = _number(test.get("mean_nasa_score"), "candidate mean NASA score")
    rmse_limit = incumbent["test_rmse_cycles"] * (
        1 - policy["promotion"]["minimum_rmse_improvement_fraction"]
    )
    nasa_limit = (
        incumbent["test_mean_nasa_score"]
        * policy["promotion"]["maximum_nasa_score_regression_ratio"]
    )
    reasons = []
    if rmse > rmse_limit:
        reasons.append("minimum_test_rmse_improvement_not_met")
    if nasa > nasa_limit:
        reasons.append("asymmetric_nasa_score_regressed")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "candidate": {"test_rmse_cycles": rmse, "test_mean_nasa_score": nasa},
        "incumbent": {
            "test_rmse_cycles": incumbent["test_rmse_cycles"],
            "test_mean_nasa_score": incumbent["test_mean_nasa_score"],
        },
        "thresholds": {"maximum_rmse_cycles": rmse_limit, "maximum_mean_nasa_score": nasa_limit},
        "test_engine_count": test_count,
        "ml_ready_manifest_sha256": data_digest,
        "model_sha256": model_digest,
    }


def _receipt_path(root: Path) -> Path:
    return root / ".azure" / "retraining-state.json"


def _policy_digest(policy: dict) -> str:
    payload = json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_receipt(path: Path, value: dict) -> None:
    baseline_azure._no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _incumbent_from_registry(client, policy: dict) -> dict:
    candidates = [
        model
        for model in client.models.list(name=policy["model_name"])
        if str(getattr(model, "version", "")).isdecimal()
    ]
    if not candidates:
        raise RetrainingError("No registered incumbent exists; retraining cannot promote.")
    model = max(candidates, key=lambda item: int(item.version))
    tags = getattr(model, "tags", {}) or {}
    if (
        not isinstance(tags, dict)
        or model.name != policy["model_name"]
        or getattr(model, "type", None) != "custom_model"
        or tags.get("ml_ready_manifest_sha256") != policy["training_asset"]["manifest_sha256"]
        or not isinstance(tags.get("baseline_model_sha256"), str)
        or not _SHA256.fullmatch(tags["baseline_model_sha256"])
    ):
        raise RetrainingError("Latest registry model does not match the approved RUL lineage.")
    try:
        incumbent = {
            "version": str(model.version),
            "model_sha256": tags["baseline_model_sha256"],
            "test_engine_count": int(
                tags.get("test_engine_count", policy["incumbent"]["test_engine_count"])
            ),
            "test_rmse_cycles": float(tags["test_rmse_cycles"]),
            "test_mean_nasa_score": float(tags["test_nasa_score_mean"]),
        }
    except (KeyError, TypeError, ValueError):
        raise RetrainingError("Latest registry model lacks validated acceptance metrics.") from None
    if (
        incumbent["test_engine_count"] != policy["incumbent"]["test_engine_count"]
        or not all(
            math.isfinite(incumbent[key]) and incumbent[key] >= 0
            for key in ("test_rmse_cycles", "test_mean_nasa_score")
        )
    ):
        raise RetrainingError("Latest registry model acceptance metadata is invalid.")
    return incumbent


def _verified_candidate(path: Path, policy: dict, expected_manifest_sha256: str) -> dict:
    expected_names = {
        "artifact-manifest.json",
        "evaluation.md",
        "feature-importance.json",
        "metrics.json",
        "model.json",
        "predictions_test.parquet",
        "predictions_validation.parquet",
        "run-metadata.json",
    }
    baseline_azure._no_links(path)
    if (
        not path.is_dir()
        or {item.name for item in path.iterdir()} != expected_names
        or any(not item.is_file() or item.is_symlink() for item in path.iterdir())
    ):
        raise RetrainingError("Candidate output inventory is invalid.")
    metrics = _read_json(path / "metrics.json")
    metadata = _read_json(path / "run-metadata.json")
    manifest = _read_json(path / "artifact-manifest.json")
    expected_digest = policy["training_asset"]["manifest_sha256"]
    manifest_entries = manifest.get("files")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("artifact_type") != "xgboost-rul-baseline"
        or manifest.get("completion_marker") != "artifact-manifest.json"
        or not isinstance(manifest_entries, list)
        or len(manifest_entries) != len(expected_names) - 1
    ):
        raise RetrainingError("Candidate artifact manifest is invalid.")
    if (
        not isinstance(expected_manifest_sha256, str)
        or not _SHA256.fullmatch(expected_manifest_sha256)
        or hashlib.sha256((path / "artifact-manifest.json").read_bytes()).hexdigest()
        != expected_manifest_sha256
    ):
        raise RetrainingError("Candidate artifact manifest checksum differs from its receipt.")
    fingerprints = {}
    for entry in manifest_entries:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256", "size_bytes"}
            or not isinstance(entry["path"], str)
            or entry["path"] not in expected_names - {"artifact-manifest.json"}
            or entry["path"] in fingerprints
            or type(entry["size_bytes"]) is not int
            or not isinstance(entry["sha256"], str)
            or not _SHA256.fullmatch(entry["sha256"])
        ):
            raise RetrainingError("Candidate artifact manifest inventory is invalid.")
        fingerprints[entry["path"]] = entry
    if set(fingerprints) != expected_names - {"artifact-manifest.json"}:
        raise RetrainingError("Candidate artifact manifest inventory is incomplete.")
    for name, entry in fingerprints.items():
        file_path = path / name
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if digest != entry["sha256"] or file_path.stat().st_size != entry["size_bytes"]:
            raise RetrainingError("Candidate artifact checksum differs from its manifest.")
    metadata_provenance = metadata.get("provenance")
    manifest_provenance = manifest.get("provenance")
    if (
        not isinstance(metrics.get("provenance"), dict)
        or metrics["provenance"].get("ml_ready_manifest_sha256") != expected_digest
        or not isinstance(metadata_provenance, dict)
        or metadata_provenance.get("ml_ready_manifest_sha256") != expected_digest
        or not isinstance(manifest_provenance, dict)
        or manifest_provenance.get("ml_ready_manifest_sha256") != expected_digest
    ):
        raise RetrainingError("Downloaded candidate lineage does not match the approved dataset.")
    model_path = path / "model.json"
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    candidate_model = metrics.get("model")
    if not isinstance(candidate_model, dict) or digest != candidate_model.get("sha256"):
        raise RetrainingError("Candidate model checksum differs from its training metrics.")
    return metrics


def _register_candidate(
    config: AzureConfig,
    policy: dict,
    incumbent: dict,
    job_name: str,
    decision: dict,
) -> dict:
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model

    with baseline_azure._client(config) as (client, _):
        current = _incumbent_from_registry(client, policy)
        if (
            current["version"] != incumbent["version"]
            or current["model_sha256"] != incumbent["model_sha256"]
        ):
            raise RetrainingError(
                "Registry incumbent changed during training; candidate was not registered."
            )
        model = client.models.create_or_update(
            Model(
                name=policy["model_name"],
                type=AssetTypes.CUSTOM_MODEL,
                path=f"azureml://jobs/{job_name}/outputs/baseline/paths/",
                description=(
                    "Acceptance-gated C-MAPSS XGBoost RUL candidate; registered only after "
                    "engine-disjoint test RMSE improvement with non-regressing asymmetric "
                    "NASA score."
                ),
                tags={
                    "framework": "xgboost",
                    "target": "uncapped_remaining_useful_life_cycles",
                    "selection": "automated_retraining_acceptance_gate",
                    "source_training_job": job_name,
                    "ml_ready_asset_name": policy["training_asset"]["name"],
                    "ml_ready_asset_version": policy["training_asset"]["version"],
                    "ml_ready_manifest_sha256": decision["ml_ready_manifest_sha256"],
                    "baseline_model_sha256": decision["model_sha256"],
                    "artifact_manifest_sha256": decision["artifact_manifest_sha256"],
                    "code_sha256": decision["code_sha256"],
                    "training_environment": decision["training_environment"],
                    "test_engine_count": str(decision["test_engine_count"]),
                    "test_rmse_cycles": str(decision["candidate"]["test_rmse_cycles"]),
                    "test_nasa_score_mean": str(decision["candidate"]["test_mean_nasa_score"]),
                    "compared_incumbent_version": incumbent["version"],
                    "compared_incumbent_model_sha256": incumbent["model_sha256"],
                },
            )
        )
        version = str(getattr(model, "version", ""))
        if (
            model.name != policy["model_name"]
            or not version.isdecimal()
            or int(version) <= int(incumbent["version"])
        ):
            raise RetrainingError("Registered model version did not advance beyond the incumbent.")
        verified = client.models.get(name=policy["model_name"], version=version)
        if (
            verified.name != policy["model_name"]
            or str(verified.version) != version
            or verified.tags.get("source_training_job") != job_name
            or verified.tags.get("baseline_model_sha256") != decision["model_sha256"]
        ):
            raise RetrainingError("Registered candidate could not be retrieved with its lineage.")
        return {"model_name": policy["model_name"], "model_version": version}


def _existing_registration(client, policy: dict, job_name: str, model_sha256: str) -> dict | None:
    matches = []
    for model in client.models.list(name=policy["model_name"]):
        tags = getattr(model, "tags", None)
        if (
            isinstance(tags, dict)
            and tags.get("source_training_job") == job_name
            and tags.get("baseline_model_sha256") == model_sha256
        ):
            matches.append(model)
    if len(matches) > 1:
        raise RetrainingError("Multiple registered versions match this training job.")
    if not matches:
        return None
    model = matches[0]
    version = str(getattr(model, "version", ""))
    if not version.isdecimal() or getattr(model, "type", None) != "custom_model":
        raise RetrainingError("Existing model registration has an invalid version or type.")
    verified = client.models.get(name=policy["model_name"], version=version)
    if (
        verified.name != policy["model_name"]
        or str(verified.version) != version
        or verified.tags.get("source_training_job") != job_name
        or verified.tags.get("baseline_model_sha256") != model_sha256
    ):
        raise RetrainingError("Existing model registration could not be verified.")
    return {"model_name": policy["model_name"], "model_version": version}


def _complete_job(
    root: Path,
    policy: dict,
    config: AzureConfig,
    receipt: dict,
    *,
    max_wait_seconds: int,
    poll_seconds: int,
) -> dict:
    job_name = receipt.get("job_name")
    run_id = receipt.get("run_id")
    if (
        not isinstance(job_name, str)
        or not _JOB_NAME.fullmatch(job_name)
        or not isinstance(run_id, str)
        or not re.fullmatch(r"[a-f0-9]{32}", run_id)
    ):
        raise RetrainingError("Retraining receipt has no valid Azure ML baseline job name.")
    if receipt.get("policy_sha256") != _policy_digest(policy):
        raise RetrainingError("Retraining policy changed; recorded job cannot be resumed safely.")
    was_registration_pending = receipt.get("status") == "registration_pending"
    if not was_registration_pending:
        receipt["status"] = "training"
        _write_receipt(_receipt_path(root), receipt)
    deadline = time.monotonic() + max_wait_seconds
    while True:
        status = baseline_azure.status(job_name, config)["status"]
        receipt["job_status"] = status
        _write_receipt(_receipt_path(root), receipt)
        if status in {"Completed", "Failed", "Canceled"}:
            break
        if time.monotonic() >= deadline:
            receipt["status"] = "training"
            _write_receipt(_receipt_path(root), receipt)
            return {"status": "training", "job_name": job_name, "resume_required": True}
        time.sleep(poll_seconds)
    if status != "Completed":
        receipt["status"] = "failed"
        _write_receipt(_receipt_path(root), receipt)
        return {"status": "failed", "job_name": job_name, "job_status": status}

    receipt["status"] = "evaluation_pending"
    _write_receipt(_receipt_path(root), receipt)
    artifacts_root = root / ".azure" / "retraining-artifacts"
    destination = artifacts_root / f"{job_name}-{run_id}"
    baseline_azure._no_links(destination)
    expected_manifest_sha256 = receipt.get("artifact_manifest_sha256")
    if destination.exists() and not expected_manifest_sha256:
        destination = artifacts_root / f"{job_name}-recovery-{uuid4().hex}"
    if not destination.exists():
        downloaded = baseline_azure.download(job_name, destination, config)
        receipt["artifact_manifest_sha256"] = downloaded["artifact_manifest_sha256"]
        _write_receipt(_receipt_path(root), receipt)
    metrics = _verified_candidate(
        destination,
        policy,
        receipt.get("artifact_manifest_sha256"),
    )
    with baseline_azure._client(config) as (client, _):
        incumbent = _incumbent_from_registry(client, policy)
    recorded_incumbent = receipt.get("incumbent")
    if (
        not isinstance(recorded_incumbent, dict)
        or incumbent["version"] != recorded_incumbent.get("version")
        or incumbent["model_sha256"] != recorded_incumbent.get("model_sha256")
    ):
        raise RetrainingError(
            "Registry incumbent changed during training; candidate was not evaluated."
        )
    decision = evaluate_candidate(policy, metrics, incumbent)
    code_digest = receipt.get("code_sha256")
    if not isinstance(code_digest, str) or not _SHA256.fullmatch(code_digest):
        raise RetrainingError("Retraining receipt lacks a verified source code hash.")
    decision["artifact_manifest_sha256"] = receipt["artifact_manifest_sha256"]
    decision["code_sha256"] = code_digest
    decision["training_environment"] = baseline_azure.ENVIRONMENT_REF
    receipt["evaluation"] = decision
    if not decision["accepted"]:
        receipt["status"] = "rejected"
        _write_receipt(_receipt_path(root), receipt)
        return {"status": "rejected", "job_name": job_name, "decision": decision}
    with baseline_azure._client(config) as (client, _):
        existing = _existing_registration(
            client, policy, job_name, decision["model_sha256"]
        )
    if existing is not None:
        receipt.update(existing)
        receipt["status"] = "registered"
        _write_receipt(_receipt_path(root), receipt)
        return {"status": "registered", "job_name": job_name, **existing, "decision": decision}
    if was_registration_pending:
        raise RetrainingError(
            "Registration outcome is uncertain and no matching model is visible; "
            "verify the registry before taking further action."
        )
    receipt["status"] = "registration_pending"
    receipt["candidate_model_sha256"] = decision["model_sha256"]
    _write_receipt(_receipt_path(root), receipt)
    registered = _register_candidate(config, policy, incumbent, job_name, decision)
    receipt.update(registered)
    receipt["status"] = "registered"
    _write_receipt(_receipt_path(root), receipt)
    return {"status": "registered", "job_name": job_name, **registered, "decision": decision}


def run_retraining(
    policy: dict,
    trigger: dict,
    config: AzureConfig,
    *,
    approve_costs: bool,
    project_root: Path = _ROOT,
    max_wait_seconds: int = 5400,
    poll_seconds: int = 30,
) -> dict:
    """Submit at most one explicitly approved CPU job, then evaluate and register conditionally."""
    if not trigger["triggered"]:
        return {"status": "skipped", "reasons": trigger["reasons"]}
    if approve_costs is not True:
        raise RetrainingError("Remote retraining requires the explicit --approve-costs flag.")
    if not 1 <= poll_seconds <= 300 or not 1 <= max_wait_seconds <= 7200:
        raise RetrainingError("Polling and wait limits are outside the supported bounds.")
    root = Path(project_root).resolve(strict=True)
    receipt_path = _receipt_path(root)
    if receipt_path.exists():
        prior = _read_json(receipt_path)
        if prior.get("status") in _ACTIVE_STATES:
            raise RetrainingError(
                "A prior retraining operation is unresolved; use resume before starting another."
            )
    with baseline_azure._client(config) as (client, _):
        incumbent = _incumbent_from_registry(client, policy)
    previous_job_name = None
    previous_baseline_receipt = root / ".azure" / "baseline-job.json"
    if previous_baseline_receipt.is_file():
        try:
            previous_job_name = _read_json(previous_baseline_receipt).get("job_name")
        except RetrainingError:
            previous_job_name = None
    receipt = {
        "schema_version": 1,
        "run_id": uuid4().hex,
        "status": "submission_pending",
        "policy_sha256": _policy_digest(policy),
        "previous_baseline_job_name": previous_job_name,
        "trigger_reasons": trigger["reasons"],
        "incumbent": incumbent,
        "asset_version": policy["training_asset"]["version"],
        "manifest_sha256": policy["training_asset"]["manifest_sha256"],
    }
    _write_receipt(receipt_path, receipt)
    try:
        submission = baseline_azure.submit(
            policy["training_asset"]["version"],
            policy["training_asset"]["manifest_sha256"],
            baseline_azure.ENVIRONMENT_VERSION,
            config,
            approve_costs=True,
            project_root=root,
        )
    except Exception:
        baseline_receipt = root / ".azure" / "baseline-job.json"
        if baseline_receipt.is_file():
            try:
                remote = _read_json(baseline_receipt)
            except RetrainingError:
                remote = {}
            if remote.get("status") in {"SubmissionPending", "SubmissionUnknown"}:
                receipt.update(
                    status="submission_unknown",
                    job_name=remote.get("job_name"),
                    code_sha256=remote.get("codehash"),
                )
            else:
                receipt["status"] = "failed"
        else:
            receipt["status"] = "failed"
        _write_receipt(receipt_path, receipt)
        raise
    receipt["job_name"] = submission["job_name"]
    receipt["code_sha256"] = submission["codehash"]
    receipt["status"] = "training"
    _write_receipt(receipt_path, receipt)
    return _complete_job(
        root,
        policy,
        config,
        receipt,
        max_wait_seconds=max_wait_seconds,
        poll_seconds=poll_seconds,
    )


def resume_retraining(
    policy: dict,
    config: AzureConfig,
    *,
    project_root: Path = _ROOT,
    max_wait_seconds: int = 5400,
    poll_seconds: int = 30,
) -> dict:
    """Resume status/evaluation for the recorded job; never submits or resubmits a job."""
    receipt_path = _receipt_path(Path(project_root))
    receipt = _read_json(receipt_path)
    if receipt.get("status") not in _ACTIVE_STATES:
        raise RetrainingError("No resumable submitted training job is recorded.")
    if receipt.get("policy_sha256") != _policy_digest(policy):
        raise RetrainingError("Retraining policy changed; recorded job cannot be resumed safely.")
    if not receipt.get("job_name") or not receipt.get("code_sha256"):
        baseline_receipt = _read_json(Path(project_root) / ".azure" / "baseline-job.json")
        if baseline_receipt.get("status") not in {
            "SubmissionPending",
            "SubmissionUnknown",
            "NotStarted",
            "Starting",
            "Provisioning",
            "Preparing",
            "Queued",
            "Running",
            "Finalizing",
            "Completed",
        }:
            raise RetrainingError(
                "Submission outcome is uncertain; inspect Azure ML before any new submission."
            )
        remote_job_name = baseline_receipt.get("job_name")
        if (
            not isinstance(remote_job_name, str)
            or not _JOB_NAME.fullmatch(remote_job_name)
            or remote_job_name == receipt.get("previous_baseline_job_name")
        ):
            raise RetrainingError(
                "No new retraining job is recorded; do not resume the unrelated baseline job."
            )
        if not receipt.get("job_name"):
            receipt["job_name"] = remote_job_name
        if not receipt.get("code_sha256"):
            receipt["code_sha256"] = baseline_receipt.get("codehash")
        _write_receipt(receipt_path, receipt)
    return _complete_job(
        Path(project_root),
        policy,
        config,
        receipt,
        max_wait_seconds=max_wait_seconds,
        poll_seconds=poll_seconds,
    )


def _summary(path: Path | None) -> dict | None:
    return _read_json(path) if path else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation in ("plan", "run"):
        command = commands.add_parser(operation)
        command.add_argument("--policy", type=Path, default=_POLICY)
        command.add_argument("--drift-summary", type=Path)
        command.add_argument("--performance-summary", type=Path)
        if operation == "run":
            command.add_argument("--approve-costs", action="store_true")
    resume = commands.add_parser("resume")
    resume.add_argument("--policy", type=Path, default=_POLICY)
    resume.add_argument("--approve-costs", action="store_true")
    args = parser.parse_args(argv)
    try:
        policy = load_policy(args.policy)
        if args.operation == "resume":
            if args.approve_costs:
                raise RetrainingError("Resume never submits training; omit --approve-costs.")
            result = resume_retraining(policy, load_config())
        else:
            trigger = evaluate_trigger(
                policy,
                drift_summary=_summary(args.drift_summary),
                performance_summary=_summary(args.performance_summary),
            )
            if args.operation == "plan":
                result = {
                    "status": "triggered" if trigger["triggered"] else "not_triggered",
                    **trigger,
                }
            else:
                result = run_retraining(
                    policy,
                    trigger,
                    load_config(),
                    approve_costs=args.approve_costs,
                )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (DataError, RetrainingError) as error:
        print(json.dumps({"status": "failed", "message": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
