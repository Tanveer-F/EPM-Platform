import hashlib
import json
from pathlib import Path

import pytest

from epm_platform.retraining import (
    RetrainingError,
    _policy_digest,
    _verified_candidate,
    evaluate_candidate,
    evaluate_trigger,
    load_policy,
    resume_retraining,
    run_retraining,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config" / "retraining.json"


def _candidate(policy, *, rmse=30.0, nasa=70.0, engine_count=707, manifest=None):
    return {
        "provenance": {
            "ml_ready_manifest_sha256": manifest or policy["training_asset"]["manifest_sha256"]
        },
        "engine_counts": {"test": engine_count},
        "model": {
            "sha256": "a" * 64,
            "configuration": {"target": "uncapped_rul"},
        },
        "test": {"overall": {"rmse": rmse, "mean_nasa_score": nasa}},
    }


def test_policy_is_bound_to_existing_baseline_and_data_asset():
    policy = load_policy(POLICY)
    assert policy["model_name"] == "epm-cmapss-rul-xgboost"
    assert policy["incumbent"]["version"] == "1"
    assert policy["training_asset"]["manifest_sha256"] == (
        "2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd"
    )


def test_drift_trigger_uses_existing_minimum_batch_and_alert_contract():
    policy = load_policy(POLICY)
    summary = {
        "sample_size": 20,
        "drift_assessable": True,
        "feature_alerts": ["sensor_03"],
        "prediction_alert": False,
    }
    assert evaluate_trigger(policy, drift_summary=summary) == {
        "triggered": True,
        "reasons": ["serving_drift_alert"],
    }
    summary["sample_size"] = 19
    assert evaluate_trigger(policy, drift_summary=summary)["triggered"] is False
    summary.update(sample_size=25, drift_assessable=False)
    assert evaluate_trigger(policy, drift_summary=summary)["triggered"] is False


def test_labeled_performance_trigger_requires_minimum_sample_and_degradation():
    policy = load_policy(POLICY)
    summary = {"sample_size": 20, "rmse": 34.0, "mean_nasa_score": 20.0}
    assert evaluate_trigger(policy, performance_summary=summary)["reasons"] == [
        "labeled_performance_regression"
    ]
    summary["sample_size"] = 19
    assert evaluate_trigger(policy, performance_summary=summary)["triggered"] is False
    summary.update(sample_size=20, rmse=10.0, mean_nasa_score=10.0)
    assert evaluate_trigger(policy, performance_summary=summary)["triggered"] is False


@pytest.mark.parametrize(
    "summary",
    [
        {
            "sample_size": True,
            "drift_assessable": True,
            "feature_alerts": [],
            "prediction_alert": True,
        },
        {
            "sample_size": 20,
            "drift_assessable": True,
            "feature_alerts": [1],
            "prediction_alert": False,
        },
        {"sample_size": 20, "drift_assessable": 1, "feature_alerts": [], "prediction_alert": False},
    ],
)
def test_invalid_monitoring_summary_is_rejected(summary):
    with pytest.raises(RetrainingError, match="Monitoring summary"):
        evaluate_trigger(load_policy(POLICY), drift_summary=summary)


def test_candidate_is_accepted_at_both_frozen_promotion_boundaries():
    policy = load_policy(POLICY)
    incumbent = policy["incumbent"]
    rmse_boundary = incumbent["test_rmse_cycles"] * 0.99
    result = evaluate_candidate(
        policy,
        _candidate(policy, rmse=rmse_boundary, nasa=incumbent["test_mean_nasa_score"]),
    )
    assert result["accepted"] is True
    assert result["test_engine_count"] == 707


def test_candidate_rejected_if_rmse_gain_or_asymmetric_score_gate_fails():
    policy = load_policy(POLICY)
    incumbent = policy["incumbent"]
    result = evaluate_candidate(
        policy,
        _candidate(
            policy,
            rmse=incumbent["test_rmse_cycles"],
            nasa=incumbent["test_mean_nasa_score"] * 1.01,
        ),
    )
    assert result["accepted"] is False
    assert result["reasons"] == [
        "minimum_test_rmse_improvement_not_met",
        "asymmetric_nasa_score_regressed",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"engine_count": 706},
        {"manifest": "b" * 64},
    ],
)
def test_candidate_with_mismatched_test_population_or_data_is_rejected(kwargs):
    policy = load_policy(POLICY)
    with pytest.raises(RetrainingError, match="Candidate data"):
        evaluate_candidate(policy, _candidate(policy, **kwargs))


def test_candidate_metric_booleans_and_infinite_values_are_rejected():
    policy = load_policy(POLICY)
    metrics = _candidate(policy)
    metrics["test"]["overall"]["rmse"] = True
    with pytest.raises(RetrainingError, match="finite nonnegative"):
        evaluate_candidate(policy, metrics)
    metrics["test"]["overall"]["rmse"] = float("inf")
    with pytest.raises(RetrainingError, match="finite nonnegative"):
        evaluate_candidate(policy, metrics)


def test_downloaded_candidate_inventory_and_hashes_are_verified(tmp_path):
    policy = load_policy(POLICY)
    output = tmp_path / "candidate"
    output.mkdir()
    model_bytes = b"model-content"
    metrics = _candidate(policy)
    metrics["model"]["sha256"] = hashlib.sha256(model_bytes).hexdigest()
    provenance = {"ml_ready_manifest_sha256": policy["training_asset"]["manifest_sha256"]}
    values = {
        "evaluation.md": b"evaluation",
        "feature-importance.json": b"{}",
        "metrics.json": json.dumps(metrics).encode(),
        "model.json": model_bytes,
        "predictions_test.parquet": b"test-predictions",
        "predictions_validation.parquet": b"validation-predictions",
        "run-metadata.json": json.dumps({"provenance": provenance}).encode(),
    }
    for name, content in values.items():
        (output / name).write_bytes(content)
    manifest = {
        "schema_version": 1,
        "artifact_type": "xgboost-rul-baseline",
        "completion_marker": "artifact-manifest.json",
        "provenance": provenance,
        "files": [
            {
                "path": name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
            for name, content in sorted(values.items())
        ],
    }
    (output / "artifact-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha256 = hashlib.sha256((output / "artifact-manifest.json").read_bytes()).hexdigest()
    verified = _verified_candidate(output, policy, manifest_sha256)
    assert verified["model"]["sha256"] == hashlib.sha256(model_bytes).hexdigest()
    (output / "metrics.json").write_text(json.dumps(metrics) + "\n", encoding="utf-8")
    with pytest.raises(RetrainingError, match="checksum"):
        _verified_candidate(output, policy, manifest_sha256)


def test_run_skips_without_trigger_and_requires_explicit_cost_approval(tmp_path):
    policy = load_policy(POLICY)
    skipped = run_retraining(
        policy,
        {"triggered": False, "reasons": []},
        config=None,
        approve_costs=False,
        project_root=tmp_path,
    )
    assert skipped["status"] == "skipped"
    with pytest.raises(RetrainingError, match="approve-costs"):
        run_retraining(
            policy,
            {"triggered": True, "reasons": ["serving_drift_alert"]},
            config=None,
            approve_costs=False,
            project_root=tmp_path,
        )


def test_resume_refuses_a_stale_baseline_job_receipt(tmp_path):
    policy = load_policy(POLICY)
    azure_state = tmp_path / ".azure"
    azure_state.mkdir()
    old_job = "epm-baseline-de82ea3141be"
    (azure_state / "baseline-job.json").write_text(
        json.dumps({"job_name": old_job, "status": "Starting"}),
        encoding="utf-8",
    )
    (azure_state / "retraining-state.json").write_text(
        json.dumps(
            {
                "status": "submission_unknown",
                "policy_sha256": _policy_digest(policy),
                "previous_baseline_job_name": old_job,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RetrainingError, match="unrelated baseline job"):
        resume_retraining(policy, config=None, project_root=tmp_path)


def test_plan_command_is_local_and_emits_trigger_decision(tmp_path, capsys):
    drift_path = tmp_path / "monitor.json"
    drift_path.write_text(
        json.dumps(
            {
                "sample_size": 20,
                "drift_assessable": True,
                "feature_alerts": [],
                "prediction_alert": True,
            }
        ),
        encoding="utf-8",
    )
    from epm_platform.retraining import main

    assert main(["plan", "--drift-summary", str(drift_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "triggered"
    assert result["reasons"] == ["serving_drift_alert"]
