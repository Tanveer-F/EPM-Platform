"""Fake-only MLflow tests also run without Torch, MLflow, credentials or a tracking server."""

import hashlib
import inspect
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from epm_platform.deep_learning import tracking

ROOT = Path(__file__).absolute().parents[2]
SECRET = "https://private.invalid/?sig=credential-sensitive"
URI = "azureml://example.invalid/mlflow/v1.0/workspaces/approved"
RUN_ID = "epm-pytorch-012345abcdef"


@pytest.fixture
def workspace():
    path = ROOT / (".pytorch-tracking-test-" + uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


@pytest.fixture(autouse=True)
def clean_tracking_environment(monkeypatch):
    for key in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_RUN_ID",
        "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING",
        "MLFLOW_ENABLE_ASYNC_LOGGING",
        "MLFLOW_ENABLE_TELEMETRY",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mlflow(monkeypatch):
    module = Mock(
        spec=[
            "active_run",
            "get_tracking_uri",
            "start_run",
            "log_params",
            "set_tags",
            "log_metrics",
            "log_artifacts",
        ]
    )
    module.active_run.return_value = None
    module.get_tracking_uri.return_value = URI
    module.finished = []
    run = SimpleNamespace(info=SimpleNamespace(run_id=RUN_ID))

    @contextmanager
    def start_run(**kwargs):
        assert module.active_run() is None
        module.active_run.return_value = run
        try:
            yield run
        except BaseException:
            module.finished.append("FAILED")
            raise
        else:
            module.finished.append("FINISHED")
        finally:
            module.active_run.return_value = None

    module.start_run.side_effect = start_run
    monkeypatch.setattr(tracking, "_load_mlflow", lambda: module)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", URI)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def artifact_bundle(root):
    root.mkdir()
    metrics = {"model": {"best_epoch": 2, "epochs_run": 3, "best_validation_rmse": 21.5}}
    for name in tracking._FILES - {"artifact-manifest.json"}:
        (root / name).write_bytes((name + "\n").encode())
    write_json(root / "metrics.json", metrics)
    manifest = {
        "schema_version": 1,
        "artifact_type": "pytorch-mlp-rul",
        "completion_marker": "artifact-manifest.json",
        "model_sha256": tracking._sha256(root / "model.pt"),
        "files": [
            {"path": path.name, "sha256": tracking._sha256(path), "size_bytes": path.stat().st_size}
            for path in sorted(root.iterdir())
        ],
    }
    write_json(root / "artifact-manifest.json", manifest)
    return metrics


@pytest.fixture
def live_inputs(workspace, monkeypatch):
    data = workspace / "data"
    data.mkdir()
    write_json(data / "manifest.json", {"verified": "synthetic feature manifest"})
    digest = hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest()
    monkeypatch.setattr(tracking, "MANIFEST_SHA256", digest)
    config, reference = workspace / "pytorch.json", workspace / "baseline-reference.json"
    write_json(config, {"device": "cpu", "threads": 2, "max_epochs": 100})
    write_json(reference, {"ml_ready_manifest_sha256": digest})
    return SimpleNamespace(
        data=data, config=config, reference=reference, output=workspace / "output"
    )


def train(inputs):
    return tracking.train_with_tracking(
        inputs.data, inputs.config, inputs.output, baseline_reference_path=inputs.reference
    )


@pytest.fixture
def fake_training(monkeypatch):
    def core(data, config, output, *, tracker, baseline_reference_path):
        tracker.log_parameters(
            {
                "device": "cpu",
                "threads": 2,
                "max_epochs": 100,
                "hidden_sizes": [64, 32],
                "source": {
                    "asset_name": "epm-cmapss-curated",
                    "asset_version": "frozen-source",
                    "manifest_sha256": "a" * 64,
                },
                "feature_config_sha256": "b" * 64,
                "pytorch_config_sha256": "c" * 64,
            }
        )
        tracker.log_epoch({"train_loss": 0.12, "train_rmse": 12.5, "val_rmse": 21.5}, step=1)
        tracker.log_final(
            {
                "validation.overall.rmse": 21.5,
                "test.overall.rmse": 22.0,
                "test.FD001.rmse": 20.0,
                "test.FD001.mae": 14.0,
            }
        )
        metrics = artifact_bundle(output)
        tracker.log_artifacts(output)
        return metrics

    function = Mock(side_effect=core)
    monkeypatch.setattr(tracking, "_load_train", lambda: function)
    return function


def test_long_provenance_is_fingerprinted_for_azure_parameter_limit(mlflow):
    value = "x" * 501
    with mlflow.start_run():
        tracking.MLflowTracker(mlflow).log_parameters({"manifest_details": value, "device": "cpu"})
    parameters = mlflow.log_params.call_args.args[0]
    assert "manifest_details" not in parameters
    assert parameters["manifest_details_sha256"] == hashlib.sha256(value.encode()).hexdigest()
    assert parameters["device"] == "cpu"
    assert all(len(str(item)) <= 500 for item in parameters.values())


def test_fingerprint_parameter_collision_is_rejected(mlflow):
    with mlflow.start_run():
        with pytest.raises(tracking.TrackingError, match="conflicts"):
            tracking.MLflowTracker(mlflow).log_parameters(
                {"item": "x" * 501, "item_sha256": "existing"}
            )
    mlflow.log_params.assert_not_called()


def test_join_azure_run_flat_params_epoch_metrics_and_artifacts_once(
    mlflow,
    fake_training,
    live_inputs,
    monkeypatch,
):
    monkeypatch.setenv("MLFLOW_RUN_ID", RUN_ID)
    result = train(live_inputs)
    assert result == {
        "status": "tracked",
        "run_id": RUN_ID,
        "best_epoch": 2,
        "artifacts_logged": 11,
    }
    mlflow.start_run.assert_called_once_with(run_id=RUN_ID, log_system_metrics=False)
    assert mlflow.finished == ["FINISHED"]
    params = mlflow.log_params.call_args_list
    assert params[0].args[0]["ml_ready_asset_name"] == tracking.ASSET_NAME
    assert params[0].args[0]["ml_ready_manifest_sha256"] == tracking.MANIFEST_SHA256
    assert params[0].args[0]["baseline_reference_sha256"] == tracking._sha256(live_inputs.reference)
    assert params[1].args[0]["hidden_sizes"] == "[64, 32]"
    assert params[1].args[0]["source.asset_name"] == "epm-cmapss-curated"
    assert all(call.kwargs == {"synchronous": True} for call in params)
    epoch, final, selected = mlflow.log_metrics.call_args_list
    assert epoch.args[0] == {"train_loss": 0.12, "train_rmse": 12.5, "val_rmse": 21.5}
    assert epoch.kwargs == {"step": 1, "synchronous": True}
    assert final.args[0] == {
        "validation_rmse": 21.5,
        "test_rmse": 22.0,
        "test_fd001_rmse": 20.0,
        "test_fd001_mae": 14.0,
    }
    assert selected.args[0] == {"best_epoch": 2, "epochs_run": 3, "best_validation_rmse": 21.5}
    assert final.kwargs == selected.kwargs == {"synchronous": True}
    tags = mlflow.set_tags.call_args.args[0]
    assert tags["source_manifest_sha256"] == "a" * 64
    assert tags["feature_config_sha256"] == "b" * 64
    mlflow.log_artifacts.assert_called_once_with(str(live_inputs.output), artifact_path="pytorch")
    assert {path.name for path in live_inputs.output.iterdir()} == tracking._FILES
    assert all(
        os.environ[key] == "false"
        for key in (
            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING",
            "MLFLOW_ENABLE_ASYNC_LOGGING",
            "MLFLOW_ENABLE_TELEMETRY",
        )
    )


def test_start_one_run_without_nested_experiment(mlflow, fake_training, live_inputs):
    train(live_inputs)
    mlflow.start_run.assert_called_once_with(log_system_metrics=False)
    assert mlflow.finished == ["FINISHED"]


def test_respect_existing_run_without_ending_it(mlflow, fake_training, live_inputs, monkeypatch):
    monkeypatch.setenv("MLFLOW_RUN_ID", RUN_ID)
    mlflow.active_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id=RUN_ID))
    train(live_inputs)
    mlflow.start_run.assert_not_called()
    assert mlflow.finished == []


def test_conflicting_active_run_fails_closed(mlflow, fake_training, live_inputs, monkeypatch):
    monkeypatch.setenv("MLFLOW_RUN_ID", RUN_ID)
    mlflow.active_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="other-run"))
    with pytest.raises(tracking.TrackingError):
        train(live_inputs)
    fake_training.assert_not_called()
    mlflow.log_params.assert_not_called()
    mlflow.start_run.assert_not_called()


@pytest.mark.parametrize(
    "uri",
    [
        None,
        "",
        "file:./mlruns",
        "http://localhost:5000",
        "azureml://",
        URI + "?sig=secret",
        "azureml://user:pass@host/path",
    ],
)
def test_cli_tracking_never_falls_back_to_local(
    uri, mlflow, fake_training, live_inputs, monkeypatch
):
    if uri is None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    else:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.get_tracking_uri.return_value = uri
    with pytest.raises(tracking.TrackingError):
        train(live_inputs)
    fake_training.assert_not_called()
    mlflow.start_run.assert_not_called()


@pytest.mark.parametrize("target", ["data", "reference"])
def test_unapproved_feature_digest_stops_before_run(
    target,
    live_inputs,
    mlflow,
    fake_training,
):
    path = live_inputs.data / "manifest.json" if target == "data" else live_inputs.reference
    write_json(path, {"ml_ready_manifest_sha256": "0" * 64})
    with pytest.raises(tracking.TrackingError):
        train(live_inputs)
    mlflow.start_run.assert_not_called()
    fake_training.assert_not_called()


@pytest.mark.parametrize("method", ["log_params", "set_tags", "log_metrics", "log_artifacts"])
def test_synchronous_failures_mark_owned_run_failed(
    method,
    mlflow,
    fake_training,
    live_inputs,
):
    getattr(mlflow, method).side_effect = RuntimeError(SECRET)
    with pytest.raises(tracking.TrackingError) as error:
        train(live_inputs)
    assert SECRET not in str(error.value)
    assert error.value.__suppress_context__
    assert mlflow.finished == ["FAILED"]
    if method == "log_artifacts":
        assert {path.name for path in live_inputs.output.iterdir()} == tracking._FILES


def test_wrapper_finalizes_artifacts_if_core_omits_callback(
    mlflow,
    live_inputs,
    monkeypatch,
):
    monkeypatch.setattr(
        tracking, "_load_train", lambda: lambda *args, **kwargs: artifact_bundle(live_inputs.output)
    )
    train(live_inputs)
    mlflow.log_artifacts.assert_called_once()
    assert mlflow.finished == ["FINISHED"]


@pytest.mark.parametrize("case", ["extra", "missing", "checksum", "traversal", "duplicate"])
def test_artifacts_require_complete_immutable_manifest(case, workspace, mlflow):
    root = workspace / "artifacts"
    artifact_bundle(root)
    if case == "extra":
        (root / "secret.txt").write_text(SECRET)
    elif case == "missing":
        (root / "model.pt").unlink()
    elif case == "checksum":
        (root / "model.pt").write_text("corrupted")
    else:
        manifest = tracking._read_json(root / "artifact-manifest.json")
        if case == "traversal":
            manifest["files"][0]["path"] = "../secret.txt"
        else:
            manifest["files"][1] = manifest["files"][0]
        write_json(root / "artifact-manifest.json", manifest)
    mlflow.active_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id=RUN_ID))
    with pytest.raises(tracking.TrackingError):
        tracking.MLflowTracker(mlflow).log_artifacts(root)
    mlflow.log_artifacts.assert_not_called()


def test_tracker_never_implicitly_creates_run(mlflow):
    with pytest.raises(tracking.TrackingError):
        tracking.MLflowTracker(mlflow).log_epoch({"train_loss": 1.0}, step=1)
    mlflow.log_metrics.assert_not_called()
    mlflow.start_run.assert_not_called()


@pytest.mark.parametrize(
    "metrics,step",
    [
        ({"train_loss": float("nan")}, 1),
        ({"train_loss": float("inf")}, 1),
        ({"train_loss": True}, 1),
        ({"train_loss": 0.1}, 101),
        ({"train_loss": 0.1}, 0),
    ],
)
def test_invalid_epoch_tracking_fails_closed(metrics, step, mlflow):
    with pytest.raises(tracking.TrackingError):
        tracking.MLflowTracker(mlflow).log_epoch(metrics, step)
    mlflow.log_metrics.assert_not_called()


def test_artifacts_cannot_be_logged_twice(workspace, mlflow):
    root = workspace / "artifacts"
    artifact_bundle(root)
    mlflow.active_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id=RUN_ID))
    tracker = tracking.MLflowTracker(mlflow)
    tracker.log_artifacts(root)
    with pytest.raises(tracking.TrackingError):
        tracker.log_artifacts(root)
    mlflow.log_artifacts.assert_called_once()


def test_cli_suppresses_connector_urls_credentials_and_raw_errors(monkeypatch, capsys):
    def fail(*args, **kwargs):
        print(SECRET)
        raise RuntimeError(SECRET)

    monkeypatch.setattr(tracking, "train_with_tracking", fail)
    assert (
        tracking.main(
            [
                "--data",
                "data",
                "--config",
                "config",
                "--output",
                "output",
                "--baseline-reference",
                "reference",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    assert json.loads(captured.out)["code"] == "pytorch_tracking_failed"


def test_installed_mlflow_supports_explicit_synchronous_api(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_TELEMETRY", "false")
    mlflow = pytest.importorskip("mlflow")
    for function in (mlflow.log_params, mlflow.log_metrics, mlflow.set_tags):
        assert "synchronous" in inspect.signature(function).parameters
    assert "log_system_metrics" in inspect.signature(mlflow.start_run).parameters
    assert "artifact_path" in inspect.signature(mlflow.log_artifacts).parameters
