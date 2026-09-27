"""CPU-only synthetic feature bundles; never train the real dataset or call cloud services."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from epm_platform.baseline import training as baseline  # noqa: E402
from epm_platform.deep_learning import training  # noqa: E402
from epm_platform.features import pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def workspace():
    root = ROOT / f".pytorch-test-{uuid.uuid4().hex}"
    root.mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root)


@pytest.fixture
def config_path(workspace):
    config = json.loads((ROOT / "config" / "pytorch.json").read_text())
    config.update(max_epochs=3, patience=2)
    path = workspace / "pytorch.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.fixture
def make_dataset(workspace, monkeypatch):
    from epm_platform.data.validation import LABEL_SCHEMA, OBSERVATION_SCHEMA

    roots = set()

    def verify_synthetic_source(root, spec_path, config):
        # Substitute only NASA-source provenance; feature verification is always real.
        assert root in roots
        assert spec_path == ROOT / "config" / "cmapss-source.json"
        assert json.loads((root / "manifest.json").read_text()) == {"synthetic_test_only": True}
        assert config == json.loads((ROOT / "config" / "features.json").read_text())

    monkeypatch.setattr(pipeline, "_verify_source", verify_synthetic_source)

    def engine(unit, life, offset, future_cut=None, future_offset=0):
        values = {"unit_id": [unit] * life, "cycle": list(range(1, life + 1))}
        for index, name in enumerate(OBSERVATION_SCHEMA.names[2:], 1):
            values[name] = [
                float(offset + unit / 4 + cycle * index / 20)
                + (future_offset if future_cut is not None and cycle > future_cut else 0)
                for cycle in range(1, life + 1)
            ]
        values["sensor_01"] = [7.0] * life
        return pa.Table.from_pydict(values, schema=OBSERVATION_SCHEMA)

    def build(name="original", test_offset=0, validation_offset=0, future_offset=0):
        source = workspace / f"source-{name}"
        source.mkdir()
        roots.add(source)
        (source / "manifest.json").write_text('{"synthetic_test_only": true}')
        for index, subset in enumerate(training.SUBSETS):
            folder = source / subset
            folder.mkdir()
            heldout = pipeline._holdout(subset, list(range(1, 6)))
            engines = []
            for unit, life in enumerate((6, 8, 10, 12, 14), 1):
                cut = pipeline._cut(subset, unit, life) if unit in heldout else None
                offset = index + (validation_offset if unit in heldout else 0)
                engines.append(engine(unit, life, offset, cut, future_offset))
            pq.write_table(pa.concat_tables(engines), folder / "train.parquet")
            pq.write_table(
                pa.concat_tables(
                    [engine(1, 4, index + test_offset), engine(2, 5, index + test_offset)]
                ),
                folder / "test.parquet",
            )
            pq.write_table(
                pa.Table.from_pydict(
                    {"unit_id": [1, 2], "rul": [20 + index + test_offset, 3 + test_offset]},
                    schema=LABEL_SCHEMA,
                ),
                folder / "test_rul.parquet",
            )
        bundle = pipeline.build_features(
            source,
            workspace / f"features-{name}",
            ROOT / "config" / "features.json",
            ROOT / "config" / "cmapss-source.json",
        )
        assert pipeline.verify_features(bundle.root)["feature_columns"] == list(
            training.FEATURE_COLUMNS
        )
        return bundle.root

    return build


def _partition(root, split="train"):
    manifest = pipeline.verify_features(root)
    return baseline._load_partition(root, split, training.FEATURE_COLUMNS, manifest)


@pytest.fixture
def completed(make_dataset, config_path, workspace):
    data = make_dataset()
    output = workspace / "output"
    metrics = training.train_pytorch(data, config_path, output)
    return data, output, metrics


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_path(workspace, metrics):
    baseline_metrics = copy.deepcopy(metrics)
    baseline_metrics["model"]["configuration"] = json.loads(
        (ROOT / "config" / "baseline.json").read_text()
    )
    original_bytes = json.dumps(baseline_metrics).encode()
    reference = {
        "schema_version": 1,
        "job_name": "epm-baseline-de82ea3141be",
        "target": "uncapped_rul",
        "ml_ready_manifest_sha256": metrics["provenance"]["ml_ready_manifest_sha256"],
        "metrics": baseline_metrics,
        "metrics_sha256": hashlib.sha256(original_bytes).hexdigest(),
    }
    path = workspace / "baseline-reference.json"
    path.write_text(json.dumps(reference))
    return path


class FakeTracker:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def _call(self, method, payload):
        self.calls.append((method, payload))
        if self.fail == method:
            raise RuntimeError("synthetic tracking failure")

    def log_parameters(self, parameters):
        self._call("parameters", parameters)

    def log_epoch(self, metrics, step):
        self._call("epoch", (step, metrics))

    def log_final(self, metrics):
        self._call("final", metrics)

    def log_artifacts(self, output_dir):
        assert {path.name for path in output_dir.iterdir()} == training._FILES
        self._call("artifacts", output_dir)


def test_architecture_exact_allowlist_and_finite_output():
    model = training.MLP()
    layers = list(model.network)
    assert [
        (layer.in_features, layer.out_features)
        for layer in layers
        if isinstance(layer, torch.nn.Linear)
    ] == [(35, 64), (64, 32), (32, 1)]
    assert sum(isinstance(layer, torch.nn.ReLU) for layer in layers) == 2
    assert [layer.p for layer in layers if isinstance(layer, torch.nn.Dropout)] == [0.1, 0.1]
    assert training.FEATURE_COLUMNS == pipeline.FEATURE_COLUMNS
    assert not set(training.FEATURE_COLUMNS) & {
        "unit_id",
        "rul",
        "split",
        "subset",
        "sample_weight",
    }
    assert "cycle" in training.FEATURE_COLUMNS
    prediction = model(torch.zeros(7, 35))
    assert prediction.shape == (7,)
    assert torch.isfinite(prediction).all()


def test_weighted_train_only_population_statistics(make_dataset):
    train = _partition(make_dataset())
    preprocessing = training.fit_preprocessing(train)
    mean = np.average(train.features, axis=0, weights=train.weights)
    variance = np.average((train.features - mean) ** 2, axis=0, weights=train.weights)
    np.testing.assert_allclose(preprocessing["feature_mean"], mean)
    nonconstant = variance > training._VARIANCE_EPSILON
    np.testing.assert_allclose(
        np.array(preprocessing["feature_scale"])[nonconstant], np.sqrt(variance[nonconstant])
    )
    assert preprocessing["target_mean"] == pytest.approx(
        np.average(train.target, weights=train.weights)
    )
    assert preprocessing["target_mean"] != pytest.approx(train.target.mean())
    assert preprocessing["fit_partition"] == "train"
    constant_index = training.FEATURE_COLUMNS.index("sensor_01")
    assert preprocessing["feature_scale"][constant_index] == 1
    assert "sensor_01" in preprocessing["zero_variance_columns"]
    assert len(preprocessing["feature_scale"]) == 35
    assert training._features_tensor(train.features, preprocessing).dtype == torch.float32


def test_validation_test_and_future_perturbations_cannot_fit_statistics(make_dataset):
    original = make_dataset()
    changed = make_dataset("changed", test_offset=100, validation_offset=1000, future_offset=2000)
    assert training.fit_preprocessing(_partition(original)) == training.fit_preprocessing(
        _partition(changed)
    )
    future = make_dataset("future", future_offset=5000)
    np.testing.assert_array_equal(
        _partition(original, "validation").features, _partition(future, "validation").features
    )


def test_preprocessing_refuses_validation_and_constant_target_scale_one(make_dataset):
    root = make_dataset()
    with pytest.raises(training.TrainingError, match="only fit"):
        training.fit_preprocessing(_partition(root, "validation"))
    train = _partition(root)
    train.target[:] = 3
    train.features[:, 0] = 4 + np.arange(len(train.target)) * 1e-12
    preprocessing = training.fit_preprocessing(train)
    assert preprocessing["target_scale"] == 1
    assert preprocessing["target_zero_variance"] is True
    assert preprocessing["feature_scale"][0] == 1


def test_engine_weights_normalized_once_and_weighted_mse(make_dataset):
    train = _partition(make_dataset())
    weights = train.weights / train.weights.mean()
    assert weights.mean() == pytest.approx(1)
    totals = [
        weights[np.array([engine == chosen for engine in train.engines])].sum()
        for chosen in set(train.engines)
    ]
    np.testing.assert_allclose(totals, totals[0])
    prediction = torch.tensor([2.0, 5.0], requires_grad=True)
    loss = training.weighted_mse(prediction, torch.tensor([1.0, 1.0]), torch.tensor([0.5, 2.0]))
    assert loss.item() == pytest.approx((0.5 + 32) / 2)
    loss.backward()
    np.testing.assert_allclose(prediction.grad.numpy(), [0.5, 8])
    with pytest.raises(training.TrainingError):
        training.weighted_mse(prediction[:, None], prediction, torch.ones(2))


def test_metrics_are_the_baseline_function_with_asymmetric_nasa():
    assert training.regression_metrics is baseline.regression_metrics
    actual, prediction = [10, 20, 30], [8, 20, 33]
    assert training.regression_metrics(actual, prediction) == baseline.regression_metrics(
        actual, prediction
    )
    assert (
        training.regression_metrics([20], [30])["nasa_score"]
        > training.regression_metrics([20], [10])["nasa_score"]
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("input_features", 34),
        ("hidden_sizes", [64, 32.0]),
        ("dropout", 0.2),
        ("device", "cuda"),
        ("seed", 43),
        ("threads", True),
        ("batch_size", 64),
        ("learning_rate", 0.01),
        ("weight_decay", 0),
        ("loss", "mse"),
        ("gradient_clip_norm", 0),
        ("preprocessing", "all_partitions"),
        ("prediction_floor", -1),
        ("target", "capped_rul"),
        ("schema_version", True),
        ("max_epochs", 101),
        ("max_epochs", 0),
        ("max_epochs", 3.0),
        ("patience", 0),
        ("patience", 13),
        ("sweep", {}),
    ],
)
def test_invalid_config_is_closed(config_path, key, value):
    config = json.loads(config_path.read_text())
    config[key] = value
    config_path.write_text(json.dumps(config))
    with pytest.raises(training.TrainingError, match="approved CPU recipe"):
        training.load_config(config_path)


def test_config_duplicate_keys_and_missing_fields(config_path):
    config_path.write_text('{"seed":42,"seed":42}')
    with pytest.raises(training.TrainingError):
        training.load_config(config_path)
    config_path.write_text("{}")
    with pytest.raises(training.TrainingError):
        training.load_config(config_path)


def test_approved_production_config():
    config = training.load_config(ROOT / "config" / "pytorch.json")
    assert config["max_epochs"] == 100
    assert config["patience"] == 12


def test_bundle_exact_inventory_safe_reload_and_metrics(completed, monkeypatch):
    data, output, metrics = completed
    assert len(list(output.iterdir())) == 11
    assert {path.name for path in output.iterdir()} == training._FILES
    manifest = json.loads((output / "artifact-manifest.json").read_text())
    assert len(manifest["files"]) == 10
    for entry in manifest["files"]:
        assert (output / entry["path"]).stat().st_size == entry["size_bytes"]
        assert _hash(output / entry["path"]) == entry["sha256"]
    assert manifest["model_sha256"] == metrics["model"]["sha256"] == _hash(output / "model.pt")
    assert manifest["provenance"]["ml_ready_manifest_sha256"] == _hash(data / "manifest.json")
    real_load = torch.load
    calls = []

    def safe_load(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", safe_load)
    loaded = training.load_model(output)
    assert loaded.model.training is False
    assert all(parameter.device.type == "cpu" for parameter in loaded.model.parameters())
    for split in ("validation", "test"):
        partition = _partition(data, split)
        saved = pq.read_table(output / f"predictions_{split}.parquet")
        prediction = training.predict_saved(partition.table, output)
        np.testing.assert_array_equal(prediction, saved["prediction"].to_numpy())
        assert np.all(prediction >= 0)
        assert metrics[split] == baseline._partition_metrics(partition, prediction)
    assert all(call == {"weights_only": True, "map_location": "cpu"} for call in calls)
    metadata = json.loads((output / "run-metadata.json").read_text())
    assert metadata["test_evaluation_count"] == 1
    assert metadata["model_frozen_before_test"] is True
    assert metadata["runtime"]["threads"] == 2
    assert metadata["runtime"]["num_workers"] == 0
    assert metadata["runtime"]["deterministic_algorithms"] is True
    assert json.loads((output / "comparison.json").read_text())["status"] == "not_provided"


def test_deterministic_training_and_test_independence(
    completed, make_dataset, config_path, workspace
):
    _, output, metrics = completed
    changed = make_dataset("changed", test_offset=30)
    second = workspace / "second"
    other = training.train_pytorch(changed, config_path, second)
    assert metrics["validation"] == other["validation"]
    assert (output / "preprocessing.json").read_bytes() == (
        second / "preprocessing.json"
    ).read_bytes()
    assert (output / "training-history.json").read_bytes() == (
        second / "training-history.json"
    ).read_bytes()
    first_state = training.load_model(output).model.state_dict()
    second_state = training.load_model(second).model.state_dict()
    for key in first_state:
        assert torch.equal(first_state[key], second_state[key])
    assert metrics["test"] != other["test"]


def test_best_checkpoint_restored_and_saved_before_single_test_read(
    make_dataset, config_path, workspace, monkeypatch
):
    data = make_dataset()
    config = json.loads(config_path.read_text())
    config["max_epochs"] = 6
    config_path.write_text(json.dumps(config))
    states = []
    scores = iter([0.0, 1.0, 2.0])

    def validation_score(model, *_):
        states.append({key: value.clone() for key, value in model.state_dict().items()})
        return next(scores)

    real_partition = baseline._load_partition
    reads = []

    def partition(root, split, *args):
        reads.append(split)
        if split == "test":
            paths = list((workspace / "output").glob(".staging-*"))
            assert len(paths) == 1
            checkpoint = torch.load(paths[0] / "model.pt", weights_only=True, map_location="cpu")
            assert all(torch.equal(value, states[0][key]) for key, value in checkpoint.items())
        return real_partition(root, split, *args)

    monkeypatch.setattr(training, "_validation_rmse", validation_score)
    monkeypatch.setattr(baseline, "_load_partition", partition)
    metrics = training.train_pytorch(data, config_path, workspace / "output")
    assert metrics["model"]["best_epoch"] == 1
    assert metrics["model"]["epochs_run"] == 3
    assert reads == ["train", "validation", "test"]
    saved = training.load_model(workspace / "output").model.state_dict()
    assert all(torch.equal(value, states[0][key]) for key, value in saved.items())
    assert any(not torch.equal(states[0][key], states[-1][key]) for key in saved)


def test_prediction_floor_and_missing_features(completed):
    data, output, _ = completed
    loaded = training.load_model(output)
    with torch.no_grad():
        loaded.model.network[-1].weight.zero_()
        loaded.model.network[-1].bias.fill_(-10000)
    np.testing.assert_array_equal(loaded.predict(np.zeros((2, 35))), np.zeros(2))
    with pytest.raises(training.TrainingError, match="missing required"):
        loaded.predict(_partition(data).table.drop(["sensor_01"]))
    with pytest.raises(training.TrainingError, match="35 feature"):
        loaded.predict(np.zeros((2, 34)))
    with pytest.raises(training.TrainingError, match="finite"):
        loaded.predict(np.full((2, 35), np.nan))


@pytest.mark.parametrize("change", ["missing_feature", "wrong_schema", "tamper"])
def test_invalid_input_fails_real_verification_before_fit(
    make_dataset, config_path, workspace, monkeypatch, change
):
    data = make_dataset()
    path = data / "train.parquet"
    table = pq.read_table(path)
    if change == "missing_feature":
        pq.write_table(table.drop(["sensor_01"]), path)
    elif change == "wrong_schema":
        manifest = json.loads((data / "manifest.json").read_text())
        manifest["feature_columns"][0] = "unit_id"
        (data / "manifest.json").write_text(json.dumps(manifest))
    else:
        path.write_bytes(path.read_bytes() + b"changed")

    def forbidden(*args):
        pytest.fail("Fitting must not run before input verification")

    monkeypatch.setattr(training, "_fit", forbidden)
    with pytest.raises(training.TrainingError):
        training.train_pytorch(data, config_path, workspace / "output")
    assert not list((workspace / "output").iterdir())


def test_existing_empty_output_allowed_but_completed_never_overwritten(
    make_dataset, config_path, workspace
):
    data = make_dataset()
    output = workspace / "output"
    output.mkdir()
    training.train_pytorch(data, config_path, output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(training.TrainingError, match="never overwritten"):
        training.train_pytorch(data, config_path, output)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_tracker_protocol_calls_complete_inventory(make_dataset, config_path, workspace):
    tracker = FakeTracker()
    metrics = training.train_pytorch(
        make_dataset(), config_path, workspace / "output", tracker=tracker
    )
    methods = [method for method, _ in tracker.calls]
    assert methods == ["parameters"] + ["epoch"] * metrics["model"]["epochs_run"] + ["final"]
    assert {path.name for path in (workspace / "output").iterdir()} == training._FILES
    tracker.log_artifacts(workspace / "output")
    assert [method for method, _ in tracker.calls].count("artifacts") == 1
    epochs = [payload for method, payload in tracker.calls if method == "epoch"]
    assert [step for step, _ in epochs] == list(range(1, len(epochs) + 1))
    assert all(set(payload) == {"train_loss", "train_rmse", "val_rmse"} for _, payload in epochs)
    final = dict(tracker.calls)["final"]
    assert final["test.overall.rmse"] == metrics["test"]["overall"]["rmse"]
    assert (
        final["validation.FD001.nasa_score"]
        == metrics["validation"]["per_subset"]["FD001"]["nasa_score"]
    )
    assert all(isinstance(value, (float, int)) for value in final.values())


@pytest.mark.parametrize("method", ["parameters", "epoch", "final"])
def test_tracker_failure_is_loud_and_not_claimed_success(
    make_dataset, config_path, workspace, method
):
    tracker = FakeTracker(fail=method)
    output = workspace / "output"
    with pytest.raises(training.TrainingError, match="Tracking failed"):
        training.train_pytorch(make_dataset(), config_path, output, tracker=tracker)
    assert tracker.calls[-1][0] == method
    assert list(output.iterdir()) == []


def test_comparison_uses_exact_reference_and_all_metrics(completed, config_path, workspace):
    data, _, metrics = completed
    reference_path = _reference_path(workspace, metrics)
    output = workspace / "compared"
    training.train_pytorch(data, config_path, output, baseline_reference_path=reference_path)
    comparison = json.loads((output / "comparison.json").read_text())
    assert comparison["status"] == "compared"
    assert comparison["baseline_reference_file_sha256"] == _hash(reference_path)
    for split in ("validation", "test"):
        for record in [comparison[split]["overall"], *comparison[split]["per_subset"].values()]:
            for name in training._METRIC_NAMES:
                if record[name]["pytorch"] is None:
                    assert record[name]["delta_pytorch_minus_baseline"] is None
                    assert record[name]["improved"] is None
                else:
                    assert record[name]["delta_pytorch_minus_baseline"] == 0
                    assert record[name]["improved"] is False
            assert record["bias"]["better"] == "lower_absolute"
            assert record["r2"]["better"] == "higher"


@pytest.mark.parametrize(
    "change", ["source", "target", "test_count", "subset_count", "nested_source", "features"]
)
def test_comparison_rejects_unfair_reference_before_fit(
    completed, config_path, workspace, monkeypatch, change
):
    data, _, metrics = completed
    path = _reference_path(workspace, metrics)
    reference = json.loads(path.read_text())
    if change == "source":
        reference["ml_ready_manifest_sha256"] = "0" * 64
    elif change == "target":
        reference["target"] = "capped_rul"
    elif change == "test_count":
        reference["metrics"]["engine_counts"]["test"] += 1
    elif change == "subset_count":
        reference["metrics"]["test"]["per_subset"]["FD001"]["count"] += 1
    elif change == "nested_source":
        reference["metrics"]["provenance"]["ml_ready_manifest_sha256"] = "0" * 64
    else:
        reference["metrics"]["model"]["feature_columns"][0] = "unit_id"
    path.write_text(json.dumps(reference))

    def forbidden(*args):
        pytest.fail("Reference must be checked before fitting")

    monkeypatch.setattr(training, "_fit", forbidden)
    with pytest.raises(training.TrainingError, match="exact dataset"):
        training.train_pytorch(
            data, config_path, workspace / "unfair", baseline_reference_path=path
        )


def test_saved_artifact_tampering_fails_before_torch_load(completed, monkeypatch):
    _, output, _ = completed
    model_path = output / "model.pt"
    model_path.write_bytes(model_path.read_bytes() + b"tampered")

    def forbidden(*args, **kwargs):
        pytest.fail("Do not deserialize before verifying hashes")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(training.TrainingError, match="integrity"):
        training.load_model(output)


def test_nonfinite_training_fails_without_completion(
    make_dataset, config_path, workspace, monkeypatch
):
    def invalid_loss(prediction, *args):
        return prediction.sum() * float("nan")

    monkeypatch.setattr(training, "weighted_mse", invalid_loss)
    with pytest.raises(training.TrainingError, match="Nonfinite training loss"):
        training.train_pytorch(make_dataset(), config_path, workspace / "output")
    assert list((workspace / "output").iterdir()) == []


def test_optimizer_loader_and_gradient_clip_recipe(
    make_dataset, config_path, workspace, monkeypatch
):
    loader_calls, optimizer_calls, clip_calls = [], [], []
    real_loader = training.DataLoader
    real_optimizer = torch.optim.AdamW
    real_clip = torch.nn.utils.clip_grad_norm_

    def loader(dataset, **kwargs):
        loader_calls.append((dataset, kwargs))
        return real_loader(dataset, **kwargs)

    def optimizer(parameters, **kwargs):
        optimizer_calls.append(kwargs)
        return real_optimizer(parameters, **kwargs)

    def clip(parameters, maximum, **kwargs):
        clip_calls.append((maximum, kwargs))
        return real_clip(parameters, maximum, **kwargs)

    monkeypatch.setattr(training, "DataLoader", loader)
    monkeypatch.setattr(torch.optim, "AdamW", optimizer)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    training.train_pytorch(make_dataset(), config_path, workspace / "output")
    assert optimizer_calls == [{"lr": 0.001, "weight_decay": 0.0001}]
    dataset, settings = loader_calls[0]
    assert settings["batch_size"] == 512
    assert settings["shuffle"] is True
    assert settings["num_workers"] == 0
    assert settings["generator"].initial_seed() == 42
    assert "sampler" not in settings
    assert dataset.tensors[2].mean().item() == pytest.approx(1)
    assert clip_calls and all(call == (5.0, {"error_if_nonfinite": True}) for call in clip_calls)


@pytest.mark.parametrize("phase", ["parameter", "validation"])
def test_nonfinite_model_or_validation_fails(
    make_dataset, config_path, workspace, monkeypatch, phase
):
    if phase == "parameter":
        real_step = torch.optim.AdamW.step

        def step(optimizer, *args, **kwargs):
            result = real_step(optimizer, *args, **kwargs)
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].fill_(float("inf"))
            return result

        monkeypatch.setattr(torch.optim.AdamW, "step", step)
    else:
        monkeypatch.setattr(training, "_validation_rmse", lambda *args: float("nan"))
    with pytest.raises(training.TrainingError, match="Nonfinite"):
        training.train_pytorch(make_dataset(), config_path, workspace / "output")
    assert list((workspace / "output").iterdir()) == []


def test_publication_manifest_is_last_and_failure_rolls_back(
    make_dataset, config_path, workspace, monkeypatch
):
    data = make_dataset()
    output = workspace / "output"
    real_open = Path.open
    observed = []

    def open_path(path, mode="r", *args, **kwargs):
        if path.parent == output and mode == "xb":
            observed.append(path.name)
            if path.name == "artifact-manifest.json":
                assert set(observed[:-1]) == training._FILES - {"artifact-manifest.json"}
                raise OSError("synthetic final publication failure")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    with pytest.raises(training.TrainingError):
        training.train_pytorch(data, config_path, output)
    assert observed[-1] == "artifact-manifest.json"
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("invalid", ["missing_key", "dtype", "nonfinite", "wrong_shape"])
def test_safe_loader_rejects_invalid_state_even_with_updated_hash(completed, invalid):
    _, output, _ = completed
    model_path = output / "model.pt"
    state = torch.load(model_path, weights_only=True, map_location="cpu")
    key = next(iter(state))
    if invalid == "missing_key":
        del state[key]
    elif invalid == "dtype":
        state[key] = state[key].double()
    elif invalid == "nonfinite":
        state[key].fill_(float("nan"))
    else:
        state[key] = torch.zeros(1)
    torch.save(state, model_path)
    manifest_path = output / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_sha256"] = _hash(model_path)
    for entry in manifest["files"]:
        if entry["path"] == "model.pt":
            entry.update(sha256=_hash(model_path), size_bytes=model_path.stat().st_size)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(training.TrainingError, match="safe-loading"):
        training.load_model(output)


def test_core_imports_and_cli_do_not_require_cloud_clients():
    code = """
import importlib.abc
import sys
class BlockCloud(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mlflow", "azure"}:
            raise AssertionError("Core imported a cloud client")
sys.meta_path.insert(0, BlockCloud())
from epm_platform.deep_learning import training
assert not any(name == "mlflow" or name.startswith("azure.") for name in sys.modules)
assert training.main(["--help"]) is None
"""
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--baseline-reference" in result.stdout


def test_cli_trains_valid_synthetic_bundle(completed, config_path, workspace):
    data, _, _ = completed
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "epm_platform.deep_learning.training",
            "--data",
            str(data),
            "--config",
            str(config_path),
            "--output",
            str(workspace / "cli"),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (workspace / "cli" / "artifact-manifest.json").is_file()
