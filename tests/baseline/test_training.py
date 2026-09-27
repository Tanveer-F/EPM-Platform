"""Small CPU-only baseline tests; fixtures stay in the project, never system temp."""

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

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xgboost as xgb

from epm_platform.baseline import training

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def workspace():
    path = ROOT / f".baseline-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


@pytest.fixture
def configuration():
    config = json.loads((ROOT / "config" / "baseline.json").read_text(encoding="utf-8"))
    config["num_boost_round"] = 12
    config["early_stopping_rounds"] = 3
    return config


@pytest.fixture
def config_path(workspace, configuration):
    path = workspace / "baseline.json"
    path.write_text(json.dumps(configuration), encoding="utf-8")
    return path


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def make_dataset(workspace, monkeypatch):
    from epm_platform.data.validation import LABEL_SCHEMA, OBSERVATION_SCHEMA
    from epm_platform.features import pipeline

    roots = set()

    def verify_synthetic_source(root, spec_path, config):
        # Only the production NASA pin is substituted; real feature verification is untouched.
        assert root in roots
        assert spec_path == ROOT / "config" / "cmapss-source.json"
        assert json.loads((root / "manifest.json").read_text()) == {"synthetic_test_only": True}
        assert config == json.loads((ROOT / "config" / "features.json").read_text())

    monkeypatch.setattr(pipeline, "_verify_source", verify_synthetic_source)

    def engine(unit, life, offset):
        values = {"unit_id": [unit] * life, "cycle": list(range(1, life + 1))}
        for index, name in enumerate(OBSERVATION_SCHEMA.names[2:], 1):
            values[name] = [
                float(offset + unit / 4 + cycle * index / 20) for cycle in range(1, life + 1)
            ]
        return pa.Table.from_pydict(values, schema=OBSERVATION_SCHEMA)

    def build(name="original", test_offset=0):
        root = workspace / f"source-{name}"
        root.mkdir()
        roots.add(root)
        (root / "manifest.json").write_text('{"synthetic_test_only": true}', encoding="utf-8")
        for index, subset in enumerate(training.SUBSETS):
            folder = root / subset
            folder.mkdir()
            pq.write_table(
                pa.concat_tables(
                    [engine(unit, life, index) for unit, life in enumerate((6, 8, 10, 12, 14), 1)]
                ),
                folder / "train.parquet",
            )
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
            root,
            workspace / f"features-{name}",
            ROOT / "config" / "features.json",
            ROOT / "config" / "cmapss-source.json",
        )
        assert pipeline.verify_features(bundle.root)["feature_columns"] == list(
            training.FEATURE_COLUMNS
        )
        return bundle.root

    return build


def _rows(split):
    rows = []
    for subset_index, subset in enumerate(training.SUBSETS):
        units = (1, 2) if split == "train" else ((3,) if split == "validation" else (1,))
        for unit in units:
            length = 6 if unit == 1 else 4
            cycles = range(1, length + 1) if split == "train" else (3,)
            for cycle in cycles:
                row = {name: float(cycle + subset_index / 10) for name in training.FEATURE_COLUMNS}
                row.update(
                    {
                        "subset": subset,
                        "unit_id": unit,
                        "cycle": cycle,
                        "split": split,
                        "rul": length - cycle if split == "train" else 3,
                        "sample_weight": 1 / length if split == "train" else 1.0,
                        "history_count": min(cycle, 10),
                    }
                )
                rows.append(row)
    return rows


def _partition(root, split, rows=None):
    rows = _rows(split) if rows is None else rows
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, root / f"{split}.parquet")
    manifest = {
        "row_counts": {split: len(rows)},
        "engine_counts": {split: len({(row["subset"], row["unit_id"]) for row in rows})},
    }
    return training._load_partition(root, split, training.FEATURE_COLUMNS, manifest)


def test_metrics_known_errors():
    result = training.regression_metrics(np.array([10, 20, 30]), np.array([8, 20, 33]))
    assert result["count"] == 3
    assert result["rmse"] == pytest.approx(np.sqrt(13 / 3))
    assert result["mae"] == pytest.approx(5 / 3)
    assert result["bias"] == pytest.approx(1 / 3)
    assert result["r2"] == pytest.approx(1 - 13 / 200)
    assert result["r2_undefined_reason"] is None
    nasa = np.expm1(2 / 13) + np.expm1(3 / 10)
    assert result["nasa_score"] == pytest.approx(nasa)
    assert result["mean_nasa_score"] == pytest.approx(nasa / 3)
    json.dumps(result, allow_nan=False)


def test_nasa_score_penalizes_overpredicting_more_severely():
    early = training.regression_metrics([20], [10])
    late = training.regression_metrics([20], [30])
    assert late["nasa_score"] > early["nasa_score"]
    assert late["nasa_score"] == pytest.approx(np.expm1(1))
    assert early["nasa_score"] == pytest.approx(np.expm1(10 / 13))


@pytest.mark.parametrize(
    "actual,prediction",
    [
        ([], []),
        ([1], [1, 2]),
        ([[1]], [[1]]),
        ([float("nan")], [1]),
        ([1], [float("inf")]),
        ([-1], [1]),
        (["not-numeric"], [1]),
    ],
)
def test_metrics_reject_invalid_arrays(actual, prediction):
    with pytest.raises(training.TrainingError):
        training.regression_metrics(actual, prediction)


@pytest.mark.parametrize(
    "actual,reason",
    [
        ([1], "fewer_than_two_observations"),
        ([2, 2], "constant_actual_rul"),
    ],
)
def test_undefined_r2_is_explicit(actual, reason):
    result = training.regression_metrics(actual, actual)
    assert result["r2"] is None
    assert result["r2_undefined_reason"] == reason
    assert result["rmse"] == result["nasa_score"] == 0
    json.dumps(result, allow_nan=False)


def test_metric_overflow_fails_without_clipping():
    with pytest.raises(training.TrainingError, match="overflow"):
        training.regression_metrics([0], [10000])


def test_approved_config_is_accepted():
    config = training.load_config(ROOT / "config" / "baseline.json")
    assert config["parameters"] == training.APPROVED_PARAMETERS
    assert config["num_boost_round"] == 600
    assert config["early_stopping_rounds"] == 50


@pytest.mark.parametrize(
    "key,value",
    [
        ("device", "cuda"),
        ("nthread", -1),
        ("tree_method", "gpu_hist"),
        ("objective", "binary:logistic"),
        ("eval_metric", "auc"),
        ("seed", 43),
        ("max_depth", 0),
        ("eta", float("nan")),
        ("nthread", True),
        ("max_bin", 256.0),
        ("extra", "../unsafe"),
    ],
)
def test_parameter_config_is_closed(config_path, configuration, key, value):
    configuration["parameters"][key] = value
    config_path.write_text(json.dumps(configuration), encoding="utf-8")
    with pytest.raises(training.TrainingError, match="approved CPU recipe"):
        training.load_config(config_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_boost_round", 601),
        ("num_boost_round", 0),
        ("num_boost_round", True),
        ("early_stopping_rounds", 0),
        ("early_stopping_rounds", 51),
        ("prediction_floor", -1),
        ("target", "capped_rul"),
        ("sweep", {}),
        ("model_name", "../../unsafe"),
        ("schema_version", True),
        ("explainability", "eval('unsafe')"),
    ],
)
def test_invalid_config_keys_and_values(config_path, configuration, key, value):
    configuration[key] = value
    config_path.write_text(json.dumps(configuration), encoding="utf-8")
    with pytest.raises(training.TrainingError):
        training.load_config(config_path)


def test_duplicate_and_missing_config_keys_fail(config_path, configuration):
    text = json.dumps(configuration)
    config_path.write_text(text[:-1] + ', "schema_version": 1}', encoding="utf-8")
    with pytest.raises(training.TrainingError):
        training.load_config(config_path)
    configuration.pop("target")
    config_path.write_text(json.dumps(configuration), encoding="utf-8")
    with pytest.raises(training.TrainingError):
        training.load_config(config_path)


def test_manifest_exact_predictor_whitelist():
    manifest = {
        "schema_version": 1,
        "dataset": "nasa-cmapss-ml-ready",
        "target": "rul",
        "feature_columns": list(training.FEATURE_COLUMNS),
    }
    assert len(training._validate_manifest(manifest)) == 35
    for forbidden in ("unit_id", "subset", "split", "rul", "sample_weight", "random_numeric"):
        modified = copy.deepcopy(manifest)
        modified["feature_columns"][0] = forbidden
        with pytest.raises(training.TrainingError):
            training._validate_manifest(modified)
    manifest["feature_columns"] = list(training.FEATURE_COLUMNS[:-1])
    with pytest.raises(training.TrainingError):
        training._validate_manifest(manifest)


def test_arbitrary_mount_name_does_not_add_hive_predictors(workspace):
    mounted = workspace / "mount=arbitrary"
    mounted.mkdir()
    partition = _partition(mounted, "test")
    assert partition.features.shape == (4, 35)
    assert "mount" not in partition.table.column_names


def test_partition_groups_balance_and_disjointness(workspace):
    train = _partition(workspace, "train")
    validation = _partition(workspace, "validation")
    test = _partition(workspace, "test")
    training._validate_disjoint(train, validation)
    assert set(train.engines) & set(test.engines)  # Original test IDs have their own namespace.
    assert len(test.engines) == 4  # Same unit_id in distinct subsets is legitimate.
    rows = _rows("validation")
    rows[0]["unit_id"] = 1
    validation = _partition(workspace, "validation", rows)
    with pytest.raises(training.TrainingError, match="disjoint"):
        training._validate_disjoint(train, validation)


@pytest.mark.parametrize(
    "split,column,value",
    [
        ("train", "sample_weight", 0),
        ("train", "sample_weight", 100),
        ("validation", "sample_weight", 2),
        ("test", "sample_weight", float("nan")),
        ("train", "sensor_03", float("inf")),
        ("test", "sensor_03", None),
        ("validation", "rul", -1),
        ("validation", "rul", 0.5),
        ("train", "unit_id", 0),
        ("test", "cycle", 2.5),
        ("test", "split", "validation"),
        ("test", "subset", "FD999"),
    ],
)
def test_invalid_partition_rejected(workspace, split, column, value):
    rows = _rows(split)
    rows[0][column] = value
    with pytest.raises(training.TrainingError):
        _partition(workspace, split, rows)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_duplicate_engine_snapshots_rejected(workspace, split):
    rows = _rows(split)
    duplicate = dict(rows[0])
    duplicate["cycle"] += 1
    rows.append(duplicate)
    with pytest.raises(training.TrainingError, match="one equally weighted snapshot"):
        _partition(workspace, split, rows)


def test_output_refuses_existing_artifacts(workspace, config_path):
    output = workspace / "output"
    output.mkdir()
    protected = output / "model.json"
    protected.write_text("must survive", encoding="utf-8")
    with pytest.raises(training.TrainingError, match="never overwritten"):
        training.train_baseline(workspace / "unused", config_path, output)
    assert protected.read_text(encoding="utf-8") == "must survive"


def test_cli_sanitizes_errors_and_has_nonzero_exit(workspace, config_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "epm_platform.baseline.training",
            "--data",
            str(workspace / "secret-token-invalid-dataset"),
            "--config",
            str(config_path),
            "--output",
            str(workspace / "output"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode != 0
    assert "Error:" in result.stderr
    assert "secret-token" not in result.stderr
    assert str(workspace) not in result.stderr
    assert "Traceback" not in result.stderr
    assert list((workspace / "output").iterdir()) == []


def test_module_import_has_no_cloud_dependency():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    code = (
        "import sys, importlib.abc\n"
        "class NoCloud(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in ('azure', 'mlflow', 'torch', 'shap'):\n"
        "            raise RuntimeError('Disallowed optional import')\n"
        "sys.meta_path.insert(0, NoCloud())\n"
        "import epm_platform.baseline.training\n"
        "assert 'epm_platform.features.pipeline' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, result.stderr


def test_cli_does_not_echo_unknown_argument_secrets(capsys):
    with pytest.raises(SystemExit) as error:
        training.main(["--secret-token=do-not-print"])
    assert error.value.code == 2
    assert "secret-token" not in capsys.readouterr().err


def test_native_early_stopping_slice_matches_best_iteration(workspace, configuration):
    train = _partition(workspace, "train")
    validation = _partition(workspace, "validation")
    columns = training.FEATURE_COLUMNS
    dtrain = training._matrix(train, columns, training=True)
    dvalidation = training._matrix(validation, columns)
    original = xgb.train(
        configuration["parameters"],
        dtrain,
        num_boost_round=configuration["num_boost_round"],
        early_stopping_rounds=configuration["early_stopping_rounds"],
        evals=[(dvalidation, "validation")],
        verbose_eval=False,
    )
    best = original.best_iteration
    assert original.num_boosted_rounds() > best + 1
    expected = original.predict(dvalidation, iteration_range=(0, best + 1))
    frozen = original[: best + 1]
    model_path = workspace / "selected.json"
    frozen.save_model(model_path)
    loaded = xgb.Booster()
    loaded.load_model(model_path)
    assert loaded.num_boosted_rounds() == best + 1
    np.testing.assert_array_equal(loaded.predict(dvalidation), expected)
    np.testing.assert_array_equal(
        training.predict_rul(loaded, dvalidation, best), np.maximum(expected, 0)
    )
    contributions = loaded.predict(dvalidation, pred_contribs=True, approx_contribs=False)
    np.testing.assert_allclose(contributions.sum(axis=1), expected, rtol=1e-6, atol=1e-6)
    importance = training._explanations(loaded, dvalidation, columns)
    assert len(importance["features"]) == 35
    assert importance["validation_count"] == 4


def test_training_artifacts_native_reload_and_empty_mount(make_dataset, workspace, config_path):
    source = make_dataset()
    mounted = workspace / "input=mounted"
    shutil.copytree(source, mounted)
    output = workspace / "output"
    output.mkdir()
    result = training.train_baseline(mounted, config_path, output)
    expected_files = {
        "model.json",
        "metrics.json",
        "run-metadata.json",
        "feature-importance.json",
        "predictions_validation.parquet",
        "predictions_test.parquet",
        "evaluation.md",
        "artifact-manifest.json",
    }
    assert {path.name for path in output.iterdir()} == expected_files
    assert json.loads((output / "metrics.json").read_text()) == result
    inventory = json.loads((output / "artifact-manifest.json").read_text())
    assert {entry["path"] for entry in inventory["files"]} == expected_files - {
        "artifact-manifest.json"
    }
    for entry in inventory["files"]:
        path = output / entry["path"]
        assert path.stat().st_size == entry["size_bytes"]
        assert _hash(path) == entry["sha256"]
    assert result["provenance"]["ml_ready_manifest_sha256"] == _hash(mounted / "manifest.json")
    assert result["provenance"]["baseline_config_file_sha256"] == _hash(config_path)
    assert result["runtime"]["packages"]["xgboost"] == xgb.__version__
    assert str(workspace) not in (output / "run-metadata.json").read_text()
    model = xgb.Booster()
    model.load_model(output / "model.json")
    best = result["model"]["best_iteration"]
    assert model.num_boosted_rounds() == len(model.get_dump()) == best + 1
    assert model.best_iteration == best
    assert model.attr("prediction_floor") == "0.0"
    assert result["model"]["tree_count"] == best + 1
    assert result["model"]["prediction_iteration_range"] == [0, best + 1]
    manifest = json.loads((mounted / "manifest.json").read_text())
    for split in ("validation", "test"):
        rows = training._load_partition(mounted, split, training.FEATURE_COLUMNS, manifest)
        matrix = training._matrix(rows, training.FEATURE_COLUMNS)
        stored = pq.ParquetFile(output / f"predictions_{split}.parquet").read()
        predictions = stored["prediction"].to_numpy()
        np.testing.assert_array_equal(predictions, np.maximum(model.predict(matrix), 0))
        np.testing.assert_array_equal(predictions, training.predict_rul(model, matrix, best))
        np.testing.assert_array_equal(stored["error"].to_numpy(), predictions - rows.target)
        assert stored.column_names == [
            "subset",
            "unit_id",
            "cycle",
            "split",
            "actual",
            "prediction",
            "error",
        ]
        assert result[split]["overall"]["count"] == rows.table.num_rows
        assert set(result[split]["per_subset"]) == set(training.SUBSETS)
    importance = json.loads((output / "feature-importance.json").read_text())
    assert len(importance["features"]) == 35
    assert importance["validation_count"] == result["row_counts"]["validation"]
    metadata = json.loads((output / "run-metadata.json").read_text())
    assert metadata["test_evaluation_count"] == 1
    assert metadata["model_frozen_before_test"] is True
    assert len(metadata["validation_rmse_history"]) == result["model"]["attempted_boosting_rounds"]
    assert "calibrated failure probability" in (output / "evaluation.md").read_text(
        encoding="utf-8"
    )


def test_training_is_deterministic_and_test_changes_do_not_affect_fit(
    make_dataset, workspace, config_path
):
    original = make_dataset()
    altered = make_dataset("altered", test_offset=37)
    for split in ("train", "validation"):
        assert (original / f"{split}.parquet").read_bytes() == (
            altered / f"{split}.parquet"
        ).read_bytes()
    results = []
    for name, source in (("first", original), ("repeat", original), ("changed", altered)):
        output = workspace / name
        results.append(training.train_baseline(source, config_path, output))
    first, repeated, changed = results
    assert first == repeated
    assert first["validation"] == changed["validation"]
    assert first["model"] == changed["model"]
    assert first["test"] != changed["test"]
    for filename in ("model.json", "feature-importance.json", "predictions_validation.parquet"):
        assert (workspace / "first" / filename).read_bytes() == (
            workspace / "changed" / filename
        ).read_bytes()
    for path in (workspace / "first").iterdir():
        assert path.read_bytes() == (workspace / "repeat" / path.name).read_bytes()


def test_real_feature_integrity_verification_precedes_fit(
    make_dataset, workspace, config_path, monkeypatch
):
    root = make_dataset()
    path = root / "test.parquet"
    path.write_bytes(path.read_bytes() + b"invalid-content")
    fit_called = False

    def fail_fit(*args, **kwargs):
        nonlocal fit_called
        fit_called = True
        raise AssertionError("Fitting unverified data is forbidden")

    monkeypatch.setattr(training.xgb, "train", fail_fit)
    output = workspace / "output"
    with pytest.raises(training.TrainingError):
        training.train_baseline(root, config_path, output)
    assert not fit_called
    assert list(output.iterdir()) == []


def test_only_validation_is_used_for_selection_and_test_is_predicted_once(
    make_dataset, workspace, config_path, monkeypatch
):
    root = make_dataset()
    native_train = training.xgb.train
    native_predict = training.predict_rul
    calls = {"fit": 0, "test": 0, "validation": 0}

    def fit(parameters, matrix, **kwargs):
        calls["fit"] += 1
        assert [name for _, name in kwargs["evals"]] == ["validation"]
        assert matrix.num_row() > kwargs["evals"][0][0].num_row()
        return native_train(parameters, matrix, **kwargs)

    def predict(model, matrix, best_iteration):
        split = "test" if matrix.num_row() == 8 else "validation"
        calls[split] += 1
        if split == "test":
            stages = list((workspace / "output").glob(".staging-*"))
            assert len(stages) == 1
            saved = stages[0] / "model.json"
            assert saved.is_file()
            loaded = xgb.Booster()
            loaded.load_model(saved)
            assert loaded.num_boosted_rounds() == model.num_boosted_rounds() == best_iteration + 1
        return native_predict(model, matrix, best_iteration)

    monkeypatch.setattr(training.xgb, "train", fit)
    monkeypatch.setattr(training, "predict_rul", predict)
    training.train_baseline(root, config_path, workspace / "output")
    assert calls == {"fit": 1, "test": 1, "validation": 1}
