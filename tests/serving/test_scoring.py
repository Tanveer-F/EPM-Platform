from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import xgboost as xgb

from epm_platform.data.validation import OBSERVATION_SCHEMA
from epm_platform.features.pipeline import features_for_engine
from epm_platform.serving.local_api import create_server
from epm_platform.serving.monitoring import load_reference, summarize_batch
from epm_platform.serving.schema import (
    FEATURE_COLUMNS,
    OBSERVATION_COLUMNS,
)
from epm_platform.serving.scoring import (
    InferenceInputError,
    InferenceService,
    _features_for_instance,
)


def _instance(cycle_count: int = 10, start_cycle: int = 1) -> dict:
    observations = []
    for cycle in range(start_cycle, start_cycle + cycle_count):
        row = {"cycle": cycle}
        row.update({f"setting_{index}": index + cycle / 10 for index in range(1, 4)})
        row.update({f"sensor_{index:02d}": index * 2 + cycle / 5 for index in range(1, 22)})
        observations.append(row)
    return {"subset": "FD001", "unit_id": 7, "observations": observations}


def _reference():
    bounds = {"mean": 0.0, "std": 1.0, "p01": -1_000_000.0, "p99": 1_000_000.0}
    return {
        "features": {name: bounds.copy() for name in FEATURE_COLUMNS},
        "prediction": bounds.copy(),
    }


class FakePredictor:
    def predict(self, features):
        return np.full(features.shape[0], 42.5)


def test_online_feature_contract_matches_causal_training_pipeline():
    instance = _instance(cycle_count=10, start_cycle=6)
    metadata, vector = _features_for_instance(instance)
    assert metadata == {"subset": "FD001", "unit_id": 7, "cycle": 15}
    values = {"unit_id": [7] * 15}
    for name in OBSERVATION_COLUMNS:
        if name == "cycle":
            values[name] = list(range(1, 16))
        else:
            values[name] = [
                (
                    float(name.split("_")[1]) * 2 + cycle / 5
                    if name.startswith("sensor_")
                    else float(name.split("_")[1]) + cycle / 10
                )
                for cycle in range(1, 16)
            ]
    expected = features_for_engine(pa.Table.from_pydict(values, schema=OBSERVATION_SCHEMA)).slice(
        14, 1
    )
    assert tuple(expected.column_names) == FEATURE_COLUMNS
    np.testing.assert_allclose(vector, [expected[name][0].as_py() for name in FEATURE_COLUMNS])


def test_partial_first_cycles_require_exact_prefix_and_preserve_history_count():
    _, vector = _features_for_instance(_instance(cycle_count=3))
    assert vector[0] == 3
    assert vector[-1] == 3


@pytest.mark.parametrize(
    "mutate,code",
    [
        (
            lambda value: value["observations"].pop(5),
            "history_must_be_contiguous_trailing_window",
        ),
        (
            lambda value: value["observations"][-1].update(sensor_01=float("nan")),
            "nonfinite_or_nonnumeric_sensor_value",
        ),
        (
            lambda value: value["observations"][-1].update(sensor_01=True),
            "nonfinite_or_nonnumeric_sensor_value",
        ),
        (
            lambda value: value["observations"][-1].update(extra=1),
            "invalid_observation_schema",
        ),
        (lambda value: value.update(subset=[]), "invalid_subset"),
        (
            lambda value: value["observations"][-1].update(cycle=12),
            "history_must_be_contiguous_trailing_window",
        ),
    ],
)
def test_invalid_instance_is_rejected(mutate, code):
    value = _instance()
    mutate(value)
    with pytest.raises(InferenceInputError) as error:
        _features_for_instance(value)
    assert error.value.code == code


def test_batch_scoring_returns_predictions_and_logs_monitor_summary(caplog):
    service = InferenceService(FakePredictor(), _reference())
    payload = {"instances": [_instance(), _instance()]}
    with caplog.at_level("INFO", logger="epm.inference"):
        result = service.score(json.dumps(payload))
    assert [row["rul_cycles"] for row in result["predictions"]] == [42.5, 42.5]
    assert "epm_inference_monitoring" in caplog.text
    assert "epm_inference_request" in caplog.text
    assert "observations" not in caplog.text


def test_duplicate_json_properties_are_rejected():
    service = InferenceService(FakePredictor(), _reference())
    with pytest.raises(InferenceInputError, match="invalid_json"):
        service.score('{"instances":[],"instances":[]}')


def test_monitoring_requires_minimum_batch_and_detects_reference_outliers():
    reference = _reference()
    reference["features"]["cycle"].update(p01=1.0, p99=10.0)
    reference["prediction"].update(p01=0.0, p99=100.0)
    features = np.ones((20, len(FEATURE_COLUMNS)))
    features[:, 0] = 20
    summary = summarize_batch(features, np.full(20, 120.0), reference)
    assert summary["drift_assessable"] is True
    assert summary["feature_alerts"] == ["cycle"]
    assert summary["prediction_alert"] is True
    assert summary["feature_out_of_reference_rates"]["cycle"] == 1

    small = summarize_batch(features[:1], np.array([120.0]), reference)
    assert small["drift_assessable"] is False
    assert small["feature_alerts"] == []
    assert small["prediction_out_of_reference_rate"] is None


def test_reference_loader_rejects_wrong_model_lineage(tmp_path):
    reference = {
        "schema_version": 1,
        "model_sha256": "bad",
        "training_manifest_sha256": "bad",
        "feature_columns": list(FEATURE_COLUMNS),
        "features": {name: _reference()["features"][name] for name in FEATURE_COLUMNS},
        "prediction": _reference()["prediction"],
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    with pytest.raises(ValueError, match="monitoring reference"):
        load_reference(path)


def test_checked_in_monitoring_reference_is_pinned_to_registered_model():
    path = (
        Path(__file__).parents[2]
        / "src"
        / "epm_platform"
        / "serving"
        / "monitoring-reference.json"
    )
    reference = load_reference(path)
    assert reference["training_row_count"] == 128_967
    assert reference["validation_prediction_row_count"] == 142


def test_local_http_api_health_scoring_and_validation():
    server = create_server(InferenceService(FakePredictor(), _reference()), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
            assert json.loads(response.read()) == {"status": "healthy"}
        request = urllib.request.Request(
            f"{base_url}/score",
            data=json.dumps({"instances": [_instance()]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            assert json.loads(response.read())["predictions"][0]["rul_cycles"] == 42.5

        invalid = urllib.request.Request(
            f"{base_url}/score",
            data=b'{"instances":[]}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(invalid, timeout=2)
        assert error.value.code == 400
        assert json.loads(error.value.read()) == {"error": "invalid_instance_count"}
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_local_api_refuses_non_loopback_bind():
    with pytest.raises(ValueError, match="loopback"):
        create_server(InferenceService(FakePredictor(), _reference()), host="0.0.0.0")


def test_local_http_api_executes_native_xgboost_inference():
    sample_vector = np.asarray(_features_for_instance(_instance())[1], dtype=np.float32)
    training_features = np.vstack([sample_vector + offset for offset in range(4)])
    training_data = xgb.DMatrix(
        training_features,
        label=np.asarray([10.0, 20.0, 30.0, 40.0]),
        feature_names=list(FEATURE_COLUMNS),
        nthread=2,
    )
    booster = xgb.train(
        {"objective": "reg:squarederror", "tree_method": "hist", "nthread": 2, "seed": 42},
        training_data,
        num_boost_round=2,
    )

    class BoosterPredictor:
        def predict(self, features):
            data = xgb.DMatrix(features, feature_names=list(FEATURE_COLUMNS), nthread=2)
            return booster.predict(data)

    server = create_server(InferenceService(BoosterPredictor(), _reference()), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/score",
            data=json.dumps({"instances": [_instance()]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            result = json.loads(response.read())
        prediction = result["predictions"][0]["rul_cycles"]
        assert isinstance(prediction, float)
        assert np.isfinite(prediction) and prediction >= 0
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
