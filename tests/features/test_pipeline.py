import hashlib
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from epm_platform.data.errors import DataError
from epm_platform.data.manifest import canonical_json, fingerprint, hash_file
from epm_platform.data.spec import OBSERVATION_COLUMNS, SUBSETS, load_spec
from epm_platform.data.validation import LABEL_SCHEMA, OBSERVATION_SCHEMA
from epm_platform.features import pipeline
from epm_platform.features.pipeline import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    METADATA_COLUMNS,
    TABLE_SCHEMA,
    build_features,
    features_for_engine,
    verify_features,
)

PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / "config" / "features.json"
SPEC = PROJECT / "config" / "cmapss-source.json"


def engine(unit=1, life=15, base=0.0):
    values = {"unit_id": [unit] * life, "cycle": list(range(1, life + 1))}
    for index, name in enumerate(OBSERVATION_COLUMNS[2:], 1):
        values[name] = [
            float(base + unit * 100 + index + cycle * index) for cycle in range(1, life + 1)
        ]
    return pa.Table.from_pydict(values, schema=OBSERVATION_SCHEMA)


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def rewrite_json(path, value):
    path.write_bytes(canonical_json(value))


def remanifest(root, change=None):
    manifest = json.loads((root / "manifest.json").read_bytes())
    for entry in manifest["files"]:
        entry.update(fingerprint(root / entry["path"]))
    if change:
        change(manifest)
    rewrite_json(root / "manifest.json", manifest)
    digest = hash_file(root / "manifest.json")
    rewrite_json(root / "_SUCCESS.json", {"manifest_sha256": digest, "version": "sha256-" + digest})


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    for index, subset in enumerate(SUBSETS):
        folder = root / subset
        folder.mkdir()
        pq.write_table(
            pa.concat_tables(
                [
                    engine(unit, life, index * 1000.0)
                    for unit, life in enumerate((12, 14, 16, 18, 150), 1)
                ]
            ),
            folder / "train.parquet",
        )
        pq.write_table(
            pa.concat_tables(
                [
                    engine(1, 13, 10000.0 + index * 1000),
                    engine(2, 6, 10000.0 + index * 1000),
                ]
            ),
            folder / "test.parquet",
        )
        pq.write_table(
            pa.Table.from_pydict({"unit_id": [1, 2], "rul": [200 + index, 0]}, schema=LABEL_SCHEMA),
            folder / "test_rul.parquet",
        )
    rewrite_json(root / "manifest.json", {"synthetic": True})
    calls = []

    def verify_source(source_root, spec_path, config):
        calls.append((source_root, spec_path, config))
        assert source_root == root
        assert spec_path == SPEC
        assert config == json.loads(CONFIG.read_text())

    monkeypatch.setattr(pipeline, "_verify_source", verify_source)
    return SimpleNamespace(
        root=root, output=tmp_path / "output", calls=calls, original=snapshot(root)
    )


@pytest.fixture
def bundle(source):
    return build_features(source.root, source.output, CONFIG, SPEC)


def test_feature_schema_exact_and_known_rolling_values():
    raw = engine(life=15)
    table = features_for_engine(raw)
    expected = (
        "cycle",
        *(f"setting_{i}" for i in range(1, 4)),
        *(f"sensor_{i:02d}" for i in range(1, 22)),
        "sensor_03_mean10",
        "sensor_03_slope10",
        "sensor_04_mean10",
        "sensor_04_slope10",
        "sensor_11_mean10",
        "sensor_11_slope10",
        "setting_1_mean10",
        "setting_2_mean10",
        "setting_3_mean10",
        "history_count",
    )
    assert FEATURE_COLUMNS == expected
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)) == 35
    assert table.schema == FEATURE_SCHEMA
    assert table.num_rows == raw.num_rows
    assert not (
        {"unit_id", "subset", "rul", "sample_weight", "split", "life", "source_rows"}
        & set(FEATURE_COLUMNS)
    )
    assert set(FEATURE_COLUMNS) & set(METADATA_COLUMNS) == {"cycle"}
    assert table["history_count"].to_pylist() == [*range(1, 10), *([10] * 6)]
    for sensor in ("sensor_03", "sensor_04", "sensor_11"):
        values = raw[sensor].to_pylist()
        assert table[f"{sensor}_slope10"][0].as_py() == 0
        for index in range(15):
            window = values[max(0, index - 9) : index + 1]
            assert table[f"{sensor}_mean10"][index].as_py() == pytest.approx(
                sum(window) / len(window)
            )
            if index:
                assert table[f"{sensor}_slope10"][index].as_py() == pytest.approx(
                    values[1] - values[0]
                )
    for setting in ("setting_1", "setting_2", "setting_3"):
        assert table[f"{setting}_mean10"][0] == raw[setting][0]
        assert table[f"{setting}_mean10"][14].as_py() == pytest.approx(
            sum(raw[setting].to_pylist()[5:15]) / 10
        )


def test_prefix_and_future_perturbation_invariance():
    original = engine(life=30)
    full = features_for_engine(original)
    for cut in (1, 2, 9, 10, 11, 20):
        assert features_for_engine(original.slice(0, cut)).equals(full.slice(0, cut))
    values = original.to_pydict()
    for name in OBSERVATION_COLUMNS[2:]:
        values[name][12:] = [value * -37.0 for value in values[name][12:]]
    changed = features_for_engine(pa.Table.from_pydict(values, schema=OBSERVATION_SCHEMA))
    assert changed.slice(0, 12).equals(full.slice(0, 12))
    assert not changed.equals(full)


def test_slope_uses_ols_not_endpoint_difference():
    values = engine(life=4).to_pydict()
    values["sensor_03"] = [0.0, 10.0, 0.0, 0.0]
    result = features_for_engine(pa.Table.from_pydict(values, schema=OBSERVATION_SCHEMA))
    assert result["sensor_03_slope10"].to_pylist() == pytest.approx([0.0, 10.0, 0.0, -1.0])


@pytest.mark.parametrize(
    "invalid",
    [
        "empty",
        "mixed",
        "gap",
        "offset",
        "reversed",
        "null",
        "nan",
        "infinity",
        "type",
        "leakage",
        "order",
    ],
)
def test_invalid_engine_input(invalid):
    table = engine(life=4)
    values = table.to_pydict()
    if invalid == "empty":
        table = table.slice(0, 0)
    elif invalid in ("mixed", "gap", "offset", "reversed", "null", "nan", "infinity"):
        if invalid == "mixed":
            values["unit_id"][2] = 2
        elif invalid == "gap":
            values["cycle"][2] = 9
        elif invalid == "offset":
            values["cycle"] = [5, 6, 7, 8]
        elif invalid == "reversed":
            values["cycle"] = [4, 3, 2, 1]
        else:
            values["sensor_01"][0] = {"null": None, "nan": float("nan"), "infinity": float("inf")}[
                invalid
            ]
        table = pa.Table.from_pydict(values, schema=OBSERVATION_SCHEMA)
    elif invalid == "type":
        table = table.set_column(1, "cycle", pa.array([1.0, 2.0, 3.0, 4.0]))
    elif invalid == "leakage":
        table = table.append_column("rul", pa.array([3, 2, 1, 0]))
    else:
        table = table.select(list(reversed(table.column_names)))
    with pytest.raises(DataError):
        features_for_engine(table)


def test_deterministic_build_immutable_reuse_and_mounted_verification(source, bundle, tmp_path):
    before = snapshot(source.root)
    assert before == source.original
    first = snapshot(bundle.root)
    times = {p: p.stat().st_mtime_ns for p in bundle.root.iterdir()}
    reused = build_features(source.root, source.output, CONFIG, SPEC)
    second = build_features(source.root, tmp_path / "second", CONFIG, SPEC)
    assert bundle.version == reused.version == second.version
    assert first == snapshot(reused.root) == snapshot(second.root)
    assert times == {p: p.stat().st_mtime_ns for p in bundle.root.iterdir()}
    assert snapshot(source.root) == before
    assert len(source.calls) == 6
    assert bundle.root == source.output / ("sha256-" + bundle.manifest_sha256)
    assert bundle.manifest_sha256 == hash_file(bundle.root / "manifest.json")
    assert list(source.output.iterdir()) == [bundle.root]
    mounted = tmp_path / "scope=value" / "arbitrary-mount"
    shutil.copytree(bundle.root, mounted)
    manifest = verify_features(mounted)
    assert manifest["feature_columns"] == list(FEATURE_COLUMNS)
    assert manifest["target"] == "rul"
    assert manifest["source"] == json.loads(CONFIG.read_text())["source"]
    assert (
        manifest["config_sha256"]
        == hashlib.sha256(canonical_json(json.loads(CONFIG.read_text()))).hexdigest()
    )
    assert sorted(snapshot(bundle.root)) == sorted(
        [
            "manifest.json",
            "_SUCCESS.json",
            "splits.json",
            "feature-summary.json",
            "train.parquet",
            "validation.parquet",
            "test.parquet",
        ]
    )
    assert str(source.root).encode() not in first["manifest.json"]


def test_targets_groups_weights_and_original_test_labels(source, bundle):
    assignments = json.loads((bundle.root / "splits.json").read_bytes())["assignments"]
    tables = {
        s: pq.ParquetFile(bundle.root / f"{s}.parquet").read()
        for s in ("train", "validation", "test")
    }
    keys = {
        s: set(zip(t["subset"].to_pylist(), t["unit_id"].to_pylist(), strict=True))
        for s, t in tables.items()
    }
    assert keys["train"].isdisjoint(keys["validation"])
    assert bundle.summary["engine_counts"] == {"train": 16, "validation": 4, "test": 8}
    assert tables["validation"].num_rows == 4
    assert tables["test"].num_rows == 8
    assert max(tables["train"]["rul"].to_pylist()) > 125
    total_weight = {}
    for split, table in tables.items():
        assert table.schema == TABLE_SCHEMA
        for row in table.to_pylist():
            entry = next(
                e
                for e in assignments
                if e["subset"] == row["subset"]
                and e["unit_id"] == row["unit_id"]
                and e["split"] == split
            )
            if split == "test":
                labels = pq.ParquetFile(source.root / row["subset"] / "test_rul.parquet").read()
                assert row["rul"] == labels["rul"][row["unit_id"] - 1].as_py()
                assert row["cycle"] == entry["source_rows"]
            else:
                assert entry["source_split"] == "train"
                assert row["rul"] == entry["source_rows"] - row["cycle"]
            if split == "train":
                key = row["subset"], row["unit_id"]
                total_weight[key] = total_weight.get(key, 0) + row["sample_weight"]
            else:
                assert row["sample_weight"] == 1.0
            if split == "validation":
                assert math.ceil(entry["source_rows"] * 0.5) <= row["cycle"]
                assert row["cycle"] <= math.floor(entry["source_rows"] * 0.8)
    assert math.fsum(tables["train"]["sample_weight"].to_pylist()) == pytest.approx(
        tables["train"].num_rows
    )
    assert list(total_weight.values()) == pytest.approx([tables["train"].num_rows / 16] * 16)
    for subset in SUBSETS:
        ordered = sorted(
            range(1, 6), key=lambda u: hashlib.sha256(f"42|{subset}|{u}".encode()).digest()
        )
        heldout = next(
            e for e in assignments if e["subset"] == subset and e["split"] == "validation"
        )
        assert heldout["unit_id"] == ordered[0]
        low, high = (
            math.ceil(heldout["source_rows"] * 0.5),
            math.floor(heldout["source_rows"] * 0.8),
        )
        number = int(hashlib.sha256(f"cut|42|{subset}|{ordered[0]}".encode()).hexdigest(), 16)
        assert heldout["cut_cycle"] == low + number % (high - low + 1)
        for split in ("train", "validation", "test"):
            for row in tables[split].to_pylist():
                if row["subset"] == subset and row["cycle"] == 1:
                    assert row["sensor_03_mean10"] == row["sensor_03"]
                    assert row["sensor_03_slope10"] == 0


def test_test_values_and_labels_never_affect_training_or_validation(source, bundle, tmp_path):
    first = snapshot(bundle.root)
    path = source.root / "FD001" / "test.parquet"
    table = pq.ParquetFile(path).read()
    values = table.to_pydict()
    for name in OBSERVATION_COLUMNS[2:]:
        values[name] = [v * 19.0 for v in values[name]]
    pq.write_table(pa.Table.from_pydict(values, schema=OBSERVATION_SCHEMA), path)
    label_path = source.root / "FD001" / "test_rul.parquet"
    pq.write_table(
        pa.Table.from_pydict({"unit_id": [1, 2], "rul": [300, 90]}, schema=LABEL_SCHEMA), label_path
    )
    changed = build_features(source.root, tmp_path / "changed", CONFIG, SPEC)
    assert first["train.parquet"] == (changed.root / "train.parquet").read_bytes()
    assert first["validation.parquet"] == (changed.root / "validation.parquet").read_bytes()
    assert first["test.parquet"] != (changed.root / "test.parquet").read_bytes()


def test_validation_uses_only_censored_prefix(source, bundle, tmp_path):
    assignments = json.loads((bundle.root / "splits.json").read_bytes())["assignments"]
    for subset in SUBSETS:
        entry = next(e for e in assignments if e["subset"] == subset and e["split"] == "validation")
        path = source.root / subset / "train.parquet"
        values = pq.ParquetFile(path).read().to_pydict()
        for i, (unit, cycle) in enumerate(zip(values["unit_id"], values["cycle"], strict=True)):
            if unit == entry["unit_id"] and cycle > entry["cut_cycle"]:
                for name in OBSERVATION_COLUMNS[2:]:
                    values[name][i] *= -89.0
        pq.write_table(pa.Table.from_pydict(values, schema=OBSERVATION_SCHEMA), path)
    changed = build_features(source.root, tmp_path / "changed", CONFIG, SPEC)
    assert snapshot(bundle.root) == snapshot(changed.root)


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.pop("features"),
        lambda c: c["features"].pop("history_count"),
        lambda c: c["features"].update(window=5),
        lambda c: c["features"].update(scaler="standard"),
        lambda c: c.update(target="capped_rul"),
        lambda c: c.update(schema_version=True),
        lambda c: c["split"].update(seed=43),
        lambda c: c["split"].update(validation_fraction=0.3),
        lambda c: c["source"].update(asset_version="latest"),
        lambda c: c.update(subsets=["FD001"]),
    ],
)
def test_configuration_is_strict(source, tmp_path, change):
    config = json.loads(CONFIG.read_text())
    change(config)
    path = tmp_path / "bad-config.json"
    rewrite_json(path, config)
    with pytest.raises(DataError, match="approved"):
        build_features(source.root, source.output, path, SPEC)
    assert not source.calls
    assert not source.output.exists()


@pytest.mark.parametrize("corruption", ["bytes", "missing", "extra", "directory", "marker"])
def test_corruption_and_existing_output_not_overwritten(source, bundle, corruption):
    path = bundle.root / "train.parquet"
    if corruption == "bytes":
        path.write_bytes(path.read_bytes() + b"corrupt")
    elif corruption == "missing":
        path.unlink()
    elif corruption == "extra":
        (bundle.root / "unexpected.txt").write_text("unexpected")
    elif corruption == "directory":
        (bundle.root / "extra").mkdir()
    else:
        rewrite_json(bundle.root / "_SUCCESS.json", {"manifest_sha256": "0" * 64, "version": "bad"})
    before = snapshot(bundle.root)
    with pytest.raises(DataError):
        verify_features(bundle.root)
    with pytest.raises(DataError):
        build_features(source.root, source.output, CONFIG, SPEC)
    assert snapshot(bundle.root) == before
    assert list(source.output.iterdir()) == [bundle.root]


@pytest.mark.parametrize(
    "path",
    [
        "../outside.parquet",
        "..\\outside.parquet",
        "/absolute.parquet",
        "C:\\outside.parquet",
        "train.parquet:stream",
        "train/../train.parquet",
    ],
)
def test_manifest_path_traversal_rejected_before_content_read(bundle, path):
    remanifest(bundle.root, lambda m: m["files"][0].update(path=path))
    with pytest.raises(DataError, match="content entry"):
        verify_features(bundle.root)


@pytest.mark.parametrize(
    "invalid",
    [
        "feature_list",
        "count",
        "leakage_column",
        "wrong_type",
        "nan",
        "infinity",
        "target",
        "weight",
        "history",
        "rolling",
        "group",
        "split",
        "extra_row",
    ],
)
def test_rehashed_invalid_schema_or_data_is_rejected(bundle, invalid):
    if invalid == "feature_list":
        remanifest(bundle.root, lambda m: m["feature_columns"].append("source_rows"))
    elif invalid == "count":
        remanifest(bundle.root, lambda m: m["row_counts"].update(train=True))
    else:
        path = bundle.root / "train.parquet"
        table = pq.ParquetFile(path).read()
        if invalid == "leakage_column":
            table = table.append_column("source_rows", table["cycle"])
        elif invalid == "wrong_type":
            table = table.set_column(
                table.schema.get_field_index("cycle"), "cycle", table["cycle"].cast(pa.float64())
            )
        elif invalid == "extra_row":
            table = pa.concat_tables([table, table.slice(0, 1)])
        else:
            values = table.to_pydict()
            field, value = {
                "nan": ("sensor_01", float("nan")),
                "infinity": ("rul", float("inf")),
                "target": ("rul", -1.0),
                "weight": ("sample_weight", 0.0),
                "history": ("history_count", 10),
                "rolling": ("sensor_03_mean10", -123.0),
                "group": ("unit_id", 999),
                "split": ("split", "test"),
            }[invalid]
            values[field][0] = value
            table = pa.Table.from_pydict(values, schema=TABLE_SCHEMA)
        pq.write_table(table, path)
        remanifest(bundle.root)
    with pytest.raises(DataError):
        verify_features(bundle.root)


@pytest.mark.parametrize(
    "invalid",
    ["duplicate", "wrong_cut", "test_contamination", "wrong_holdout", "label_mapping", "missing"],
)
def test_rehashed_assignment_tampering_is_rejected(bundle, invalid):
    path = bundle.root / "splits.json"
    value = json.loads(path.read_bytes())
    assignments = value["assignments"]
    if invalid == "duplicate":
        assignments.append(assignments[0])
    elif invalid == "missing":
        assignments.pop()
    elif invalid == "wrong_cut":
        next(e for e in assignments if e["split"] == "validation")["cut_cycle"] += 1
    elif invalid == "test_contamination":
        next(e for e in assignments if e["source_split"] == "test")["split"] = "train"
    elif invalid == "label_mapping":
        next(e for e in assignments if e["source_split"] == "test")["supplied_rul"] += 1
    else:
        entry = next(e for e in assignments if e["split"] == "validation")
        entry.update(split="train", cut_cycle=None)
    rewrite_json(path, value)
    remanifest(bundle.root)
    with pytest.raises(DataError):
        verify_features(bundle.root)


def test_failed_build_cleans_private_staging(source, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(pq, "write_table", fail)
    before = snapshot(source.root)
    with pytest.raises(DataError, match="no unverified"):
        build_features(source.root, source.output, CONFIG, SPEC)
    assert not list(source.output.iterdir())
    assert snapshot(source.root) == before


def test_producer_refuses_source_descendant_output(source):
    before = snapshot(source.root)
    with pytest.raises(DataError, match="immutable"):
        build_features(source.root, source.root / "features", CONFIG, SPEC)
    assert snapshot(source.root) == before


def test_real_source_verifier_and_spec_are_mandatory(tmp_path, monkeypatch):
    calls = []
    root = tmp_path / "source"
    root.mkdir()
    rewrite_json(root / "manifest.json", {"not": "the approved pinned source"})

    def verify_curated(source_root, spec):
        calls.append((source_root, spec))
        return {}

    monkeypatch.setattr(pipeline, "verify_curated", verify_curated)
    with pytest.raises(DataError, match="pinned digest"):
        build_features(root, tmp_path / "out", CONFIG, SPEC)
    assert calls == [(root, load_spec(SPEC))]
    assert not (tmp_path / "out").exists()
    with pytest.raises(DataError, match="source specification"):
        build_features(root, tmp_path / "out", CONFIG, tmp_path / "missing-spec.json")


def test_original_test_label_mapping_must_not_be_reordered(source):
    path = source.root / "FD001" / "test_rul.parquet"
    pq.write_table(
        pa.Table.from_pydict({"unit_id": [2, 1], "rul": [0, 200]}, schema=LABEL_SCHEMA), path
    )
    with pytest.raises(DataError, match="map exactly"):
        build_features(source.root, source.output, CONFIG, SPEC)
    assert not list(source.output.iterdir())


def test_duplicate_configuration_keys_are_rejected(source, tmp_path):
    path = tmp_path / "duplicate.json"
    text = CONFIG.read_text().replace('"seed": 42', '"seed": 99, "seed": 42')
    path.write_text(text)
    with pytest.raises(DataError, match="duplicate keys"):
        build_features(source.root, source.output, path, SPEC)
    assert not source.output.exists()


def test_holdout_rounds_up_and_does_not_depend_on_input_order():
    units = list(range(1, 7))
    heldout = pipeline._holdout("FD001", units)
    assert len(heldout) == 2
    assert heldout == pipeline._holdout("FD001", list(reversed(units)))


def test_small_invalid_holdout_or_censor_interval_fails():
    with pytest.raises(DataError, match="at least two"):
        pipeline._holdout("FD001", [1])
    with pytest.raises(DataError, match="censor interval"):
        pipeline._cut("FD001", 1, 1)


def test_assignment_row_count_is_bounded_by_actual_partition(bundle):
    path = bundle.root / "splits.json"
    value = json.loads(path.read_bytes())
    next(e for e in value["assignments"] if e["split"] == "train")["source_rows"] = 2**31 - 1
    rewrite_json(path, value)
    remanifest(bundle.root)
    with pytest.raises(DataError, match="exceeds the available"):
        verify_features(bundle.root)


def test_source_reverified_before_any_finalization(source, monkeypatch):
    initial_check = pipeline._verify_source
    count = 0

    def verify_source(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise DataError("Source changed during feature construction")
        initial_check(*args)

    monkeypatch.setattr(pipeline, "_verify_source", verify_source)
    with pytest.raises(DataError, match="Source changed"):
        build_features(source.root, source.output, CONFIG, SPEC)
    assert count == 2
    assert not list(source.output.iterdir())
    assert snapshot(source.root) == source.original


@pytest.mark.parametrize("split", ["validation", "test"])
def test_endpoint_feature_values_use_only_correct_engine_prefix(source, bundle, split):
    assignments = json.loads((bundle.root / "splits.json").read_bytes())["assignments"]
    output = pq.ParquetFile(bundle.root / f"{split}.parquet").read()
    for index, entry in enumerate(e for e in assignments if e["split"] == split):
        table = pq.ParquetFile(
            source.root / entry["subset"] / f"{entry['source_split']}.parquet"
        ).read()
        indices = [
            i for i, unit in enumerate(table["unit_id"].to_pylist()) if unit == entry["unit_id"]
        ]
        prefix = table.take(indices)
        if split == "validation":
            prefix = prefix.slice(0, entry["cut_cycle"])
        expected = features_for_engine(prefix).slice(prefix.num_rows - 1, 1)
        assert output.select(FEATURE_COLUMNS).slice(index, 1).equals(expected)


@pytest.mark.parametrize("split", ["validation", "test"])
@pytest.mark.parametrize(
    "field,value",
    [("rul", -1.0), ("sensor_03_mean10", float("nan")), ("cycle", 1), ("sample_weight", 0.25)],
)
def test_rehashed_endpoint_corruption_is_rejected(bundle, split, field, value):
    path = bundle.root / f"{split}.parquet"
    values = pq.ParquetFile(path).read().to_pydict()
    values[field][0] = value
    pq.write_table(pa.Table.from_pydict(values, schema=TABLE_SCHEMA), path)
    remanifest(bundle.root)
    with pytest.raises(DataError):
        verify_features(bundle.root)
