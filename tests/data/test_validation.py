import math

import pyarrow as pa
import pytest

from epm_platform.data.manifest import canonical_json
from epm_platform.data.spec import OBSERVATION_COLUMNS
from epm_platform.data.validation import (
    LABEL_SCHEMA,
    OBSERVATION_SCHEMA,
    ValidationError,
    validate_label_table,
    validate_labels,
    validate_observation_table,
    validate_observations,
    validate_split_integrity,
)


def write_observations(root, lines, name="observations.txt"):
    path = root / name
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def parsed(root, row, *, base=10.0, name="observations.txt", cycles=3):
    path = write_observations(root, [row(1, cycle, base) for cycle in range(1, cycles + 1)], name)
    return validate_observations(path, expected_rows=cycles, expected_units=1)


def labels(root, count=1):
    path = root / "rul.txt"
    path.write_text("0\n" * count, encoding="ascii")
    return validate_labels(path, expected_units=count)


def test_lossless_typed_observations_and_quality(data_workspace, observation_row):
    lines = [observation_row(unit, cycle) for unit in (1, 2) for cycle in (1, 2, 3)]
    path = write_observations(data_workspace, [" \t" + line + "  " for line in lines])
    result = validate_observations(path, expected_rows=6, expected_units=2)
    assert result.table.schema == OBSERVATION_SCHEMA
    assert result.table.column_names == list(OBSERVATION_COLUMNS)
    assert len(result.table.column_names) == 26
    assert result.table.to_pylist() == [
        dict(
            zip(
                OBSERVATION_COLUMNS,
                [*map(int, line.split()[:2]), *map(float, line.split()[2:])],
                strict=True,
            )
        )
        for line in lines
    ]
    assert result.report["status"] == "passed"
    assert result.report["columns"]["sensor_01"]["constant"]
    assert result.report["columns"]["sensor_01"]["population_stddev"] == 0.0
    assert result.report["columns"]["sensor_02"]["distinct_count"] == 2
    assert result.report["columns"]["setting_1"]["min"] == -0.125
    assert math.copysign(1.0, result.table["setting_2"][0].as_py()) == -1.0
    assert result.report["columns"]["cycle"]["mean"] == pytest.approx(2.0)
    assert result.report["columns"]["cycle"]["population_stddev"] == pytest.approx(math.sqrt(2 / 3))
    assert result.report["rules"]["ascii"]["checks"] == 6
    assert result.report["rules"]["unique_unit_cycle"]["checks"] == 6
    assert result.report["cycles"]["per_unit"][1]["last_cycle"] == 3
    assert result.report["warnings"][0]["rule"] == "zero_variance"
    canonical_json(result.report)


@pytest.mark.parametrize("token", ["NA", "NaN", "null", "None", "n/a", "<NA>"])
def test_missing_tokens_fail_with_counts(data_workspace, observation_row, token):
    tokens = observation_row().split()
    tokens[7] = token
    path = write_observations(data_workspace, [" ".join(tokens)])
    with pytest.raises(ValidationError) as raised:
        validate_observations(path, expected_rows=1, expected_units=1)
    report = raised.value.quality_report
    assert report["counts"]["null_values"] == 1
    assert report["columns"]["sensor_03"]["null_count"] == 1
    assert report["rules"]["no_missing_values"]["examples"] == [{"row": 1, "column": "sensor_03"}]
    assert b"NaN" not in canonical_json(report)


@pytest.mark.parametrize("token", ["inf", "-Infinity", "+INF", "1e999"])
def test_nonfinite_tokens_rejected(data_workspace, observation_row, token):
    tokens = observation_row().split()
    tokens[2] = token
    path = write_observations(data_workspace, [" ".join(tokens)])
    with pytest.raises(ValidationError) as raised:
        validate_observations(path, expected_rows=1, expected_units=1)
    assert raised.value.quality_report["counts"]["nonfinite_values"] == 1
    assert b"Infinity" not in canonical_json(raised.value.quality_report)


@pytest.mark.parametrize("token", ["oops", "1_000", "0x10", "1,2", "1D3", "--1"])
def test_strict_numeric_tokens(data_workspace, observation_row, token):
    tokens = observation_row().split()
    tokens[-1] = token
    path = write_observations(data_workspace, [" ".join(tokens)])
    with pytest.raises(ValidationError) as raised:
        validate_observations(path, expected_rows=1, expected_units=1)
    assert raised.value.quality_report["rules"]["numeric_tokens"]["failures"] == 1


@pytest.mark.parametrize(
    ("column", "token", "rule"),
    [
        (0, "1.0", "integer_tokens"),
        (1, "1e0", "integer_tokens"),
        (0, "0", "int32_range"),
        (0, "-1", "int32_range"),
        (1, "0", "int32_range"),
        (1, "-1", "int32_range"),
        (0, "2147483648", "int32_range"),
        (1, "2147483648", "int32_range"),
    ],
)
def test_positive_int32_identifiers(data_workspace, observation_row, column, token, rule):
    tokens = observation_row().split()
    tokens[column] = token
    path = write_observations(data_workspace, [" ".join(tokens)])
    with pytest.raises(ValidationError) as raised:
        validate_observations(path, expected_rows=1, expected_units=1)
    assert raised.value.quality_report["rules"][rule]["failures"] == 1


@pytest.mark.parametrize("width", [0, 25, 27])
def test_exact_column_width(data_workspace, observation_row, width):
    tokens = (observation_row().split() + ["0"])[:width]
    path = write_observations(data_workspace, [" ".join(tokens)])
    with pytest.raises(ValidationError) as raised:
        validate_observations(path, expected_rows=1, expected_units=1)
    assert raised.value.quality_report["rules"]["column_count"]["failures"] == 1


def test_non_ascii_is_not_decoded_or_repaired(data_workspace, observation_row):
    path = data_workspace / "numeric.txt"
    path.write_bytes(observation_row().encode("ascii") + b"\xa0\n")
    with pytest.raises(ValidationError) as raised:
        validate_observations(path, expected_rows=1, expected_units=1)
    assert raised.value.quality_report["rules"]["ascii"]["failures"] == 1


@pytest.mark.parametrize(
    ("units_cycles", "rule"),
    [
        ([(1, 2)], "cycles_start_at_one"),
        ([(1, 1), (1, 3)], "continuous_cycles"),
        ([(1, 1), (1, 2), (1, 1)], "continuous_cycles"),
        ([(1, 1), (3, 1)], "contiguous_unit_ids"),
        ([(2, 1), (1, 1)], "unit_block_order"),
        ([(1, 1), (2, 1), (1, 2)], "no_unit_reappearance"),
    ],
)
def test_engine_blocks_and_continuous_cycles(data_workspace, observation_row, units_cycles, rule):
    path = write_observations(data_workspace, [observation_row(u, c) for u, c in units_cycles])
    with pytest.raises(ValidationError) as raised:
        validate_observations(
            path, expected_rows=len(units_cycles), expected_units=len({u for u, _ in units_cycles})
        )
    assert raised.value.quality_report["rules"][rule]["failures"] >= 1


def test_exact_and_conflicting_duplicates_reported_separately(data_workspace, observation_row):
    path = write_observations(
        data_workspace, [observation_row(), observation_row(), observation_row(base=20)]
    )
    with pytest.raises(ValidationError) as raised:
        validate_observations(path, expected_rows=3, expected_units=1)
    counts = raised.value.quality_report["counts"]
    assert counts["duplicate_records"] == 1
    assert counts["duplicate_unit_cycles"] == 2
    assert counts["conflicting_unit_cycles"] == 1


@pytest.mark.parametrize(
    ("rows", "units", "rule"),
    [
        (2, 1, "expected_rows"),
        (1, 2, "expected_units"),
    ],
)
def test_reviewed_counts_enforced(data_workspace, observation_row, rows, units, rule):
    path = write_observations(data_workspace, [observation_row()])
    with pytest.raises(ValidationError) as raised:
        validate_observations(path, expected_rows=rows, expected_units=units)
    assert raised.value.quality_report["rules"][rule]["failures"] == 1


def test_errors_have_bounded_examples(data_workspace, observation_row):
    path = write_observations(data_workspace, [observation_row()] * 100)
    with pytest.raises(ValidationError) as raised:
        validate_observations(path, expected_rows=100, expected_units=1)
    report = raised.value.quality_report
    assert report["rules"]["unique_records"]["failures"] == 99
    assert len(report["rules"]["unique_records"]["examples"]) == 5
    assert len(str(raised.value)) < 300


@pytest.mark.parametrize("exists", [True, False])
def test_empty_and_unreadable_inputs_have_json_safe_reports(data_workspace, exists):
    path = data_workspace / "empty.txt"
    if exists:
        path.write_bytes(b"")
    with pytest.raises(ValidationError) as raised:
        validate_observations(path, expected_rows=1, expected_units=1)
    report = raised.value.quality_report
    assert report["columns"]["cycle"]["mean"] is None
    assert report["columns"]["cycle"]["population_stddev"] is None
    assert report["cycles"]["min"] is None
    canonical_json(report)


def test_extreme_finite_values_do_not_overflow_quality_json(data_workspace, observation_row):
    lines = []
    for cycle, value in enumerate(("1.79e308", "-1.79e308"), 1):
        tokens = observation_row(cycle=cycle).split()
        tokens[-1] = value
        lines.append(" ".join(tokens))
    result = validate_observations(
        write_observations(data_workspace, lines), expected_rows=2, expected_units=1
    )
    assert result.report["columns"]["sensor_21"]["mean"] == 0
    assert result.report["columns"]["sensor_21"]["population_stddev"] == 1.79e308
    canonical_json(result.report)


def test_labels_are_separate_int32_and_aligned_by_line(data_workspace):
    path = data_workspace / "rul.txt"
    path.write_text("  27 \n0\n2147483647\n27\n", encoding="ascii")
    result = validate_labels(path, expected_units=4, expected_rows=4)
    assert result.table.schema == LABEL_SCHEMA
    assert result.table.to_pydict() == {"unit_id": [1, 2, 3, 4], "rul": [27, 0, 2147483647, 27]}
    assert result.report["status"] == "passed"


@pytest.mark.parametrize(
    "text", ["-1\n", "1.0\n", "2147483648\n", "NaN\n", "inf\n", "1 2\n", "\n", ""]
)
def test_invalid_labels_fail(data_workspace, text):
    path = data_workspace / "rul.txt"
    path.write_text(text, encoding="ascii")
    with pytest.raises(ValidationError):
        validate_labels(path, expected_units=1)


def test_label_cardinality_is_exact(data_workspace):
    path = data_workspace / "rul.txt"
    path.write_text("3\n4\n", encoding="ascii")
    with pytest.raises(ValidationError) as raised:
        validate_labels(path, expected_units=3, expected_rows=2)
    assert raised.value.quality_report["rules"]["rul_cardinality"]["failures"] == 1


def test_split_local_numeric_ids_are_not_leakage(data_workspace, observation_row):
    train = parsed(data_workspace, observation_row, name="train.txt").table
    test = parsed(data_workspace, observation_row, base=20, name="test.txt").table
    report = validate_split_integrity(train, test, labels(data_workspace).table)
    assert report["status"] == "passed"
    assert report["counts"]["copied_test_trajectories"] == 0


@pytest.mark.parametrize("full", [True, False])
def test_copied_test_trajectory_ignores_unit_id(data_workspace, observation_row, full):
    train_lines = [observation_row(1, cycle, 99) for cycle in (1, 2, 3)]
    train_lines += [observation_row(2, cycle, 10) for cycle in range(1, 3 if full else 4)]
    train = validate_observations(
        write_observations(data_workspace, train_lines, "train.txt"),
        expected_rows=len(train_lines),
        expected_units=2,
    ).table
    test = parsed(data_workspace, observation_row, name="test.txt", cycles=2).table
    with pytest.raises(ValidationError) as raised:
        validate_split_integrity(train, test, labels(data_workspace).table)
    report = raised.value.quality_report
    assert report["counts"]["copied_test_trajectories"] == 1
    assert report["rules"]["no_copied_test_trajectory"]["examples"][0]["match"] == (
        "full" if full else "prefix"
    )
    assert report["rules"]["no_copied_test_unit"]["examples"][0]["train_unit_id"] == 2


def test_longer_test_is_not_exact_training_prefix(data_workspace, observation_row):
    train = parsed(data_workspace, observation_row, name="train.txt", cycles=2).table
    test = parsed(data_workspace, observation_row, name="test.txt", cycles=3).table
    assert validate_split_integrity(train, test, labels(data_workspace).table)["status"] == "passed"


def test_signed_zero_cannot_evade_copy_detection(data_workspace, observation_row):
    train = parsed(data_workspace, observation_row, name="train.txt").table
    lines = [observation_row(cycle=cycle).replace("-0.0", "0.0") for cycle in (1, 2)]
    test = validate_observations(
        write_observations(data_workspace, lines, "test.txt"), expected_rows=2, expected_units=1
    ).table
    with pytest.raises(ValidationError):
        validate_split_integrity(train, test, labels(data_workspace).table)


def test_integrity_rejects_incorrect_rul_unit_alignment(data_workspace, observation_row):
    train = parsed(data_workspace, observation_row, name="train.txt").table
    test = parsed(data_workspace, observation_row, base=20, name="test.txt").table
    wrong = pa.Table.from_arrays(
        [pa.array([2], type=pa.int32()), pa.array([3], type=pa.int32())], schema=LABEL_SCHEMA
    )
    with pytest.raises(ValidationError) as raised:
        validate_split_integrity(train, test, wrong)
    assert raised.value.quality_report["rules"]["rul_alignment"]["failures"] == 1
    with pytest.raises(ValidationError):
        validate_label_table(wrong, expected_units=1)


def test_typed_schema_and_invalid_values_are_rechecked(data_workspace, observation_row):
    good = parsed(data_workspace, observation_row).table
    assert validate_observation_table(good, expected_rows=3, expected_units=1).report["status"] == (
        "passed"
    )
    bad_schema = good.set_column(0, "unit_id", pa.array([1, 1, 1], type=pa.int64()))
    with pytest.raises(ValidationError):
        validate_observation_table(bad_schema, expected_rows=3, expected_units=1)
    for value in (None, float("nan"), float("inf")):
        arrays = [good.column(index) for index in range(26)]
        arrays[2] = pa.array([value, 0.0, 0.0], type=pa.float64())
        bad_value = pa.Table.from_arrays(arrays, schema=OBSERVATION_SCHEMA)
        with pytest.raises(ValidationError):
            validate_observation_table(bad_value, expected_rows=3, expected_units=1)
