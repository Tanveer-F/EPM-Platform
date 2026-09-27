"""Lossless C-MAPSS parsing and bounded, deterministic data-quality checks."""

import hashlib
import math
import re
import struct
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path

import pyarrow as pa

from epm_platform.data.errors import DataError
from epm_platform.data.spec import LABEL_COLUMNS, OBSERVATION_COLUMNS

OBSERVATION_SCHEMA = pa.schema(
    [
        pa.field(name, pa.int32() if index < 2 else pa.float64(), nullable=False)
        for index, name in enumerate(OBSERVATION_COLUMNS)
    ]
)
LABEL_SCHEMA = pa.schema([pa.field(name, pa.int32(), nullable=False) for name in LABEL_COLUMNS])
_INT32_MAX = 2**31 - 1
_INTEGER = re.compile(r"[+-]?[0-9]+")
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_MISSING = frozenset({"na", "n/a", "null", "none", "nan", "+nan", "-nan", "<na>"})
_INFINITY = frozenset({"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"})
_MAX_EXAMPLES = 5


class ValidationError(DataError):
    """A rejected input, with a JSON-safe report and bounded examples."""

    def __init__(self, quality_report: dict):
        self.quality_report = quality_report
        failed = [
            name for name, rule in quality_report.get("rules", {}).items() if rule["failures"]
        ]
        super().__init__("C-MAPSS validation failed: " + ", ".join(failed[:5]) + ".")


@dataclass(frozen=True)
class ValidationResult:
    table: pa.Table
    report: dict


@dataclass
class _Checks:
    rules: dict = field(default_factory=dict)

    def test(self, name: str, passed: bool, **example: int | str) -> None:
        rule = self.rules.setdefault(name, {"checks": 0, "failures": 0, "examples": []})
        rule["checks"] += 1
        if not passed:
            rule["failures"] += 1
            if example and len(rule["examples"]) < _MAX_EXAMPLES:
                rule["examples"].append(example)

    def finish(self, report: dict) -> dict:
        report["rules"] = self.rules
        report["status"] = "failed" if any(r["failures"] for r in self.rules.values()) else "passed"
        if report["status"] != "passed":
            raise ValidationError(report)
        return report


def _parse(
    path: Path, names: tuple[str, ...], integer_minima: dict[int, int], checks: _Checks
) -> tuple[list[tuple], list[int], int, list[int], list[int]]:
    rows, row_numbers = [], []
    nulls, nonfinite = [0] * len(names), [0] * len(names)
    source_rows = 0
    try:
        with path.open("rb") as stream:
            for source_rows, raw in enumerate(stream, 1):
                try:
                    line = raw.decode("ascii")
                except UnicodeDecodeError:
                    checks.test("ascii", False, row=source_rows)
                    continue
                checks.test("ascii", True)
                tokens = line.split()
                checks.test("column_count", len(tokens) == len(names), row=source_rows)
                if len(tokens) != len(names):
                    continue
                parsed = []
                for column, token in enumerate(tokens):
                    example = {"row": source_rows, "column": names[column]}
                    missing = token.lower() in _MISSING
                    checks.test("no_missing_values", not missing, **example)
                    if missing:
                        nulls[column] += 1
                        parsed.append(None)
                        continue
                    if token.lower() in _INFINITY:
                        nonfinite[column] += 1
                        checks.test("finite_values", False, **example)
                        parsed.append(None)
                        continue
                    integer = column in integer_minima
                    pattern = _INTEGER if integer else _NUMBER
                    numeric = pattern.fullmatch(token) is not None
                    checks.test(
                        "integer_tokens" if integer else "numeric_tokens", numeric, **example
                    )
                    if not numeric:
                        parsed.append(None)
                        continue
                    try:
                        value = int(token) if integer else float(token)
                    except (ValueError, OverflowError):
                        checks.test("int32_range" if integer else "finite_values", False, **example)
                        parsed.append(None)
                        continue
                    if integer:
                        valid = integer_minima[column] <= value <= _INT32_MAX
                        checks.test("int32_range", valid, **example)
                    else:
                        valid = math.isfinite(value)
                        checks.test("finite_values", valid, **example)
                        if not valid:
                            nonfinite[column] += 1
                    parsed.append(value if valid else None)
                if all(value is not None for value in parsed):
                    rows.append(tuple(parsed))
                    row_numbers.append(source_rows)
        checks.test("readable_source", True)
    except OSError:
        checks.test("readable_source", False)
    return rows, row_numbers, source_rows, nulls, nonfinite


def _statistics(
    rows: list[tuple], names: tuple[str, ...], nulls: list[int], nonfinite: list[int]
) -> dict:
    result = {}
    for index, name in enumerate(names):
        values = [row[index] for row in rows]
        count = len(values)
        minimum, maximum = (min(values), max(values)) if values else (None, None)
        mean, stddev = None, None
        if values:
            # Scaling only the quality calculation avoids overflow, never changes stored values.
            scale = max(abs(minimum), abs(maximum))
            if scale:
                normalized_mean = math.fsum(value / scale for value in values) / count
                mean = scale * normalized_mean
                variance = math.fsum((value / scale - normalized_mean) ** 2 for value in values)
                stddev = scale * min(1.0, math.sqrt(variance / count))
            else:
                mean = stddev = 0.0
        distinct = len(set(values))
        result[name] = {
            "count": count,
            "null_count": nulls[index],
            "nonfinite_count": nonfinite[index],
            "min": minimum,
            "max": maximum,
            "mean": mean,
            "population_stddev": stddev,
            "distinct_count": distinct,
            "constant": bool(count and distinct == 1),
        }
    return result


def _base_report(
    kind: str,
    rows: list[tuple],
    names: tuple[str, ...],
    source_rows: int,
    expected_rows: int,
    expected_units: int,
    nulls: list[int],
    nonfinite: list[int],
    checks: _Checks,
) -> dict:
    checks.test("nonempty", source_rows > 0)
    checks.test("expected_rows", source_rows == expected_rows)
    checks.test("all_rows_parsed", len(rows) == source_rows)
    units = {row[0] for row in rows}
    checks.test("expected_units", len(units) == expected_units)
    checks.test("contiguous_unit_ids", sorted(units) == list(range(1, len(units) + 1)))
    columns = _statistics(rows, names, nulls, nonfinite)
    constants = [name for name, summary in columns.items() if summary["constant"]]
    return {
        "kind": kind,
        "rows": {"source": source_rows, "parsed": len(rows), "expected": expected_rows},
        "units": {
            "count": len(units),
            "expected": expected_units,
            "min": min(units) if units else None,
            "max": max(units) if units else None,
        },
        "counts": {"null_values": sum(nulls), "nonfinite_values": sum(nonfinite)},
        "columns": columns,
        "warnings": [{"rule": "zero_variance", "columns": constants}] if constants else [],
    }


def _observation_report(
    rows: list[tuple],
    row_numbers: list[int],
    source_rows: int,
    expected_rows: int,
    expected_units: int,
    nulls: list[int],
    nonfinite: list[int],
    checks: _Checks,
) -> dict:
    report = _base_report(
        "observations",
        rows,
        OBSERVATION_COLUMNS,
        source_rows,
        expected_rows,
        expected_units,
        nulls,
        nonfinite,
        checks,
    )
    seen_records, seen_keys, units = set(), {}, {}
    previous_unit, previous_cycle = None, None
    duplicates = duplicate_keys = conflicts = 0
    for number, record in zip(row_numbers, rows, strict=True):
        unit, cycle = record[:2]
        example = {"row": number, "unit_id": unit, "cycle": cycle}
        duplicate = record in seen_records
        duplicate_key = (unit, cycle) in seen_keys
        conflict = duplicate_key and seen_keys[unit, cycle] != record
        checks.test("unique_records", not duplicate, **example)
        checks.test("unique_unit_cycle", not duplicate_key, **example)
        checks.test("no_conflicting_unit_cycle", not conflict, **example)
        duplicates += duplicate
        duplicate_keys += duplicate_key
        conflicts += conflict
        seen_records.add(record)
        seen_keys.setdefault((unit, cycle), record)
        if unit != previous_unit:
            checks.test("unit_block_order", unit == (previous_unit or 0) + 1, **example)
            checks.test("no_unit_reappearance", unit not in units, **example)
            checks.test("cycles_start_at_one", cycle == 1, **example)
        else:
            checks.test("continuous_cycles", cycle == previous_cycle + 1, **example)
        summary = units.setdefault(
            unit,
            {
                "unit_id": unit,
                "rows": 0,
                "first_cycle": cycle,
                "last_cycle": cycle,
                "min_cycle": cycle,
                "max_cycle": cycle,
            },
        )
        summary["rows"] += 1
        summary["last_cycle"] = cycle
        summary["min_cycle"] = min(summary["min_cycle"], cycle)
        summary["max_cycle"] = max(summary["max_cycle"], cycle)
        previous_unit, previous_cycle = unit, cycle
    report["cycles"] = {
        "min": min((row[1] for row in rows), default=None),
        "max": max((row[1] for row in rows), default=None),
        "per_unit": [units[unit] for unit in sorted(units)],
    }
    report["counts"].update(
        {
            "duplicate_records": duplicates,
            "duplicate_unit_cycles": duplicate_keys,
            "conflicting_unit_cycles": conflicts,
        }
    )
    return checks.finish(report)


def _label_report(
    rows: list[tuple],
    source_rows: int,
    expected_rows: int,
    expected_units: int,
    nulls: list[int],
    nonfinite: list[int],
    checks: _Checks,
) -> dict:
    report = _base_report(
        "test_rul",
        rows,
        LABEL_COLUMNS,
        source_rows,
        expected_rows,
        expected_units,
        nulls,
        nonfinite,
        checks,
    )
    checks.test("rul_cardinality", source_rows == expected_units)
    checks.test(
        "rul_index_alignment", [row[0] for row in rows] == list(range(1, expected_units + 1))
    )
    report["alignment"] = "source_line_i_to_test_unit_id_i"
    report["counts"]["duplicate_records"] = len(rows) - len(set(rows))
    return checks.finish(report)


def _table(rows: list[tuple], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array([row[index] for row in rows], type=column.type)
            for index, column in enumerate(schema)
        ],
        schema=schema,
    )


def validate_observations(
    path: Path, *, expected_rows: int, expected_units: int
) -> ValidationResult:
    """Parse ASCII observations without sorting/repair; reject on any failed rule."""
    checks = _Checks()
    rows, numbers, source_rows, nulls, nonfinite = _parse(
        path,
        OBSERVATION_COLUMNS,
        {0: 1, 1: 1},
        checks,
    )
    report = _observation_report(
        rows, numbers, source_rows, expected_rows, expected_units, nulls, nonfinite, checks
    )
    return ValidationResult(_table(rows, OBSERVATION_SCHEMA), report)


def validate_labels(
    path: Path, *, expected_units: int, expected_rows: int | None = None
) -> ValidationResult:
    """Keep supplied RUL separate; source line i labels contiguous test unit i."""
    checks = _Checks()
    values, numbers, source_rows, nulls, nonfinite = _parse(path, ("rul",), {0: 0}, checks)
    rows = [(number, value[0]) for number, value in zip(numbers, values, strict=True)]
    report = _label_report(
        rows,
        source_rows,
        expected_units if expected_rows is None else expected_rows,
        expected_units,
        [0, *nulls],
        [0, *nonfinite],
        checks,
    )
    return ValidationResult(_table(rows, LABEL_SCHEMA), report)


def _typed_rows(table: pa.Table, schema: pa.Schema, checks: _Checks) -> list[tuple]:
    checks.test("typed_schema", table.schema.equals(schema, check_metadata=True))
    if not table.schema.equals(schema, check_metadata=True):
        checks.finish({"kind": "parquet_schema"})
    rows = list(zip(*(column.to_pylist() for column in table.columns), strict=True))
    return rows


def _check_typed_values(
    rows: list[tuple], names: tuple[str, ...], integer_minima: dict[int, int], checks: _Checks
) -> tuple[list[tuple], list[int], list[int], list[int]]:
    good, numbers = [], []
    nulls, nonfinite = [0] * len(names), [0] * len(names)
    for number, row in enumerate(rows, 1):
        valid = True
        for index, value in enumerate(row):
            example = {"row": number, "column": names[index]}
            checks.test("no_missing_values", value is not None, **example)
            if value is None:
                nulls[index] += 1
                valid = False
            elif not math.isfinite(value):
                nonfinite[index] += 1
                checks.test("finite_values", False, **example)
                valid = False
            elif index in integer_minima:
                in_range = integer_minima[index] <= value <= _INT32_MAX
                checks.test("int32_range", in_range, **example)
                valid = valid and in_range
            else:
                checks.test("finite_values", True)
        if valid:
            good.append(row)
            numbers.append(number)
    return good, numbers, nulls, nonfinite


def validate_observation_table(
    table: pa.Table, *, expected_rows: int, expected_units: int
) -> ValidationResult:
    """Validate the exact curated schema, values, engine blocks and cycle sequences."""
    checks = _Checks()
    rows = _typed_rows(table, OBSERVATION_SCHEMA, checks)
    good, numbers, nulls, nonfinite = _check_typed_values(
        rows,
        OBSERVATION_COLUMNS,
        {0: 1, 1: 1},
        checks,
    )
    report = _observation_report(
        good, numbers, len(rows), expected_rows, expected_units, nulls, nonfinite, checks
    )
    return ValidationResult(table, report)


def validate_label_table(
    table: pa.Table, *, expected_units: int, expected_rows: int | None = None
) -> ValidationResult:
    """Validate int32 RUL and exact line-order unit alignment in curated Parquet."""
    checks = _Checks()
    rows = _typed_rows(table, LABEL_SCHEMA, checks)
    good, _, nulls, nonfinite = _check_typed_values(rows, LABEL_COLUMNS, {0: 1, 1: 0}, checks)
    report = _label_report(
        good,
        len(rows),
        expected_units if expected_rows is None else expected_rows,
        expected_units,
        nulls,
        nonfinite,
        checks,
    )
    return ValidationResult(table, report)


def _trajectory_row(record: tuple) -> bytes:
    # Numeric equality includes -0.0 == 0.0; this normalization affects hashes only.
    return struct.pack(">i24d", record[1], *(0.0 if value == 0 else value for value in record[2:]))


def validate_split_integrity(train: pa.Table, test: pa.Table, labels: pa.Table) -> dict:
    """Check validated tables within ONE subset, allowing split-local numeric ID reuse.

    Compare complete test trajectories with equal-length training prefixes, excluding
    unit_id. Memory is bounded by test engines, not all possible training prefixes.
    """
    checks = _Checks()
    train_rows = _typed_rows(train, OBSERVATION_SCHEMA, checks)
    test_rows = _typed_rows(test, OBSERVATION_SCHEMA, checks)
    label_rows = _typed_rows(labels, LABEL_SCHEMA, checks)
    test_ids = list(dict.fromkeys(row[0] for row in test_rows))
    checks.test("rul_alignment", [row[0] for row in label_rows] == test_ids)
    checks.test("contiguous_test_ids", test_ids == list(range(1, len(test_ids) + 1)))
    checks.test("nonempty_splits", bool(train_rows and test_rows and label_rows))
    signatures: dict[tuple[int, bytes], list[int]] = {}
    for unit, records in groupby(test_rows, key=lambda row: row[0]):
        digest, length = hashlib.sha256(), 0
        for record in records:
            digest.update(_trajectory_row(record))
            length += 1
        signatures.setdefault((length, digest.digest()), []).append(unit)
    lengths = {length for length, _ in signatures}
    copied, comparisons, train_units = set(), 0, 0
    for train_unit, records in groupby(train_rows, key=lambda row: row[0]):
        train_units += 1
        digest = hashlib.sha256()
        trajectory = list(records)
        for length, record in enumerate(trajectory, 1):
            digest.update(_trajectory_row(record))
            if length not in lengths:
                continue
            comparisons += 1
            matches = signatures.get((length, digest.digest()), [])
            checks.test(
                "no_copied_test_trajectory",
                not matches,
                train_unit_id=train_unit,
                cycles=length,
                match="full" if length == len(trajectory) else "prefix",
            )
            for test_unit in matches:
                copied.add(test_unit)
                checks.test(
                    "no_copied_test_unit",
                    False,
                    train_unit_id=train_unit,
                    test_unit_id=test_unit,
                    cycles=length,
                )
    return checks.finish(
        {
            "kind": "train_test_integrity",
            "unit_id_scope": "subset_and_split",
            "trajectory_comparison": "exact_test_to_training_prefix_excluding_unit_id",
            "counts": {
                "train_units": train_units,
                "test_units": len(test_ids),
                "rul_rows": len(label_rows),
                "trajectory_comparisons": comparisons,
                "copied_test_trajectories": len(copied),
            },
        }
    )
