"""Cloud-agnostic, content-addressed C-MAPSS features for uncapped RUL regression."""

import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from epm_platform.data.curation import verify_curated
from epm_platform.data.errors import DataError
from epm_platform.data.manifest import canonical_json, fingerprint, hash_file, write_json_new
from epm_platform.data.spec import OBSERVATION_COLUMNS, SUBSETS, load_spec
from epm_platform.data.validation import LABEL_SCHEMA, OBSERVATION_SCHEMA

_SETTINGS = ("setting_1", "setting_2", "setting_3")
_TREND_SENSORS = ("sensor_03", "sensor_04", "sensor_11")
FEATURE_COLUMNS = (
    "cycle",
    *_SETTINGS,
    *(f"sensor_{number:02d}" for number in range(1, 22)),
    *(f"{sensor}_{stat}10" for sensor in _TREND_SENSORS for stat in ("mean", "slope")),
    *(f"{setting}_mean10" for setting in _SETTINGS),
    "history_count",
)
METADATA_COLUMNS = ("subset", "unit_id", "cycle", "split", "rul", "sample_weight")
FEATURE_SCHEMA = pa.schema(
    [
        pa.field(name, pa.int32() if name in ("cycle", "history_count") else pa.float64(), False)
        for name in FEATURE_COLUMNS
    ]
)
TABLE_SCHEMA = pa.schema(
    [pa.field("subset", pa.string(), False), pa.field("unit_id", pa.int32(), False)]
    + list(FEATURE_SCHEMA)
    + [
        pa.field("split", pa.string(), False),
        pa.field("rul", pa.float64(), False),
        pa.field("sample_weight", pa.float64(), False),
    ]
)
_APPROVED_CONFIG = {
    "schema_version": 1,
    "recipe_version": "1",
    "source": {
        "asset_name": "epm-cmapss-curated",
        "asset_version": "d-xjsfezqpozevgso26sct6tpodm",
        "manifest_sha256": "ba6452660f76495349daf4853f4dee1b511a299ead831d150dab757340f33640",
    },
    "subsets": list(SUBSETS),
    "target": "uncapped_rul",
    "features": {
        "window": 10,
        "trend_sensors": list(_TREND_SENSORS),
        "setting_means": True,
        "history_count": True,
    },
    "split": {
        "strategy": "engine_hash_holdout",
        "seed": 42,
        "validation_fraction": 0.2,
        "validation_cut_min_fraction": 0.5,
        "validation_cut_max_fraction": 0.8,
    },
    "weighting": "equal_training_engine",
}
_SPLITS = ("train", "validation", "test")
_CONTENT_PATHS = tuple(
    sorted(
        [f"{s}.parquet" for s in _SPLITS]
        + [
            "splits.json",
            "feature-summary.json",
        ]
    )
)
_ALL_PATHS = frozenset((*_CONTENT_PATHS, "manifest.json", "_SUCCESS.json"))
_SHA256 = re.compile(r"[a-f0-9]{64}")
_VALIDATION_POLICY = (
    "One deterministic endpoint at 50-80% of retrospectively known life per held-out engine; "
    "on-policy offline validation is selection-biased and does not establish field performance."
)


@dataclass(frozen=True)
class FeatureBundle:
    root: Path
    version: str
    manifest_sha256: str
    summary: dict


def _check_config(config: dict) -> None:
    if canonical_json(config) != canonical_json(_APPROVED_CONFIG):
        raise DataError("Feature configuration must exactly match the approved recipe and source.")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataError("Feature JSON must not contain duplicate keys.")
        result[key] = value
    return result


def _read_json(path: Path, *, canonical: bool = True) -> dict:
    raw = path.read_bytes()
    result = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(result, dict) or (canonical and canonical_json(result) != raw):
        raise DataError("Feature JSON must be a canonical object with a trailing newline.")
    return result


def _reject_link(path: Path) -> None:
    if path.is_symlink() or path.is_junction():
        raise DataError("Feature bundles must not contain symbolic links or junctions.")


def _verify_source(source_root: Path, source_spec_path: Path, config: dict) -> None:
    verify_curated(source_root, load_spec(source_spec_path))
    if hash_file(source_root / "manifest.json") != config["source"]["manifest_sha256"]:
        raise DataError("Curated source manifest does not match the approved pinned digest.")


def _require_schema(table: pa.Table, schema: pa.Schema) -> None:
    if not table.schema.equals(schema, check_metadata=True) or any(
        column.null_count for column in table.columns
    ):
        raise DataError("Invalid feature/source table schema, metadata, or null values.")


def _rolling(values: list[float], cycles: list[int], *, slope: bool) -> tuple[list, list]:
    means, slopes = [], []
    for end in range(1, len(values) + 1):
        start = max(0, end - 10)
        window = values[start:end]
        mean = math.fsum(window) / len(window)
        means.append(mean)
        if not slope:
            continue
        if len(window) == 1:
            slopes.append(0.0)
            continue
        # Local cycle coordinates keep OLS stable and independent of any future rows.
        xs = [cycle - cycles[start] for cycle in cycles[start:end]]
        x_mean = math.fsum(xs) / len(xs)
        slopes.append(
            math.fsum((x - x_mean) * (y - mean) for x, y in zip(xs, window, strict=True))
            / math.fsum((x - x_mean) ** 2 for x in xs)
        )
    return means, slopes


def features_for_engine(table: pa.Table) -> pa.Table:
    """Return exactly 35 causal features for one ordered engine prefix starting at cycle 1.

    The input must contain only the original typed observation columns. No target,
    split, engine lifetime, or information beyond each row enters the calculation.
    """
    _require_schema(table, OBSERVATION_SCHEMA)
    if not table.num_rows:
        raise DataError("An engine prefix must not be empty.")
    values = table.to_pydict()
    units, cycles = values["unit_id"], values["cycle"]
    if (
        units[0] < 1
        or any(unit != units[0] for unit in units)
        or cycles != list(range(1, len(cycles) + 1))
    ):
        raise DataError("Features require a single engine with contiguous cycles starting at 1.")
    if any(not math.isfinite(value) for name in OBSERVATION_COLUMNS[2:] for value in values[name]):
        raise DataError("Source feature values must be finite.")
    result = {name: values[name] for name in OBSERVATION_COLUMNS[1:]}
    for sensor in _TREND_SENSORS:
        result[f"{sensor}_mean10"], result[f"{sensor}_slope10"] = _rolling(
            values[sensor], cycles, slope=True
        )
    for setting in _SETTINGS:
        result[f"{setting}_mean10"] = _rolling(values[setting], cycles, slope=False)[0]
    result["history_count"] = [min(cycle, 10) for cycle in cycles]
    if any(not math.isfinite(value) for column in result.values() for value in column):
        raise DataError("Derived feature values must be finite.")
    return pa.Table.from_pydict(result, schema=FEATURE_SCHEMA)


def _engines(table: pa.Table):
    _require_schema(table, OBSERVATION_SCHEMA)
    units = table["unit_id"].to_pylist()
    start, expected_unit = 0, 1
    while start < len(units):
        unit = units[start]
        if unit != expected_unit:
            raise DataError("Source engines must be contiguous and ordered by unit ID.")
        end = start + 1
        while end < len(units) and units[end] == unit:
            end += 1
        yield unit, table.slice(start, end - start)
        start, expected_unit = end, expected_unit + 1


def _holdout(subset: str, units: list[int]) -> set[int]:
    ordered = sorted(
        units,
        key=lambda unit: (hashlib.sha256(f"42|{subset}|{unit}".encode()).digest(), unit),
    )
    count = math.ceil(len(units) * 0.2)
    if not units or count >= len(units):
        raise DataError("Each subset needs at least two training-source engines for holdout.")
    return set(ordered[:count])


def _cut(subset: str, unit: int, life: int) -> int:
    low, high = math.ceil(life * 0.5), math.floor(life * 0.8)
    if low > high:
        raise DataError("Engine lifetime cannot support the approved validation censor interval.")
    number = int.from_bytes(hashlib.sha256(f"cut|42|{subset}|{unit}".encode()).digest(), "big")
    return low + number % (high - low + 1)


def _partition(
    features: pa.Table, assignment: dict, total_rows: int, total_engines: int
) -> pa.Table:
    split, life = assignment["split"], assignment["source_rows"]
    if split != "train":
        features = features.slice(features.num_rows - 1, 1)
    count = features.num_rows
    targets = (
        [float(assignment["supplied_rul"])]
        if split == "test"
        else [float(life - cycle) for cycle in features["cycle"].to_pylist()]
    )
    weight = total_rows / total_engines / life if split == "train" else 1.0
    return pa.Table.from_arrays(
        [
            pa.array([assignment["subset"]] * count),
            pa.array([assignment["unit_id"]] * count, type=pa.int32()),
        ]
        + list(features.columns)
        + [pa.array([split] * count), pa.array(targets), pa.array([weight] * count)],
        schema=TABLE_SCHEMA,
    )


def _counts(tables: dict[str, pa.Table]) -> tuple[dict, dict]:
    return (
        {split: table.num_rows for split, table in tables.items()},
        {
            split: len(
                set(zip(table["subset"].to_pylist(), table["unit_id"].to_pylist(), strict=True))
            )
            for split, table in tables.items()
        },
    )


def _summary(tables: dict[str, pa.Table], assignments: list[dict]) -> dict:
    rows, engines = _counts(tables)
    by_subset = {}
    for subset in SUBSETS:
        entries = [entry for entry in assignments if entry["subset"] == subset]
        by_subset[subset] = {
            "row_counts": {
                split: sum(
                    e["source_rows"] if split == "train" else 1
                    for e in entries
                    if e["split"] == split
                )
                for split in _SPLITS
            },
            "engine_counts": {
                split: sum(e["split"] == split for e in entries) for split in _SPLITS
            },
        }
    return {
        "schema_version": 1,
        "status": "passed",
        "feature_count": len(FEATURE_COLUMNS),
        "row_counts": rows,
        "engine_counts": engines,
        "by_subset": by_subset,
        "target_ranges": {
            split: {"min": min(table["rul"].to_pylist()), "max": max(table["rul"].to_pylist())}
            for split, table in tables.items()
        },
        "training_weight_sum": math.fsum(tables["train"]["sample_weight"].to_pylist()),
        "validation_policy": _VALIDATION_POLICY,
    }


def _check_assignments(value: dict) -> list[dict]:
    if set(value) != {"schema_version", "strategy", "seed", "assignments"} or canonical_json(
        {k: v for k, v in value.items() if k != "assignments"}
    ) != canonical_json({"schema_version": 1, "strategy": "engine_hash_holdout", "seed": 42}):
        raise DataError("Invalid split assignment header.")
    entries = value["assignments"]
    if not isinstance(entries, list) or not entries:
        raise DataError("Split assignments must be a nonempty list.")
    keys = []
    for entry in entries:
        required = {"subset", "unit_id", "source_split", "source_rows", "split", "cut_cycle"}
        if isinstance(entry, dict) and entry.get("source_split") == "test":
            required.add("supplied_rul")
        if (
            not isinstance(entry, dict)
            or set(entry) != required
            or entry["subset"] not in SUBSETS
            or entry["source_split"] not in ("train", "test")
            or entry["split"] not in _SPLITS
            or any(
                type(entry[k]) is not int or not 0 < entry[k] < 2**31
                for k in ("unit_id", "source_rows")
            )
        ):
            raise DataError("Invalid source engine assignment.")
        key = (entry["subset"], entry["source_split"] == "test", entry["unit_id"])
        keys.append(key)
        if entry["source_split"] == "test":
            if (
                entry["split"] != "test"
                or entry["cut_cycle"] is not None
                or type(entry["supplied_rul"]) is not int
                or not 0 <= entry["supplied_rul"] < 2**31
            ):
                raise DataError("Test engines and original endpoint labels must remain test-only.")
        elif entry["split"] == "test":
            raise DataError("Training-source engines cannot enter the test partition.")
        elif entry["split"] == "validation":
            if type(entry["cut_cycle"]) is not int or entry["cut_cycle"] != _cut(
                entry["subset"], entry["unit_id"], entry["source_rows"]
            ):
                raise DataError("Validation cut does not match the deterministic censor policy.")
        elif entry["cut_cycle"] is not None:
            raise DataError("Training engines must retain their complete trajectory.")
    if keys != sorted(set(keys)):
        raise DataError("Engine assignments must be unique and in canonical order.")
    for subset in SUBSETS:
        for source_split in ("train", "test"):
            group = [
                e for e in entries if e["subset"] == subset and e["source_split"] == source_split
            ]
            units = [e["unit_id"] for e in group]
            if not units or units != list(range(1, len(units) + 1)):
                raise DataError("Source engine assignments are incomplete or noncontiguous.")
            if source_split == "train":
                heldout = _holdout(subset, units)
                if any(
                    e["split"] != ("validation" if e["unit_id"] in heldout else "train")
                    for e in group
                ):
                    raise DataError("Engine split does not match the deterministic hash holdout.")
    return entries


def _check_tables(tables: dict[str, pa.Table], assignments: list[dict]) -> None:
    train_entries = [e for e in assignments if e["split"] == "train"]
    total_rows = sum(e["source_rows"] for e in train_entries)
    for split, table in tables.items():
        _require_schema(table, TABLE_SCHEMA)
        data = table.to_pydict()
        if any(
            not math.isfinite(v)
            for name in (*FEATURE_COLUMNS, "rul", "sample_weight")
            for v in data[name]
        ):
            raise DataError("Features, targets, and weights must be finite.")
        expected = [e for e in assignments if e["split"] == split]
        offset = 0
        for entry in expected:
            life = entry["source_rows"]
            count = life if split == "train" else 1
            if count > table.num_rows - offset:
                raise DataError("Engine assignment exceeds the available partition rows.")
            end = offset + count
            cycles = (
                list(range(1, life + 1))
                if split == "train"
                else [entry["cut_cycle"] if split == "validation" else life]
            )
            if (
                data["subset"][offset:end] != [entry["subset"]] * count
                or data["unit_id"][offset:end] != [entry["unit_id"]] * count
                or data["split"][offset:end] != [split] * count
                or data["cycle"][offset:end] != cycles
                or data["history_count"][offset:end] != [min(c, 10) for c in cycles]
            ):
                raise DataError(
                    "Partition rows violate source assignment, group boundary, or cycles."
                )
            labels = (
                [float(entry["supplied_rul"])]
                if split == "test"
                else [float(life - cycle) for cycle in cycles]
            )
            weight = total_rows / len(train_entries) / life if split == "train" else 1.0
            if data["rul"][offset:end] != labels or any(
                not math.isclose(v, weight, rel_tol=1e-12, abs_tol=0)
                for v in data["sample_weight"][offset:end]
            ):
                raise DataError(
                    "Targets or engine-balanced sample weights disagree with assignments."
                )
            if split == "train":
                observations = pa.Table.from_arrays(
                    [table[name].slice(offset, count) for name in OBSERVATION_COLUMNS],
                    schema=OBSERVATION_SCHEMA,
                )
                if (
                    not table.select(FEATURE_COLUMNS)
                    .slice(offset, count)
                    .equals(features_for_engine(observations))
                ):
                    raise DataError("Training rolling features are not the approved causal recipe.")
            offset = end
        if offset != table.num_rows:
            raise DataError("Partition row counts disagree with engine assignments.")


def _verify_features(root: Path) -> dict:
    _reject_link(root)
    if not root.is_dir():
        raise DataError("Feature bundle directory is missing.")
    children = list(root.iterdir())
    for child in children:
        _reject_link(child)
        if not child.is_file():
            raise DataError("Feature bundles contain only the exact flat file inventory.")
    if {child.name for child in children} != _ALL_PATHS:
        raise DataError("Feature bundle inventory is incomplete or unexpected.")
    manifest = _read_json(root / "manifest.json")
    config = manifest["config"]
    _check_config(config)
    fixed = {
        "schema_version": 1,
        "recipe_version": "1",
        "dataset": "nasa-cmapss-ml-ready",
        "source": config["source"],
        "config": config,
        "config_sha256": hashlib.sha256(canonical_json(config)).hexdigest(),
        "feature_columns": list(FEATURE_COLUMNS),
        "metadata_columns": list(METADATA_COLUMNS),
        "target": "rul",
    }
    if set(manifest) != {
        *fixed,
        "writer",
        "row_counts",
        "engine_counts",
        "files",
    } or canonical_json({k: manifest[k] for k in fixed}) != canonical_json(fixed):
        raise DataError("Feature manifest schema or approved feature/target contract is invalid.")
    writer = manifest["writer"]
    if (
        not isinstance(writer, dict)
        or set(writer) != {"name", "version", "compression"}
        or writer["name"] != "pyarrow"
        or writer["compression"] != "zstd"
        or not isinstance(writer["version"], str)
        or not writer["version"]
    ):
        raise DataError("Feature writer provenance is invalid.")
    entries = manifest["files"]
    if not isinstance(entries, list) or len(entries) != len(_CONTENT_PATHS):
        raise DataError("Feature content inventory must be an exact list.")
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256", "size_bytes"}
            or not isinstance(entry["path"], str)
            or entry["path"] not in _CONTENT_PATHS
            or not isinstance(entry["sha256"], str)
            or not _SHA256.fullmatch(entry["sha256"])
            or type(entry["size_bytes"]) is not int
            or entry["size_bytes"] <= 0
        ):
            raise DataError(
                "Invalid content entry; only exact flat allowlisted paths are accepted."
            )
    if [entry["path"] for entry in entries] != list(_CONTENT_PATHS):
        raise DataError("Feature content paths must be unique and sorted.")
    digest = hash_file(root / "manifest.json")
    if _read_json(root / "_SUCCESS.json") != {
        "manifest_sha256": digest,
        "version": "sha256-" + digest,
    }:
        raise DataError("Feature completion marker does not match the manifest digest.")
    for entry in entries:
        if fingerprint(root / entry["path"]) != {k: entry[k] for k in ("sha256", "size_bytes")}:
            raise DataError("Feature content hash or size disagrees with its manifest.")
    assignments = _check_assignments(_read_json(root / "splits.json"))
    tables = {split: pq.ParquetFile(root / f"{split}.parquet").read() for split in _SPLITS}
    _check_tables(tables, assignments)
    row_counts, engine_counts = _counts(tables)
    if canonical_json(manifest["row_counts"]) != canonical_json(row_counts) or canonical_json(
        manifest["engine_counts"]
    ) != canonical_json(engine_counts):
        raise DataError("Feature manifest counts do not match the partition tables.")
    if canonical_json(_read_json(root / "feature-summary.json")) != canonical_json(
        _summary(tables, assignments)
    ):
        raise DataError("Feature summary does not match the verified partition tables.")
    return manifest


def verify_features(root: Path) -> dict:
    """Read-only integrity and semantic verification; mounted folder names are unrestricted."""
    try:
        return _verify_features(Path(root))
    except DataError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
        pa.ArrowException,
    ):
        raise DataError("Feature bundle is unreadable or malformed.") from None


def build_features(
    source_root: Path,
    output_root: Path,
    config_path: Path,
    source_spec_path: Path,
) -> FeatureBundle:
    """Verify pinned Phase 2 input, build privately, then atomically finalize without overwrite."""
    staging = None
    try:
        source_root, output_root = Path(source_root), Path(output_root)
        config = _read_json(Path(config_path), canonical=False)
        _check_config(config)
        _verify_source(source_root, Path(source_spec_path), config)
        if output_root.resolve() == source_root.resolve() or source_root.resolve() in (
            output_root.resolve().parents
        ):
            raise DataError("Feature output cannot be inside the immutable curated source.")
        _reject_link(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        staging = output_root / (".features-" + uuid4().hex)
        staging.mkdir()
        assignments, observations = [], {}
        for subset in SUBSETS:
            train = list(_engines(pq.ParquetFile(source_root / subset / "train.parquet").read()))
            heldout = _holdout(subset, [unit for unit, _ in train])
            for unit, engine in train:
                split = "validation" if unit in heldout else "train"
                cut = _cut(subset, unit, engine.num_rows) if split == "validation" else None
                assignments.append(
                    {
                        "subset": subset,
                        "unit_id": unit,
                        "source_split": "train",
                        "source_rows": engine.num_rows,
                        "split": split,
                        "cut_cycle": cut,
                    }
                )
                observations[(subset, "train", unit)] = engine.slice(0, cut) if cut else engine
            labels = pq.ParquetFile(source_root / subset / "test_rul.parquet").read()
            _require_schema(labels, LABEL_SCHEMA)
            test = list(_engines(pq.ParquetFile(source_root / subset / "test.parquet").read()))
            if labels["unit_id"].to_pylist() != [unit for unit, _ in test]:
                raise DataError(
                    "Original test RUL labels must map exactly to observed test unit IDs."
                )
            for (unit, engine), rul in zip(test, labels["rul"].to_pylist(), strict=True):
                assignments.append(
                    {
                        "subset": subset,
                        "unit_id": unit,
                        "source_split": "test",
                        "source_rows": engine.num_rows,
                        "split": "test",
                        "cut_cycle": None,
                        "supplied_rul": rul,
                    }
                )
                observations[(subset, "test", unit)] = engine
        splits = {
            "schema_version": 1,
            "strategy": "engine_hash_holdout",
            "seed": 42,
            "assignments": assignments,
        }
        _check_assignments(splits)
        train_entries = [e for e in assignments if e["split"] == "train"]
        total_rows = sum(e["source_rows"] for e in train_entries)
        pieces = {split: [] for split in _SPLITS}
        for entry in assignments:
            engine = observations[(entry["subset"], entry["source_split"], entry["unit_id"])]
            pieces[entry["split"]].append(
                _partition(
                    features_for_engine(engine),
                    entry,
                    total_rows,
                    len(train_entries),
                )
            )
        tables = {
            split: pa.concat_tables(parts).combine_chunks() for split, parts in pieces.items()
        }
        for split, table in tables.items():
            pq.write_table(
                table,
                staging / f"{split}.parquet",
                row_group_size=65536,
                compression="zstd",
                compression_level=3,
                use_dictionary=False,
                write_statistics=True,
                version="2.6",
                data_page_version="1.0",
                store_schema=True,
            )
        summary = _summary(tables, assignments)
        write_json_new(staging / "splits.json", splits)
        write_json_new(staging / "feature-summary.json", summary)
        manifest = {
            "schema_version": 1,
            "recipe_version": "1",
            "dataset": "nasa-cmapss-ml-ready",
            "source": config["source"],
            "config": config,
            "config_sha256": hashlib.sha256(canonical_json(config)).hexdigest(),
            "feature_columns": list(FEATURE_COLUMNS),
            "metadata_columns": list(METADATA_COLUMNS),
            "target": "rul",
            "row_counts": summary["row_counts"],
            "engine_counts": summary["engine_counts"],
            "writer": {"name": "pyarrow", "version": pa.__version__, "compression": "zstd"},
            "files": [{"path": name, **fingerprint(staging / name)} for name in _CONTENT_PATHS],
        }
        write_json_new(staging / "manifest.json", manifest)
        digest = hash_file(staging / "manifest.json")
        version = "sha256-" + digest
        write_json_new(staging / "_SUCCESS.json", {"manifest_sha256": digest, "version": version})
        verify_features(staging)
        _verify_source(source_root, Path(source_spec_path), config)
        final = output_root / version
        if final.exists() or final.is_symlink() or final.is_junction():
            verify_features(final)
        else:
            try:
                staging.rename(final)
                staging = None
            except OSError:
                if not final.exists():
                    raise
                verify_features(final)
        if hash_file(final / "manifest.json") != digest or final.name != version:
            raise DataError(
                "Existing finalized feature version does not match its directory identity."
            )
        return FeatureBundle(final, version, digest, summary)
    except DataError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
        pa.ArrowException,
    ):
        raise DataError("Feature build failed; no unverified bundle was finalized.") from None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
