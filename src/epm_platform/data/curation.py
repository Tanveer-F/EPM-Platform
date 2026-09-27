"""Content-addressed, immutable, lossless C-MAPSS Parquet bundles."""

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from epm_platform.data.errors import DataError
from epm_platform.data.manifest import canonical_json, fingerprint, hash_file, write_json_new
from epm_platform.data.spec import LABEL_COLUMNS, OBSERVATION_COLUMNS, SUBSETS, SourceSpec
from epm_platform.data.validation import (
    validate_label_table,
    validate_labels,
    validate_observation_table,
    validate_observations,
    validate_split_integrity,
)

_RECIPE_VERSION = "1"
_ROW_GROUP_SIZE = 65536
_SHA256 = re.compile(r"[a-f0-9]{64}")
_PARQUET_PATHS = tuple(
    sorted(
        f"{subset}/{split}.parquet" for subset in SUBSETS for split in ("train", "test", "test_rul")
    )
)
_CONTENT_PATHS = tuple(sorted((*_PARQUET_PATHS, "data-quality.json")))
_ALL_PATHS = frozenset((*_CONTENT_PATHS, "manifest.json", "_SUCCESS.json"))
_TRANSFORMATIONS = [
    "ASCII whitespace parsing without sorting or numeric cleanup",
    "Naming unit_id, cycle, setting_1..3 and sensor_01..21",
    "Typing unit_id/cycle as int32 and settings/sensors as float64",
    "Aligning supplied RUL line i to test unit_id i in a separate int32 table",
]
_PRESERVATION = {
    "imputed_values": 0,
    "dropped_rows": 0,
    "dropped_columns": 0,
    "reordered_rows": 0,
    "scaled_values": 0,
    "scaling": False,
    "clipped_values": 0,
    "denoised_values": 0,
    "derived_training_targets": 0,
    "combined_splits": False,
}


@dataclass(frozen=True)
class CuratedBundle:
    root: Path
    version: str
    manifest_sha256: str
    quality_report: dict


def _manifest_header(spec: SourceSpec) -> dict:
    return {
        "schema_version": 1,
        "dataset": "nasa-cmapss",
        "source_archive_sha256": spec.archive_sha256,
        "source_spec_sha256": spec.spec_sha256,
        "recipe_version": _RECIPE_VERSION,
        "writer": {"name": "pyarrow", "version": pa.__version__, "compression": "zstd"},
        "subsets": list(SUBSETS),
        "observation_columns": list(OBSERVATION_COLUMNS),
        "label_columns": list(LABEL_COLUMNS),
    }


def _quality_header(spec: SourceSpec) -> dict:
    return {
        "schema_version": 1,
        "dataset": "nasa-cmapss",
        "status": "passed",
        "source_notes": list(spec.source_notes),
        "count_authority": "reviewed source specification and original source records, not README",
        "transformations": list(_TRANSFORMATIONS),
        "preservation": dict(_PRESERVATION),
    }


def _local_path(root: Path, relative: str) -> Path:
    return root.joinpath(*relative.split("/"))


def _reject_link(path: Path) -> None:
    if path.is_symlink() or path.is_junction():
        raise DataError("Curated bundles must not contain symbolic links or junctions.")


def _inventory(root: Path) -> None:
    _reject_link(root)
    if not root.is_dir():
        raise DataError("Curated bundle directory is missing.")
    files, directories = set(), set()
    pending = [root]
    while pending:
        for path in pending.pop().iterdir():
            _reject_link(path)
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                if relative not in SUBSETS:
                    raise DataError("Unexpected directory in curated bundle.")
                directories.add(relative)
                pending.append(path)
            elif path.is_file():
                if relative not in _ALL_PATHS:
                    raise DataError("Unexpected file in curated bundle.")
                files.add(relative)
            else:
                raise DataError("Unsupported filesystem entry in curated bundle.")
    if files != _ALL_PATHS or directories != set(SUBSETS):
        raise DataError("Curated bundle inventory is incomplete.")


def _read_json(path: Path) -> dict:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise DataError("Curated JSON must be a canonical object with a trailing newline.")
    return value


def _check_saved_report(saved: dict, actual: dict) -> None:
    if not isinstance(saved, dict) or saved.get("status") != "passed":
        raise DataError("Curated data-quality status is not passed.")
    rules = saved.get("rules")
    if not isinstance(rules, dict) or not rules:
        raise DataError("Curated data-quality checks are missing.")
    for rule in rules.values():
        if (
            not isinstance(rule, dict)
            or set(rule) != {"checks", "failures", "examples"}
            or type(rule["checks"]) is not int
            or rule["checks"] < 1
            or type(rule["failures"]) is not int
            or rule["failures"] != 0
            or rule["examples"] != []
        ):
            raise DataError("Curated data-quality checks are invalid.")
    # Text-only parsing rules differ, but the complete data summaries must agree.
    if canonical_json({k: v for k, v in saved.items() if k != "rules"}) != canonical_json(
        {k: v for k, v in actual.items() if k != "rules"}
    ):
        raise DataError("Curated data-quality summaries do not match the Parquet values.")


def _verify_bundle(root: Path, spec: SourceSpec, *, check_name: bool) -> dict:
    _inventory(root)
    manifest = _read_json(root / "manifest.json")
    expected_header = _manifest_header(spec)
    if set(manifest) != {*expected_header, "files"} or canonical_json(
        {k: v for k, v in manifest.items() if k != "files"}
    ) != canonical_json(expected_header):
        raise DataError("Curated manifest does not match the pinned source and curation recipe.")
    entries = manifest["files"]
    if not isinstance(entries, list) or len(entries) != len(_CONTENT_PATHS):
        raise DataError("Curated manifest content inventory is invalid.")
    paths = []
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
            raise DataError("Curated manifest contains an invalid content entry.")
        paths.append(entry["path"])
    if paths != list(_CONTENT_PATHS):
        raise DataError("Curated manifest paths must be unique, exact and sorted.")
    digest = hash_file(root / "manifest.json")
    version = "sha256-" + digest
    if check_name and root.name != version:
        raise DataError("Curated directory name does not match its manifest version.")
    if _read_json(root / "_SUCCESS.json") != {"manifest_sha256": digest, "version": version}:
        raise DataError("Curated completion marker does not match its manifest.")
    for entry in entries:
        if fingerprint(_local_path(root, entry["path"])) != {
            "sha256": entry["sha256"],
            "size_bytes": entry["size_bytes"],
        }:
            raise DataError("Curated content hash or size does not match its manifest.")
    quality = _read_json(root / "data-quality.json")
    header = _quality_header(spec)
    if (
        set(quality) != {*header, "subsets"}
        or canonical_json({k: v for k, v in quality.items() if k != "subsets"})
        != canonical_json(header)
        or not isinstance(quality["subsets"], dict)
        or set(quality["subsets"]) != set(SUBSETS)
    ):
        raise DataError("Curated data-quality metadata is invalid or not passed.")
    for subset in SUBSETS:
        metadata = spec.subsets[subset]
        saved = quality["subsets"][subset]
        if not isinstance(saved, dict) or set(saved) != {"train", "test", "test_rul", "integrity"}:
            raise DataError("Curated subset data-quality reports are incomplete.")
        tables = {}
        for split in ("train", "test", "test_rul"):
            table = pq.ParquetFile(root / subset / f"{split}.parquet").read()
            if split == "test_rul":
                result = validate_label_table(
                    table, expected_rows=metadata["rul_rows"], expected_units=metadata["test_units"]
                )
            else:
                result = validate_observation_table(
                    table,
                    expected_rows=metadata[f"{split}_rows"],
                    expected_units=metadata[f"{split}_units"],
                )
            _check_saved_report(saved[split], result.report)
            tables[split] = table
        integrity = validate_split_integrity(tables["train"], tables["test"], tables["test_rul"])
        if canonical_json(saved["integrity"]) != canonical_json(integrity):
            raise DataError("Curated train/test integrity report does not match its tables.")
    return manifest


def verify_curated(root: Path, spec: SourceSpec) -> dict:
    """Read-only verification of exact inventory, identity, hashes, quality and tables.

    Return the parsed manifest, or raise DataError; never trust the marker alone.
    """
    try:
        return _verify_bundle(root, spec, check_name=True)
    except DataError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, pa.ArrowException):
        raise DataError("Curated bundle is unreadable or malformed.") from None


def curate(raw_root: Path, destination_root: Path, spec: SourceSpec) -> CuratedBundle:
    """Verify raw provenance, validate every split and atomically publish one local version.

    Only the private staging directory is removed on failure. Existing versions are
    verified and reused without modifying any of their files.
    """
    from epm_platform.data.source import verify_raw

    verify_raw(raw_root, spec)
    staging = None
    try:
        _reject_link(destination_root)
        destination_root.mkdir(parents=True, exist_ok=True)
        candidate = destination_root / (".curation-" + uuid4().hex)
        candidate.mkdir()
        staging = candidate
        quality = {**_quality_header(spec), "subsets": {}}
        for subset in SUBSETS:
            metadata = spec.subsets[subset]
            results = {
                split: validate_observations(
                    raw_root / "files" / f"{split}_{subset}.txt",
                    expected_rows=metadata[f"{split}_rows"],
                    expected_units=metadata[f"{split}_units"],
                )
                for split in ("train", "test")
            }
            results["test_rul"] = validate_labels(
                raw_root / "files" / f"RUL_{subset}.txt",
                expected_rows=metadata["rul_rows"],
                expected_units=metadata["test_units"],
            )
            integrity = validate_split_integrity(
                results["train"].table,
                results["test"].table,
                results["test_rul"].table,
            )
            quality["subsets"][subset] = {
                **{split: result.report for split, result in results.items()},
                "integrity": integrity,
            }
            (staging / subset).mkdir()
            for split, result in results.items():
                pq.write_table(
                    result.table,
                    staging / subset / f"{split}.parquet",
                    row_group_size=_ROW_GROUP_SIZE,
                    compression="zstd",
                    compression_level=3,
                    use_dictionary=False,
                    write_statistics=True,
                    version="2.6",
                    data_page_version="1.0",
                    store_schema=True,
                )
        write_json_new(staging / "data-quality.json", quality)
        manifest = {
            **_manifest_header(spec),
            "files": [
                {"path": relative, **fingerprint(_local_path(staging, relative))}
                for relative in _CONTENT_PATHS
            ],
        }
        write_json_new(staging / "manifest.json", manifest)
        digest = hash_file(staging / "manifest.json")
        version = "sha256-" + digest
        write_json_new(staging / "_SUCCESS.json", {"manifest_sha256": digest, "version": version})
        _verify_bundle(staging, spec, check_name=False)
        final = destination_root / version
        if final.exists() or final.is_symlink() or final.is_junction():
            verify_curated(final, spec)
        else:
            try:
                staging.rename(final)
                staging = None
            except OSError:
                # Another writer may have completed the identical version first.
                if not final.exists():
                    raise
                verify_curated(final, spec)
        return CuratedBundle(final, version, digest, quality)
    except DataError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, pa.ArrowException):
        raise DataError("C-MAPSS curation failed; no unverified bundle was finalized.") from None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
