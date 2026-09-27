import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from epm_platform.data import curation
from epm_platform.data.curation import curate, verify_curated
from epm_platform.data.errors import DataError
from epm_platform.data.manifest import canonical_json, fingerprint, hash_file
from epm_platform.data.spec import OBSERVATION_COLUMNS, SUBSETS
from epm_platform.data.validation import LABEL_SCHEMA, OBSERVATION_SCHEMA, ValidationError


def snapshot(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def remanifest(root, change=None):
    manifest = json.loads((root / "manifest.json").read_bytes())
    if change:
        change(manifest)
    (root / "manifest.json").write_bytes(canonical_json(manifest))
    digest = hash_file(root / "manifest.json")
    version = "sha256-" + digest
    new_root = root.with_name(version)
    if root != new_root:
        root.rename(new_root)
    (new_root / "_SUCCESS.json").write_bytes(
        canonical_json(
            {
                "manifest_sha256": digest,
                "version": version,
            }
        )
    )
    return new_root


def rehash_content(root, relative):
    def change(manifest):
        for entry in manifest["files"]:
            if entry["path"] == relative:
                entry.update(fingerprint(root.joinpath(*relative.split("/"))))

    return remanifest(root, change)


def test_deterministic_lossless_bundle_and_immutable_reuse(synthetic_bundle):
    source = synthetic_bundle
    raw_before = snapshot(source.raw)
    first = curate(source.raw, source.destination, source.spec)
    first_bytes = snapshot(first.root)
    first_times = {path: path.stat().st_mtime_ns for path in first.root.rglob("*")}
    repeated = curate(source.raw, source.destination, source.spec)
    independent = curate(source.raw, source.destination.with_name("scope=value"), source.spec)
    assert first.root == repeated.root
    assert first.version == repeated.version == independent.version
    assert first.manifest_sha256 == hash_file(first.root / "manifest.json")
    assert first.version == "sha256-" + first.manifest_sha256
    assert first.root == source.destination / first.version
    assert first_bytes == snapshot(repeated.root) == snapshot(independent.root)
    assert first_times == {path: path.stat().st_mtime_ns for path in first.root.rglob("*")}
    assert snapshot(source.raw) == raw_before
    assert len(source.calls) == 3
    assert list(source.destination.iterdir()) == [first.root]
    manifest = verify_curated(first.root, source.spec)
    assert set(manifest) == {
        "schema_version",
        "dataset",
        "source_archive_sha256",
        "source_spec_sha256",
        "recipe_version",
        "writer",
        "subsets",
        "observation_columns",
        "label_columns",
        "files",
    }
    assert manifest["writer"] == {
        "name": "pyarrow",
        "version": pa.__version__,
        "compression": "zstd",
    }
    assert manifest["recipe_version"] == "1"
    assert manifest["observation_columns"] == list(OBSERVATION_COLUMNS)
    assert manifest["label_columns"] == ["unit_id", "rul"]
    expected_files = sorted(
        [
            f"{subset}/{split}.parquet"
            for subset in SUBSETS
            for split in ("train", "test", "test_rul")
        ]
        + ["data-quality.json"]
    )
    assert [entry["path"] for entry in manifest["files"]] == expected_files
    assert len(first_bytes) == 15
    assert first.quality_report["status"] == "passed"
    assert first.quality_report["source_notes"] == list(source.spec.source_notes)
    assert first.quality_report["preservation"]["imputed_values"] == 0
    assert first.quality_report["preservation"]["dropped_rows"] == 0
    assert first.quality_report["preservation"]["reordered_rows"] == 0
    assert first.quality_report["preservation"]["scaling"] is False
    assert len(first.quality_report["transformations"]) == 4
    for subset in SUBSETS:
        for split, count in (("train", 6), ("test", 3)):
            path = first.root / subset / f"{split}.parquet"
            table = pq.read_table(path)
            assert table.schema == OBSERVATION_SCHEMA
            assert table.num_rows == count
            assert table.to_pylist() == [
                dict(
                    zip(
                        OBSERVATION_COLUMNS,
                        [*map(int, line.split()[:2]), *map(float, line.split()[2:])],
                        strict=True,
                    )
                )
                for line in (source.raw / "files" / f"{split}_{subset}.txt")
                .read_text()
                .splitlines()
            ]
            parquet = pq.ParquetFile(path)
            assert parquet.metadata.num_row_groups == 1
            assert parquet.metadata.row_group(0).column(2).compression == "ZSTD"
            assert first.quality_report["subsets"][subset][split]["rows"]["parsed"] == count
        rul = pq.read_table(first.root / subset / "test_rul.parquet")
        assert rul.schema == LABEL_SCHEMA
        assert rul.to_pydict() == {"unit_id": [1, 2], "rul": [12, 0]}
        assert not (first.root / subset / "train_rul.parquet").exists()
        assert first.quality_report["subsets"][subset]["integrity"]["status"] == "passed"
    serialized = (first.root / "manifest.json").read_bytes()
    assert serialized == canonical_json(manifest)
    assert str(source.destination).encode() not in serialized


@pytest.mark.parametrize("corruption", ["bytes", "extra_file", "extra_directory", "missing_file"])
def test_corrupt_existing_version_is_rejected_not_overwritten(synthetic_bundle, corruption):
    source = synthetic_bundle
    result = curate(source.raw, source.destination, source.spec)
    path = result.root / "FD001" / "train.parquet"
    if corruption == "bytes":
        path.write_bytes(path.read_bytes() + b"corrupt")
    elif corruption == "extra_file":
        (result.root / "unexpected.txt").write_text("conflict", encoding="ascii")
    elif corruption == "extra_directory":
        (result.root / "unexpected").mkdir()
    else:
        path.unlink()
    before = snapshot(result.root)
    with pytest.raises(DataError):
        verify_curated(result.root, source.spec)
    with pytest.raises(DataError):
        curate(source.raw, source.destination, source.spec)
    assert snapshot(result.root) == before
    assert list(source.destination.iterdir()) == [result.root]


@pytest.mark.parametrize(
    "path",
    [
        "../outside.parquet",
        "/absolute.parquet",
        "FD001/../train.parquet",
        "FD001\\train.parquet",
        "FD005/train.parquet",
    ],
)
def test_manifest_paths_are_allowlisted_before_reading(synthetic_bundle, path):
    source = synthetic_bundle
    result = curate(source.raw, source.destination, source.spec)
    root = remanifest(result.root, lambda value: value["files"][0].update(path=path))
    with pytest.raises(DataError, match="content entry"):
        verify_curated(root, source.spec)


@pytest.mark.parametrize(
    "corruption", ["duplicate", "unsorted", "extra_key", "negative_size", "bool_size", "wrong_sha"]
)
def test_manifest_content_entries_are_exact(synthetic_bundle, corruption):
    source = synthetic_bundle
    result = curate(source.raw, source.destination, source.spec)

    def change(manifest):
        entries = manifest["files"]
        if corruption == "duplicate":
            entries[1] = entries[0]
        elif corruption == "unsorted":
            entries.reverse()
        elif corruption == "extra_key":
            entries[0]["uri"] = "not permitted"
        elif corruption == "negative_size":
            entries[0]["size_bytes"] = -1
        elif corruption == "bool_size":
            entries[0]["size_bytes"] = True
        else:
            entries[0]["sha256"] = "not a digest"

    root = remanifest(result.root, change)
    with pytest.raises(DataError):
        verify_curated(root, source.spec)


@pytest.mark.parametrize("corruption", ["recipe", "schema", "writer", "field"])
def test_manifest_recipe_and_schema_are_pinned(synthetic_bundle, corruption):
    source = synthetic_bundle
    result = curate(source.raw, source.destination, source.spec)

    def change(manifest):
        if corruption == "recipe":
            manifest["recipe_version"] = "2"
        elif corruption == "schema":
            manifest["schema_version"] = True
        elif corruption == "writer":
            manifest["writer"]["compression"] = "snappy"
        else:
            manifest["timestamp"] = "not permitted"

    root = remanifest(result.root, change)
    with pytest.raises(DataError, match="recipe"):
        verify_curated(root, source.spec)


@pytest.mark.parametrize("field", ["archive_sha256", "spec_sha256"])
def test_verification_requires_pinned_source_hashes(synthetic_bundle, field):
    source = synthetic_bundle
    result = curate(source.raw, source.destination, source.spec)
    with pytest.raises(DataError, match="pinned source"):
        verify_curated(result.root, replace(source.spec, **{field: "0" * 64}))


@pytest.mark.parametrize("corruption", ["name", "marker", "noncanonical", "malformed"])
def test_name_marker_and_canonical_json_are_required(synthetic_bundle, corruption):
    source = synthetic_bundle
    result = curate(source.raw, source.destination, source.spec)
    root = result.root
    if corruption == "name":
        new_root = root.with_name("wrong-name")
        root.rename(new_root)
        root = new_root
    elif corruption == "marker":
        (root / "_SUCCESS.json").write_bytes(
            canonical_json({"manifest_sha256": "0" * 64, "version": result.version})
        )
    elif corruption == "noncanonical":
        path = root / "manifest.json"
        path.write_bytes(path.read_bytes().rstrip())
    else:
        (root / "manifest.json").write_bytes(b"not JSON\n")
    with pytest.raises(DataError):
        verify_curated(root, source.spec)


@pytest.mark.parametrize("corruption", ["status", "summary", "rule", "preservation"])
def test_quality_report_is_verified_even_after_rehashing(synthetic_bundle, corruption):
    source = synthetic_bundle
    result = curate(source.raw, source.destination, source.spec)
    path = result.root / "data-quality.json"
    quality = json.loads(path.read_bytes())
    if corruption == "status":
        quality["status"] = "failed"
    elif corruption == "summary":
        quality["subsets"]["FD001"]["train"]["columns"]["sensor_01"]["mean"] = -99.0
    elif corruption == "rule":
        quality["subsets"]["FD001"]["train"]["rules"]["ascii"]["failures"] = 1
    else:
        quality["preservation"]["imputed_values"] = 1
    path.write_bytes(canonical_json(quality))
    root = rehash_content(result.root, "data-quality.json")
    with pytest.raises(DataError, match="data-quality"):
        verify_curated(root, source.spec)


@pytest.mark.parametrize("corruption", ["schema", "count", "value", "labels"])
def test_parquet_values_schema_and_counts_are_rechecked(synthetic_bundle, corruption):
    source = synthetic_bundle
    result = curate(source.raw, source.destination, source.spec)
    relative = "FD001/test_rul.parquet" if corruption == "labels" else "FD001/train.parquet"
    path = result.root.joinpath(*relative.split("/"))
    table = pq.read_table(path)
    if corruption == "schema":
        table = table.set_column(0, "unit_id", table["unit_id"].cast(pa.int64()))
    elif corruption == "count":
        table = table.slice(1)
    elif corruption == "value":
        arrays = list(table.columns)
        arrays[2] = pa.array([float("nan")] * table.num_rows, type=pa.float64())
        table = pa.Table.from_arrays(arrays, schema=OBSERVATION_SCHEMA)
    else:
        table = table.take(pa.array([1, 0]))
    pq.write_table(table, path, compression="zstd")
    root = rehash_content(result.root, relative)
    with pytest.raises(DataError):
        verify_curated(root, source.spec)


@pytest.mark.parametrize("link_kind", ["is_symlink", "is_junction"])
def test_link_inventory_rejection_on_all_platforms(synthetic_bundle, monkeypatch, link_kind):
    source = synthetic_bundle
    result = curate(source.raw, source.destination, source.spec)
    linked = result.root / "FD001" / "train.parquet"
    original = getattr(Path, link_kind)
    monkeypatch.setattr(Path, link_kind, lambda path: path == linked or original(path))
    with pytest.raises(DataError, match="links or junctions"):
        verify_curated(result.root, source.spec)


def test_actual_symlink_is_rejected_when_os_permits(synthetic_bundle):
    source = synthetic_bundle
    result = curate(source.raw, source.destination, source.spec)
    path = result.root / "FD001" / "train.parquet"
    target = source.destination / "outside.parquet"
    target.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("Creating symlinks requires an OS privilege on this host.")
    with pytest.raises(DataError, match="links or junctions"):
        verify_curated(result.root, source.spec)


def test_invalid_late_subset_cleans_only_own_staging(synthetic_bundle):
    source = synthetic_bundle
    unrelated = source.destination / ".curation-unrelated"
    unrelated.mkdir(parents=True)
    (unrelated / "sentinel").write_bytes(b"preserve another writer")
    path = source.raw / "files" / "train_FD004.txt"
    lines = path.read_text().splitlines()
    lines[-1] = lines[-2]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    with pytest.raises(ValidationError) as raised:
        curate(source.raw, source.destination, source.spec)
    assert raised.value.quality_report["status"] == "failed"
    assert list(source.destination.iterdir()) == [unrelated]
    assert (unrelated / "sentinel").read_bytes() == b"preserve another writer"


def test_staging_collision_does_not_remove_another_writer(synthetic_bundle, monkeypatch):
    source = synthetic_bundle
    occupied = source.destination / ".curation-occupied"
    occupied.mkdir(parents=True)
    (occupied / "sentinel").write_bytes(b"another writer owns this directory")
    monkeypatch.setattr(curation, "uuid4", lambda: SimpleNamespace(hex="occupied"))
    with pytest.raises(DataError):
        curate(source.raw, source.destination, source.spec)
    assert (occupied / "sentinel").read_bytes() == b"another writer owns this directory"
    assert list(source.destination.iterdir()) == [occupied]


def test_raw_verification_must_pass_before_staging(synthetic_bundle, monkeypatch):
    source = synthetic_bundle

    def reject(*args):
        raise DataError("Raw integrity failed.")

    monkeypatch.setattr(source.source, "verify_raw", reject)
    with pytest.raises(DataError, match="Raw integrity"):
        curate(source.raw, source.destination, source.spec)
    assert not source.destination.exists()


def test_writer_failure_never_finalizes_partial_bundle(synthetic_bundle, monkeypatch):
    source = synthetic_bundle
    original = curation.pq.write_table
    calls = 0

    def fail_later(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("synthetic write failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(curation.pq, "write_table", fail_later)
    with pytest.raises(DataError, match="no unverified bundle"):
        curate(source.raw, source.destination, source.spec)
    assert list(source.destination.iterdir()) == []


def test_completion_marker_exists_before_atomic_rename(synthetic_bundle, monkeypatch):
    source = synthetic_bundle
    original = Path.rename
    renamed = []

    def inspect_rename(path, target):
        if path.name.startswith(".curation-"):
            assert set(snapshot(path)) == {
                *[
                    f"{subset}/{split}.parquet"
                    for subset in SUBSETS
                    for split in ("train", "test", "test_rul")
                ],
                "manifest.json",
                "data-quality.json",
                "_SUCCESS.json",
            }
            marker = json.loads((path / "_SUCCESS.json").read_bytes())
            assert marker["version"] == target.name
            renamed.append(target)
        return original(path, target)

    monkeypatch.setattr(Path, "rename", inspect_rename)
    result = curate(source.raw, source.destination, source.spec)
    assert renamed == [result.root]
    verify_curated(result.root, source.spec)
