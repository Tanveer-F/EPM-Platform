"""Small synthetic, project-local fixtures; no downloads or cloud services."""

import hashlib
import importlib
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest

from epm_platform.data.manifest import fingerprint
from epm_platform.data.spec import SUBSETS, SourceSpec


@pytest.fixture
def data_workspace():
    path = Path(__file__).resolve().parents[2] / (".data-test-" + uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


@pytest.fixture
def observation_row():
    def row(unit=1, cycle=1, base=10.0):
        sensors = [42.0, base + (-1) ** cycle * 0.03125]
        sensors.extend(base + cycle * 0.1 + index * 0.01 for index in range(19))
        return " ".join(map(str, [unit, cycle, -0.125, -0.0, 100.0, *sensors]))

    return row


@pytest.fixture
def synthetic_bundle(data_workspace, observation_row, monkeypatch):
    raw = data_workspace / "raw"
    files = raw / "files"
    files.mkdir(parents=True)
    subsets = {}
    for index, subset in enumerate(SUBSETS):
        train = [
            observation_row(unit, cycle, 10.0 + 100 * index + unit)
            for unit in (1, 2)
            for cycle in (1, 2, 3)
        ]
        test = [
            observation_row(unit, cycle, 1000.0 + 100 * index + unit)
            for unit, cycles in ((1, (1, 2)), (2, (1,)))
            for cycle in cycles
        ]
        (files / f"train_{subset}.txt").write_text("\n".join(train) + "\n", encoding="ascii")
        (files / f"test_{subset}.txt").write_text("\n".join(test) + "\n", encoding="ascii")
        (files / f"RUL_{subset}.txt").write_text("12\n0\n", encoding="ascii")
        subsets[subset] = {
            "train_rows": 6,
            "train_units": 2,
            "test_rows": 3,
            "test_units": 2,
            "rul_rows": 2,
            "conditions": 1,
            "fault_modes": 1,
        }
    (files / "readme.txt").write_bytes(b"Synthetic unchanged Windows-1252 \x96 notes")
    (files / "Damage Propagation Modeling.pdf").write_bytes(b"%PDF-synthetic-fixture")
    archive = raw / "CMAPSSData.zip"
    archive.write_bytes(b"synthetic raw verifier is mocked; not a real archive")
    archive_info = fingerprint(archive)
    spec = SourceSpec(
        archive_name=archive.name,
        archive_sha256=archive_info["sha256"],
        archive_size_bytes=archive_info["size_bytes"],
        catalog_url="https://example.invalid/catalog",
        download_url="https://example.invalid/archive",
        files={path.name: fingerprint(path) for path in files.iterdir()},
        subsets=subsets,
        source_notes=("Original numeric records override documentation counts.",),
        spec_sha256=hashlib.sha256(b"synthetic-source-spec").hexdigest(),
    )
    calls = []

    def verify_raw(actual_root, actual_spec):
        assert actual_root == raw
        assert actual_spec is spec
        calls.append((actual_root, actual_spec))
        return {"status": "verified-synthetic"}

    module_name = "epm_platform.data.source"
    try:
        source = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        source = ModuleType(module_name)
        monkeypatch.setitem(sys.modules, module_name, source)
    monkeypatch.setattr(source, "verify_raw", verify_raw, raising=False)
    return SimpleNamespace(
        raw=raw, spec=spec, destination=data_workspace / "curated", calls=calls, source=source
    )
