import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from epm_platform.data import __main__ as cli
from epm_platform.data.errors import DataError
from epm_platform.data.reporting import render_quality_report
from epm_platform.data.spec import SUBSETS, load_spec

SPEC_PATH = Path(__file__).resolve().parents[2] / "config" / "cmapss-source.json"
VERSION = "sha256-" + "a" * 64


def arguments(tmp_path, action):
    return [
        "--spec",
        str(SPEC_PATH),
        "--data-root",
        str(tmp_path / "data"),
        "--state-dir",
        str(tmp_path / "state"),
        action,
    ]


def quality_report():
    observation = {
        "rows": {"parsed": 2},
        "units": {"count": 1},
        "counts": {
            "null_values": 0,
            "nonfinite_values": 0,
            "duplicate_records": 0,
            "duplicate_unit_cycles": 0,
            "conflicting_unit_cycles": 0,
        },
        "warnings": [{"rule": "zero_variance", "columns": ["sensor_01"]}],
    }
    return {
        "status": "passed",
        "transformations": ["Whitespace parsing"],
        "subsets": {
            name: {
                "train": deepcopy(observation),
                "test": deepcopy(observation),
                "integrity": {"counts": {"copied_test_trajectories": 0, "rul_rows": 1}},
            }
            for name in SUBSETS
        },
    }


def test_report_is_concise_and_preserves_exceptions():
    spec = load_spec(SPEC_PATH)
    text = render_quality_report(quality_report(), spec, VERSION)
    assert "| FD001 | 2 | 1 | 2 | 1 |" in text
    assert "Retained unchanged" in text
    assert "FD004" in text and "249" in text and "248" in text
    assert "No rows/columns dropped" in text
    assert "X-Amz" not in text


def test_acquire_receipt_does_not_require_azure(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "acquire", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(
        cli, "load_config", lambda: pytest.fail("No Azure config for local acquire")
    )
    assert cli.main(arguments(tmp_path, "acquire") + ["--archive", "original.zip"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["original_members"] == 14
    assert receipt["jobs_submitted"] is False
    assert calls[0][1]["archive_path"] == Path("original.zip")
    assert (tmp_path / "state" / "data-acquisition.json").is_file()


def test_curate_receipt_and_summary_use_validated_counts(tmp_path, monkeypatch, capsys):
    bundle = SimpleNamespace(
        version=VERSION, manifest_sha256="a" * 64, quality_report=quality_report()
    )
    monkeypatch.setattr(cli, "curate", lambda *args: bundle)
    monkeypatch.setattr(cli, "load_config", lambda: pytest.fail("No Azure config for local curate"))
    assert cli.main(arguments(tmp_path, "curate")) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["curated_version"] == VERSION
    assert "asset_version" not in receipt
    assert receipt["train_rows"] == 8
    assert receipt["test_rows"] == 8
    assert receipt["supplied_test_labels"] == 4
    assert (tmp_path / "state" / "data-quality-summary.md").is_file()
    assert json.loads((tmp_path / "state" / "data-preparation.json").read_text()) == receipt


def test_publish_requires_explicit_approval_before_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda: pytest.fail("Must reject before Azure config"))
    assert cli.main(arguments(tmp_path, "publish") + ["--version", VERSION]) == 1
    assert "--approve-azure-writes" in json.loads(capsys.readouterr().out)["message"]
    assert not (tmp_path / "state" / "data-publication.json").exists()


@pytest.mark.parametrize(
    "action,method,receipt_name",
    [
        ("publish", "publish", "data-publication.json"),
        ("verify", "verify_publication", "data-verification.json"),
    ],
)
def test_cloud_command_routes_exactly_one_operation(
    tmp_path, monkeypatch, capsys, action, method, receipt_name
):
    import epm_platform.data.publishing as publishing

    calls = []
    configuration = object()
    monkeypatch.setattr(cli, "load_config", lambda: configuration)

    def selected(raw, curated, spec, config):
        calls.append((raw, curated, spec, config))
        return {"asset_name": "epm-cmapss-curated", "asset_version": VERSION}

    monkeypatch.setattr(publishing, method, selected)
    other = "verify_publication" if method == "publish" else "publish"
    monkeypatch.setattr(publishing, other, lambda *a: pytest.fail("Wrong cloud operation"))
    args = arguments(tmp_path, action) + ["--version", VERSION]
    if action == "publish":
        args += ["--approve-azure-writes"]
    assert cli.main(args) == 0
    assert calls[0][1] == tmp_path / "data" / "curated" / "cmapss" / VERSION
    assert calls[0][3] is configuration
    assert json.loads(capsys.readouterr().out)["stage"] == action
    assert (tmp_path / "state" / receipt_name).exists()


@pytest.mark.parametrize("version", ["../escape", "latest", "sha256-short", "A" * 64])
def test_explicit_valid_version_required(tmp_path, version):
    with pytest.raises(SystemExit) as caught:
        cli.main(arguments(tmp_path, "verify") + ["--version", version])
    assert caught.value.code == 2


def test_data_failure_has_nonzero_exit_and_no_success_receipt(tmp_path, monkeypatch, capsys):
    def fail(*args):
        error = DataError("Invalid source cycles.")
        error.quality_report = {"status": "failed", "reason": "cycle_gap"}
        raise error

    monkeypatch.setattr(cli, "curate", fail)
    assert cli.main(arguments(tmp_path, "curate")) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert not (tmp_path / "state" / "data-preparation.json").exists()
    assert (tmp_path / "state" / "failed-data-quality.json").is_file()


def test_mutable_state_cannot_pollute_raw_data(tmp_path, monkeypatch, capsys):
    args = arguments(tmp_path, "acquire")
    args[args.index("--state-dir") + 1] = str(tmp_path / "data" / "raw" / "cmapss")
    monkeypatch.setattr(
        cli, "acquire", lambda *a, **kw: pytest.fail("Must reject before acquisition")
    )
    assert cli.main(args) == 1
    assert "outside raw and curated" in json.loads(capsys.readouterr().out)["message"]


def test_atomic_receipt_leaves_no_temporary_files(tmp_path):
    path = tmp_path / "state" / "receipt.json"
    cli._write_state(path, b"old")
    cli._write_state(path, b"new")
    assert path.read_bytes() == b"new"
    assert list(path.parent.glob(".data-state-*")) == []
