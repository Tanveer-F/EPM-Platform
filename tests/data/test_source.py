import hashlib
import io
import json
import stat
import urllib.error
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from epm_platform.data.errors import DataError
from epm_platform.data.manifest import hash_file
from epm_platform.data.source import _check_download_url, acquire, verify_raw
from epm_platform.data.spec import EXPECTED_MEMBERS, load_spec

SPEC_PATH = Path(__file__).resolve().parents[2] / "config" / "cmapss-source.json"


def make_archive(tmp_path, *, extra=None, symlink=False, duplicate=False):
    files = {name: (name + "\r\n").encode("ascii") for name in EXPECTED_MEMBERS}
    files["readme.txt"] = b"README \x96 original encoding\r\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            if symlink and name == "readme.txt":
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, content)
            else:
                archive.writestr(name, content)
        if extra:
            archive.writestr(extra, b"unexpected")
        if duplicate:
            with pytest.warns(UserWarning, match="Duplicate"):
                archive.writestr("readme.txt", files["readme.txt"])
    payload = buffer.getvalue()
    path = tmp_path / "original.zip"
    path.write_bytes(payload)
    spec = replace(
        load_spec(SPEC_PATH),
        archive_sha256=hashlib.sha256(payload).hexdigest(),
        archive_size_bytes=len(payload),
        files={
            name: {"sha256": hashlib.sha256(value).hexdigest(), "size_bytes": len(value)}
            for name, value in files.items()
        },
    )
    return path, spec, files


def test_original_bytes_and_cached_bundle_are_unchanged(tmp_path):
    archive, spec, files = make_archive(tmp_path)
    original_digest = hash_file(archive)
    root = acquire(spec, tmp_path / "raw", archive)
    before = {p.relative_to(root): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    assert (root / spec.archive_name).read_bytes() == archive.read_bytes()
    for name, data in files.items():
        assert (root / "files" / name).read_bytes() == data
    assert acquire(spec, tmp_path / "raw", archive) == root
    assert hash_file(archive) == original_digest
    assert before == {
        p.relative_to(root): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()
    }
    manifest = verify_raw(root, spec)
    assert manifest["source"]["download_url"] == spec.download_url
    assert "X-Amz" not in (root / "raw-manifest.json").read_text()
    assert not list((tmp_path / "raw").glob(".acquire-*"))


@pytest.mark.parametrize("damage", ["member", "archive", "manifest", "extra", "missing"])
def test_corrupted_cached_bundle_fails_without_overwrite(tmp_path, damage):
    archive, spec, _ = make_archive(tmp_path)
    root = acquire(spec, tmp_path / "raw", archive)
    paths = {
        "member": root / "files" / "readme.txt",
        "archive": root / spec.archive_name,
        "manifest": root / "raw-manifest.json",
        "extra": root / "unexpected.txt",
    }
    if damage == "missing":
        (root / "files" / "readme.txt").unlink()
    else:
        paths[damage].write_bytes(b"changed")
    with pytest.raises(DataError):
        acquire(spec, tmp_path / "raw", archive)
    if damage != "missing":
        assert paths[damage].read_bytes() == b"changed"


@pytest.mark.parametrize("option", ["traversal", "absolute", "backslash", "duplicate", "symlink"])
def test_unsafe_archives_never_finalize(tmp_path, option):
    extras = {
        "traversal": "../escaped.txt",
        "absolute": "/escaped.txt",
        "backslash": "..\\escaped.txt",
    }
    archive, spec, _ = make_archive(
        tmp_path,
        extra=extras.get(option),
        symlink=option == "symlink",
        duplicate=option == "duplicate",
    )
    with pytest.raises(DataError):
        acquire(spec, tmp_path / "raw", archive)
    assert list((tmp_path / "raw").iterdir()) == []
    assert not (tmp_path / "escaped.txt").exists()


def test_archive_hash_mismatch_does_not_modify_input_or_publish(tmp_path):
    archive, spec, _ = make_archive(tmp_path)
    archive.write_bytes(b"not the pinned source")
    with pytest.raises(DataError, match="does not match"):
        acquire(spec, tmp_path / "raw", archive)
    assert archive.read_bytes() == b"not the pinned source"
    assert list((tmp_path / "raw").iterdir()) == []


def test_invalid_zip_with_matching_transport_hash_is_rejected(tmp_path):
    archive, spec, _ = make_archive(tmp_path)
    data = b"not a zip archive"
    archive.write_bytes(data)
    spec = replace(
        spec, archive_sha256=hashlib.sha256(data).hexdigest(), archive_size_bytes=len(data)
    )
    with pytest.raises(DataError, match="CRC"):
        acquire(spec, tmp_path / "raw", archive)
    assert list((tmp_path / "raw").iterdir()) == []


@pytest.mark.parametrize(
    "url",
    [
        "http://data.nasa.gov/file.zip",
        "https://example.com/file.zip",
        "https://data.nasa.gov.evil.example/file.zip",
        "https://user:password@data.nasa.gov/file.zip",
        "https://data.nasa.gov:8443/file.zip",
        "https://127.0.0.1/file.zip",
        "file:///secret",
    ],
)
def test_only_publisher_https_redirects_are_allowed(url):
    with pytest.raises(DataError):
        _check_download_url(url)


def test_signed_publisher_get_download_is_preserved_without_query_logging(tmp_path, monkeypatch):
    archive, spec, _ = make_archive(tmp_path)
    payload = archive.read_bytes()
    final_url = (
        "https://data-nasa-bucket-production.s3.us-east-1.amazonaws.com/legacy/"
        "CMAPSSData.zip?X-Amz-Security-Token=DO_NOT_RECORD"
    )

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(payload))}

        def geturl(self):
            return final_url

    def open_response(request, timeout):
        assert request.get_method() == "GET"
        assert request.full_url == spec.download_url
        assert timeout == 60
        return Response(payload)

    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *_: SimpleNamespace(open=open_response)
    )
    root = acquire(spec, tmp_path / "raw")
    assert "DO_NOT_RECORD" not in (root / "raw-manifest.json").read_text()
    verify_raw(root, spec)


def test_download_errors_never_disclose_signed_query(tmp_path, monkeypatch):
    _, spec, _ = make_archive(tmp_path)

    def fail(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://data.nasa.gov/file?secret=DO_NOT_RECORD", 403, "Forbidden", None, None
        )

    monkeypatch.setattr("urllib.request.build_opener", lambda *_: SimpleNamespace(open=fail))
    with pytest.raises(DataError) as caught:
        acquire(spec, tmp_path / "raw")
    assert "403" in str(caught.value)
    assert "DO_NOT_RECORD" not in str(caught.value)
    assert list((tmp_path / "raw").iterdir()) == []


def test_download_size_guard_and_cleanup(tmp_path, monkeypatch):
    _, spec, _ = make_archive(tmp_path)

    class Response(io.BytesIO):
        headers = {}

        def geturl(self):
            return spec.download_url

    monkeypatch.setattr(
        "urllib.request.build_opener",
        lambda *_: SimpleNamespace(
            open=lambda *_args, **_kw: Response(b"x" * (spec.archive_size_bytes + 1))
        ),
    )
    with pytest.raises(DataError, match="exceeds"):
        acquire(spec, tmp_path / "raw")
    assert list((tmp_path / "raw").iterdir()) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("archive_sha256", "bad"),
        ("download_url", "https://example.com/data.zip"),
        ("archive_size_bytes", True),
        ("archive_size_bytes", 1024**3),
        ("schema_version", 2),
        ("schema_version", True),
    ],
)
def test_source_spec_rejects_invalid_contract(tmp_path, field, value):
    document = json.loads(SPEC_PATH.read_text())
    document[field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(document))
    with pytest.raises(DataError):
        load_spec(path)
