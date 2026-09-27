"""Acquire the pinned NASA archive and preserve its original bytes write-once."""

import hashlib
import shutil
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from epm_platform.data.errors import DataError
from epm_platform.data.manifest import canonical_json, fingerprint, write_json_new
from epm_platform.data.spec import DOWNLOAD_HOSTS, SourceSpec

_CHUNK = 1024 * 1024


def _check_download_url(url: str) -> None:
    try:
        parsed = urllib.parse.urlsplit(url)
        valid = (
            parsed.scheme == "https"
            and parsed.hostname in DOWNLOAD_HOSTS
            and parsed.port in (None, 443)
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        valid = False
    if not valid:
        raise DataError("Dataset download redirected outside the approved HTTPS publisher hosts.")


class _PublisherRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_download_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(spec: SourceSpec, destination: Path) -> None:
    _check_download_url(spec.download_url)
    opener = urllib.request.build_opener(_PublisherRedirect())
    request = urllib.request.Request(spec.download_url)
    try:
        # NASA issues a signed GET redirect. Never use HEAD or record the redirected query string.
        with opener.open(request, timeout=60) as response, destination.open("xb") as output:
            _check_download_url(response.geturl())
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != spec.archive_size_bytes:
                raise DataError("Dataset response length differs from the pinned archive.")
            total = 0
            for chunk in iter(lambda: response.read(_CHUNK), b""):
                total += len(chunk)
                if total > spec.archive_size_bytes:
                    raise DataError("Dataset response exceeds the pinned archive size.")
                output.write(chunk)
    except urllib.error.HTTPError as error:
        raise DataError(f"NASA archive download failed (HTTP {error.code}).") from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        if isinstance(error, DataError):
            raise
        raise DataError(
            f"NASA download failed ({type(error).__name__}); "
            "check network/TLS connectivity and retry."
        ) from None


def _manifest(spec: SourceSpec) -> dict:
    files = {
        spec.archive_name: {
            "sha256": spec.archive_sha256,
            "size_bytes": spec.archive_size_bytes,
        },
        **{f"files/{name}": info for name, info in spec.files.items()},
    }
    return {
        "schema_version": 1,
        "dataset": "nasa-cmapss",
        "archive_sha256": spec.archive_sha256,
        "source": {"catalog_url": spec.catalog_url, "download_url": spec.download_url},
        "files": dict(sorted(files.items())),
    }


def verify_raw(root: Path, spec: SourceSpec) -> dict:
    """Verify the archive, every member and the exact source inventory without changing anything."""
    expected = _manifest(spec)
    try:
        if not root.is_dir() or root.is_symlink() or root.is_junction():
            raise DataError("Raw bundle must be a real directory, not a link.")
        if root.name != spec.archive_sha256:
            raise DataError("Raw bundle directory does not match the pinned archive digest.")
        actual = set()
        for path in root.rglob("*"):
            if path.is_symlink() or path.is_junction():
                raise DataError("Raw bundle contains a link.")
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                if relative != "files":
                    raise DataError("Raw bundle contains an unexpected directory.")
            elif path.is_file():
                actual.add(relative)
            else:
                raise DataError("Raw bundle contains an unsupported filesystem entry.")
        if actual != set(expected["files"]) | {"raw-manifest.json"}:
            raise DataError("Raw bundle inventory differs from the pinned source.")
        if (root / "raw-manifest.json").read_bytes() != canonical_json(expected):
            raise DataError("Raw provenance manifest differs from the pinned source.")
        for name, info in expected["files"].items():
            if fingerprint(root / name) != info:
                raise DataError(f"Raw file integrity check failed: {name}.")
    except OSError:
        raise DataError("Cannot read the complete raw bundle.") from None
    return expected


def _extract(archive_path: Path, destination: Path, spec: SourceSpec) -> None:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) != len(spec.files) or {item.filename for item in members} != set(
                spec.files
            ):
                raise DataError("Archive members differ from the reviewed source inventory.")
            for item in members:
                mode = item.external_attr >> 16
                if (
                    item.is_dir()
                    or stat.S_ISLNK(mode)
                    or item.flag_bits & 1
                    or "/" in item.filename
                    or "\\" in item.filename
                    or ":" in item.filename
                    or item.filename in (".", "..")
                ):
                    raise DataError("Archive contains an unsafe, linked or encrypted member.")
                expected = spec.files[item.filename]
                if item.file_size != expected["size_bytes"]:
                    raise DataError("Archive member size differs from the reviewed source.")
                digest = hashlib.sha256()
                total = 0
                with (
                    archive.open(item) as source,
                    (destination / item.filename).open("xb") as output,
                ):
                    for chunk in iter(lambda: source.read(_CHUNK), b""):
                        total += len(chunk)
                        if total > expected["size_bytes"]:
                            raise DataError("Archive member exceeded its expected size.")
                        digest.update(chunk)
                        output.write(chunk)
                if total != expected["size_bytes"] or digest.hexdigest() != expected["sha256"]:
                    raise DataError("Archive member checksum differs from the reviewed source.")
    except (zipfile.BadZipFile, RuntimeError, OSError):
        raise DataError("Archive extraction or CRC verification failed.") from None


def acquire(spec: SourceSpec, destination_root: Path, archive_path: Path | None = None) -> Path:
    """Download or import an exactly matching archive; reuse intact bundles, never overwrite."""
    destination_root.mkdir(parents=True, exist_ok=True)
    final = destination_root / spec.archive_sha256
    if final.exists():
        verify_raw(final, spec)
        return final
    stage = Path(tempfile.mkdtemp(prefix=".acquire-", dir=destination_root))
    try:
        target_archive = stage / spec.archive_name
        if archive_path is None:
            _download(spec, target_archive)
        else:
            if not archive_path.is_file() or archive_path.is_symlink():
                raise DataError("The supplied archive must be a regular file.")
            if fingerprint(archive_path) != {
                "sha256": spec.archive_sha256,
                "size_bytes": spec.archive_size_bytes,
            }:
                raise DataError("The supplied archive does not match the reviewed NASA archive.")
            with archive_path.open("rb") as source, target_archive.open("xb") as output:
                shutil.copyfileobj(source, output, _CHUNK)
        if fingerprint(target_archive) != {
            "sha256": spec.archive_sha256,
            "size_bytes": spec.archive_size_bytes,
        }:
            raise DataError("Downloaded archive checksum or size does not match the pinned source.")
        (stage / "files").mkdir()
        _extract(target_archive, stage / "files", spec)
        write_json_new(stage / "raw-manifest.json", _manifest(spec))
        try:
            stage.rename(final)
        except FileExistsError:
            verify_raw(final, spec)
        verify_raw(final, spec)
        return final
    finally:
        if stage.exists():
            shutil.rmtree(stage)
