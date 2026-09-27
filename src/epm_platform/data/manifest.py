"""Deterministic serialization and integrity primitives."""

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, str | int]:
    return {"sha256": hash_file(path), "size_bytes": path.stat().st_size}


def write_json_new(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json(value))
