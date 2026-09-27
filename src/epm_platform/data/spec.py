"""The reviewed source specification and original C-MAPSS column contract."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from epm_platform.data.errors import DataError
from epm_platform.data.manifest import canonical_json

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
OBSERVATION_COLUMNS = (
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *(f"sensor_{number:02d}" for number in range(1, 22)),
)
LABEL_COLUMNS = ("unit_id", "rul")
SOURCE_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
CATALOG_URL = "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"
DOWNLOAD_HOSTS = frozenset(
    {"data.nasa.gov", "data-nasa-bucket-production.s3.us-east-1.amazonaws.com"}
)
EXPECTED_MEMBERS = frozenset(
    {"readme.txt", "Damage Propagation Modeling.pdf"}
    | {f"{kind}_{subset}.txt" for subset in SUBSETS for kind in ("train", "test", "RUL")}
)
_SHA256 = re.compile(r"[a-f0-9]{64}")


@dataclass(frozen=True)
class SourceSpec:
    archive_name: str
    archive_sha256: str
    archive_size_bytes: int
    catalog_url: str
    download_url: str
    files: dict[str, dict]
    subsets: dict[str, dict[str, int]]
    source_notes: tuple[str, ...]
    spec_sha256: str


def _positive(value: object) -> bool:
    return type(value) is int and value > 0


def load_spec(path: Path) -> SourceSpec:
    """Load reviewed fingerprints; never discover or silently refresh hashes at runtime."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (
            type(raw["schema_version"]) is not int
            or raw["schema_version"] != 1
            or raw["dataset"] != "nasa-cmapss"
            or raw["archive_name"] != "CMAPSSData.zip"
            or raw["download_url"] != SOURCE_URL
            or raw["catalog_url"] != CATALOG_URL
            or not _SHA256.fullmatch(raw["archive_sha256"])
            or not _positive(raw["archive_size_bytes"])
            or raw["archive_size_bytes"] > 64 * 1024 * 1024
            or set(raw["files"]) != EXPECTED_MEMBERS
            or set(raw["subsets"]) != set(SUBSETS)
        ):
            raise ValueError
        for info in raw["files"].values():
            if not _positive(info["size_bytes"]) or not _SHA256.fullmatch(info["sha256"]):
                raise ValueError
        if sum(info["size_bytes"] for info in raw["files"].values()) > 128 * 1024 * 1024:
            raise ValueError
        for info in raw["subsets"].values():
            keys = (
                "train_rows",
                "train_units",
                "test_rows",
                "test_units",
                "rul_rows",
                "conditions",
                "fault_modes",
            )
            if any(not _positive(info[key]) for key in keys):
                raise ValueError
            if (
                info["rul_rows"] != info["test_units"]
                or info["train_rows"] < info["train_units"]
                or info["test_rows"] < info["test_units"]
            ):
                raise ValueError
        notes = raw["source_notes"]
        if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise DataError("Invalid or unreadable C-MAPSS source specification.") from None
    return SourceSpec(
        archive_name=raw["archive_name"],
        archive_sha256=raw["archive_sha256"],
        archive_size_bytes=raw["archive_size_bytes"],
        catalog_url=raw["catalog_url"],
        download_url=raw["download_url"],
        files=raw["files"],
        subsets=raw["subsets"],
        source_notes=tuple(notes),
        spec_sha256=hashlib.sha256(canonical_json(raw)).hexdigest(),
    )
