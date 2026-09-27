"""Explicit local preparation and approved Azure publication commands."""

import argparse
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from epm_platform.config import ConfigurationError, load_config
from epm_platform.data.curation import curate
from epm_platform.data.errors import DataError
from epm_platform.data.manifest import canonical_json
from epm_platform.data.reporting import render_quality_report
from epm_platform.data.source import acquire
from epm_platform.data.spec import SUBSETS, load_spec


def _version(value: str) -> str:
    if not re.fullmatch(r"sha256-[a-f0-9]{64}", value):
        raise argparse.ArgumentTypeError(
            "Version must be sha256- followed by 64 lowercase hex digits."
        )
    return value


def _write_state(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".data-state-", delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="C-MAPSS Phase 2 data preparation; no ML workloads."
    )
    parser.add_argument("--spec", type=Path, default=Path("config") / "cmapss-source.json")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--state-dir", type=Path, default=Path(".azure"))
    actions = parser.add_subparsers(dest="action", required=True)
    acquisition = actions.add_parser("acquire", help="Download or import the pinned original ZIP.")
    acquisition.add_argument("--archive", type=Path, help="Already-downloaded matching NASA ZIP.")
    actions.add_parser("curate", help="Validate and produce deterministic Parquet; local only.")
    publication = actions.add_parser(
        "publish", help="Upload verified bytes and register one version."
    )
    publication.add_argument("--version", required=True, type=_version)
    publication.add_argument("--approve-azure-writes", action="store_true")
    verification = actions.add_parser(
        "verify", help="Read-only remote byte and asset verification."
    )
    verification.add_argument("--version", required=True, type=_version)
    args = parser.parse_args(argv)
    try:
        if args.action == "publish" and not args.approve_azure_writes:
            raise DataError(
                "Publishing requires explicit --approve-azure-writes after access review."
            )
        spec = load_spec(args.spec)
        raw_base = args.data_root / "raw" / "cmapss"
        curated_base = args.data_root / "curated" / "cmapss"
        state_path = args.state_dir.resolve()
        if any(state_path.is_relative_to(root.resolve()) for root in (raw_base, curated_base)):
            raise DataError("Mutable receipts must remain outside raw and curated dataset roots.")
        if args.action == "acquire":
            acquire(spec, raw_base, archive_path=args.archive)
            result = {
                "source_archive_sha256": spec.archive_sha256,
                "original_members": len(spec.files),
            }
            receipt_name = "data-acquisition.json"
        elif args.action == "curate":
            bundle = curate(raw_base / spec.archive_sha256, curated_base, spec)
            result = {
                "source_archive_sha256": spec.archive_sha256,
                "curated_version": bundle.version,
                "curated_manifest_sha256": bundle.manifest_sha256,
                "subsets": list(SUBSETS),
                "train_rows": sum(
                    bundle.quality_report["subsets"][s]["train"]["rows"]["parsed"] for s in SUBSETS
                ),
                "test_rows": sum(
                    bundle.quality_report["subsets"][s]["test"]["rows"]["parsed"] for s in SUBSETS
                ),
                "supplied_test_labels": sum(
                    bundle.quality_report["subsets"][s]["integrity"]["counts"]["rul_rows"]
                    for s in SUBSETS
                ),
            }
            _write_state(
                args.state_dir / "data-quality-summary.md",
                render_quality_report(bundle.quality_report, spec, bundle.version).encode("utf-8"),
            )
            receipt_name = "data-preparation.json"
        else:
            from epm_platform.data.publishing import publish, verify_publication

            operation = publish if args.action == "publish" else verify_publication
            result = operation(
                raw_base / spec.archive_sha256, curated_base / args.version, spec, load_config()
            )
            receipt_name = (
                "data-publication.json" if args.action == "publish" else "data-verification.json"
            )
        result = {
            **result,
            "status": "passed",
            "stage": args.action,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "jobs_submitted": False,
        }
        _write_state(args.state_dir / receipt_name, canonical_json(result))
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (DataError, ConfigurationError) as error:
        report = getattr(error, "quality_report", None)
        if isinstance(report, dict):
            _write_state(args.state_dir / "failed-data-quality.json", canonical_json(report))
        print(json.dumps({"status": "failed", "stage": args.action, "message": str(error)}))
        return 1
    except OSError:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "stage": args.action,
                    "message": "Local file operation failed; check paths and permissions.",
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
