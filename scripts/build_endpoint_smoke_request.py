"""Create a representative endpoint payload from verified Phase 2 training data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from epm_platform.data.curation import verify_curated
from epm_platform.data.spec import load_spec
from epm_platform.serving.schema import MAX_OBSERVATIONS_PER_INSTANCE, OBSERVATION_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SPEC = ROOT / "config" / "cmapss-source.json"
SOURCE_MANIFEST_SHA256 = "ba6452660f76495349daf4853f4dee1b511a299ead831d150dab757340f33640"


def build(output: Path, sample_size: int = 20) -> dict:
    spec = load_spec(SOURCE_SPEC)
    source_root = ROOT / "data" / "curated" / "cmapss" / f"sha256-{SOURCE_MANIFEST_SHA256}"
    verify_curated(source_root, spec)
    digest = hashlib.sha256((source_root / "manifest.json").read_bytes()).hexdigest()
    if digest != SOURCE_MANIFEST_SHA256:
        raise ValueError("Curated C-MAPSS source manifest differs from the approved Phase 2 asset.")
    table = pq.ParquetFile(source_root / "FD001" / "train.parquet").read(
        columns=["unit_id", *OBSERVATION_COLUMNS]
    )

    units = table["unit_id"].to_pylist()
    instances = []
    start = 0
    while start < len(units) and len(instances) < sample_size:
        end = start + 1
        while end < len(units) and units[end] == units[start]:
            end += 1
        if end - start >= MAX_OBSERVATIONS_PER_INSTANCE:
            rows = table.slice(end - MAX_OBSERVATIONS_PER_INSTANCE, MAX_OBSERVATIONS_PER_INSTANCE)
            observations = []
            columns = rows.to_pydict()
            for index in range(rows.num_rows):
                observations.append({name: columns[name][index] for name in OBSERVATION_COLUMNS})
            instances.append(
                {"subset": "FD001", "unit_id": units[start], "observations": observations}
            )
        start = end
    if len(instances) != sample_size:
        raise ValueError("Verified Phase 2 data did not provide enough representative engines.")

    payload = {"instances": instances}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"instances": len(instances), "history_cycles_each": 10, "subset": "FD001"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()
    if not 20 <= args.sample_size <= 100:
        parser.error("--sample-size must be from 20 to 100 for drift monitoring.")
    print(json.dumps(build(args.output, args.sample_size), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
