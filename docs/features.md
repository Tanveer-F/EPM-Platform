# Phase 3: causal run-to-failure benchmark features

This cloud-agnostic pipeline builds one pooled, **uncapped remaining-useful-life (RUL)
regression** dataset from all four run-to-failure benchmark subsets. It does not fit a model, scaler,
imputer, operating-condition classifier, or feature selector. No Azure SDK is imported.
Phase 2 remains the immutable, lossless source of observations and supplied test labels.

## Reviewed inputs and public API

The complete approved recipe is `config\features.json`. Missing keys, extra keys,
changed types, alternative feature/split options, and unpinned source identities are
rejected rather than defaulted. A recipe change requires an explicit implementation
and configuration review; this is not a general-purpose feature configuration engine.

The pinned source is:

- Asset: `epm-cmapss-curated:d-xjsfezqpozevgso26sct6tpodm`.
- Manifest SHA-256:
  `ba6452660f76495349daf4853f4dee1b511a299ead831d150dab757340f33640`.
- Local bundle: `data\curated\cmapss\sha256-<source digest>`.
- Source specification: `config\cmapss-source.json`.

`epm_platform.features.pipeline` exports:

```python
from pathlib import Path
from epm_platform.features.pipeline import build_features, verify_features

bundle = build_features(
    source_root=Path("data") / "curated" / "cmapss" / (
        "sha256-ba6452660f76495349daf4853f4dee1b511a299ead831d150dab757340f33640"
    ),
    output_root=Path("data") / "ml-ready" / "cmapss",
    config_path=Path("config") / "features.json",
    source_spec_path=Path("config") / "cmapss-source.json",
)
manifest = verify_features(bundle.root)
```

`build_features(...) -> FeatureBundle` returns a frozen dataclass with `root: Path`,
`version: str`, `manifest_sha256: str`, and `summary: dict`.
`verify_features(root: Path) -> dict` returns the verified manifest or raises `DataError`.
The pure `features_for_engine(table: pyarrow.Table) -> pyarrow.Table` accepts exactly
one typed Phase 2 engine prefix, ordered from cycle 1, and preserves every prefix row.

## Engine separation and supervision

An original engine key is `(subset, source_split, unit_id)`. Numeric unit IDs are
**split-local**: NASA test unit 1 is not NASA training unit 1. Subset is also part of the
key; engines from different subsets never share rolling history.

For each subset, original training engines are sorted lexicographically by the full
SHA-256 digest of UTF-8 `42|{subset}|{unit_id}` (unit ID breaks any digest tie). The first
`ceil(engine_count * 0.2)` form validation; all remaining engines form training. No
rows from an engine cross the training/validation boundary. Neither test observations
nor test labels influence split selection, censor selection, training features, or
training sample weights.

| Partition | Rows retained | Target `rul` |
| --- | --- | --- |
| Training | Every observed row of each assigned full training engine | Full training-engine `max(cycle) - cycle` |
| Validation | Exactly one endpoint of a censored prefix per held-out training engine | Full training-engine `max(cycle) - cut_cycle` |
| Test | Exactly the last observed row per original test engine | Original supplied RUL aligned by unit ID |

For validation, let `life = max(cycle)` of the full held-out training trajectory,
`low = ceil(0.5 * life)`, and `high = floor(0.8 * life)`. Interpret the full SHA-256
digest of `cut|42|{subset}|{unit_id}` as a big-endian integer `h` and choose
`cut_cycle = low + h % (high - low + 1)`. Only observations through this cut enter
feature computation. The complete lifetime is retrospective supervision/censoring
information, never a model feature. Tiny invalid inputs without a feasible split or
censor interval are rejected, not silently changed.

Training weights are `(total_training_rows / training_engine_count) / rows_in_engine`.
Their global mean is one, and every training engine has equal total weight across the
pooled dataset. Validation and test weights are exactly one. Weights are metadata,
not predictor columns.

**No per-row test labels are derived.** Earlier observed test rows provide causal
history only; supplied endpoint RUL is not propagated backward. RUL is never capped,
clipped, normalized, or converted to a health score. A training failure row has RUL 0.

## Exactly 35 numeric features

`FEATURE_COLUMNS` is the authoritative ordered tuple. The order is:

1. `cycle` (1).
2. `setting_1`, `setting_2`, `setting_3` (3).
3. `sensor_01` through `sensor_21`, including constant sensors (21).
4. `sensor_03_mean10`, `sensor_03_slope10`, `sensor_04_mean10`,
   `sensor_04_slope10`, `sensor_11_mean10`, `sensor_11_slope10` (6).
5. `setting_1_mean10`, `setting_2_mean10`, `setting_3_mean10` (3).
6. `history_count` (1).

All windows are trailing **10 rows including the current row**, within one engine.
Partial windows preserve early rows. Means use only the available rows. Slopes are
ordinary least squares against local observed cycle coordinates, with zero for a
single observation. `history_count = min(cycle, 10)` because Phase 2 cycles start at
one and are contiguous. There are no centered windows, future values, scaling,
imputation, sensor dropping, or fitted transformations. A longer engine suffix cannot
change any feature already computed from its prefix.

`cycle` and `history_count` are int32; the other features are float64. All values must
be finite and non-null. No engine ID, subset ID, lifetime, target, partition label, or
weight is included in `FEATURE_COLUMNS`.

Parquet column order is `subset`, `unit_id`, the 35 feature columns, then `split`, `rul`,
`sample_weight`: **40 unique columns**. Metadata are explicitly
`("subset", "unit_id", "cycle", "split", "rul", "sample_weight")`, with strings for
subset/split, int32 for unit/cycle, and float64 for target/weight. `cycle` intentionally
serves as both metadata and a feature and is stored only once. Consumers must select
`manifest["feature_columns"]`, not infer predictors by dropping a target or by selecting
all numeric columns. The table/manifest target name is always `rul`; `uncapped_rul` is
the recipe's supervision policy, not a second target column.

## Immutable bundle and verification

There are exactly seven files, with only three pooled partitions:

```text
sha256-<full manifest digest>\
    train.parquet
    validation.parquet
    test.parquet
    splits.json
    feature-summary.json
    manifest.json
    _SUCCESS.json
```

`splits.json` records every original source engine, assigned partition, observed source
row count, and validation cut. For full training-source trajectories `source_rows` is
the retrospective lifetime; for test it is only the observed prefix length. Original
test endpoint labels are retained as `supplied_rul` for mapping checks. These audit
fields never enter the feature table. Assignments are canonically ordered by subset,
training source before test source, then unit ID.

The manifest includes schema version 1, recipe version `"1"`, dataset
`nasa-cmapss-ml-ready`, exact source asset identity and digest, full configuration and
its canonical JSON digest, ordered feature/metadata columns, target, per-partition
row/engine counts, writer provenance, and a sorted `files` list of the five content
paths with SHA-256 and byte size. The summary includes per-subset counts, target
ranges, weight sum, and the validation-policy caveat. No absolute path or timestamp
enters any bundle metadata.

Canonical JSON and hashing use `epm_platform.data.manifest`. The hash of the complete
canonical `manifest.json` is the bundle digest. `_SUCCESS.json` records that digest
and `sha256-<digest>` version. The producer validates the source using the existing
`load_spec` and `verify_curated` APIs and checks the approved manifest digest before
building and again before finalization. It stages privately under the output root,
verifies the completed bundle, and finalizes by atomic directory rename. An existing
version is verified and reused without modifying it; corruption is rejected, never
repaired or overwritten. Failed private staging directories are removed. Output inside
the immutable source is forbidden.

`verify_features` checks the exact inventory, canonical JSON, marker/digest agreement,
allowlisted paths, hashes, sizes, approved schema/configuration, finite values, counts,
source-qualified engine assignments, deterministic holdout/cuts, endpoint-only rules,
RUL mapping, and engine-balanced weights. It also recomputes training rolling features
from the retained observations. Added leakage columns, duplicate assignments, test
engines assigned to training/validation, reordered engine groups, and malformed tables
are rejected. Symbolic links, junctions, directories, and path traversal entries are
not permitted inside a bundle.

The verifier deliberately accepts an **arbitrary root basename**, including mounted
input folders: identity is checked through the manifest and marker. The producer alone
enforces the local finalized `sha256-<digest>` directory name. Parquet is read using
`ParquetFile`, so mount directories such as `scope=value` cannot inject inferred Hive
partition columns.

Hashes establish integrity, not an independent signature. Validation/test partitions
retain only endpoints, so their discarded historical windows cannot be recomputed
from these partitions alone. Their provenance comes from the pinned, verified source
and producer recipe; independent source-to-feature verification is a deterministic
rebuild. Bitwise reproduction is expected with the same recipe and PyArrow writer
version/options, not promised across arbitrary library versions. The writer version
is recorded in the manifest; a reader need not use that exact version.

## Interpretation and limitations

The 50-80%-of-life validation rule is **on-policy offline evaluation with retrospective
selection bias**. Real deployment does not know an engine's eventual failure time;
this simulation selects a specific lifetime region and does not establish performance
at arbitrary ages, under changed operating conditions, or on real maintenance data.
Pooled subsets span different conditions and fault modes, without condition-specific
scaling or a fitted health representation. Low offline error would not demonstrate
perfect health estimation, diagnostic certainty, safe operational decisions, or field
readiness. Original test labels remain evaluation-only; no model selection is done by
this feature pipeline.

## Focused validation

Run only the feature suite and owned-code lint from the project root (the artifact
parent must exist before using pytest's explicit project-local base directory):

```powershell
New-Item -ItemType Directory -Path .artifacts -Force | Out-Null
.\.venv\Scripts\python.exe -m pytest tests\features\test_pipeline.py --basetemp .artifacts\feature-tests -q
.\.venv\Scripts\python.exe -m ruff check src\epm_platform\features\pipeline.py tests\features\test_pipeline.py
```

Tests exercise exact schema/counts, OLS windows, partial history, invalid inputs,
future-perturbation/prefix invariance, engine boundaries, uncapped labels, original test
mapping, deterministic split/censor/build behavior, balanced weights, independent test
inputs, mounted-folder verification, source immutability, staging cleanup, corruption,
rehashed semantic tampering, and path traversal rejection. Synthetic builds replace
only source verification; the production entry point uses real Phase 2 validation.

Final focused validation: **77 tests passed**; Ruff lint and format checks passed on
both owned Python files. The real bundle also passed verification with the final code.

### Verified local dataset build

The pinned source was built successfully using PyArrow 25.0.1 at:

`data\ml-ready\cmapss\sha256-2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd`

The manifest SHA-256 is
`2f284013d4f9b82ea24b310ee6c2a426d85d73b81cca7ca6dceedafdb0dd41dd`.

| Partition | Rows | Engines | Uncapped RUL range |
| --- | ---: | ---: | ---: |
| Train | 128,967 | 567 | 0–542 |
| Validation | 142 | 142 | 28–170 |
| Test | 707 | 707 | 6–195 |

| Subset | Training rows | Training engines | Validation endpoints | Test endpoints |
| --- | ---: | ---: | ---: | ---: |
| FD001 | 16,711 | 80 | 20 | 100 |
| FD002 | 43,035 | 208 | 52 | 259 |
| FD003 | 20,236 | 80 | 20 | 100 |
| FD004 | 48,985 | 199 | 50 | 248 |

All partitions have the same 35 predictor columns. Training sample weights sum to
128,967 (mean one). An independent second producer invocation under an explicitly
project-local artifact directory reproduced the identical manifest digest and the
SHA-256/size fingerprints of **all seven files**. Both builds passed real source and
feature verification; fingerprints of every source file were unchanged. The second
build directory was removed after comparison. This establishes local reproduction,
not cloud asset verification or publication.
