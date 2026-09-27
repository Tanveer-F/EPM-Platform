# C-MAPSS data-quality report

**Status: passed** — all four original subsets remain separate.

- Source: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- Original archive SHA-256: `74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f`
- Curated version: `sha256-ba6452660f76495349daf4853f4dee1b511a299ead831d150dab757340f33640`

| Subset | Train rows | Train engines | Test rows | Test engines / RUL labels |
|---|---:|---:|---:|---:|
| FD001 | 20,631 | 100 | 13,096 | 100 |
| FD002 | 53,759 | 260 | 33,991 | 259 |
| FD003 | 24,720 | 100 | 16,596 | 100 |
| FD004 | 61,249 | 249 | 41,214 | 248 |

## Checks

- null values: **0**.
- nonfinite values: **0**.
- duplicate records: **0**.
- duplicate unit cycles: **0**.
- conflicting unit cycles: **0**.
- Exact test trajectories copied from training, including prefixes: **0**.
- Schema, integer IDs/cycles, unit blocks, cycle start/order/continuity, expected counts and supplied test-RUL alignment passed.
- Numeric engine IDs are scoped by `(subset, split, unit_id)`; reuse across files is expected.
- Per-column min/max/mean/population standard deviation and distinct counts are retained in the bundle's `data-quality.json`.

## Applied transformations

- ASCII whitespace parsing without sorting or numeric cleanup.
- Naming unit_id, cycle, setting_1..3 and sensor_01..21.
- Typing unit_id/cycle as int32 and settings/sensors as float64.
- Aligning supplied RUL line i to test unit_id i in a separate int32 table.
- No rows/columns dropped, imputation, reordering, clipping, scaling, denoising, random splitting or derived training targets.

## Warnings retained without cleaning

- FD001/train: zero_variance — `setting_3`, `sensor_01`, `sensor_05`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`. Retained unchanged.
- FD001/test: zero_variance — `setting_3`, `sensor_01`, `sensor_05`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`. Retained unchanged.
- FD003/train: zero_variance — `setting_3`, `sensor_01`, `sensor_05`, `sensor_16`, `sensor_18`, `sensor_19`. Retained unchanged.
- FD003/test: zero_variance — `setting_3`, `sensor_01`, `sensor_05`, `sensor_16`, `sensor_18`, `sensor_19`. Retained unchanged.

## Source assumptions and documented exceptions

- NASA catalog and archive README reverse FD004 engine counts. Archive filenames and contents are authoritative: train=249 and test=248. No records are moved to match the documentation.
- The final sensor label in NASA documentation is a typo: 26 total columns minus unit/cycle/3 settings means 21 sensors, confirmed in all eight observation files.
- Source README is Windows-1252; numeric observation and RUL files are ASCII. Original bytes, PDF and whitespace are preserved.
- NASA catalog lists License not specified; PCoE requests acknowledgement of NASA and data contributors. Public availability is not a claim of an unrestricted license.
- The supplied test RUL vector is aligned by line order to contiguous test unit IDs 1..N. It remains separate from observations; no training RUL is derived.

No model, experiment, endpoint, monitoring or retraining workflow is part of this report.
