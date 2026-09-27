# Phase 2 validation evidence

## Status

**Phase 2 complete — verified 2026-09-26.** Original raw bytes are preserved locally and in Azure; curated Parquet passes validation; the Azure ML data asset is registered and independently verified. Repeat publication performed no uploads or new registration. At the Phase 2 completion checkpoint, Phase 3+ work had not started.

Registered reference: `azureml:epm-cmapss-curated:d-xjsfezqpozevgso26sct6tpodm` (`uri_folder`), through credential-free datastore `epm_cmapss_curated` in the existing Phase 1 workspace.

## Verified source and local dataset

- Authoritative NASA catalog and official ZIP GET verified; no mirror used.
- Archive: 12,425,978 bytes; CRC validation passed.
- Archive SHA-256: `74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f`.
- Original ZIP plus all 14 members preserved byte-for-byte, with per-file fingerprints and unchanged README encoding/whitespace/PDF.
- Curated content version: `sha256-ba6452660f76495349daf4853f4dee1b511a299ead831d150dab757340f33640`.
- Independent rebuild in a different output directory produced the identical manifest/version.
- 160,359 training observations; 104,897 test observations; 707 supplied test-RUL labels.
- All eight observation files have 26 columns. Schema, finite values, positive integer ID/cycle domains, duplicates/conflicting keys, unit consistency, cycle order/continuity, split integrity and RUL alignment passed.
- No missing/nonfinite values, duplicate records, duplicate/conflicting unit-cycle keys or exact copied train/test trajectories/prefixes found.
- Constant channels retained. FD004 count reversal and sensor-label typo in source documentation explicitly recorded; data was not altered to match the prose.
- No imputation, clipping, rescaling, row reordering, dropped rows/columns, denoising, random splitting or derived training targets.

See the generated [concise quality report](data-quality-report.md); full statistics and rules are in each immutable bundle's `data-quality.json`.

## Engineering and cloud checks

| Check | Evidence |
|---|---|
| Python data tests | Final Phase 2 suite: **252 passed, 1 skipped**. The real-symlink test lacks Windows privilege; simulated symlink/junction rejection passed |
| Foundation regressions | The existing 92 Python foundation tests passed in integrated runs; final infrastructure/script regression checks: 42 passed |
| Ruff / dependency consistency | Passed |
| Data-access Bicep contracts | 17 passed; only two private containers and two container-scoped grants |
| IaC security scan | Final Checkov run: 9 passed, 0 failed, 6 previously reviewed Phase 1 development exceptions; no new exception |
| ARM preview | Final preview: **NoChange** for both containers and both grants. Existing foundation resources are ignored, not modified. Explicit encryption defaults match the live containers |
| Container/access deployment | Succeeded in the existing workspace account; live role inspection confirmed the two container-scoped Blob Data Contributor grants. No new account/service/compute |
| Azure data registration | Passed at `2026-09-26T09:38:45Z`. Earlier 71/48-character labels encountered different lookup/registration limits; owner-approved 28-character alias succeeded. Full checksums and storage paths were never shortened |
| Remote byte/inventory verification | **16 raw + 15 curated objects** streamed and SHA-256/ETag verified with exact prefix inventories. Completion markers verified; no unrelated files uploaded |
| Registered version readback | CLI/SDK readback confirmed name, `uri_folder`, matching source/manifest tags and datastore; version inventory contains exactly the verified alias |
| Idempotent republish / read-only verify | Read-only verification passed at `09:46:25Z`; repeat publication passed at `09:52:02Z`, with **0 uploads, 31 reuses, no new registration** |
| Final compute/job check | Allocated=0, target=0, min=0, max=1; workspace job count=0 |
| Source hygiene | No actual subscription/publishing-principal IDs in source/configuration/docs; immutable JSON metadata has no host paths or signed credentials; temporary deployment-parameter file removed |

## Preserved artifacts, replay and costs

- Raw bundle: **57,778,000 bytes**, including original ZIP, 14 original members and provenance manifest.
- Curated bundle: **10,816,155 bytes**, including 12 Parquet files, quality evidence, manifest and completion marker.
- Combined versioned Blob payload: **68,594,155 bytes** (about 68.6 MB), excluding service metadata and any future retained versions. Storage/transactions and possible egress are usage-based; no Azure compute was used for Phase 2 and no billing-total claim is made.
- Bytes were uploaded before the registry-label compatibility failure. Successful registration resumed by verifying/reusing all 31 objects, demonstrating recovery without overwrites or duplicate uploads.
- Final CLI curation replay at `09:53:20Z` produced the same full-hex version and updated preparation receipt. Source-spec SHA-256 remains `c67ff21406a8cdd1a72e88ec9b8e329c61e42cd3c6d9598e171e60483bcdcb9c`.

Sanitized local evidence lives in ignored `.azure\data-acquisition.json`, `data-preparation.json`, `data-publication.json`, `data-verification.json`, `data-access-what-if.json` and `data-quality-summary.md`. The tracked concise quality report is generated from the immutable bundle report. Full Parquet/source data stays out of source control.

## Important limits

Source fingerprints are recorded from the official download, not a NASA-published signature. The NASA catalog does not specify a license; attribution is retained and no unrestricted-license claim is made.

Data assets do not physically lock Blob contents. The approved design uses application-level no-overwrite, content-addressed paths, completion markers and streamed checksum/ETag verification—not WORM/legal holds.

No Phase 3+ code or workload has been started. No model, experiment, feature-engineering workflow, endpoint, monitoring/retraining or CI/CD is part of Phase 2.
