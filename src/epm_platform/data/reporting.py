"""Concise human-readable evidence from the validated, immutable quality report."""

from epm_platform.data.spec import SUBSETS, SourceSpec


def render_quality_report(report: dict, spec: SourceSpec, version: str) -> str:
    lines = [
        "# C-MAPSS data-quality report",
        "",
        f"**Status: {report['status']}** — all four original subsets remain separate.",
        "",
        f"- Source: {spec.catalog_url}",
        f"- Original archive SHA-256: `{spec.archive_sha256}`",
        f"- Curated version: `{version}`",
        "",
        "| Subset | Train rows | Train engines | Test rows | Test engines / RUL labels |",
        "|---|---:|---:|---:|---:|",
    ]
    totals = {
        "null_values": 0,
        "nonfinite_values": 0,
        "duplicate_records": 0,
        "duplicate_unit_cycles": 0,
        "conflicting_unit_cycles": 0,
    }
    copied = 0
    warnings = []
    for subset in SUBSETS:
        parts = report["subsets"][subset]
        train, test = parts["train"], parts["test"]
        lines.append(
            f"| {subset} | {train['rows']['parsed']:,} | {train['units']['count']} | "
            f"{test['rows']['parsed']:,} | {test['units']['count']} |"
        )
        copied += parts["integrity"]["counts"]["copied_test_trajectories"]
        for split in ("train", "test"):
            for key in totals:
                totals[key] += parts[split]["counts"][key]
            for warning in parts[split]["warnings"]:
                warnings.append(
                    f"- {subset}/{split}: {warning['rule']} — "
                    + ", ".join(f"`{name}`" for name in warning.get("columns", []))
                    + ". Retained unchanged."
                )
    lines += ["", "## Checks", ""]
    lines.extend(f"- {key.replace('_', ' ')}: **{value}**." for key, value in totals.items())
    lines += [
        f"- Exact test trajectories copied from training, including prefixes: **{copied}**.",
        "- Schema, integer IDs/cycles, unit blocks, cycle start/order/continuity, expected counts "
        "and supplied test-RUL alignment passed.",
        "- Numeric engine IDs are scoped by `(subset, split, unit_id)`; "
        "reuse across files is expected.",
        "- Per-column min/max/mean/population standard deviation and distinct counts are retained "
        "in the bundle's `data-quality.json`.",
        "",
        "## Applied transformations",
        "",
    ]
    lines.extend(f"- {transformation}." for transformation in report["transformations"])
    lines += [
        "- No rows/columns dropped, imputation, reordering, clipping, scaling, denoising, "
        "random splitting or derived training targets.",
        "",
        "## Warnings retained without cleaning",
        "",
        *(warnings or ["- None."]),
        "",
        "## Source assumptions and documented exceptions",
        "",
    ]
    lines.extend(f"- {note}" for note in spec.source_notes)
    lines += [
        "",
        "No model, experiment, endpoint, monitoring or retraining workflow is part of this report.",
        "",
    ]
    return "\n".join(lines)
