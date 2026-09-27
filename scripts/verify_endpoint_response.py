"""Compare endpoint predictions with the same registered artifact scored locally."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from epm_platform.serving.scoring import InferenceService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()

    service = InferenceService.load(
        args.model_dir,
        Path(__file__).parents[1]
        / "src"
        / "epm_platform"
        / "serving"
        / "monitoring-reference.json",
    )
    expected = service.score(args.request.read_bytes())["predictions"]
    try:
        actual = json.loads(args.response.read_text(encoding="utf-8-sig"))["predictions"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        raise ValueError("Azure endpoint response was not a valid predictions document.") from None
    if len(expected) != len(actual):
        raise ValueError("Azure endpoint prediction count differs from the local result.")

    differences = []
    for local, remote in zip(expected, actual, strict=True):
        if (
            {key: remote.get(key) for key in ("subset", "unit_id", "cycle")}
            != {key: local[key] for key in ("subset", "unit_id", "cycle")}
            or isinstance(remote.get("rul_cycles"), bool)
            or not isinstance(remote.get("rul_cycles"), (int, float))
            or not math.isfinite(remote["rul_cycles"])
        ):
            raise ValueError("Azure endpoint returned an invalid prediction record.")
        differences.append(abs(float(remote["rul_cycles"]) - local["rul_cycles"]))
    if max(differences, default=0.0) > 0.05:
        raise ValueError(
            "Azure predictions differ from the registered local model by >0.05 cycles."
        )
    print(
        json.dumps(
            {
                "status": "verified",
                "prediction_count": len(expected),
                "max_abs_difference_cycles": max(differences, default=0.0),
                "mean_prediction_cycles": float(np.mean([row["rul_cycles"] for row in actual])),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
