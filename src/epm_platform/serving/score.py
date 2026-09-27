"""Azure ML Managed Online Endpoint entry point and local model loader."""

import os
from pathlib import Path

from epm_platform.serving.scoring import InferenceService

_SERVICE: InferenceService | None = None
_REFERENCE = Path(__file__).with_name("monitoring-reference.json")


def init() -> None:
    global _SERVICE
    model_root = os.environ.get("AZUREML_MODEL_DIR")
    if not model_root:
        raise RuntimeError("Azure ML did not provide the registered model mount.")
    _SERVICE = InferenceService.load(Path(model_root), _REFERENCE)


def run(raw_data):
    if _SERVICE is None:
        raise RuntimeError("The inference service has not been initialized.")
    return _SERVICE.score(raw_data)
