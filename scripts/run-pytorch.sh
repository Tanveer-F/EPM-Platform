#!/usr/bin/env bash
set -euo pipefail
runtime="$(mktemp -d /tmp/epm-torch-XXXXXXXX)"
trap 'rm -rf "$runtime"' EXIT
conda create --yes --quiet --override-channels --channel conda-forge --prefix "$runtime" python=3.12.10 pip=25.2
"$runtime/bin/python" -m pip --isolated install --disable-pip-version-check --no-cache-dir --only-binary=:all: --index-url https://pypi.org/simple -r config/runtime-requirements.txt
"$runtime/bin/python" -m pip check
"$runtime/bin/python" -c "import torch; assert torch.version.cuda is None; print('CPU-only PyTorch:',torch.__version__)"
"$runtime/bin/python" -m epm_platform.deep_learning.tracking "$@"
