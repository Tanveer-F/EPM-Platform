#!/usr/bin/env bash
set -euo pipefail
# The pinned curated Azure image supplies conda; model code runs in an isolated Python 3.12 env.
runtime="$(mktemp -d /tmp/epm-runtime-XXXXXXXX)"
trap 'rm -rf "$runtime"' EXIT
conda create --yes --quiet --override-channels --channel conda-forge --prefix "$runtime" python=3.12.10 pip=25.2
"$runtime/bin/python" -m pip --isolated install --disable-pip-version-check --no-cache-dir --only-binary=:all: --index-url https://pypi.org/simple -r config/runtime-requirements.txt
"$runtime/bin/python" -m pip check
"$runtime/bin/python" -m epm_platform.baseline.training "$@"
