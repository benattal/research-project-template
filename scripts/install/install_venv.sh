#!/usr/bin/env bash
# Create ./venv, install the Python requirements, and run the static tests.
#
# Usage: bash scripts/install/install_venv.sh
#   PYTHON=<interpreter>  use a specific base interpreter (default python3)
#
# TODO(template): load the environment modules the project needs before this
# script (docs/environments/index.md) and pin versions in requirements.txt.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

if [ ! -x venv/bin/python ]; then
    "${PYTHON:-python3}" -m venv venv
fi
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
venv/bin/python -m pytest -q tests/architecture tests/unit
