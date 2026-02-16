#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-weaviate"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/apps/weaviate/python/requirements.txt"

echo "Python environment is ready: ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
