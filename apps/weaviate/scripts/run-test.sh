#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-weaviate"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing virtualenv: ${VENV_DIR}"
  echo "Run: make weaviate-py-setup"
  exit 1
fi

echo "Waiting for Weaviate to be ready..."
for _ in {1..30}; do
  if curl -sf "http://localhost:8080/v1/.well-known/ready" >/dev/null; then
    break
  fi
  sleep 2
done

source "${VENV_DIR}/bin/activate"
python "${ROOT_DIR}/apps/weaviate/python/test_embedding_to_weaviate.py"
