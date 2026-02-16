#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-weaviate}"
RELEASE_NAME="${RELEASE_NAME:-weaviate}"
DELETE_NAMESPACE="${DELETE_NAMESPACE:-false}"

if ! command -v helm >/dev/null 2>&1; then
  echo "[ERROR] helm not found"
  exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "[ERROR] kubectl not found"
  exit 1
fi

helm uninstall "${RELEASE_NAME}" -n "${NAMESPACE}" || true

if [[ "${DELETE_NAMESPACE}" == "true" ]]; then
  kubectl delete ns "${NAMESPACE}" --wait=true
fi

echo "[OK] Weaviate removed"
