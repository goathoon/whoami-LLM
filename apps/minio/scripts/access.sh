#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-minio}"
RELEASE_NAME="${RELEASE_NAME:-minio}"
API_PORT="${API_PORT:-9000}"
CONSOLE_PORT="${CONSOLE_PORT:-9001}"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "[ERROR] kubectl not found"
  exit 1
fi

if ! kubectl -n "${NAMESPACE}" get svc "${RELEASE_NAME}" >/dev/null 2>&1; then
  echo "[ERROR] service ${RELEASE_NAME} not found in namespace ${NAMESPACE}"
  exit 1
fi

if ! kubectl -n "${NAMESPACE}" get svc "${RELEASE_NAME}-console" >/dev/null 2>&1; then
  echo "[ERROR] service ${RELEASE_NAME}-console not found in namespace ${NAMESPACE}"
  exit 1
fi

ROOT_USER="$(sed -n 's/^rootUser:[[:space:]]*//p' apps/minio/values.yaml | head -n1)"
ROOT_PASSWORD="$(sed -n 's/^rootPassword:[[:space:]]*//p' apps/minio/values.yaml | head -n1)"

echo "[INFO] MinIO API:     http://127.0.0.1:${API_PORT}"
echo "[INFO] MinIO Console: http://127.0.0.1:${CONSOLE_PORT}"
echo "[INFO] Access Key: ${ROOT_USER:-<check apps/minio/values.yaml>}"
echo "[INFO] Secret Key: ${ROOT_PASSWORD:-<check apps/minio/values.yaml>}"
echo "[INFO] Press Ctrl+C to stop port-forward"

kubectl -n "${NAMESPACE}" port-forward svc/"${RELEASE_NAME}" "${API_PORT}":9000 2>&1 &
API_PID=$!
kubectl -n "${NAMESPACE}" port-forward svc/"${RELEASE_NAME}-console" "${CONSOLE_PORT}":9001 2>&1 &
CONSOLE_PID=$!

cleanup() {
  kill "${API_PID}" "${CONSOLE_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

wait
