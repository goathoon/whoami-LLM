#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-weaviate}"
RELEASE_NAME="${RELEASE_NAME:-weaviate}"

HTTP_PORT="${HTTP_PORT:-8080}"    # local
GRPC_PORT="${GRPC_PORT:-50051}"   # local

# svc names (default based on your output)
HTTP_SVC="${HTTP_SVC:-weaviate}"
GRPC_SVC="${GRPC_SVC:-weaviate-grpc}"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "[ERROR] kubectl not found"
  exit 1
fi

# Validate services exist
if ! kubectl -n "${NAMESPACE}" get svc "${HTTP_SVC}" >/dev/null 2>&1; then
  echo "[ERROR] service ${HTTP_SVC} not found in namespace ${NAMESPACE}"
  exit 1
fi

if ! kubectl -n "${NAMESPACE}" get svc "${GRPC_SVC}" >/dev/null 2>&1; then
  echo "[WARN] service ${GRPC_SVC} not found in namespace ${NAMESPACE} (gRPC port-forward will be skipped)"
  GRPC_SVC=""
fi

echo "[INFO] Weaviate HTTP:  http://127.0.0.1:${HTTP_PORT}"
echo "[INFO] Ready check:    curl -s http://127.0.0.1:${HTTP_PORT}/v1/.well-known/ready && echo"
if [[ -n "${GRPC_SVC}" ]]; then
  echo "[INFO] Weaviate gRPC:  127.0.0.1:${GRPC_PORT}"
fi
echo "[INFO] Press Ctrl+C to stop port-forward(s)"

# Start port-forwards
pids=()

kubectl -n "${NAMESPACE}" port-forward "svc/${HTTP_SVC}" "${HTTP_PORT}:80" >/dev/null 2>&1 &
pids+=("$!")

if [[ -n "${GRPC_SVC}" ]]; then
  kubectl -n "${NAMESPACE}" port-forward "svc/${GRPC_SVC}" "${GRPC_PORT}:50051" >/dev/null 2>&1 &
  pids+=("$!")
fi

cleanup() {
  echo
  echo "[INFO] Stopping port-forward(s)..."
  for pid in "${pids[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT INT TERM

wait
