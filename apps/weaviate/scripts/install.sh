#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-weaviate}"
RELEASE_NAME="${RELEASE_NAME:-weaviate}"
VALUES_FILE="${VALUES_FILE:-apps/weaviate/values.yaml}"
MINIKUBE_PROFILE="${MINIKUBE_PROFILE:-minikube}"
WEAVIATE_CHART_VERSION="${WEAVIATE_CHART_VERSION:-}"

if ! command -v helm >/dev/null 2>&1; then
  echo "[ERROR] helm command not found"
  exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "[ERROR] kubectl command not found"
  exit 1
fi

if ! command -v minikube >/dev/null 2>&1; then
  echo "[ERROR] minikube command not found"
  exit 1
fi

if [[ "$(minikube status -p "${MINIKUBE_PROFILE}" --format='{{.Host}}' 2>/dev/null || true)" != "Running" ]]; then
  echo "[ERROR] minikube profile ${MINIKUBE_PROFILE} is not running"
  exit 1
fi

helm repo add weaviate https://weaviate.github.io/weaviate-helm/ >/dev/null 2>&1 || true
helm repo update weaviate >/dev/null

kubectl get ns "${NAMESPACE}" >/dev/null 2>&1 || kubectl create ns "${NAMESPACE}" >/dev/null

HELM_ARGS=(
  upgrade --install "${RELEASE_NAME}" weaviate/weaviate
  --namespace "${NAMESPACE}"
  --values "${VALUES_FILE}"
  --wait
  --timeout 3m
)

if [[ -n "${WEAVIATE_CHART_VERSION}" ]]; then
  HELM_ARGS+=(--version "${WEAVIATE_CHART_VERSION}")
fi

helm "${HELM_ARGS[@]}"

echo "[OK] Weaviate installed"
echo " - namespace: ${NAMESPACE}"
echo " - release:   ${RELEASE_NAME}"
