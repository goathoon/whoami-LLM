# k8s-local-ai-platform

Minikube 기반 로컬 Kubernetes 환경에서 AI 관련 컴포넌트(MinIO, Weaviate)를 Helm으로 설치/테스트하는 저장소입니다.

## 현재 구성

```text
.
├── Makefile
├── scripts/
│   └── setup-cluster.sh
└── apps/
    ├── minio/
    │   ├── values.yaml
    │   └── scripts/
    │       ├── install.sh
    │       ├── access.sh
    │       └── uninstall.sh
    └── weaviate/
        ├── values.yaml
        ├── README.md
        ├── scripts/
        │   ├── install.sh
        │   ├── access.sh
        │   ├── setup-python.sh
        │   ├── run-test.sh
        │   └── uninstall.sh
        └── python/
            ├── requirements.txt
            └── test_embedding_to_weaviate.py
```

## 사전 요구사항

- `minikube`
- `kubectl`
- `helm`
- (Weaviate Python 테스트 시) `python3`, `venv`

## 빠른 시작

### 1) 클러스터 준비

```bash
make cluster-setup
```

옵션 예시:

```bash
make cluster-setup MINIKUBE_PROFILE=local MINIKUBE_CPUS=6 MINIKUBE_MEMORY=12288 MINIKUBE_DISK_SIZE=40g
make cluster-setup MINIKUBE_DRIVER=docker
```

### 2) MinIO 설치/접속/제거

```bash
make minio-install
make minio-access
make minio-status
make minio-logs
make minio-uninstall
```

- 기본 네임스페이스: `minio`
- 기본 포트포워딩:
  - API: `http://127.0.0.1:9000`
  - Console: `http://127.0.0.1:9001`
- 기본 계정은 `apps/minio/values.yaml`의 `rootUser`, `rootPassword`를 사용합니다.

### 3) Weaviate 설치/접속/제거

```bash
make weaviate-install
make weaviate-access
make weaviate-status
make weaviate-logs
make weaviate-uninstall
```

- 기본 네임스페이스: `weaviate`
- 기본 포트포워딩:
  - HTTP: `http://127.0.0.1:8080`
  - gRPC: `127.0.0.1:50051` (서비스 존재 시)
- Readiness 체크:

```bash
curl -s http://127.0.0.1:8080/v1/.well-known/ready && echo
```

### 4) Weaviate Python 테스트

`Makefile`에는 Python 환경/테스트용 타겟이 없으므로 스크립트를 직접 실행합니다.

```bash
bash apps/weaviate/scripts/setup-python.sh
bash apps/weaviate/scripts/run-test.sh
```

또는 이미 가상환경이 준비된 경우:

```bash
make weaviate-test
```

## 설정 파일

- MinIO Helm values: `apps/minio/values.yaml`
- Weaviate Helm values: `apps/weaviate/values.yaml`

네임스페이스/릴리즈명/values 경로는 `Makefile` 변수 오버라이드로 변경할 수 있습니다.

예시:

```bash
make minio-install NAMESPACE=storage RELEASE_NAME=minio-local
make weaviate-install WEAVIATE_NAMESPACE=vector WEAVIATE_RELEASE_NAME=weaviate-local
```
