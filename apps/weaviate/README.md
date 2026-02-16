# Weaviate + Sentence Transformers 로컬 테스트

## 1) Weaviate를 K8s에 설치

```bash
make weaviate-install
```

## 2) 로컬 포트로 접속 연결(포트포워딩)

```bash
make weaviate-access
```

헬스체크:

```bash
curl http://localhost:8080/v1/.well-known/ready
```

## 3) Python 환경 구성

```bash
bash apps/weaviate/scripts/setup-python.sh
```

가상환경 위치: `.venv-weaviate`

## 4) 임베딩 생성 + Vector DB 저장 테스트

```bash
bash apps/weaviate/scripts/run-test.sh
```
혹은 
```bash
make weaviate-test
````

테스트 스크립트는 아래를 수행합니다.

- `sentence-transformers/all-MiniLM-L6-v2` 모델로 텍스트 임베딩 생성
- Weaviate 컬렉션(`TextDocument`) 생성 (없을 때만)
- 텍스트/메타데이터 + 벡터 저장
- 질의 문장 임베딩으로 `near_vector` 검색 후 Top-3 출력

## 5) 상태/로그 확인

```bash
make weaviate-status
make weaviate-logs
```

## 6) 제거

```bash
make weaviate-uninstall
```
