# k8s-local-ai-platform

Velog 글을 기반으로 로컬에서 Personal RAG를 실행해
"이 개발자는 누구인가?"를 답하는 CLI 프로젝트입니다.

## Local CLI Quick Start

상세 가이드는 `apps/whoami-llm/README.md`를 참고하세요.

### 1) 요구사항

- `python3` (권장 3.10+)
- `pip`
- `cmake`
- `git` (submodule 초기화용)
- 모델 파일: `apps/whoami-llm/qwen.gguf`

### 2) 저장소/서브모듈 준비

처음 clone:

```bash
git clone --recurse-submodules <REPO_URL>
cd k8s-local-ai-platform
```

이미 clone 되어 있다면:

```bash
git submodule sync --recursive
git submodule update --init --recursive apps/whoami-llm/llama.cpp
```

### 3) qwen.gguf 다운로드

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download <HF_REPO> <GGUF_FILE> --local-dir apps/whoami-llm
mv apps/whoami-llm/<GGUF_FILE> apps/whoami-llm/qwen.gguf
```

대체(`curl`):

```bash
curl -L "https://huggingface.co/<HF_REPO>/resolve/main/<GGUF_FILE>?download=true" \
  -o apps/whoami-llm/qwen.gguf
```

### 4) Python 환경 설치

```bash
python3 -m venv .venv-whoami
source .venv-whoami/bin/activate
pip install -e apps/whoami-llm
```

### 5) llama-cli 빌드

```bash
cmake -S apps/whoami-llm/llama.cpp -B apps/whoami-llm/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build apps/whoami-llm/llama.cpp/build --config Release --target llama-cli -j8
```

권장 실행파일: `apps/whoami-llm/llama-cli-cpu`

### 6) 인덱스 생성

```bash
whoami-llm build --blog https://velog.io/@<username>/posts
whoami-llm chunk --blog https://velog.io/@<username>/posts
whoami-llm embed --blog https://velog.io/@<username>/posts
```

### 7) RAG 실행

```bash
whoami-llm rag "이 개발자는 어떤 엔지니어인가?" \
  --blog https://velog.io/@<username>/posts \
  --retrieval-mode auto \
  --llama-cli "$(python3 -c 'from pathlib import Path; print((Path("apps/whoami-llm/llama-cli-cpu")).resolve())')"
```

## Related Docs

- 상세 로컬 CLI 가이드: `apps/whoami-llm/README.md`
- Retrieval 설계: `apps/whoami-llm/docs/retrieval-architecture.md`
