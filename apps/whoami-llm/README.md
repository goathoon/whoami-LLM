# whoami-llm (Local CLI Only)

Velog 글 기반으로 개인 RAG를 구성하고, `llama-cli`로 최종 답변을 생성하는 로컬 전용 가이드입니다.

## 범위

- 포함: `whoami-llm` CLI (`build`, `chunk`, `embed`, `search`, `rag`)
- 제외: Kubernetes/MinIO/Weaviate 배포 절차

## 사전 요구사항

- `python3` (권장 3.10+)
- `pip`
- `cmake` (llama.cpp 빌드용)
- `apps/whoami-llm/qwen.gguf` 파일

## 설치

저장소 루트에서:

```bash
python3 -m venv .venv-whoami
source .venv-whoami/bin/activate
pip install -e apps/whoami-llm
```

## llama-cli 빌드

저장소 루트에서:

```bash
cmake -S apps/whoami-llm/llama.cpp -B apps/whoami-llm/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build apps/whoami-llm/llama.cpp/build --config Release --target llama-cli -j8
```

### 권장 실행파일

이 저장소에는 CPU 강제/단일턴 옵션이 적용된 래퍼가 포함되어 있습니다.

```text
apps/whoami-llm/llama-cli-cpu
```

- 내부 호출: `apps/whoami-llm/llama.cpp/build/bin/llama-cli --device none --single-turn`
- 목적:
  - macOS Metal 초기화 오류 회피 (`failed to create command queue`)
  - `rag` 실행 시 인터랙티브 모드 고정 방지

## 빠른 시작

`<username>`을 본인 Velog 계정으로 바꿔 실행:

```bash
whoami-llm build --blog https://velog.io/@<username>/posts
whoami-llm chunk --blog https://velog.io/@<username>/posts
whoami-llm embed --blog https://velog.io/@<username>/posts
```

검색 확인:

```bash
whoami-llm search "이 개발자는 어떤 기술 스택을 주로 다루나?" \
  --blog https://velog.io/@<username>/posts \
  --top-k 5 \
  --retrieval-mode auto
```

최종 RAG 답변 생성:

```bash
whoami-llm rag "이 개발자는 어떤 엔지니어인가?" \
  --blog https://velog.io/@<username>/posts \
  --retrieval-mode auto \
  --llama-cli "$(python3 -c 'from pathlib import Path; print((Path(\"apps/whoami-llm/llama.cpp/build/bin/llama-cli\")).resolve())')"
```

- `--retrieval-mode`: `auto`(기본), `semantic`, `persona`
  - `auto`: 질문이 성향/철학형이면 질의 확장 + 다양성 기반 검색을 자동 사용
  - `semantic`: 기존 단일 semantic 검색
  - `persona`: 성향/의사결정 관점 검색을 강제로 사용

## Retrieval 개선 (성향/철학 질문 대응)

상세 설계 문서: `apps/whoami-llm/docs/retrieval-architecture.md`

기존 semantic 검색은 질의와 가장 유사한 일부 문단만 고르기 때문에, 아래 질문에 약할 수 있습니다.

- "이 개발자는 어떤 생각을 가지고 있는가?"
- "문제를 어떤 기준으로 판단하는가?"

현재 `auto` / `persona` 모드에서는 다음 전략을 사용합니다.

1. 질의 확장
   - 원 질의 + `가치관/원칙`, `왜 그렇게 설계했는지`, `트레이드오프`, `회고/배운점` 등 관점 질의로 확장
2. 다중 결과 융합
   - 확장 질의별 검색 결과를 RRF(Reciprocal Rank Fusion)로 합쳐 편향을 줄임
3. 회고성 단서 가중치
   - `왜`, `교훈`, `실수`, `트레이드오프`, `판단` 같은 문장이 포함된 청크에 소폭 가중
4. 다양성 선택(MMR)
   - 비슷한 내용의 중복 청크 대신, 서로 다른 맥락의 근거를 우선 선택

추천:

- 스택/키워드 탐색 질의: `--retrieval-mode semantic`
- 인물/성향/의사결정 질의: `--retrieval-mode auto` 또는 `persona`

예시:

```bash
whoami-llm rag "이 개발자는 어떤 생각을 가지고 있는가?" \
  --blog https://velog.io/@<username>/posts \
  --retrieval-mode auto \
  --top-k 8
```

macOS Metal 초기화 오류가 있으면 래퍼를 대신 사용하세요:

```bash
--llama-cli "$(python3 -c 'from pathlib import Path; print((Path(\"apps/whoami-llm/llama-cli-cpu\")).resolve())')"
```

## 검증된 전체 실행 예시 (`goat_hoon`)

아래는 실제 실행이 확인된 순서입니다.

```bash
python3 -m venv .venv-whoami
source .venv-whoami/bin/activate
pip install -e apps/whoami-llm

cmake -S apps/whoami-llm/llama.cpp -B apps/whoami-llm/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build apps/whoami-llm/llama.cpp/build --config Release --target llama-cli -j8

whoami-llm build --blog https://velog.io/@goat_hoon/posts
whoami-llm chunk --blog https://velog.io/@goat_hoon/posts
whoami-llm embed --blog https://velog.io/@goat_hoon/posts

whoami-llm rag "이 개발자는 어떤 엔지니어인가?" \
  --blog https://velog.io/@goat_hoon/posts \
  --llama-cli "$(python3 -c 'from pathlib import Path; print((Path(\"apps/whoami-llm/llama.cpp/build/bin/llama-cli\")).resolve())')"
```

## 모델 선택 방식

- 기본:
  - `apps/whoami-llm/qwen.gguf`를 고정 사용
- `--model` 지정:
  - 다른 로컬 GGUF 파일로 override 가능

예시:

```bash
whoami-llm rag "요약해줘" \
  --blog https://velog.io/@<username>/posts \
  --model /absolute/path/to/model.gguf \
  --llama-cli "$(python3 -c 'from pathlib import Path; print((Path(\"apps/whoami-llm/llama.cpp/build/bin/llama-cli\")).resolve())')"
```

## 자주 쓰는 옵션

- `whoami-llm chunk`
  - `--target-tokens`, `--overlap-tokens`, `--min-tokens`
- `whoami-llm embed`
  - `--model` (임베딩 모델), `--batch-size`, `--no-normalize`
- `whoami-llm rag`
  - `--top-k`, `--max-tokens`, `--temperature`, `--context-chars`

## 출력 데이터 위치

실행 결과 인덱스/메타 파일은 `whoami_llm.storage` 모듈이 관리하는 data 디렉토리에 저장됩니다.
(`blog`의 username 기준으로 `*_index/` 폴더 생성)

## 트러블슈팅

- `ModuleNotFoundError`:
  - 가상환경 활성화 후 `pip install -e apps/whoami-llm` 재실행
- `llama-cli` 실행 실패:
  - `--llama-cli`에 실행파일 절대경로 지정
  - Python 경로 해석 예시:
    - `--llama-cli "$(python3 -c 'from pathlib import Path; print((Path(\"apps/whoami-llm/llama.cpp/build/bin/llama-cli\")).resolve())')"`
  - macOS Metal 오류가 보이면 `apps/whoami-llm/llama-cli-cpu` 사용
- 모델 파일 경로 오류:
  - 기본 파일 `apps/whoami-llm/qwen.gguf` 존재 여부 확인
  - 필요 시 `--model /absolute/path/to/model.gguf` 사용
- `rag`에서 Hugging Face 네트워크 호출 오류:
  - `embed`를 먼저 1회 실행한 뒤
  - 기본은 오프라인 모드(`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`)
  - 온라인이 필요하면 `HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0`로 override
