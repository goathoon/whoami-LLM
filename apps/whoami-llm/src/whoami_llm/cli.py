import typer
import json
import shutil
from pathlib import Path

from whoami_llm.extract.velog_rss_description import description_to_text
from whoami_llm.storage.document_store import write_documents,documents_file
from whoami_llm.storage.jsonl_store import save_posts, posts_file
from whoami_llm.storage.chunk_store import write_chunks, chunks_file
from whoami_llm.storage.index_store import faiss_index_file, meta_file, embed_info_file

from whoami_llm.velog.rss import fetch_posts, extract_username
from whoami_llm.velog.rss import extract_username

from whoami_llm.chunking.chunker import ChunkConfig, chunk_text, count_tokens
from whoami_llm.embedding.faiss_builder import EmbedConfig, build_faiss_index
from whoami_llm.storage.index_store import faiss_index_file, meta_file, embed_info_file
from whoami_llm.search.faiss_searcher import search_faiss_advanced
from whoami_llm.llm.llama_cli_runner import run_llama_cli


app = typer.Typer()
DEFAULT_RAG_MODEL_PATH = Path(__file__).resolve().parents[2] / "qwen.gguf"
DEFAULT_LLAMA_CLI_PATH = Path(__file__).resolve().parents[2] / "llama-cli-cpu"

def _print_chunk_config(cfg: ChunkConfig):
    typer.echo(
        "Chunk config -> "
        f"target_tokens={cfg.target_tokens}, "
        f"overlap_tokens={cfg.overlap_tokens}, "
        f"min_tokens={cfg.min_tokens}"
    )


def _load_posts_from_file(pfile):
    posts = []
    with open(pfile, "r", encoding="utf-8") as f:
        for line in f:
            posts.append(json.loads(line))
    return posts


def _extract_docs_from_posts(posts, min_chars: int):
    docs: list[dict] = []
    warn_count = 0

    for i, p in enumerate(posts, start=1):
        title = p.get("title")
        url = p.get("link")
        published = p.get("pub_date")
        desc = p.get("description")

        text = description_to_text(desc)
        char_count = len(text)

        if char_count < min_chars:
            warn_count += 1
            typer.echo(f"[warn] [{i}/{len(posts)}] Short text ({char_count} chars): {url}")

        docs.append(
            {
                "source": "rss_description",
                "url": url,
                "title": title,
                "published": published,
                "text": text,
                "char_count": char_count,
            }
        )

        typer.echo(f"[{i}/{len(posts)}] Extracted {char_count:,} chars from RSS description.")

    return docs, warn_count


def _build_rag_prompt(query: str, results: list, context_chars: int) -> str:
    blocks: list[str] = []
    for i, res in enumerate(results, start=1):
        meta = res.meta
        title = meta.get("title") or "(no title)"
        url = meta.get("url") or ""
        text = (meta.get("text") or "").strip().replace("\n", " ")
        text = text[:context_chars]
        blocks.append(
            f"[CONTEXT {i}]\n"
            f"title: {title}\n"
            f"url: {url}\n"
            f"text: {text}\n"
        )

    context = "\n".join(blocks)
    return (
        "You are an expert technical interviewer and writing coach.\n\n"
        "Goal:\n"
        "Evaluate the blog author’s engineering profile and strengths using:\n"
        "1) General software engineering knowledge (your training),\n"
        "2) The supplied blog context as primary evidence.\n\n"
        "Rules:\n"
        "- You MAY use general knowledge to interpret and frame what the context implies (e.g., what a technique typically signals, common tradeoffs).\n"
        "- You MUST NOT invent specific facts about the author that are not supported by the context.\n"
        "- Separate what is directly evidenced vs. what is inferred.\n"
        "- When you infer, explicitly mark it as an inference and explain why it follows from the context.\n"
        "- If key details are missing, state what is missing and provide reasonable assumptions or alternative interpretations.\n\n"
        "Output format (Korean):\n"
        "1) 한 줄 요약 (이 개발자는 어떤 엔지니어인가)\n"
        "2) Evidence 기반 강점 5개\n"
        "   - 각 항목마다: 주장 → 근거 [CONTEXT n] → 해석(일반지식 기반)\n"
        "3) 기술적 깊이/성숙도 평가 (0~5 스케일로)\n"
        "   - 성능/운영, 분산시스템, 데이터/DB, 테스트/품질, 커뮤니케이션/문서화\n"
        "   - 각 점수마다 근거 [CONTEXT n] 또는 \"컨텍스트 부족\"\n"
        "4) 리스크/한계 3개 (컨텍스트로 관찰되는 약점)\n"
        "5) 다음 성장 방향 5개 (가장 임팩트 큰 순)\n"
        "6) 면접에서 물어볼 질문 8개 (컨텍스트 기반으로 깊이를 검증하는 질문)\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer in Korean:"

    )


@app.command()
def ingest(blog: str = typer.Option(..., "--blog")):
    typer.echo("🔎 Fetching Velog posts...")
    posts = fetch_posts(blog)
    username = extract_username(blog)

    typer.echo(f"Found {len(posts)} posts.")
    path = save_posts(username, posts)
    typer.echo(f"Saved -> {path}")


@app.command()
def extract(
    blog: str = typer.Option(..., "--blog"),
    limit: int = typer.Option(0, "--limit", help="0이면 전부, 아니면 상위 N개만 처리"),
    min_chars: int = typer.Option(800, "--min-chars", help="description 텍스트 최소 길이 경고 기준"),
):
    username = extract_username(blog)
    pfile = posts_file(username)

    if not pfile.exists():
        raise typer.BadParameter(f"posts file not found: {pfile}. Run ingest first.")

    posts = _load_posts_from_file(pfile)
    if limit > 0:
        posts = posts[:limit]

    typer.echo(f"Building documents from RSS descriptions: {len(posts)} posts")
    docs, warn_count = _extract_docs_from_posts(posts, min_chars=min_chars)

    out = write_documents(username, docs)
    typer.echo(f"Saved -> {out}")
    if warn_count:
        typer.echo(f"Warnings: {warn_count} posts had text shorter than {min_chars} chars.")


@app.command()
def build(
    blog: str = typer.Option(..., "--blog"),
    limit: int = typer.Option(0, "--limit", help="0이면 전부, 아니면 상위 N개만 처리"),
    min_chars: int = typer.Option(800, "--min-chars", help="description 텍스트 최소 길이 경고 기준"),
):
    """
    One-shot: ingest + extract
    """
    # 1) ingest (RSS fetch -> posts.jsonl)
    typer.echo("🔎 Fetching Velog posts...")
    posts = fetch_posts(blog)
    username = extract_username(blog)

    typer.echo(f"Found {len(posts)} posts.")
    ppath = save_posts(username, posts)
    typer.echo(f"Saved -> {ppath}")

    # 2) extract (posts -> documents.jsonl)
    if limit > 0:
        posts_dicts = [p.__dict__ for p in posts[:limit]]
    else:
        posts_dicts = [p.__dict__ for p in posts]

    typer.echo(f"Building documents from RSS descriptions: {len(posts_dicts)} posts")
    docs, warn_count = _extract_docs_from_posts(posts_dicts, min_chars=min_chars)

    out = write_documents(username, docs)
    typer.echo(f"Saved -> {out}")
    if warn_count:
        typer.echo(f"Warnings: {warn_count} posts had text shorter than {min_chars} chars.")

@app.command()
def chunk(
    blog: str = typer.Option(..., "--blog"),
    target_tokens: int = typer.Option(250, "--target-tokens", help="권장 500~800"),
    overlap_tokens: int = typer.Option(100, "--overlap-tokens"),
    min_tokens: int = typer.Option(200, "--min-tokens"),
):
    """
    documents.jsonl -> chunks.jsonl
    """
    username = extract_username(blog)
    dfile = documents_file(username)
    if not dfile.exists():
        raise typer.BadParameter(f"documents file not found: {dfile}. Run extract/build first.")

    cfg = ChunkConfig(
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
    )
    _print_chunk_config(cfg)

    rows: list[dict] = []
    total_docs = 0
    total_chunks = 0

    with open(dfile, "r", encoding="utf-8") as f:
        for doc_idx, line in enumerate(f, start=1):
            doc = json.loads(line)
            total_docs += 1

            url = doc.get("url")
            title = doc.get("title")
            published = doc.get("published")
            text = doc.get("text") or ""

            chunks = chunk_text(text, cfg)
            for c_idx, c in enumerate(chunks, start=1):
                rows.append(
                    {
                        "source": doc.get("source", "rss_description"),
                        "doc_id": doc_idx,
                        "chunk_id": c_idx,
                        "url": url,
                        "title": title,
                        "published": published,
                        "text": c,
                        "token_count": count_tokens(c),
                    }
                )

            total_chunks += len(chunks)
            typer.echo(f"[doc {doc_idx}] chunks={len(chunks)} url={url}")

    out = write_chunks(username, rows)
    typer.echo(f"Total docs: {total_docs}")
    typer.echo(f"Total chunks created: {total_chunks}")
    typer.echo(f"Saved -> {out}")


@app.command()
def embed(
    blog: str = typer.Option(..., "--blog"),
    model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2"),
    batch_size: int = typer.Option(64),
    no_normalize: bool = typer.Option(False),
):
    username = extract_username(blog)
    cfile = chunks_file(username)

    if not cfile.exists():
        raise typer.BadParameter("Run chunk first")

    cfg = EmbedConfig(
        model_name=model,
        batch_size=batch_size,
        normalize=not no_normalize,
    )

    build_faiss_index(
        chunks_path=cfile,
        index_path=faiss_index_file(username),
        meta_path=meta_file(username),
        info_path=embed_info_file(username),
        cfg=cfg,
    )

    typer.echo("✅ Embedding + FAISS index build done.")

@app.command()
def search(
    query: str = typer.Argument(..., help="검색 질의 (예: MongoDB)"),
    blog: str = typer.Option(..., "--blog"),
    top_k: int = typer.Option(5, "--top-k"),
    model: str | None = typer.Option(None, "--model", help="(옵션) 임베딩 모델 override"),
    retrieval_mode: str = typer.Option(
        "auto",
        "--retrieval-mode",
        help="retrieval 모드: auto | semantic | persona",
    ),
    show_chars: int = typer.Option(280, "--show-chars", help="본문 미리보기 길이"),
):
    """
    query -> embedding -> FAISS top-k -> meta 출력
    """
    username = extract_username(blog)

    idx_path = faiss_index_file(username)
    m_path = meta_file(username)
    info_path = embed_info_file(username)

    results = search_faiss_advanced(
        query=query,
        index_path=idx_path,
        meta_path=m_path,
        info_path=info_path,
        top_k=top_k,
        model_override=model,
        retrieval_mode=retrieval_mode,
    )

    if not results:
        typer.echo("No results.")
        raise typer.Exit(code=0)

    typer.echo(f'🔎 Query: "{query}" (top_k={top_k})')
    typer.echo("-" * 80)

    for res in results:
        meta = res.meta
        title = meta.get("title") or "(no title)"
        url = meta.get("url") or ""
        text = (meta.get("text") or "").replace("\n", " ").strip()
        preview = text[:show_chars] + ("…" if len(text) > show_chars else "")

        typer.echo(f"[{res.rank}] score={res.score:.4f}")
        typer.echo(f"    title: {title}")
        if url:
            typer.echo(f"    url:   {url}")
        typer.echo(f"    text:  {preview}")
        typer.echo("-" * 80)


@app.command()
def rag(
    query: str = typer.Argument(..., help="최종 질의 (예: 이 개발자는 어떤 엔지니어인가?)"),
    blog: str = typer.Option(..., "--blog"),
    top_k: int = typer.Option(5, "--top-k"),
    retrieval_mode: str = typer.Option(
        "auto",
        "--retrieval-mode",
        help="retrieval 모드: auto | semantic | persona",
    ),
    model: str | None = typer.Option(None, "--model", help="로컬 GGUF 파일 경로 (기본: whoami-llm/qwen.gguf)"),
    llama_cli: str = typer.Option(
        str(DEFAULT_LLAMA_CLI_PATH) if DEFAULT_LLAMA_CLI_PATH.exists() else "llama-cli",
        "--llama-cli",
        help="llama-cli 실행파일 경로",
    ),
    max_tokens: int = typer.Option(256, "--max-tokens"),
    temperature: float = typer.Option(0.2, "--temperature"),
    context_chars: int = typer.Option(1200, "--context-chars"),
):
    """
    query -> FAISS 검색 -> context 구성 -> llama-cli로 최종 답변 생성
    """
    username = extract_username(blog)

    results = search_faiss_advanced(
        query=query,
        index_path=faiss_index_file(username),
        meta_path=meta_file(username),
        info_path=embed_info_file(username),
        top_k=top_k,
        retrieval_mode=retrieval_mode,
    )

    if not results:
        typer.echo("No retrieval results. Run build/chunk/embed first.")
        raise typer.Exit(code=1)

    model_path = Path(model).expanduser().resolve() if model else DEFAULT_RAG_MODEL_PATH.resolve()
    if not model_path.exists():
        raise typer.BadParameter(f"model file not found: {model_path}")

    if "/" in llama_cli or "\\" in llama_cli or llama_cli.startswith("."):
        llama_cli_path = Path(llama_cli).expanduser().resolve()
        if not llama_cli_path.exists():
            raise typer.BadParameter(f"llama-cli file not found: {llama_cli_path}")
        llama_cli = str(llama_cli_path)
    else:
        if shutil.which(llama_cli) is None:
            raise typer.BadParameter(
                f"llama-cli executable not found in PATH: {llama_cli}. "
                "Use --llama-cli /absolute/path/to/llama-cli"
            )

    prompt = _build_rag_prompt(query=query, results=results, context_chars=context_chars)
    typer.echo("Generating final answer with llama-cli...")
    answer = run_llama_cli(
        llama_cli=llama_cli,
        model_path=model_path,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    typer.echo("-" * 80)
    typer.echo(answer)
    typer.echo("-" * 80)


if __name__ == "__main__":
    app()
