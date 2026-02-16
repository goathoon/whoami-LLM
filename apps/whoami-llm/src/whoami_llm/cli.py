import typer
import json

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
from whoami_llm.search.faiss_searcher import search_faiss


app = typer.Typer()

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
    show_chars: int = typer.Option(280, "--show-chars", help="본문 미리보기 길이"),
):
    """
    query -> embedding -> FAISS top-k -> meta 출력
    """
    username = extract_username(blog)

    idx_path = faiss_index_file(username)
    m_path = meta_file(username)
    info_path = embed_info_file(username)

    results = search_faiss(
        query=query,
        index_path=idx_path,
        meta_path=m_path,
        info_path=info_path,
        top_k=top_k,
        model_override=model,
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


if __name__ == "__main__":
    app()
