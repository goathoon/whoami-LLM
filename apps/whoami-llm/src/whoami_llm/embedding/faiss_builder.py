from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import faiss  # faiss-cpu


@dataclass(frozen=True)
class EmbedConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool = True  # cosine(=inner product) 쓰려면 정규화 권장


def read_chunks_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        raise FileNotFoundError(f"chunks jsonl not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_faiss_index(
    *,
    chunks_path: Path,
    index_path: Path,
    meta_path: Path,
    info_path: Path,
    cfg: EmbedConfig,
) -> None:
    # 1) load chunks
    chunks = list(read_chunks_jsonl(chunks_path))
    if not chunks:
        raise ValueError(f"No chunks found in: {chunks_path}")

    texts = [c.get("text", "") or "" for c in chunks]

    # 2) embed
    model = SentenceTransformer(cfg.model_name)
    emb = model.encode(
        texts,
        batch_size=cfg.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    if emb.ndim != 2 or emb.shape[0] != len(chunks):
        raise RuntimeError(f"Unexpected embedding shape: {emb.shape}")

    if cfg.normalize:
        faiss.normalize_L2(emb)  # cosine similarity == inner product (정규화 시)

    dim = int(emb.shape[1])

    # 3) index (cosine용: IndexFlatIP)
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # 4) save files
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "doc_id": c.get("doc_id"),
                        "chunk_id": c.get("chunk_id"),
                        "url": c.get("url"),
                        "title": c.get("title"),
                        "published": c.get("published"),
                        "token_count": c.get("token_count"),
                        "text": c.get("text"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": cfg.model_name,
                "batch_size": cfg.batch_size,
                "normalize": cfg.normalize,
                "chunks_path": str(chunks_path),
                "vector_dim": dim,
                "num_vectors": int(index.ntotal),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
