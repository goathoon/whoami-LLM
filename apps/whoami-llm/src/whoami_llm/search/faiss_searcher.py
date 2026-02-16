from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

import faiss


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    meta: dict[str, Any]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"meta file not found: {path}")

    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_embed_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def search_faiss(
    *,
    query: str,
    index_path: Path,
    meta_path: Path,
    info_path: Path,
    top_k: int = 5,
    model_override: str | None = None,
) -> list[SearchResult]:
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    # 1) load index + meta + embed_info
    index = faiss.read_index(str(index_path))
    meta = load_jsonl(meta_path)
    info = load_embed_info(info_path)

    normalize = bool(info.get("normalize", True))
    model_name = model_override or info.get("model_name") or "sentence-transformers/all-MiniLM-L6-v2"

    # 2) embed query
    model = SentenceTransformer(model_name)
    q = model.encode([query], convert_to_numpy=True).astype(np.float32)  # (1, dim)

    if normalize:
        faiss.normalize_L2(q)

    # 3) search (IndexFlatIP 기준: score=inner product, normalize 했으면 cosine)
    k = min(top_k, len(meta))
    scores, ids = index.search(q, k)  # (1,k), (1,k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results: list[SearchResult] = []
    for r, (score, idx) in enumerate(zip(scores, ids), start=1):
        if idx < 0:
            continue
        if idx >= len(meta):
            # 메타가 꼬였을 때 방어
            continue
        results.append(SearchResult(rank=r, score=float(score), meta=meta[idx]))
    return results
