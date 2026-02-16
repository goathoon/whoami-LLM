from __future__ import annotations

import json
import re
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


_PROFILE_QUERY_PATTERNS = [
    r"어떤\s*생각",
    r"가치관",
    r"철학",
    r"성향",
    r"스타일",
    r"관점",
    r"판단",
    r"의사결정",
    r"어떤\s*엔지니어",
    r"who\s+is",
    r"profile",
    r"mindset",
]

_REFLECTIVE_HINTS = [
    "왜",
    "배운",
    "교훈",
    "회고",
    "실수",
    "트레이드오프",
    "선택",
    "중요",
    "원칙",
    "판단",
    "느꼈",
    "고민",
    "문제",
    "해결",
    "협업",
    "테스트",
    "운영",
]


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


def _load_search_assets(
    *,
    index_path: Path,
    meta_path: Path,
    info_path: Path,
    model_override: str | None = None,
):
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(str(index_path))
    meta = load_jsonl(meta_path)
    info = load_embed_info(info_path)

    normalize = bool(info.get("normalize", True))
    model_name = model_override or info.get("model_name") or "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    return index, meta, normalize, model


def _encode(model: SentenceTransformer, texts: list[str], normalize: bool) -> np.ndarray:
    vec = model.encode(texts, convert_to_numpy=True).astype(np.float32)
    if normalize:
        faiss.normalize_L2(vec)
    return vec


def _is_profile_query(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in _PROFILE_QUERY_PATTERNS)


def _expand_profile_queries(query: str) -> list[str]:
    return [
        query,
        f"{query} 가치관 원칙",
        f"{query} 왜 그렇게 설계했는지",
        f"{query} 트레이드오프 의사결정",
        f"{query} 회고 배운 점 실수",
        f"{query} 협업 테스트 운영 관점",
    ]


def _reflective_boost(query: str, meta_row: dict[str, Any]) -> float:
    text = (meta_row.get("text") or "").lower()
    title = (meta_row.get("title") or "").lower()
    query_terms = [t for t in re.findall(r"[a-z0-9가-힣_]+", query.lower()) if len(t) > 1]

    hint_hits = sum(1 for w in _REFLECTIVE_HINTS if (w in text or w in title))
    query_hits = sum(1 for t in query_terms if t in text or t in title)

    # Keep boost conservative to avoid overriding semantic relevance.
    return min(0.25, 0.015 * hint_hits + 0.01 * query_hits)


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def _mmr_select(
    *,
    candidate_indices: list[int],
    candidate_scores: dict[int, float],
    candidate_vecs: np.ndarray,
    query_vec: np.ndarray,
    top_k: int,
    diversity_lambda: float,
) -> list[int]:
    if not candidate_indices:
        return []

    selected: list[int] = []
    pool: list[int] = list(range(len(candidate_indices)))
    pos_by_idx = {idx: pos for pos, idx in enumerate(candidate_indices)}

    q_sims = (candidate_vecs @ query_vec.reshape(-1, 1)).reshape(-1)

    while pool and len(selected) < top_k:
        best_pos = None
        best_score = -1e9
        for pos in pool:
            idx = candidate_indices[pos]
            relevance = float(candidate_scores.get(idx, 0.0) + 0.35 * q_sims[pos])
            if not selected:
                mmr = relevance
            else:
                sel_pos = [pos_by_idx[s] for s in selected]
                redundancy = float(np.max(candidate_vecs[pos] @ candidate_vecs[sel_pos].T))
                mmr = diversity_lambda * relevance - (1.0 - diversity_lambda) * redundancy
            if mmr > best_score:
                best_score = mmr
                best_pos = pos
        if best_pos is None:
            break
        selected.append(candidate_indices[best_pos])
        pool.remove(best_pos)

    return selected


def search_faiss(
    *,
    query: str,
    index_path: Path,
    meta_path: Path,
    info_path: Path,
    top_k: int = 5,
    model_override: str | None = None,
) -> list[SearchResult]:
    # 1) load index + meta + embed_info + model
    index, meta, normalize, model = _load_search_assets(
        index_path=index_path,
        meta_path=meta_path,
        info_path=info_path,
        model_override=model_override,
    )

    # 2) embed query
    q = _encode(model, [query], normalize=normalize)

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


def search_faiss_advanced(
    *,
    query: str,
    index_path: Path,
    meta_path: Path,
    info_path: Path,
    top_k: int = 5,
    model_override: str | None = None,
    retrieval_mode: str = "auto",  # auto | semantic | persona
    candidate_k: int = 40,
    diversity_lambda: float = 0.75,
) -> list[SearchResult]:
    mode = retrieval_mode.lower().strip()
    if mode not in {"auto", "semantic", "persona"}:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")

    if mode == "semantic" or (mode == "auto" and not _is_profile_query(query)):
        return search_faiss(
            query=query,
            index_path=index_path,
            meta_path=meta_path,
            info_path=info_path,
            top_k=top_k,
            model_override=model_override,
        )

    index, meta, normalize, model = _load_search_assets(
        index_path=index_path,
        meta_path=meta_path,
        info_path=info_path,
        model_override=model_override,
    )
    if not meta:
        return []

    expanded = _expand_profile_queries(query) if mode in {"auto", "persona"} else [query]
    expanded_vecs = _encode(model, expanded, normalize=normalize)

    pool_k = min(max(top_k, candidate_k), len(meta))
    rank_lists: list[list[int]] = []
    best_semantic: dict[int, float] = {}

    for vec in expanded_vecs:
        scores, ids = index.search(vec.reshape(1, -1), pool_k)
        row_scores = scores[0].tolist()
        row_ids = ids[0].tolist()
        ranked_ids: list[int] = []
        for score, idx in zip(row_scores, row_ids):
            if idx < 0 or idx >= len(meta):
                continue
            ranked_ids.append(int(idx))
            best_semantic[idx] = max(float(score), best_semantic.get(idx, -1e9))
        rank_lists.append(ranked_ids)

    fused: dict[int, float] = {}
    for ranked in rank_lists:
        for r, idx in enumerate(ranked, start=1):
            fused[idx] = fused.get(idx, 0.0) + _rrf_score(r)

    for idx in list(fused.keys()):
        fused[idx] += 0.15 * float(best_semantic.get(idx, 0.0))
        fused[idx] += _reflective_boost(query, meta[idx])

    candidate_indices = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[:pool_k]
    if not candidate_indices:
        return []

    candidate_texts = [(meta[i].get("text") or "") for i in candidate_indices]
    candidate_vecs = _encode(model, candidate_texts, normalize=normalize)
    query_vec = _encode(model, [query], normalize=normalize)[0]

    selected = _mmr_select(
        candidate_indices=candidate_indices,
        candidate_scores=fused,
        candidate_vecs=candidate_vecs,
        query_vec=query_vec,
        top_k=min(top_k, len(candidate_indices)),
        diversity_lambda=diversity_lambda,
    )

    out: list[SearchResult] = []
    for rank, idx in enumerate(selected, start=1):
        out.append(SearchResult(rank=rank, score=float(fused.get(idx, 0.0)), meta=meta[idx]))
    return out
