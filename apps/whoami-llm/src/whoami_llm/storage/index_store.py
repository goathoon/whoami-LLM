from pathlib import Path
from whoami_llm.storage.jsonl_store import ensure_data_dir

def index_dir(username: str) -> Path:
    data_dir = ensure_data_dir()
    safe = username.strip().lstrip("@")
    d = data_dir / f"{safe}_index"
    d.mkdir(parents=True, exist_ok=True)
    return d

def faiss_index_file(username: str) -> Path:
    return index_dir(username) / "index.faiss"

def meta_file(username: str) -> Path:
    return index_dir(username) / "meta.jsonl"

def embed_info_file(username: str) -> Path:
    return index_dir(username) / "embed_info.json"
