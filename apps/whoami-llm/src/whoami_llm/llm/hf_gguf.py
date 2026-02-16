from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


@dataclass(frozen=True)
class GGUFFile:
    repo_id: str
    filename: str
    size_bytes: int


def _resolve_size(sibling: object) -> int:
    # huggingface_hub version에 따라 size 정보 위치가 달라질 수 있어 방어적으로 처리
    size = getattr(sibling, "size", None)
    if isinstance(size, int):
        return size

    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, dict):
        lfs_size = lfs.get("size")
        if isinstance(lfs_size, int):
            return lfs_size

    return 0


def find_smallest_gguf(repo_id: str, pattern: str = "*.gguf") -> GGUFFile:
    api = HfApi()
    info = api.model_info(repo_id)

    candidates: list[GGUFFile] = []
    for sibling in info.siblings or []:
        name = getattr(sibling, "rfilename", "")
        if not name or not name.lower().endswith(".gguf"):
            continue
        if not fnmatch(name, pattern):
            continue
        size = _resolve_size(sibling)
        candidates.append(GGUFFile(repo_id=repo_id, filename=name, size_bytes=size))

    if not candidates:
        raise FileNotFoundError(f"No GGUF files found in repo={repo_id} pattern={pattern}")

    # size 정보 없는 파일은 뒤로 보낸다
    candidates.sort(key=lambda x: (x.size_bytes <= 0, x.size_bytes, x.filename))
    return candidates[0]


def download_gguf(
    repo_id: str,
    filename: str,
    local_dir: Path,
    token: str | None = None,
) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
        token=token,
    )
    return Path(local_path)
