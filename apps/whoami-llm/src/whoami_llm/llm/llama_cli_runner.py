from __future__ import annotations

import subprocess
from pathlib import Path


def run_llama_cli(
    *,
    llama_cli: str,
    model_path: Path,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    cmd = [
        llama_cli,
        "-m",
        str(model_path),
        "-p",
        prompt,
        "-n",
        str(max_tokens),
        "--temp",
        str(temperature),
        "--no-display-prompt",
    ]

    proc = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "llama-cli failed\n"
            f"cmd={' '.join(cmd)}\n"
            f"stderr={proc.stderr.strip()}"
        )

    return proc.stdout.strip()
