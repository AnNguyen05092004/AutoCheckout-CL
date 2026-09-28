"""Small file helpers shared by the tools: atomic JSON writes and checksums."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def load_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | os.PathLike, obj: Any, *, indent: int | None = None) -> None:
    """Write JSON atomically: a crash (or a Spot preemption) never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)
            if indent is not None:
                f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def md5_file(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
