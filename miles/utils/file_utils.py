from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, Literal

import torch


def atomic_write_text(path: str | Path, text: str) -> None:
    _atomic_write(Path(path), mode="w", write=lambda file: file.write(text))


def atomic_torch_save(path: str | Path, obj: Any) -> None:
    _atomic_write(Path(path), mode="wb", write=lambda file: torch.save(obj, file))


def _atomic_write(path: Path, *, mode: Literal["w", "wb"], write: Callable[[IO[Any]], Any]) -> None:
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, mode) as file:
            write(file)
            # mkstemp opens 0600, which leaves a file one uid wrote unreadable to the next one
            os.fchmod(file.fileno(), 0o644)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
