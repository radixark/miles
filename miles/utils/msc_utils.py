"""Checkpoint paths that may be Multi-Storage Client (MSC) URLs.

Megatron reads and writes ``msc://`` checkpoints when ``--enable-msc`` is set. Miles inspects
``--load`` and ``--ref-load`` before handing them to Megatron, so those checks go through these
helpers instead of ``pathlib`` / ``os.path``, which cannot see an ``msc://`` URL.
"""

from pathlib import Path
from typing import Any

MSC_PREFIX = "msc://"


def is_msc_path(path: str | Path) -> bool:
    return str(path).startswith(MSC_PREFIX)


def storage_path(path: str | Path) -> Any:
    """A ``pathlib.Path``, or a ``multistorageclient.Path`` with the same interface for an ``msc://`` URL."""
    if is_msc_path(path):
        import multistorageclient as msc

        return msc.Path(str(path))
    return Path(path)


def storage_reader(path: str | Path) -> Any:
    """A torch.distributed.checkpoint reader for a local or ``msc://`` checkpoint directory."""
    if is_msc_path(path):
        import multistorageclient as msc

        return msc.torch.MultiStorageFileSystemReader(str(path))
    from torch.distributed.checkpoint import FileSystemReader

    return FileSystemReader(path)
