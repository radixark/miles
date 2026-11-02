"""Checkpoint paths that may be Multi-Storage Client (MSC) URLs.

Megatron reads and writes ``msc://`` checkpoints when ``--enable-msc`` is set. Miles inspects
``--ref-load`` before handing it to Megatron, so those checks go through these helpers instead
of ``pathlib`` / ``os.path``, which cannot see an ``msc://`` URL.
"""

from pathlib import Path
from typing import Any

MSC_PREFIX = "msc://"


def is_msc_path(path: str | Path) -> bool:
    return str(path).startswith(MSC_PREFIX)


def validate_msc_checkpoint_args(args) -> None:
    """Megatron arguments only. Reject ``msc://`` checkpoint paths that Megatron or Miles would not read through MSC."""
    if is_msc_path(args.load or ""):
        raise ValueError(
            f"--load {args.load}: resuming from an msc:// checkpoint is not supported yet, because the "
            "rollout state is restored from a local path; use --ref-load for an msc:// initial checkpoint."
        )
    if is_msc_path(args.ref_load or "") and not args.enable_msc:
        raise ValueError(
            f"--ref-load {args.ref_load} is an msc:// path, which Megatron reads only with --enable-msc; "
            "without it Megatron finds no checkpoint and starts from random weights."
        )


def enable_megatron_msc(args) -> None:
    """Turn Megatron's MSC checkpoint I/O on in this process when ``--enable-msc`` is set.

    Megatron enables it inside ``parse_args``, which runs only in the driver; trainer workers
    receive the parsed args and must enable it themselves.
    """
    if not args.enable_msc:
        return
    from megatron.core.msc_utils import MultiStorageClientFeature

    MultiStorageClientFeature.enable()
    if not MultiStorageClientFeature.is_enabled():
        raise RuntimeError("--enable-msc is set but multi-storage-client is not available in this trainer worker")


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
