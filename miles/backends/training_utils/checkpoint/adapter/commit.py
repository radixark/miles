"""Write an adapter directory on the trainer, then commit it from the one process that owns the root.

``<name>`` is a relative symlink to a physical directory ``.<name>.<commit_id>/``; user names cannot start with
``.``, so internal entries never collide with them. Only the committer swaps links, recovers and deletes.
"""

import hashlib
import json
import logging
import os
import secrets
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch.distributed as dist

from miles.backends.training_utils.checkpoint.adapter.adapter_state import gloo_group

logger = logging.getLogger(__name__)

TENANT_FILENAME = "tenant.json"
_DELETING_SUFFIX = ".deleting"
_COMMIT_ID_NBYTES = 8


class OwnershipError(PermissionError):
    pass


@dataclass(frozen=True)
class UncommittedAdapterDir:
    path: Path
    commit_id: str


@dataclass(frozen=True)
class CommittedAdapterDir:
    path: Path
    commit_id: str


def tenant_digest(tenant: str) -> str:
    return hashlib.sha256(tenant.encode()).hexdigest()


def _physical_dir(root: Path, name: str, commit_id: str) -> Path:
    return root / f".{name}.{commit_id}"


def _broadcast_from_rank0(value):
    if not dist.is_initialized():
        return value
    box = [value]
    dist.broadcast_object_list(box, src=0, group=gloo_group())
    return box[0]


def raise_if_any_rank_failed(error: BaseException | None) -> None:
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        # waits for every writer; a rank-local failure must not leave the others in a collective
        dist.all_gather_object(
            errors, None if error is None else f"{type(error).__name__}: {error}", group=gloo_group()
        )
    failed = [(rank, e) for rank, e in enumerate(errors) if e is not None]
    if failed:
        raise RuntimeError("adapter write failed: " + "; ".join(f"rank {rank}: {e}" for rank, e in failed))


@contextmanager
def write_adapter_dir(root: Path, name: str, *, tenant: str | None = None):
    """All ranks write into the yielded physical directory; the result is not visible until committed."""
    assert not name.startswith("."), f"adapter name {name!r} must not start with '.'"
    commit_id = _broadcast_from_rank0(secrets.token_hex(_COMMIT_ID_NBYTES))
    path = _physical_dir(Path(root), name, commit_id)
    prepare_error = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            path.mkdir(parents=True)
            if tenant is not None:
                (path / TENANT_FILENAME).write_text(json.dumps({"tenant_digest": tenant_digest(tenant)}))
        except Exception as exc:
            prepare_error = exc
    raise_if_any_rank_failed(prepare_error)
    error = None
    try:
        yield UncommittedAdapterDir(path, commit_id)
    except Exception as exc:
        error = exc
    raise_if_any_rank_failed(error)


def _link_target(link: Path) -> Path | None:
    if not link.is_symlink():
        return None
    return link.parent / os.readlink(link)


def commit_adapter_dir(
    root: Path, name: str, written: UncommittedAdapterDir, *, overwrite: bool
) -> CommittedAdapterDir:
    """Atomically point ``<name>`` at ``written``; the replaced commit is removed afterwards."""
    root, link = Path(root), Path(root) / name
    delete_unreferenced_adapter_dirs(root, name, keep=written.path.name)
    previous = _link_target(link)
    assert not link.exists() or link.is_symlink(), f"{link} exists and is not an adapter link"
    if previous is not None and not overwrite:
        raise FileExistsError(f"adapter {name!r} already exists")
    staging_link = root / f".{name}.{written.commit_id}.link"
    staging_link.symlink_to(written.path.name)
    os.replace(staging_link, link)
    _fsync_dir(root)
    if previous is not None and previous.exists():
        _retire(previous)
    return CommittedAdapterDir(written.path, written.commit_id)


def resolve_adapter_dir(root: Path, name: str, *, tenant: str | None = None) -> CommittedAdapterDir | None:
    """The committed directory behind ``name``; ``tenant`` must own it when given."""
    target = _link_target(Path(root) / name)
    if target is None or not target.is_dir():
        return None
    if tenant is not None:
        owner = json.loads((target / TENANT_FILENAME).read_text())["tenant_digest"]
        if owner != tenant_digest(tenant):
            raise OwnershipError(f"adapter {name!r} does not belong to this tenant")
    return CommittedAdapterDir(target, target.name.rsplit(".", 1)[1])


def delete_adapter_dir(root: Path, name: str, *, tenant: str | None = None) -> None:
    committed = resolve_adapter_dir(root, name, tenant=tenant)
    assert committed is not None, f"adapter {name!r} does not exist"
    os.unlink(Path(root) / name)
    _retire(committed.path)


def delete_unreferenced_adapter_dirs(root: Path, name: str | None = None, *, keep: str | None = None) -> None:
    """Remove physical directories no link in ``root`` points to, retired directories and dangling links.

    Referenced-ness is decided by the links, not by name prefixes, so recovering ``foo`` never touches ``foo.bar``.
    """
    root = Path(root)
    if not root.is_dir():
        return
    referenced = set()
    for entry in root.iterdir():
        if entry.is_symlink():
            target = _link_target(entry)
            if target.exists():
                referenced.add(target.name)
            elif name is None or entry.name == name:
                entry.unlink()
    for entry in root.iterdir():
        if not entry.name.startswith(".") or entry.is_symlink() or not entry.is_dir():
            continue
        if entry.name == keep:
            continue
        if entry.name.endswith(_DELETING_SUFFIX) or entry.name not in referenced:
            if name is None or _physical_name(entry.name) == name:
                _rmtree(entry)


def _physical_name(dirname: str) -> str:
    stem = dirname.removesuffix(_DELETING_SUFFIX)
    return stem[1:].rsplit(".", 1)[0]


def _retire(path: Path) -> None:
    retired = path.with_name(path.name + _DELETING_SUFFIX)
    os.replace(path, retired)
    _rmtree(retired)


def _rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError:
        # left for the next recovery pass; a failed cleanup must not fail the commit that triggered it
        logger.exception(f"failed to remove {path}")


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
