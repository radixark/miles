"""Write an adapter directory on the trainer, then commit it from the one process that owns the root.

``<name>`` is a relative symlink to a physical directory ``.<name>.<commit_id>/``; user names cannot start with
``.``, so internal entries never collide with them. Only the committer swaps links, recovers and deletes.
"""

import hashlib
import json
import logging
import os
import re
import secrets
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch.distributed as dist

from miles.backends.training_utils.checkpoint.adapter.adapter_state import gloo_group, raise_if_any_rank_failed

logger = logging.getLogger(__name__)

TENANT_FILENAME = "tenant.json"
_DELETING_SUFFIX = ".deleting"
_STAGING_SUFFIX = ".link"
_COMMIT_ID_NBYTES = 8
# what this library creates in a root: .<name>.<commit_id>, its retired form and a crashed commit's staging link
_INTERNAL_ENTRY = re.compile(rf"^\.(?P<name>.+)\.[0-9a-f]{{{2 * _COMMIT_ID_NBYTES}}}(?P<suffix>\.deleting|\.link)?$")


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


def _assert_user_name(name: str) -> None:
    assert not name.startswith("."), f"adapter name {name!r} must not start with '.'"
    assert name and "/" not in name, f"adapter name {name!r} must be one path component"


@contextmanager
def write_adapter_dir(root: Path, name: str, *, tenant: str | None = None):
    """All ranks write into the yielded physical directory; the result is not visible until committed."""
    _assert_user_name(name)
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
    assert written.path == _physical_dir(root, name, written.commit_id), f"{written.path} was not written for {name!r}"
    delete_unreferenced_adapter_dirs(root, name, keep=written.path.name)
    previous = _link_target(link)
    assert not link.exists() or link.is_symlink(), f"{link} exists and is not an adapter link"
    if previous is not None and not overwrite:
        raise FileExistsError(f"adapter {name!r} already exists")
    staging_link = root / f".{name}.{written.commit_id}{_STAGING_SUFFIX}"
    staging_link.symlink_to(written.path.name)
    os.replace(staging_link, link)
    _fsync_dir(root)
    if previous is not None and previous.exists():
        _retire(previous)
    return CommittedAdapterDir(written.path, written.commit_id)


def resolve_adapter_dir(root: Path, name: str, *, tenant: str | None) -> CommittedAdapterDir | None:
    """The committed directory behind ``name``; ``tenant`` must own it. RL, which has no tenants, passes None."""
    _assert_user_name(name)
    target = _link_target(Path(root) / name)
    if target is None or not target.is_dir():
        return None
    if tenant is not None:
        tenant_file = target / TENANT_FILENAME
        # an RL adapter has no tenant and belongs to no Tinker tenant
        owner = json.loads(tenant_file.read_text())["tenant_digest"] if tenant_file.is_file() else None
        if owner != tenant_digest(tenant):
            raise OwnershipError(f"adapter {name!r} does not belong to this tenant")
    return CommittedAdapterDir(target, target.name.rsplit(".", 1)[1])


def delete_adapter_dir(root: Path, name: str, *, tenant: str | None) -> None:
    committed = resolve_adapter_dir(root, name, tenant=tenant)
    if committed is None:
        raise FileNotFoundError(f"adapter {name!r} does not exist")
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
        if not entry.is_symlink():
            continue
        internal = _INTERNAL_ENTRY.match(entry.name)
        if internal is not None and internal["suffix"] == _STAGING_SUFFIX:
            # a staging link never made it to its name; it references nothing
            if name is None or internal["name"] == name:
                entry.unlink()
        elif _link_target(entry).exists():
            referenced.add(_link_target(entry).name)
        elif name is None or entry.name == name:
            entry.unlink()
    for entry in root.iterdir():
        internal = _INTERNAL_ENTRY.match(entry.name)
        if internal is None or internal["suffix"] == _STAGING_SUFFIX or entry.is_symlink() or entry.name == keep:
            continue
        if internal["suffix"] == _DELETING_SUFFIX or entry.name not in referenced:
            if name is None or internal["name"] == name:
                _rmtree(entry)


def _retire(path: Path) -> None:
    retired = path.with_name(path.name + _DELETING_SUFFIX)
    try:
        os.replace(path, retired)
    except OSError:
        # the link already moved; recovery removes the unreferenced dir, so the commit stands
        logger.exception(f"failed to retire {path}")
        return
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
