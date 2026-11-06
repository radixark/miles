"""Write an adapter directory on the trainer, then commit it from the one process that owns the root.

``<name>`` is a relative symlink to a physical directory ``.<name>.<commit_id>/``; user names cannot start with
``.``, so internal entries never collide with them. Only the committer swaps links, recovers and deletes.
"""

import fcntl
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
from typing import TextIO


from miles.backends.training_utils.checkpoint.adapter.adapter_state import (
    broadcast_from_rank0,
    is_rank0,
    raise_if_any_rank_failed,
)

logger = logging.getLogger(__name__)

TENANT_FILENAME = "tenant.json"
_DELETING_SUFFIX = ".deleting"
_STAGING_SUFFIX = ".link"
_ADOPTING_SUFFIX = ".adopting"
_COMMIT_ID_NBYTES = 8
# what this library creates in a root: .<name>.<commit_id>, its retired form and a crashed commit's or adoption's
# staging link
_INTERNAL_ENTRY = re.compile(
    rf"^\.(?P<name>.+)\.[0-9a-f]{{{2 * _COMMIT_ID_NBYTES}}}(?P<suffix>\.deleting|\.link|\.adopting)?$"
)


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


def format_sglang_lora_name(model_id: str, name: str, commit_id: str) -> str:
    """sglang's adapter name; a re-save under the same name never hits the previous commit's cached weights."""
    return f"{model_id}@{name}@{commit_id}"


def _physical_dir(root: Path, name: str, commit_id: str) -> Path:
    return root / f".{name}.{commit_id}"


def _assert_user_name(name: str) -> None:
    assert not name.startswith("."), f"adapter name {name!r} must not start with '.'"
    assert name and "/" not in name, f"adapter name {name!r} must be one path component"


@contextmanager
def write_adapter_dir(root: Path, name: str, *, tenant_digest: str | None = None):
    """All ranks write into the yielded physical directory; the result is not visible until committed, and a failed
    write leaves nothing behind.

    ``tenant_digest``: the committer digests the tenant's key, so the key itself never reaches the writing ranks.
    """
    _assert_user_name(name)
    commit_id = broadcast_from_rank0(secrets.token_hex(_COMMIT_ID_NBYTES))
    path = _physical_dir(Path(root), name, commit_id)
    prepare_error = None
    if is_rank0():
        try:
            path.mkdir(parents=True)
            try:
                if tenant_digest is not None:
                    (path / TENANT_FILENAME).write_text(json.dumps({"tenant_digest": tenant_digest}))
            except Exception:
                _rmtree(path)
                raise
        except Exception as exc:
            prepare_error = exc
    raise_if_any_rank_failed(prepare_error)
    error = None
    try:
        yield UncommittedAdapterDir(path, commit_id)
        if is_rank0():
            # the commit must follow a durable write; DCP syncs only its own files
            for entry in (*path.iterdir(), path):
                _fsync(entry)
    except Exception as exc:
        error = exc
    try:
        raise_if_any_rank_failed(error)
    except RuntimeError:
        if is_rank0():
            _rmtree(path)
        raise


def _link_target(link: Path) -> Path | None:
    if not link.is_symlink():
        return None
    return link.parent / os.readlink(link)


def commit_adapter_dir(
    root: Path, name: str, written: UncommittedAdapterDir, *, overwrite: bool
) -> CommittedAdapterDir:
    """Atomically point ``<name>`` at ``written``; the replaced commit is renamed ``.deleting``.

    The committer removes it later with ``delete_retired_adapter_dirs``, outside any lock it holds.
    """
    root, link = Path(root), Path(root) / name
    assert written.path == _physical_dir(root, name, written.commit_id), f"{written.path} was not written for {name!r}"
    if not written.path.is_dir():
        # a storage condition, not an invariant: another node's write may not be visible yet over NFS
        raise FileNotFoundError(f"{written.path} does not exist")
    delete_unreferenced_adapter_dirs(root, name, keep=written.path.name)
    if _link_target(link) == written.path:
        # a retry after the link swap; the commit already stands
        return CommittedAdapterDir(written.path, written.commit_id)
    staging_link = root / f".{name}.{written.commit_id}{_STAGING_SUFFIX}"
    try:
        if os.path.lexists(link) and not link.is_symlink() and not link.is_dir():
            # a file the library did not write: overwrite replaces commits, never a user's data
            raise FileExistsError(f"{link} is a file, not an adapter")
        if not overwrite and os.path.lexists(link):
            raise FileExistsError(f"adapter {name!r} already exists")
        if link.is_dir() and not link.is_symlink():
            _adopt_pre_commit_dir(root, name)
        previous = _link_target(link)
        if previous is not None and not _is_library_dir(root, name, previous):
            raise _foreign_link_error(link, previous)
        staging_link.symlink_to(written.path.name)
        os.replace(staging_link, link)
    except BaseException:
        # nothing points at the written dir before the swap, so a failed commit would leave it on disk
        if os.path.lexists(staging_link):
            staging_link.unlink()
        _rmtree(written.path)
        raise
    _fsync(root)
    if previous is not None and previous.exists():
        _retire(previous)
    return CommittedAdapterDir(written.path, written.commit_id)


def _adopt_pre_commit_dir(root: Path, name: str) -> None:
    """Give a checkpoint saved before commits (a plain ``<name>/`` dir) the committed layout, so an overwrite
    replaces and retires it like any commit."""
    physical = _physical_dir(root, name, secrets.token_hex(_COMMIT_ID_NBYTES))
    # linked before the rename, so recovery finishes an adoption that stops between the rename and the swap
    staging_link = root / f"{physical.name}{_ADOPTING_SUFFIX}"
    staging_link.symlink_to(physical.name)
    os.rename(root / name, physical)
    os.replace(staging_link, root / name)
    _fsync(root)


def resolve_adapter_dir(root: Path, name: str, *, tenant: str | None) -> CommittedAdapterDir | None:
    """The committed directory behind ``name``; ``tenant`` must own it. RL, which has no tenants, passes None."""
    _assert_user_name(name)
    target = _link_target(Path(root) / name)
    if target is not None and not _is_library_dir(Path(root), name, target):
        raise _foreign_link_error(Path(root) / name, target)
    if target is None or not target.is_dir():
        return None
    if tenant is not None:
        tenant_file = target / TENANT_FILENAME
        # an RL adapter has no tenant and belongs to no Tinker tenant
        owner = json.loads(tenant_file.read_text())["tenant_digest"] if tenant_file.is_file() else None
        if owner != tenant_digest(tenant):
            raise OwnershipError(f"adapter {name!r} does not belong to this tenant")
    return CommittedAdapterDir(target, target.name.rsplit(".", 1)[1])


def _is_library_dir(root: Path, name: str, path: Path) -> bool:
    internal = _INTERNAL_ENTRY.match(path.name)
    return path.parent == root and internal is not None and internal["name"] == name and internal["suffix"] is None


def _foreign_link_error(link: Path, target: Path) -> ValueError:
    # e.g. a user's symlink to a shared adapter; replacing it must never retire the shared dir
    return ValueError(f"{link} points to {target}, which this library did not write; refusing to replace or delete it")


def delete_adapter_dir(root: Path, name: str, *, tenant: str | None) -> None:
    committed = resolve_adapter_dir(root, name, tenant=tenant)
    if committed is None:
        raise FileNotFoundError(f"adapter {name!r} does not exist")
    os.unlink(Path(root) / name)
    _retire(committed.path)


def delete_unreferenced_adapter_dirs(root: Path, name: str | None = None, *, keep: str | None = None) -> None:
    """Remove physical directories no link in ``root`` points to, retired directories and dangling links to them.

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
        if internal is not None and internal["suffix"] == _ADOPTING_SUFFIX:
            if name is None or internal["name"] == name:
                link = root / internal["name"]
                if _link_target(entry).exists() and not os.path.lexists(link):
                    # the pre-commit dir was renamed and is reachable only through this link
                    os.replace(entry, link)
                    referenced.add(_link_target(link).name)
                else:
                    entry.unlink()
        elif internal is not None and internal["suffix"] == _STAGING_SUFFIX:
            # a staging link never made it to its name; it references nothing
            if name is None or internal["name"] == name:
                entry.unlink()
        elif _link_target(entry).exists():
            referenced.add(_link_target(entry).name)
        elif (name is None or entry.name == name) and _is_library_dir(root, entry.name, _link_target(entry)):
            # a dangling foreign link (e.g. to an unmounted share) is the user's, like a live one
            entry.unlink()
    for entry in root.iterdir():
        internal = _INTERNAL_ENTRY.match(entry.name)
        if internal is None or internal["suffix"] == _STAGING_SUFFIX or entry.is_symlink() or entry.name == keep:
            continue
        if internal["suffix"] == _DELETING_SUFFIX:
            # a commit leaves them to delete_retired_adapter_dirs, which may be removing them right now
            if name is None:
                _rmtree(entry)
        elif entry.name not in referenced and (name is None or internal["name"] == name):
            _rmtree(entry)


def _retire(path: Path) -> None:
    try:
        os.replace(path, path.with_name(path.name + _DELETING_SUFFIX))
    except OSError:
        # the link already moved; recovery removes the unreferenced dir, so the commit stands
        logger.exception(f"failed to retire {path}")


def lock_checkpoint_root(root: Path) -> TextIO:
    """Hold an exclusive lock on ``root`` for the caller's lifetime: one committer per root."""
    Path(root).mkdir(parents=True, exist_ok=True)
    handle = open(Path(root) / ".lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"another process already commits under {root}") from error
    except OSError as error:
        handle.close()
        raise RuntimeError(f"cannot lock {root}: one committer per root needs a filesystem with flock") from error
    return handle


def delete_retired_adapter_dirs(root: Path) -> None:
    """Remove the dirs commits and deletes retired (``*.deleting``); safe without the committer's lock."""
    root = Path(root)
    if not root.is_dir():
        return
    for entry in root.iterdir():
        internal = _INTERNAL_ENTRY.match(entry.name)
        if internal is not None and internal["suffix"] == _DELETING_SUFFIX and not entry.is_symlink():
            _rmtree(entry)


def _rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError:
        # left for the next recovery pass; a failed cleanup must not fail the commit that triggered it
        logger.exception(f"failed to remove {path}")


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
