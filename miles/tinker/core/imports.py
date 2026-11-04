"""file:// imports: a tenant's adapter files, copied from beneath the import root into a gateway-owned dir that the
trainer then reads, so the tenant cannot change what it loads after the checks."""

import errno
import os
import shutil
import stat
import uuid
from collections.abc import Callable
from pathlib import Path

from miles.tinker.core.types import UserInputError

# gateway-owned copies of file:// imports, under the checkpoint root so every trainer rank can read them
IMPORT_COPIES_DIRNAME = ".imports"
_COPY_CHUNK_BYTES = 1 << 24


def _open_dir_beneath(root: str, relative: str) -> int:
    """fd of the directory ``root/relative``, each component opened without following a symlink."""
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    for part in [] if relative == "." else relative.split(os.sep):
        try:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
        except OSError as error:
            # a symlinked directory fails with ELOOP on Linux, ENOTDIR on macOS
            if error.errno == errno.ENOTDIR and stat.S_ISLNK(os.lstat(part, dir_fd=fd).st_mode):
                error = OSError(errno.ELOOP, os.strerror(errno.ELOOP), part)
            raise error from None
        finally:
            os.close(fd)
        fd = child
    return fd


def _copy_adapter_files(
    source_fd: int,
    copy_dir: str,
    wanted: Callable[[str], bool],
    max_bytes: int,
    json_max_bytes: int,
    refused_entries: dict[str, str],
) -> None:
    """Copy the files a reader opens, ``max_bytes`` in total and ``json_max_bytes`` per JSON; anything else (README.md,
    .cache/, a pickled DCP or .pt) stays behind."""
    names = os.listdir(source_fd)
    # e.g. a checkpoint store mounted under the import root, which realpath checks cannot see
    for entry, why in refused_entries.items():
        if entry in names:
            raise UserInputError(f"the import {why} ({entry})")
    budget = max_bytes
    for name in names:
        if not wanted(name):
            continue
        # O_NONBLOCK: opening a FIFO would otherwise wait for a writer
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=source_fd)
        try:
            # checked on the raw fd: wrapping a directory's fd raises without closing it
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise UserInputError(f"{name} is not a regular file")
            # sized before reading: a sparse file would otherwise fill the checkpoint disk every tenant shares
            if os.fstat(fd).st_size > budget:
                raise UserInputError(f"{name} would bring the import past {max_bytes} bytes, more than any adapter")
            # a JSON is parsed whole by the gateway and every trainer rank
            if name.endswith(".json") and os.fstat(fd).st_size > json_max_bytes:
                raise UserInputError(f"{name} is larger than {json_max_bytes} bytes, more than any adapter's")
            # a file growing while it is copied is still held to its cap
            limit = min(budget, json_max_bytes) if name.endswith(".json") else budget
            source = open(fd, "rb", closefd=False)
            with source, open(os.path.join(copy_dir, name), "wb") as copy:
                while chunk := source.read(min(_COPY_CHUNK_BYTES, limit + 1)):
                    if len(chunk) > limit:
                        raise UserInputError(f"{name} grew past {limit} bytes while it was copied")
                    copy.write(chunk)
                    budget -= len(chunk)
                    limit -= len(chunk)
        finally:
            os.close(fd)


def copy_import(
    uri: str,
    import_root: str,
    checkpoint_root: str,
    wanted: Callable[[str], bool],
    *,
    max_bytes: int,
    json_max_bytes: int,
    refused_entries: dict[str, str],
) -> str:
    """Copy a file:// import's adapter files, opened beneath the import root without following symlinks, into a
    gateway-owned dir under the checkpoint root; returns that dir."""
    root = os.path.normpath(import_root)
    copy_dir = os.path.join(checkpoint_root, IMPORT_COPIES_DIRNAME, uuid.uuid4().hex)
    os.makedirs(copy_dir)
    try:
        path = os.path.normpath(uri.removeprefix("file://"))
        if not path.startswith(root + os.sep):
            raise UserInputError(f"{uri} is outside the import root {import_root}")
        source_fd = _open_dir_beneath(root, os.path.relpath(path, root))
        try:
            _copy_adapter_files(
                source_fd,
                copy_dir,
                wanted,
                max_bytes,
                json_max_bytes,
                refused_entries,
            )
        finally:
            os.close(source_fd)
    except BaseException as error:
        shutil.rmtree(copy_dir)
        if isinstance(error, OSError) and error.errno == errno.ELOOP:
            raise UserInputError(f"{uri} is or holds a symlink; import a copy of the adapter's files") from None
        if isinstance(error, OSError | ValueError):
            # ValueError: a NUL byte or a lone surrogate in the tenant's path
            raise UserInputError(f"cannot read {uri}: {error}") from None
        raise
    return copy_dir


def clear_import_copies(checkpoint_root: str) -> None:
    """Remove the copies a previous gateway left behind (a crash between a copy and its load)."""
    copies = Path(checkpoint_root) / IMPORT_COPIES_DIRNAME
    if copies.exists():
        shutil.rmtree(copies)
