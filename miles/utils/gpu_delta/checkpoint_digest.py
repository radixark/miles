"""Content identity of the immutable safetensors base shared by delta senders and receivers."""

import errno
import fcntl
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile


def _fingerprint(paths):
    # ctime also catches an in-place rewrite with a restored mtime, inode catches replacement.
    identities = []
    for path in paths:
        stat = path.stat()
        identities.append([path.name, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
    return identities


def _cached(record, fingerprint):
    try:
        entry = json.loads(record.read_bytes())
    except (FileNotFoundError, ValueError):
        return None
    if not isinstance(entry, dict) or entry.get("files") != fingerprint:
        return None
    digest = entry.get("sha256")
    return digest if isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest) else None


def _shard_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _get_or_create(record, paths):
    fingerprint = _fingerprint(paths)
    digest = _cached(record, fingerprint)
    if digest is not None:
        return digest
    # The lock inode survives atomic replacement of the cache record. Only one rank reads the model.
    with record.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fingerprint = _fingerprint(paths)
        digest = _cached(record, fingerprint)
        if digest is None:
            with ThreadPoolExecutor(max_workers=min(4, len(paths))) as pool:
                encoded = json.dumps(list(zip((p.name for p in paths), pool.map(_shard_hash, paths), strict=True)))
            if _fingerprint(paths) != fingerprint:
                raise ValueError("GPU-delta checkpoint changed while hashing its base shards")
            digest = hashlib.sha256(encoded.encode()).hexdigest()
            with NamedTemporaryFile(mode="w", dir=record.parent, prefix=record.name + ".", delete=False) as pending:
                try:
                    json.dump({"files": fingerprint, "sha256": digest}, pending)
                    pending.flush()
                    os.replace(pending.name, record)
                finally:
                    Path(pending.name).unlink(missing_ok=True)
        return digest


def checkpoint_content_digest(checkpoint, files):
    """Hash selected shard bytes once, keeping the record beside the shards or in a local cache for read-only mounts."""
    checkpoint = Path(checkpoint)
    paths = sorted(Path(file) for file in files)
    if not paths:
        raise ValueError("GPU-delta checkpoint has no base shards")
    try:
        return _get_or_create(checkpoint / ".miles-gpu-delta.sha256", paths)
    except OSError as error:
        if error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
            raise
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "miles" / "gpu-delta"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(os.fsencode(checkpoint.resolve())).hexdigest()
    return _get_or_create(cache / f"{key}.sha256", paths)
