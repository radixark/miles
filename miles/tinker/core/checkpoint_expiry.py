"""When saved checkpoints expire: the gateway-private ``checkpoint_root/checkpoint_expiry.json``.

An expiry belongs to one commit of a checkpoint name. An entry for a commit that is no longer live is stale and
never deletes the live one, so a crash between a commit and its expiry write leaves the new commit not expiring."""

import json
import os
from pathlib import Path
from typing import NamedTuple

from miles.tinker.core.input_validation import validate_ttl_seconds
from miles.tinker.core.utils import parse_tinker_path

EXPIRY_FILENAME = "checkpoint_expiry.json"


class Expiry(NamedTuple):
    commit_id: str
    expires_at: float


class CheckpointExpiry:
    def __init__(self, checkpoint_root: str) -> None:
        self._path = os.path.join(checkpoint_root, EXPIRY_FILENAME)
        self._entries: dict[str, Expiry] = {}
        if os.path.exists(self._path):
            for tinker_path, entry in json.loads(Path(self._path).read_text()).items():
                # an entry that does not parse is an error, never a deletion
                parse_tinker_path(tinker_path)
                commit_id, expires_at = entry["commit_id"], entry["expires_at"]
                assert isinstance(commit_id, str), f"{self._path}: {tinker_path} has commit id {commit_id!r}"
                assert isinstance(expires_at, int | float), f"{self._path}: {tinker_path} expires at {expires_at!r}"
                self._entries[tinker_path] = Expiry(commit_id, float(expires_at))

    def entry(self, tinker_path: str) -> Expiry | None:
        return self._entries.get(tinker_path)

    def expires_at(self, tinker_path: str, commit_id: str) -> float | None:
        """When this commit expires; None when it does not, including under another commit's stale entry."""
        entry = self._entries.get(tinker_path)
        return entry.expires_at if entry is not None and entry.commit_id == commit_id else None

    def set(self, tinker_path: str, commit_id: str, ttl_seconds: int | None, now: float) -> None:
        """``ttl_seconds`` None: the checkpoint no longer expires."""
        validate_ttl_seconds(ttl_seconds)
        if ttl_seconds is None:
            self.drop(tinker_path)
        else:
            self._write(self._entries | {tinker_path: Expiry(commit_id, now + ttl_seconds)})

    def expired(self, now: float) -> list[tuple[str, Expiry]]:
        return sorted((path, entry) for path, entry in self._entries.items() if entry.expires_at <= now)

    def drop(self, tinker_path: str) -> None:
        if tinker_path in self._entries:
            self._write({path: entry for path, entry in self._entries.items() if path != tinker_path})

    def _write(self, entries: dict[str, Expiry]) -> None:
        temporary = f"{self._path}.tmp"
        with open(temporary, "w") as f:
            json.dump({path: entry._asdict() for path, entry in entries.items()}, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, self._path)
        # only now: a failed write leaves the expiries as the file holds them, so the sweep never acts on one it lost
        self._entries = entries
        # the rename itself is durable only once the directory is
        directory = os.open(os.path.dirname(self._path), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
