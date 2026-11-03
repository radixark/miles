"""When saved checkpoints expire: the gateway-private ``checkpoint_root/checkpoint_expiry.json`` (tinker path ->
expires_at, unix seconds). Only the gateway, under its trainer lock, reads and rewrites it; a checkpoint never
carries its own expiry, so one that crashed between commit and this write never expires (it is never deleted by
mistake)."""

import json
import os

from miles.tinker.core.input_validation import validate_ttl_seconds
from miles.tinker.core.utils import parse_tinker_path

EXPIRY_FILENAME = "checkpoint_expiry.json"


class CheckpointExpiry:
    def __init__(self, checkpoint_root: str) -> None:
        self._path = os.path.join(checkpoint_root, EXPIRY_FILENAME)
        self._expires_at: dict[str, float] = {}
        if os.path.exists(self._path):
            for tinker_path, expires_at in json.loads(open(self._path).read()).items():
                # an entry that does not parse is an error, never a deletion
                parse_tinker_path(tinker_path)
                assert isinstance(expires_at, int | float), f"{self._path}: {tinker_path} expires at {expires_at!r}"
                self._expires_at[tinker_path] = float(expires_at)

    def expires_at(self, tinker_path: str) -> float | None:
        return self._expires_at.get(tinker_path)

    def set(self, tinker_path: str, ttl_seconds: int | None, now: float) -> None:
        """``ttl_seconds`` None: the checkpoint no longer expires."""
        validate_ttl_seconds(ttl_seconds)
        if ttl_seconds is None:
            if self._expires_at.pop(tinker_path, None) is None:
                return
        else:
            self._expires_at[tinker_path] = now + ttl_seconds
        self._write()

    def expired(self, now: float) -> list[str]:
        return sorted(path for path, expires_at in self._expires_at.items() if expires_at <= now)

    def drop(self, tinker_path: str) -> None:
        if self._expires_at.pop(tinker_path, None) is not None:
            self._write()

    def _write(self) -> None:
        temporary = f"{self._path}.tmp"
        with open(temporary, "w") as f:
            json.dump(self._expires_at, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, self._path)
        # the rename itself is durable only once the directory is
        directory = os.open(os.path.dirname(self._path), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
