# doc-dev: docs/developer/multi-process-session-server.md
"""Process-stable session→owner mapping for the multi-process session server.

Decides which worker owns a ``session_id`` (see ``worker_index_for_session``
for the hash choice), so the router and every worker agree on each session's
owner with no coordination between them.

Stdlib only (``hashlib``), so a headless worker or router can import it
without pulling in FastAPI.
"""

from __future__ import annotations

import hashlib


def worker_index_for_session(session_id: str, n_worker: int) -> int:
    """Map *session_id* to a worker index in ``range(n_worker)``.

    Uses blake2b (not the builtin ``hash()``, which PYTHONHASHSEED salts per
    process) so the router and every worker derive the same owner.
    """
    if n_worker < 1:
        raise ValueError(f"n_worker must be >= 1, got {n_worker}")
    digest = hashlib.blake2b(session_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % n_worker
