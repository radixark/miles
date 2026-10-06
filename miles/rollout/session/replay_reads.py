"""Background reads of SGLang output-store bundles behind session records.

A training chat response that opted into the output store carries
``choices[0].meta_info.output_store_ref``. The session server starts reading the
bundle as soon as the response arrives, without holding up the reply, and the
read removes the bundle, so no later path (a discarded response, a deleted
session, a rolled-back record) has an object left to clean up. ``/samples``
waits for its records' reads and then assembles from the arrays synchronously.
"""

import asyncio
import functools
import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor

from miles.rollout.generate_utils.output_store import (
    OUTPUT_STORE_REF_KEY,
    ReplayOutputs,
    output_store_enabled,
    read_replay_outputs,
)
from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.types import SessionRecord
from miles.utils.object_store import BaseObjectStore, MooncakeObjectStore

logger = logging.getLogger(__name__)

# Arbitrary until measured; bounds the reads in flight per session server.
_MAX_READ_WORKERS = 8


class ReplayReadError(RuntimeError):
    """A record's output-store bundle could not be read; an infrastructure failure, not an assembly error."""


class ReplayReader:
    def __init__(self, store: BaseObjectStore) -> None:
        self._store = store
        self._executor = ThreadPoolExecutor(max_workers=_MAX_READ_WORKERS, thread_name_prefix="output-store-read")

    def start(self, response: dict) -> "asyncio.Future[ReplayOutputs] | None":
        """Start reading the bundle a chat response references; ``None`` if it references none.

        Runs before ``extract_completion`` validates the response, so a malformed
        one is left for that check to reject.
        """
        meta_info = response.get("choices", [{}])[0].get("meta_info")
        output_store_ref = meta_info.get(OUTPUT_STORE_REF_KEY) if isinstance(meta_info, dict) else None
        if output_store_ref is None:
            return None
        read = asyncio.get_running_loop().run_in_executor(
            self._executor, functools.partial(read_replay_outputs, output_store_ref, store=self._store)
        )
        read.add_done_callback(_log_failed_read)
        return read

    def close(self) -> None:
        self._executor.shutdown(wait=False)


def create_replay_reader(config: SessionServerConfig) -> ReplayReader | None:
    if not output_store_enabled(config):
        return None
    store = MooncakeObjectStore(
        init_kwargs=config.mooncake_store_init_kwargs or {},
        replica_num=config.mooncake_replica_num,
        contribute_segment=False,
    )
    return ReplayReader(store)


async def wait_for_replay_reads(records_of: Callable[[], Iterable[SessionRecord]]) -> None:
    """Wait until no record has a read in flight, including records committed meanwhile.

    Raises ``ReplayReadError`` for the first failed read, so the caller never
    assembles from a record whose arrays are missing.
    """
    while pending := [
        record.replay_read for record in records_of() if record.replay_read and not record.replay_read.done()
    ]:
        await asyncio.wait(pending)
    for record in records_of():
        if record.replay_read is not None and (error := record.replay_read.exception()) is not None:
            raise ReplayReadError(f"reading the output-store bundle of a session record failed: {error!r}") from error


def _log_failed_read(read: asyncio.Future) -> None:
    # Also marks the exception retrieved for reads whose response was never recorded.
    if not read.cancelled() and (error := read.exception()) is not None:
        logger.error("Reading an output-store bundle for a session record failed", exc_info=error)
