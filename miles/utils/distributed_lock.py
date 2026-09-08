import logging
import time
from contextlib import AbstractContextManager, nullcontext
from types import TracebackType

import torch.distributed as dist
from torch.distributed import Store, TCPStore

from miles.utils.distributed_utils import get_gloo_group
from miles.utils.misc import get_current_node_ip

logger = logging.getLogger(__name__)


class StoreTicketLock:
    """FIFO mutual exclusion over a ``torch.distributed`` store."""

    def __init__(
        self,
        *,
        store: Store,
        prefix: str,
        poll_interval: float = 0.01,
        warn_interval: float = 60.0,
    ) -> None:
        self._store = store
        self._next_key = f"{prefix}/next"
        self._serving_key = f"{prefix}/serving"
        self._poll_interval = poll_interval
        self._warn_interval = warn_interval
        self._held_ticket: int | None = None

    def __enter__(self) -> "StoreTicketLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.release()

    def acquire(self) -> None:
        assert self._held_ticket is None, f"lock is already held with ticket {self._held_ticket}"

        ticket = self._store.add(self._next_key, 1) - 1
        last_warned = time.monotonic()
        while (serving := self._store.add(self._serving_key, 0)) != ticket:
            if time.monotonic() - last_warned >= self._warn_interval:
                logger.warning(f"Waiting for {self._serving_key}: holding ticket {ticket}, now serving {serving}")
                last_warned = time.monotonic()
            time.sleep(self._poll_interval)

        self._held_ticket = ticket

    def release(self) -> None:
        assert self._held_ticket is not None, f"lock {self._serving_key} is not held, cannot release"

        self._held_ticket = None
        self._store.add(self._serving_key, 1)


def create_world_ticket_lock(
    *,
    prefix: str,
    participates: bool = True,
    poll_interval: float = 0.01,
) -> AbstractContextManager:
    rank = dist.get_rank()
    store = TCPStore(host_name="0.0.0.0", port=0, is_master=True, wait_for_workers=False) if rank == 0 else None
    address = [f"{get_current_node_ip()}:{store.port}"] if store is not None else [None]
    dist.broadcast_object_list(address, src=0, group=get_gloo_group())

    if not participates:
        return nullcontext(store)

    if rank != 0:
        assert store is None, f"rank {rank} must not host the store"
        host, port = address[0].rsplit(":", 1)
        store = TCPStore(host_name=host, port=int(port), is_master=False)

    return StoreTicketLock(store=store, prefix=prefix, poll_interval=poll_interval)
