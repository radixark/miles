from collections.abc import Callable
from datetime import timedelta

import torch
import torch.distributed as dist

from miles.utils.test_utils.det_process_group import DetProcessGroup


class DeferredWork(dist.Work):
    def __init__(self, complete: Callable[[], None]) -> None:
        super().__init__()
        self.complete = complete
        self.wait_count = 0
        self.error: Exception | None = None
        self.success = True
        self.timeouts: list[timedelta | None] = []

    def wait(self, timeout: timedelta | None = None) -> bool:
        self.wait_count += 1
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        if self.success:
            self.complete()
        return self.success


class DeferredBackend:
    def __init__(self) -> None:
        self.sent: list[torch.Tensor] = []
        self.works: list[DeferredWork] = []

    def send(self, tensors: list[torch.Tensor], dst_rank: int, tag: int) -> DeferredWork:
        work = DeferredWork(lambda: self.sent.extend(tensor.clone() for tensor in tensors))
        self.works.append(work)
        return work

    def recv(self, tensors: list[torch.Tensor], src_rank: int, tag: int) -> DeferredWork:
        def complete() -> None:
            for tensor in tensors:
                tensor.fill_(src_rank)

        work = DeferredWork(complete)
        self.works.append(work)
        return work


class DeferredProcessGroup(DetProcessGroup):
    def __init__(self) -> None:
        dist.ProcessGroup.__init__(self, 0, 4)
        self._inner = DeferredBackend()
        self._coalesced_p2p_works = None
