from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import (
    RemoteShard,
    assert_tensors_fit,
)


class RolloutEngineWriteThreads:
    """One write thread per rollout engine, so a stuck engine holds up only its own writes."""

    def __init__(self, thread_name_prefix: str) -> None:
        self._thread_name_prefix = thread_name_prefix
        self._executors_by_rollout_engine_ind: dict[int, ThreadPoolExecutor] = {}

    def restart(self) -> None:
        """Called at connect: a rollout engine index now names a new engine, which must not queue behind the writes
        of the one it replaced."""
        for executor in self._executors_by_rollout_engine_ind.values():
            executor.shutdown(wait=False)
        self._executors_by_rollout_engine_ind = {}

    def write_each_shard(
        self,
        remote_shards: Sequence[RemoteShard],
        tensors_by_name: dict[str, torch.Tensor],
        write_shard: Callable[[RemoteShard, dict[str, torch.Tensor]], None],
    ) -> list[Future]:
        """Runs `write_shard` for each shard on its rollout engine's thread; returns one future per shard. Raises
        before writing anything if a shard lacks one of the weights or holds it in a different number of bytes."""
        for remote_shard in remote_shards:
            assert_tensors_fit(remote_shard, tensors_by_name)
        return [
            self._get_or_create_executor(remote_shard.rollout_engine_ind).submit(
                write_shard, remote_shard, tensors_by_name
            )
            for remote_shard in remote_shards
        ]

    def _get_or_create_executor(self, rollout_engine_ind: int) -> ThreadPoolExecutor:
        if rollout_engine_ind not in self._executors_by_rollout_engine_ind:
            self._executors_by_rollout_engine_ind[rollout_engine_ind] = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"{self._thread_name_prefix}-rollout-engine-{rollout_engine_ind}"
            )
        return self._executors_by_rollout_engine_ind[rollout_engine_ind]
