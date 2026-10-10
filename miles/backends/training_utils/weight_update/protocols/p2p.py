import concurrent.futures
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future
from typing import NamedTuple

import torch
import torch.distributed as dist
from sglang.srt.distributed.parallel_state import ParallelismContext, RankParallelismConfig

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.transports.mooncake import MooncakeTransport, RemoteShard
from miles.backends.training_utils.weight_update.protocols.utils.model_param_stager import ModelParamStager
from miles.backends.training_utils.weight_update.protocols.utils.model_replica import (
    ModelReplicas,
    assert_replica_matches_shard,
    query_rollout_engine_rank_configs,
)
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    assign_rollout_engine_ranks,
)
from miles.utils.distributed_utils import get_gloo_group


class _ReplicaTarget(NamedTuple):
    model_replica: torch.nn.Module
    remote_shards: list[RemoteShard]
    parallelism_config: RankParallelismConfig


class UpdateWeightP2P(WeightTransferProtocol):
    """P2P weight transfer over the updater's bucketed all-gather + HF conversion,
    and a single set of shared CPU pinned buffers for P2P writes.

    Compute transfer_ready_params once (same for all engine ranks)
    For each engine rank:
        wait for the previous rank's writes, load_weights(shared buffer) → P2P write
    after_base_weights waits for every write and fails the update if any failed
    """

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        if args.sglang_pp_size != 1:
            raise NotImplementedError("Rollout pipeline parallelism is not tested yet.")
        self.global_rank = dist.get_rank(group=get_gloo_group())
        self._model_registered = False
        self._model_param_stager = ModelParamStager()
        self._pending_writes: list[tuple[RemoteShard, Future]] = []
        self._model_replicas = ModelReplicas(model_path=args.hf_checkpoint)
        self._transport: MooncakeTransport | None = None
        self._replica_targets: list[_ReplicaTarget] = []

    def after_base_weights(self) -> None:
        """Wait for every write of this update; fail the update if any write failed or is still running."""
        if not self.is_sender:
            return
        pending_writes, self._pending_writes = self._pending_writes, []
        _raise_if_any_write_failed(pending_writes, timeout=self.args.p2p_transfer_timeout)
        self._model_param_stager.assert_all_done()

    def begin_sync(
        self, weight_version: int, iter_buckets: Callable[..., Iterator[list[tuple[str, torch.Tensor]]]]
    ) -> bool:
        """Register shared CPU pinned memory with P2P on the first sync."""
        if self.is_sender and not self._model_registered:
            for tensor in self._model_replicas.shared_params_dict.values():
                self._transport.register_memory(tensor)
            self._model_registered = True
        return True

    def send_bucket(self, converted_named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Stage incoming tensors; when all shards for a param are collected,
        load into shared buffer and P2P-write per engine rank.

        Only calls load_weights() with complete accumulated tensors, preventing
        partial writes that would corrupt the shared buffer when different engine
        ranks have different EP expert-to-local mappings.
        """
        if not self.is_sender or not converted_named_tensors:
            return
        # `ready_hf_tensors`` here are the complete tensors ready to be transferred.
        transfer_ready_params, ready_hf_tensors = self._model_param_stager.get_transfer_ready_params(
            converted_named_tensors,
            param_mapper=self._model_replicas.param_mapper,
            params_dict=self._model_replicas.shared_params_dict,
        )

        if transfer_ready_params and ready_hf_tensors:
            tensors_by_name = {name: self._model_replicas.shared_params_dict[name] for name in transfer_ready_params}
            previous_rank_writes: list[Future] = []
            for target in self._replica_targets:
                # loading overwrites the shared buffer the previous rank's writes read from
                concurrent.futures.wait(previous_rank_writes)
                with ParallelismContext(target.parallelism_config):
                    target.model_replica.load_weights(ready_hf_tensors)

                previous_rank_writes = self._transport.write(target.remote_shards, tensors_by_name)
                self._pending_writes += zip(target.remote_shards, previous_rank_writes, strict=True)

        converted_named_tensors.clear()

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        """Connects this trainer rank to the rollout engines handed over: assigns it rollout engine ranks over them
        and the iterator's placement, queries their configs and Mooncake shards, and checks each model replica
        against the weights its ranks publish. Replicas, the transport and the registered buffer carry over from
        earlier connects."""
        self.rollout_engines = rollout_engines
        assignments = assign_rollout_engine_ranks(parallel_state, placement, engine_gpu_counts)
        self.is_sender = bool(assignments)

        if self.is_sender:
            configs_by_rollout_engine_rank = query_rollout_engine_rank_configs(rollout_engines, assignments)
            if self._transport is None:
                self._transport = MooncakeTransport()
            remote_shards_by_rollout_engine_rank = self._transport.connect(rollout_engines, assignments)
            self._model_param_stager = ModelParamStager()
            self._replica_targets = []
            for rollout_engine_rank, remote_shards in remote_shards_by_rollout_engine_rank.items():
                config = configs_by_rollout_engine_rank[rollout_engine_rank]
                model_replica = self._model_replicas.get_or_build(config)
                for remote_shard in remote_shards:
                    assert_replica_matches_shard(
                        model_replica,
                        remote_shard.published_nbytes_by_name,
                        published_by=f"rollout engine {remote_shard.rollout_engine_ind} rank {rollout_engine_rank}",
                    )
                self._replica_targets.append(_ReplicaTarget(model_replica, remote_shards, config.parallelism))


def _raise_if_any_write_failed(pending_writes: list[tuple[RemoteShard, Future]], timeout: float) -> None:
    _, unfinished_writes = concurrent.futures.wait([write for _, write in pending_writes], timeout=timeout)
    failures = []
    for remote_shard, write in pending_writes:
        if write in unfinished_writes:
            reason = f"still running after {timeout}s"
        elif write.exception() is not None:
            reason = repr(write.exception())
        else:
            continue
        failures.append(
            f"rollout engine {remote_shard.rollout_engine_ind} rank {remote_shard.rollout_engine_rank} "
            f"(session {remote_shard.session_id}): {reason}"
        )
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(pending_writes)} p2p writes failed:\n" + "\n".join(failures))
