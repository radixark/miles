import concurrent.futures
from argparse import Namespace
from collections.abc import Sequence
from concurrent.futures import Future
from typing import NamedTuple

import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.transports.mooncake import MooncakeTransport, RemoteShard
from miles.backends.training_utils.weight_update.protocols.utils.model_param_stager import ModelParamStager
from miles.backends.training_utils.weight_update.protocols.utils.model_replica import (
    ModelReplica,
    ModelReplicas,
    assert_replica_matches_shard,
    pack_into_buffers,
    query_rollout_engine_rank_configs,
)
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    assign_rollout_engine_ranks,
)
from miles.backends.training_utils.weight_update.protocols.utils.transfer_buffers import TransferBuffers
from miles.utils.distributed_utils import get_gloo_group

# one is loaded while the writes of the other are in flight
_NUM_TRANSFER_BUFFERS = 2


class _ReplicaTarget(NamedTuple):
    model_replica: ModelReplica
    remote_shards: list[RemoteShard]


class UpdateWeightP2P(WeightTransferProtocol):
    """Writes weight updates straight into the rollout engines' GPU memory over Mooncake.

    Each sender loads the HF tensors of the updater's buckets with a model replica of each rollout engine rank it
    sends to, into transfer buffers in the bytes that rank's loader would write, and writes those bytes to the
    rank's published addresses. The end of the base weights waits for every write and fails the update if any
    failed or is still running.
    """

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        if args.sglang_pp_size != 1:
            raise NotImplementedError("Rollout pipeline parallelism is not tested yet.")
        self.global_rank = dist.get_rank(group=get_gloo_group())
        self._model_param_stager = ModelParamStager()
        self._pending_writes: list[tuple[RemoteShard, Future]] = []
        self._model_replicas = ModelReplicas(model_path=args.hf_checkpoint)
        self._transport: MooncakeTransport | None = None
        self._transfer_buffers: TransferBuffers | None = None
        self._replica_targets: list[_ReplicaTarget] = []

    def after_base_weights(self) -> None:
        """Wait for every write of this update; fail the update if any write failed or is still running."""
        if not self.is_sender:
            return
        pending_writes, self._pending_writes = self._pending_writes, []
        _raise_if_any_write_failed(pending_writes, timeout=self.args.p2p_transfer_timeout)
        self._model_param_stager.assert_all_done()

    def send_bucket(self, converted_named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Loads the params this bucket completes into transfer buffers, a group at a time, and writes them to every
        rollout engine rank this sender serves."""
        if not self.is_sender or not converted_named_tensors:
            return
        ready_hf_tensors_by_param_name = self._model_param_stager.get_transfer_ready_params(
            converted_named_tensors,
            param_mapper=self._model_replicas.param_mapper,
            params_dict=self._model_replicas.transfer_buffer_param_layouts,
        )
        for param_names in pack_into_buffers(
            ready_hf_tensors_by_param_name,
            self._model_replicas.transfer_buffer_param_layouts,
            self._transfer_buffers.buffer_nbytes,
        ):
            hf_tensors = [hf_tensor for name in param_names for hf_tensor in ready_hf_tensors_by_param_name[name]]
            for target in self._replica_targets:
                buffer = self._transfer_buffers.acquire()
                param_bytes_by_name = target.model_replica.load_into(buffer, param_names, hf_tensors)
                writes = self._transport.write(target.remote_shards, param_bytes_by_name)
                self._transfer_buffers.release_after(buffer, writes)
                self._pending_writes += zip(target.remote_shards, writes, strict=True)

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
        against the weights its ranks publish. Replicas, the transport and the transfer buffers carry over from
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
                self._replica_targets.append(_ReplicaTarget(model_replica, remote_shards))
            if self._transfer_buffers is None:
                self._transfer_buffers = self._create_transfer_buffers()

    def _create_transfer_buffers(self) -> TransferBuffers:
        largest_param_nbytes = max(
            layout.occupied_nbytes for layout in self._model_replicas.transfer_buffer_param_layouts.values()
        )
        return TransferBuffers(
            _NUM_TRANSFER_BUFFERS,
            max(self.args.update_weight_buffer_size, largest_param_nbytes),
            device=self._transport.transfer_buffer_device,
            register_memory=self._transport.register_memory,
        )


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
