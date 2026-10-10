import logging
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
    query_rollout_engine_rank_configs,
)
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    assign_rollout_engine_ranks,
)
from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)


class _ReplicaTarget(NamedTuple):
    model_replica: torch.nn.Module
    remote_shards: list[RemoteShard]
    parallelism_config: RankParallelismConfig


class UpdateWeightP2P(WeightTransferProtocol):
    """P2P weight transfer over the updater's bucketed all-gather + HF conversion,
    and a single set of shared CPU pinned buffers for P2P writes.

    Compute transfer_ready_params once (same for all engine ranks)
    For each engine rank:
        load_weights(shared buffer) → P2P write
        where the last rank's write runs in the background
    after_base_weights waits for the background writes
    """

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        if args.sglang_pp_size != 1:
            raise NotImplementedError("Rollout pipeline parallelism is not tested yet.")
        self.global_rank = dist.get_rank(group=get_gloo_group())
        self._model_registered = False
        self._model_param_stager = ModelParamStager()
        self._last_rank_writes: list[Future] = []

    def after_base_weights(self) -> None:
        """Wait for all background P2P writes to complete."""
        if not self.is_sender:
            return
        for write in self._last_rank_writes:
            try:
                write.result(timeout=self.args.p2p_transfer_timeout)
            except Exception as e:
                logger.error(f"[P2P] Transfer future failed: {e}")
        self._last_rank_writes = []
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
            last_idx = len(self._replica_targets) - 1
            for i, target in enumerate(self._replica_targets):
                with ParallelismContext(target.parallelism_config):
                    target.model_replica.load_weights(ready_hf_tensors)

                writes = self._transport.write(target.remote_shards, tensors_by_name)
                if i == last_idx:
                    # Last engine rank: its writes run in the background, as the weight will no longer be overwritten
                    self._last_rank_writes += writes
                else:
                    # Non-last engine rank needs to be fully written to target before next update can happen.
                    for write in writes:
                        write.result()

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
        and the iterator's placement, queries their configs and Mooncake shards, and builds a model replica per
        shard layout."""
        self.rollout_engines = rollout_engines
        assignments = assign_rollout_engine_ranks(parallel_state, placement, engine_gpu_counts)
        self.is_sender = bool(assignments)

        if self.is_sender:
            configs_by_rollout_engine_rank = query_rollout_engine_rank_configs(rollout_engines, assignments)
            self._transport = MooncakeTransport(num_write_workers=self.args.p2p_transfer_num_workers)
            remote_shards_by_rollout_engine_rank = self._transport.connect(rollout_engines, assignments)
            self._model_replicas = ModelReplicas(model_path=self.args.hf_checkpoint)
            self._replica_targets: list[_ReplicaTarget] = []
            for rollout_engine_rank, remote_shards in remote_shards_by_rollout_engine_rank.items():
                config = configs_by_rollout_engine_rank[rollout_engine_rank]
                model_replica = self._model_replicas.get_or_build(config)
                self._replica_targets.append(_ReplicaTarget(model_replica, remote_shards, config.parallelism))
