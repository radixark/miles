import logging
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.hf_weight_iterator.checkpoint_towers import is_tower_key
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.transports.mooncake import MooncakeTransport, RemoteShard
from miles.backends.training_utils.weight_update.protocols.utils.loader_probe import HfTensorSpec
from miles.backends.training_utils.weight_update.protocols.utils.model_param_stager import ModelParamStager
from miles.backends.training_utils.weight_update.protocols.utils.model_replica import (
    ModelReplica,
    ModelReplicas,
    RolloutEngineRankConfig,
    assert_replica_matches_shard,
    compute_param_group_nbytes,
    pack_into_buffers,
    query_rollout_engine_rank_configs,
    query_runner_roles,
)
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    assign_rollout_engine_ranks,
)
from miles.backends.training_utils.weight_update.protocols.utils.transfer_buffers import TransferBuffers
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.hf_utils.config import load_hf_config

logger = logging.getLogger(__name__)

# one is loaded while the writes of the other are in flight
_NUM_TRANSFER_BUFFERS = 2


class _ReplicaTarget(NamedTuple):
    model_replica: ModelReplica
    remote_shards: list[RemoteShard]


@dataclass
class _RunnerSender:
    """Per-connection write targets and per-sync staging for one runner role, backed by cached replicas."""

    runner_role: str
    model_replicas: ModelReplicas
    replica_targets: list[_ReplicaTarget]
    model_param_stager: ModelParamStager | None = None

    def loads_hf_tensor(self, hf_name: str) -> bool:
        return hf_name in self.model_replicas.hf_name_mapping.param_names_by_hf_name


class UpdateWeightP2P(WeightTransferProtocol):
    """Write target and supported MTP draft weights to published GPU addresses over Mooncake."""

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        if args.sglang_pp_size != 1:
            raise NotImplementedError("Rollout pipeline parallelism is not tested yet.")
        self.global_rank = dist.get_rank(group=get_gloo_group())
        self._hf_tensor_specs: dict[str, HfTensorSpec] | None = None
        self._model_replicas_by_runner_role: dict[str, ModelReplicas] = {}
        self._transport: MooncakeTransport | None = None
        self._transfer_buffers: TransferBuffers | None = None
        self._runner_senders: list[_RunnerSender] = []

    def begin_sync(
        self, weight_version: int, iter_buckets: Callable[..., Iterator[list[tuple[str, torch.Tensor]]]]
    ) -> bool:
        """Map new replicas, reset staging and grow the transfer buffers if a param group no longer fits; all ranks
        must join the first iterator pass's collectives."""
        if self._hf_tensor_specs is None:
            self._hf_tensor_specs = {
                hf_name: HfTensorSpec(tuple(tensor.shape), tensor.dtype)
                for bucket in iter_buckets(materialize=True)
                for hf_name, tensor in bucket
            }
        if self.is_sender:
            for runner_sender in self._runner_senders:
                runner_sender.model_replicas.map_hf_names(self._hf_tensor_specs)
                runner_sender.model_param_stager = ModelParamStager(runner_sender.model_replicas.hf_name_mapping)
            buffer_nbytes = self._compute_transfer_buffer_nbytes()
            if self._transfer_buffers is None:
                self._transfer_buffers = self._create_transfer_buffers(buffer_nbytes)
            elif self._transfer_buffers.buffer_nbytes < buffer_nbytes:
                self._transfer_buffers.release()
                self._transfer_buffers = self._create_transfer_buffers(buffer_nbytes)
        if self._hf_tensors_unchecked:
            self._assert_every_hf_tensor_is_loaded()
            self._hf_tensors_unchecked = False
        return True

    def after_base_weights(self) -> None:
        """Wait for every write of this update; fail the update if any write failed or is still running."""
        if not self.is_sender:
            return
        self._transfer_buffers.wait_for_all_writes()
        for runner_sender in self._runner_senders:
            runner_sender.model_param_stager.assert_all_done()

    def send_bucket(self, converted_named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Write complete parameter groups to every runner whose loader takes them, as the engine's own updates do."""
        if not self.is_sender or not converted_named_tensors:
            return
        unknown_hf_names = [hf_name for hf_name, _ in converted_named_tensors if hf_name not in self._hf_tensor_specs]
        assert not unknown_hf_names, (
            f"{unknown_hf_names[:5]} ({len(unknown_hf_names)} in all) were not in the trainer's first pass, so no "
            "replica was mapped for them"
        )
        for runner_sender in self._runner_senders:
            self._send_runner_bucket(
                runner_sender,
                [
                    (hf_name, tensor)
                    for hf_name, tensor in converted_named_tensors
                    if runner_sender.loads_hf_tensor(hf_name)
                ],
            )
        converted_named_tensors.clear()

    def _send_runner_bucket(
        self, runner_sender: _RunnerSender, converted_named_tensors: list[tuple[str, torch.Tensor]]
    ) -> None:
        ready_hf_tensors_by_param_group = runner_sender.model_param_stager.stage(converted_named_tensors)
        for param_groups in pack_into_buffers(
            ready_hf_tensors_by_param_group,
            runner_sender.model_replicas.transfer_buffer_param_layouts,
            self._transfer_buffers.buffer_nbytes,
        ):
            param_names = [param_name for param_group in param_groups for param_name in param_group]
            hf_tensors = [
                hf_tensor for param_group in param_groups for hf_tensor in ready_hf_tensors_by_param_group[param_group]
            ]
            for target in runner_sender.replica_targets:
                buffer = self._transfer_buffers.acquire()
                param_bytes_by_name = target.model_replica.load_into(buffer, param_names, hf_tensors)
                writes = self._transport.write(target.remote_shards, param_bytes_by_name)
                self._transfer_buffers.release_after(buffer, list(zip(target.remote_shards, writes, strict=True)))

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        """Assign and validate write targets, reusing replicas, mappings, transport and buffers across reconnects."""
        self.rollout_engines = rollout_engines
        assignments = assign_rollout_engine_ranks(parallel_state, placement, engine_gpu_counts)
        self.is_sender = bool(assignments)
        # ranks of one PP stage send the same HF tensors; with PP gathered, every rank sends all of them
        self._pp_stage, self._pp_stages = (
            (0, 1) if placement.gather_pp else (parallel_state.pp.rank, parallel_state.pp.size)
        )
        self._hf_tensors_unchecked = True

        if self.is_sender:
            runner_roles = query_runner_roles(rollout_engines, assignments, selector)
            configs_by_runner_role = {
                runner_role: query_rollout_engine_rank_configs(rollout_engines, assignments, runner_role)
                for runner_role in runner_roles
            }
            if self._transport is None:
                self._transport = MooncakeTransport()
            remote_shards_by_runner_role = self._transport.connect(rollout_engines, assignments, runner_roles)
            self._runner_senders = [
                self._connect_runner(
                    runner_role, configs_by_runner_role[runner_role], remote_shards_by_runner_role[runner_role]
                )
                for runner_role in runner_roles
            ]

    def _connect_runner(
        self,
        runner_role: str,
        configs_by_rollout_engine_rank: dict[int, RolloutEngineRankConfig],
        remote_shards_by_rollout_engine_rank: dict[int, list[RemoteShard]],
    ) -> _RunnerSender:
        if runner_role not in self._model_replicas_by_runner_role:
            self._model_replicas_by_runner_role[runner_role] = ModelReplicas(
                model_path=self.args.hf_checkpoint, transfer_buffer_device=self._transport.transfer_buffer_device
            )
        model_replicas = self._model_replicas_by_runner_role[runner_role]
        replica_targets = []
        for rollout_engine_rank, remote_shards in remote_shards_by_rollout_engine_rank.items():
            model_replica = model_replicas.get_or_build(configs_by_rollout_engine_rank[rollout_engine_rank])
            for remote_shard in remote_shards:
                assert_replica_matches_shard(
                    model_replica,
                    remote_shard.published_nbytes_by_name,
                    published_by=(
                        f"the {runner_role} of rollout engine {remote_shard.rollout_engine_ind} "
                        f"rank {rollout_engine_rank}"
                    ),
                )
            replica_targets.append(_ReplicaTarget(model_replica, remote_shards))
        return _RunnerSender(runner_role, model_replicas, replica_targets)

    def _assert_every_hf_tensor_is_loaded(self) -> None:
        """Fail the first sync after a connect if no rollout engine rank loads an HF tensor the trainer sends: the
        weight it was meant for would stay stale. Each sender loads only its own engine ranks (EP splits the experts),
        so all ranks OR their views of their PP stage's tensors. Collective."""
        hf_names = sorted(self._hf_tensor_specs)
        max_hf_names = torch.tensor(len(hf_names))
        dist.all_reduce(max_hf_names, op=dist.ReduceOp.MAX, group=get_gloo_group())
        loaded = torch.zeros(self._pp_stages, int(max_hf_names), dtype=torch.uint8)
        if self.is_sender:
            loaded[self._pp_stage, : len(hf_names)] = torch.tensor(
                [any(sender.loads_hf_tensor(name) for sender in self._runner_senders) for name in hf_names]
            )
        dist.all_reduce(loaded, op=dist.ReduceOp.MAX, group=get_gloo_group())
        if not self.is_sender:
            return

        # towers are resent unchanged from the checkpoint; a language-only engine has none
        unloaded = [
            name
            for name, is_loaded in zip(hf_names, loaded[self._pp_stage, : len(hf_names)].tolist(), strict=True)
            if not is_loaded and not is_tower_key(name)
        ]
        # tied engines take the head from the embedding and ignore the trainer's copy
        if (
            any(name.endswith("lm_head.weight") for name in unloaded)
            and load_hf_config(self.args.hf_checkpoint).tie_word_embeddings
        ):
            unloaded = [name for name in unloaded if not name.endswith("lm_head.weight")]
        if (
            unloaded
            and self.args.mtp_num_layers
            and all(sender.runner_role != "draft" for sender in self._runner_senders)
        ):
            logger.info(
                f"p2p skips {len(unloaded)} HF tensors no rollout engine loads, e.g. {unloaded[:5]}: without a draft "
                "runner the trainer's MTP layers have no loader"
            )
            return
        assert not unloaded, (
            f"no rollout engine loads {len(unloaded)} HF tensors the trainer sends, e.g. {unloaded[:5]}, so the "
            "weights they were meant for would stay stale; the trainer's export and the rollout engines' loaders "
            "disagree on these names or formats"
        )

    def _compute_transfer_buffer_nbytes(self) -> int:
        largest_param_group_nbytes = max(
            compute_param_group_nbytes(param_group, runner_sender.model_replicas.transfer_buffer_param_layouts)
            for runner_sender in self._runner_senders
            for param_group in runner_sender.model_param_stager.param_groups
        )
        return max(self.args.update_weight_buffer_size, largest_param_group_nbytes)

    def _create_transfer_buffers(self, buffer_nbytes: int) -> TransferBuffers:
        return TransferBuffers(
            _NUM_TRANSFER_BUFFERS,
            buffer_nbytes,
            device=self._transport.transfer_buffer_device,
            register_memory=self._transport.register_memory,
            unregister_memory=self._transport.unregister_memory,
            write_timeout=self.args.p2p_transfer_timeout,
        )
