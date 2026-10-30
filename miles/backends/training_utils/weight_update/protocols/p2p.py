import logging
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update import checksum_utils
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.p2p_rollout_cell_updater import _P2PRolloutCellUpdater
from miles.backends.training_utils.weight_update.protocols.transports.mooncake import MooncakeTransport
from miles.backends.training_utils.weight_update.protocols.transports.nixl import NixlTransport
from miles.backends.training_utils.weight_update.protocols.transports.p2p_transport import P2PTransport
from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import RemoteShard
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
from miles.backends.training_utils.weight_update.protocols.utils.transfer_buffers import PendingWrite, TransferBuffers
from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)

# one is loaded while the writes of the other are in flight
_NUM_TRANSFER_BUFFERS = 2


class _ReplicaTarget(NamedTuple):
    model_replica: ModelReplica
    remote_shards: list[RemoteShard]


@dataclass
class _RunnerSender:
    """Per-connection write targets and per-sync staging for one runner role, backed by cached replicas."""

    model_replicas: ModelReplicas
    replica_targets: list[_ReplicaTarget]
    model_param_stager: ModelParamStager | None = None

    def loads_hf_tensor(self, hf_name: str) -> bool:
        return hf_name in self.model_replicas.hf_name_mapping.param_names_by_hf_name


# ============================== transfer protocol ==============================


class UpdateWeightP2P(WeightTransferProtocol):
    """Write target and supported MTP draft weights to published GPU addresses over Mooncake or NIXL."""

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        if args.sglang.common_value("pp_size") != 1:
            raise NotImplementedError("Rollout pipeline parallelism is not tested yet.")
        self.global_rank = dist.get_rank(group=get_gloo_group())
        self._hf_tensor_specs: dict[str, HfTensorSpec] | None = None
        self._model_replicas_by_runner_role: dict[str, ModelReplicas] = {}
        self._transport: P2PTransport | None = None
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
                self._log_hf_names_no_runner_loads()
                self._transfer_buffers = self._create_transfer_buffers(buffer_nbytes)
            elif self._transfer_buffers.buffer_nbytes < buffer_nbytes:
                self._transfer_buffers.release()
                self._transfer_buffers = self._create_transfer_buffers(buffer_nbytes)
        return True

    def after_base_weights(self) -> None:
        """Wait for healthy cells and isolate cells with failed or timed-out writes."""
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
        cell_updaters_of_cell_id = self.cell_updaters_of_cell_id
        engine_cell_ids = self._engine_cell_ids

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
                remote_shards = [
                    shard
                    for shard in target.remote_shards
                    if not self.cell_updaters_of_cell_id[self._engine_cell_ids[shard.rollout_engine_ind]].is_errored
                ]
                if not remote_shards:
                    continue
                param_bytes_by_name = target.model_replica.load_into(buffer, param_names, hf_tensors)
                sent_checksums = (
                    checksum_utils.compute_send_checksums(param_bytes_by_name, list(param_bytes_by_name))
                    if self.args.check_weight_transfer_checksum
                    else None
                )
                writes = self._transport.write(
                    remote_shards,
                    param_bytes_by_name,
                    run_write=lambda remote_shard, write, sent_checksums=sent_checksums: cell_updaters_of_cell_id[
                        engine_cell_ids[remote_shard.rollout_engine_ind]
                    ].run_write(write, remote_shard=remote_shard, sent_checksums=sent_checksums),
                )
                pending_writes = [
                    PendingWrite(
                        future=write,
                        cell_updater=self.cell_updaters_of_cell_id[self._engine_cell_ids[shard.rollout_engine_ind]],
                    )
                    for shard, write in zip(remote_shards, writes, strict=True)
                ]
                self._transfer_buffers.release_after(buffer, pending_writes)

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        engine_cell_ids: Sequence[str],
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        """Assign and validate write targets, reusing replicas, mappings, transport and buffers across reconnects."""
        assert len(engine_cell_ids) == len(rollout_engines), (
            f"[P2P-Shared] {len(engine_cell_ids)} cell ids for {len(rollout_engines)} rollout engines; "
            f"the per-engine metadata must describe the same engines"
        )

        self.rollout_engines = rollout_engines
        self._engine_cell_ids = tuple(engine_cell_ids)
        self.cell_updaters_of_cell_id = {
            cell_id: _P2PRolloutCellUpdater(args=self.args, cell_id=cell_id, api_client=api_client)
            for api_client, cell_id in zip(rollout_engines, engine_cell_ids, strict=True)
        }
        self._runner_senders = []
        assignments = assign_rollout_engine_ranks(parallel_state, placement, engine_gpu_counts)
        self.is_sender = bool(assignments)

        if self.is_sender:
            runner_roles = query_runner_roles(self.cell_updaters_of_cell_id, engine_cell_ids, assignments, selector)
            configs_by_runner_role = {
                runner_role: query_rollout_engine_rank_configs(
                    self.cell_updaters_of_cell_id, engine_cell_ids, assignments, runner_role
                )
                for runner_role in runner_roles
            }
            if not any(configs_by_runner_role.values()):
                self.is_sender = False
                return
            if self._transport is None:
                self._transport = P2PTransport(
                    NixlTransport(write_timeout=self.args.p2p_transfer_timeout)
                    if self.args.p2p_transport == NixlTransport.name
                    else MooncakeTransport()
                )
            remote_shards_by_runner_role = self._transport.connect(
                self.cell_updaters_of_cell_id, engine_cell_ids, assignments, runner_roles
            )
            for runner_role in runner_roles:
                if (
                    runner_sender := self._connect_runner(
                        runner_role=runner_role,
                        configs_by_rollout_engine_rank=configs_by_runner_role[runner_role],
                        remote_shards_by_rollout_engine_rank=remote_shards_by_runner_role[runner_role],
                        engine_cell_ids=engine_cell_ids,
                    )
                ) is not None:
                    self._runner_senders.append(runner_sender)
            self.is_sender = bool(self._runner_senders)

    def _connect_runner(
        self,
        runner_role: str,
        configs_by_rollout_engine_rank: dict[int, RolloutEngineRankConfig],
        remote_shards_by_rollout_engine_rank: dict[int, list[RemoteShard]],
        engine_cell_ids: Sequence[str],
    ) -> _RunnerSender | None:
        if runner_role not in self._model_replicas_by_runner_role:
            self._model_replicas_by_runner_role[runner_role] = ModelReplicas(
                model_path=self.args.hf_checkpoint, transfer_buffer_device=self._transport.transfer_buffer_device
            )
        model_replicas = self._model_replicas_by_runner_role[runner_role]
        replica_targets = []
        for rollout_engine_rank, all_remote_shards in remote_shards_by_rollout_engine_rank.items():
            remote_shards = [
                shard
                for shard in all_remote_shards
                if not self.cell_updaters_of_cell_id[engine_cell_ids[shard.rollout_engine_ind]].is_errored
            ]
            if not remote_shards:
                continue
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
        return _RunnerSender(model_replicas, replica_targets) if replica_targets else None

    def _log_hf_names_no_runner_loads(self) -> None:
        hf_names_no_runner_loads = sorted(
            hf_name
            for hf_name in self._hf_tensor_specs
            if not any(runner_sender.loads_hf_tensor(hf_name) for runner_sender in self._runner_senders)
        )
        if hf_names_no_runner_loads:
            logger.info(
                f"p2p skips {len(hf_names_no_runner_loads)} HF tensors the rollout engines' loaders ignore, e.g. "
                f"{hf_names_no_runner_loads[:5]}"
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
