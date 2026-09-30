import logging
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from typing import NamedTuple

import torch
import torch.distributed as dist
from sglang.srt import server_args as server_args_module
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import ParallelismContext, RankParallelismConfig
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.model_loader import get_model
from sglang.srt.model_loader.parameter_mapper import ParameterMapper
from sglang.srt.server_args import ServerArgs

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.utils import ModelParamStager
from miles.utils.distributed_utils import get_gloo_group

from .p2p_rollout_cell_updater import _P2PRolloutCellUpdater
from .p2p_transfer_utils import (
    P2PTransferManager,
    RemoteTransferPlan,
    RemoteWeightInfo,
    TransferTaskP2PMeta,
    create_transfer_engine,
    query_remote_weight_infos,
    register_cpu_memory,
)

logger = logging.getLogger(__name__)


# ============================== transfer protocol ==============================


class UpdateWeightP2P(WeightTransferProtocol):
    """P2P weight transfer over the updater's bucketed all-gather + HF conversion,
    and a single set of shared CPU pinned buffers for P2P writes.

    Compute transfer_ready_params once (same for all rollout engine ranks)
    For each rollout engine rank:
        load_weights(shared buffer) → P2P write
        where the last rank's write is submitted to a background thread
    wait_transfers() at finish to collect all background writes
    """

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.transfer_plan = RemoteTransferPlan(args)
        self.global_rank = dist.get_rank(group=get_gloo_group())
        self._model_registered = False
        self._model_param_stager = ModelParamStager()
        self._cell_updaters_of_rollout_engine_ind: dict[int, _P2PRolloutCellUpdater] = {}
        self.transfer_manager = P2PTransferManager(
            num_workers=getattr(args, "p2p_transfer_num_workers", 4),
            transfer_timeout=getattr(args, "p2p_transfer_timeout", 30.0),
        )

    def after_base_weights(self) -> None:
        """Wait for all background P2P writes to complete."""
        if not self.is_sender:
            return
        self.transfer_manager.wait_transfers()
        self._model_param_stager.assert_all_done()

    def begin_sync(
        self, weight_version: int, iter_buckets: Callable[..., Iterator[list[tuple[str, torch.Tensor]]]]
    ) -> bool:
        """Register shared CPU pinned memory with P2P on the first sync."""
        if self.is_sender and not self._model_registered:
            self._weight_memory_registry = register_cpu_memory(
                self._cpu_replicas.shared_params_dict, self._transfer_engine
            )
            self._model_registered = True
        return True

    def send_bucket(self, converted_named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Stage incoming tensors; when all shards for a param are collected,
        load into shared buffer and P2P-write per rollout engine rank.

        Only calls load_weights() with complete accumulated tensors, preventing
        partial writes that would corrupt the shared buffer when different rollout engine
        ranks have different EP expert-to-local mappings.
        """
        if not self.is_sender or not converted_named_tensors:
            return
        # `ready_hf_tensors`` here are the complete tensors ready to be transferred.
        transfer_ready_params, ready_hf_tensors = self._model_param_stager.get_transfer_ready_params(
            converted_named_tensors,
            param_mapper=self._cpu_replicas.shared_param_mapper,
            params_dict=self._cpu_replicas.shared_params_dict,
        )

        if transfer_ready_params and ready_hf_tensors:
            last_idx = len(self._rollout_engine_rank_infos) - 1
            for i, meta in enumerate(self._rollout_engine_rank_infos):
                meta.model_replica.load_weights(ready_hf_tensors)

                # Last rollout engine rank: fire-and-forget all sessions to background,
                # as the weight will no longer be overwritten
                for cell_updater in meta.target_cell_updaters:
                    cell_updater.submit_write(
                        rollout_engine_rank=meta.rollout_engine_rank,
                        names=transfer_ready_params,
                        weight_memory_registry=self._weight_memory_registry,
                        transfer_engine=self._transfer_engine,
                        transfer_manager=self.transfer_manager,
                    )

                if i != last_idx:
                    # Non-last rollout engine rank needs to be fully written to target before next update can happen.
                    for cell_updater in meta.target_cell_updaters:
                        cell_updater.wait_for_pending_writes()

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
        """``connect`` here will:

        - Create a transfer plan that maps each training rank to its target
          rollout rank(s) based on GPU counts and parallelism configuration.
        - Query remote rollout engines for their weight memory registration
          info (addresses and sizes for RDMA writes).
        - Query remote parallelism config and construct a local CPU model
          replica that mirrors the target's sharding layout, enabling correct
          weight format conversion before transfer.
        """
        self.rollout_engines = rollout_engines
        self._cell_updaters_of_rollout_engine_ind = {
            rollout_engine_ind: _P2PRolloutCellUpdater(rollout_engine_ind=rollout_engine_ind)
            for rollout_engine_ind in range(len(rollout_engines))
        }

        self.is_sender = self.transfer_plan._gathered_dp_rank < self.transfer_plan._rollout_num_gpus

        if self.is_sender:
            self.group_name = f"miles-p2p_{self.transfer_plan._gathered_dp_rank}"
            targets = self.transfer_plan.plan_p2p()
            (
                self.remote_weight_infos_by_session_id,
                targets_to_session_id,
                self.session_id_to_server_args,
            ) = query_remote_weight_infos(rollout_engines, targets)

            targets_grouped_by_rollout_engine_rank: dict[int, list] = {}
            for target in targets:
                targets_grouped_by_rollout_engine_rank.setdefault(target.rollout_engine_rank, []).append(target)

            # Create ONE transfer engine for all rollout engine ranks
            self._transfer_engine = create_transfer_engine()
            self._cpu_replicas = _CPUReplicasManager(model_path=self.args.hf_checkpoint)
            # in self._rollout_engine_rank_infos: tuple of
            # - single CPU replica shared among all sessions
            # - related remote weight info
            self._rollout_engine_rank_infos: list[_RolloutEngineRankInfo] = []

            _assign_p2p_targets(
                self._cell_updaters_of_rollout_engine_ind,
                targets=targets,
                targets_to_session_id=targets_to_session_id,
                remote_weight_infos_by_session_id=self.remote_weight_infos_by_session_id,
            )

            for rollout_engine_rank, rank_targets in targets_grouped_by_rollout_engine_rank.items():
                first_target = rank_targets[0]
                session_id = targets_to_session_id[(first_target.rollout_engine_ind, first_target.rollout_engine_rank)]
                parallelism_config = RankParallelismConfig.from_dict(
                    self.remote_weight_infos_by_session_id[session_id][1]
                )
                server_args = self.session_id_to_server_args[session_id]

                model_replica = self._cpu_replicas.create_replica(
                    parallelism_config=parallelism_config, server_args=server_args
                )

                rank_cell_updaters = [
                    self._cell_updaters_of_rollout_engine_ind[target.rollout_engine_ind] for target in rank_targets
                ]

                self._rollout_engine_rank_infos.append(
                    _RolloutEngineRankInfo(
                        rollout_engine_rank=rollout_engine_rank,
                        model_replica=model_replica,
                        target_cell_updaters=rank_cell_updaters,
                    )
                )


def _assign_p2p_targets(
    cell_updaters: dict[int, _P2PRolloutCellUpdater],
    targets: list[TransferTaskP2PMeta],
    targets_to_session_id: dict[tuple[int, int], str],
    remote_weight_infos_by_session_id: dict[str, tuple],
) -> None:
    for target in targets:
        session_id = targets_to_session_id[(target.rollout_engine_ind, target.rollout_engine_rank)]
        cell_targets = cell_updaters[target.rollout_engine_ind].targets_by_rollout_engine_rank
        assert target.rollout_engine_rank not in cell_targets
        cell_targets[target.rollout_engine_rank] = RemoteWeightInfo(
            session_id, remote_weight_infos_by_session_id[session_id][0]
        )


class _RolloutEngineRankInfo(NamedTuple):
    rollout_engine_rank: int
    model_replica: torch.nn.Module
    target_cell_updaters: list[_P2PRolloutCellUpdater]


# ================================= cpu replica =================================


class _CPUReplicasManager:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self.replicas: list[torch.nn.Module] = []
        self.shared_params_dict: dict[str, torch.Tensor] = {}
        self.shared_param_mapper: ParameterMapper | None = None

    def create_replica(self, parallelism_config: RankParallelismConfig, server_args: ServerArgs) -> torch.nn.Module:
        first_rollout_engine_rank = not self.replicas
        model_replica = _create_cpu_replica(
            parallelism_config,
            self._model_path,
            server_args,
            shared_params_dict=self.shared_params_dict,
            first_rollout_engine_rank=first_rollout_engine_rank,
        )
        if first_rollout_engine_rank:
            self.shared_params_dict = dict(model_replica.named_parameters())
            self.shared_param_mapper = ParameterMapper.from_model(model_replica)
        self.replicas.append(model_replica)
        return model_replica


def _create_cpu_replica(
    parallelism_config: RankParallelismConfig,
    model_path: str,
    server_args: ServerArgs,
    shared_params_dict: dict[str, torch.Tensor],
    first_rollout_engine_rank: bool = False,
) -> torch.nn.Module:
    """Create a CPU model replica that loads the right shard and skips post_load_weights."""
    load_config = LoadConfig(
        load_format="dummy",
        model_loader_extra_config=None,
        rl_quant_profile=server_args.rl_quant_profile,
    )
    server_args_module.set_global_server_args_for_scheduler(server_args)
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()

    # Monkey-patch the loader-level post_load_weights to no-op BEFORE get_model,
    # because get_model() calls post_load_weights() internally (loader.py:1310)
    # which may invoke CUDA-only kernels (e.g., per_tensor_quant_fp8 for FP8 models).
    # This is safe because the rollout engine runs post_load_weights on its own GPU
    # after RDMA transfer, at end_weight_update.
    from sglang.srt.model_loader import loader as model_loader_module

    original_post_load_weights = model_loader_module.post_load_weights
    model_loader_module.post_load_weights = lambda *args, **kwargs: None
    try:
        with ParallelismContext(parallelism_config):
            model = get_model(
                model_config=ModelConfig(model_path),
                load_config=load_config,
                device_config=DeviceConfig(device="cpu"),
            )
    finally:
        model_loader_module.post_load_weights = original_post_load_weights

    # Also patch the instance method for subsequent load_weights() calls
    # (deepseek_weight_loader.py:342 calls self.post_load_weights() at the end).
    if hasattr(model, "post_load_weights"):
        model.post_load_weights = lambda *args, **kwargs: None

    if first_rollout_engine_rank:
        for param in model.parameters():
            param.data = param.data.pin_memory()
    else:
        for name, param in model.named_parameters():
            assert name in shared_params_dict, f"[P2P-Shared] Parameter {name} not found in shared buffers"
            param.data = shared_params_dict[name]

    return model
