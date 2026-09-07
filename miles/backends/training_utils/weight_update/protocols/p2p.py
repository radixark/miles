import json
import logging
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future
from typing import Any

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
from miles.backends.training_utils.weight_update.inference_cell_health import InferenceCellHealth
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.p2p_inference_cell_updater import (
    P2PInferenceCellUpdater,
    TransferEngineMeta,
)
from miles.backends.training_utils.weight_update.utils import ModelParamStager
from miles.utils.distributed_utils import get_gloo_group

from .p2p_transfer_utils import (
    P2PTransferManager,
    RemoteTransferPlan,
    RemoteWeightInfo,
    create_transfer_engine,
    query_remote_weight_infos,
    register_cpu_memory,
)

logger = logging.getLogger(__name__)

_PLACEMENT_PARALLELISM_FIELDS = frozenset({"global_rank", "local_rank"})


class UpdateWeightP2P(WeightTransferProtocol):
    """P2P weight transfer over the updater's bucketed all-gather + HF conversion,
    and a single set of shared CPU pinned buffers for P2P writes.

    Compute transfer_ready_params once (same for all engine ranks)
    For each engine rank:
        load_weights(shared buffer) → P2P write
        where the last rank's write is submitted to a background thread
    each inference cell collects its own writes at finish
    """

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.transfer_plan = RemoteTransferPlan(args)
        self.global_rank = dist.get_rank(group=get_gloo_group())
        self._model_registered = False
        self._model_param_stager = ModelParamStager()
        self.transfer_manager = P2PTransferManager(
            num_workers=getattr(args, "p2p_transfer_num_workers", 4),
            transfer_timeout=getattr(args, "p2p_transfer_timeout", 30.0),
        )
        self._transfer_engine: Any | None = None
        self._shared_params_dict: dict[str, torch.Tensor] = {}
        self._shared_param_mapper: ParameterMapper | None = None
        self._weight_memory_registry: dict[str, tuple[int, int, int]] = {}
        self._replicas_by_representation: dict[str, torch.nn.Module] = {}
        self.inference_cell_health = InferenceCellHealth()
        self._cell_updaters_by_cell_id: dict[str, P2PInferenceCellUpdater] = {}
        self._unfinished_writes: list[Future] = []
        self.remote_weight_infos_by_session_id: dict[str, tuple] = {}
        self.session_id_to_server_args: dict[str, ServerArgs] = {}
        # in self._transfer_engine_meta_list: tuple of
        # - single CPU replica shared among all sessions
        # - related remote weight info
        self._transfer_engine_meta_list: list[TransferEngineMeta] = []

    def after_base_weights(self) -> None:
        """Wait for all background P2P writes to complete."""
        if not self.is_sender:
            return
        for cell_updater in self._cell_updaters_by_cell_id.values():
            cell_updater.wait_for_pending_writes()
        self._model_param_stager.assert_all_done()

    def begin_sync(
        self, weight_version: int, iter_buckets: Callable[..., Iterator[list[tuple[str, torch.Tensor]]]]
    ) -> bool:
        """Register shared CPU pinned memory with P2P on the first sync."""
        if self.is_sender and not self._model_registered and self._shared_params_dict:
            self._weight_memory_registry = register_cpu_memory(self._shared_params_dict, self._transfer_engine)
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
            param_mapper=self._shared_param_mapper,
            params_dict=self._shared_params_dict,
        )

        if transfer_ready_params and ready_hf_tensors:
            last_idx = len(self._transfer_engine_meta_list) - 1
            for i, meta in enumerate(self._transfer_engine_meta_list):
                meta.model_replica.load_weights(ready_hf_tensors)

                # Last engine rank: fire-and-forget all sessions to background,
                # as the weight will no longer be overwritten
                submitted = []
                for cell_updater in meta.cell_updaters:
                    future = cell_updater.submit_write(
                        engine_rank=meta.engine_rank,
                        names=transfer_ready_params,
                        weight_memory_registry=self._weight_memory_registry,
                    )
                    if future is not None:
                        submitted.append((cell_updater, future))

                if i != last_idx:
                    # Non-last engine rank needs to be fully written to target before next update can happen.
                    for cell_updater, future in submitted:
                        cell_updater.wait_for_write(future)

        converted_named_tensors.clear()

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
        """``connect`` here will:

        - Create a transfer plan that maps each training rank to its target
          rollout rank(s) based on GPU counts and parallelism configuration.
        - Query remote rollout engines for their weight memory registration
          info (addresses and sizes for RDMA writes).
        - Query remote parallelism config and construct a local CPU model
          replica that mirrors the target's sharding layout, enabling correct
          weight format conversion before transfer.
        """
        assert engine_gpu_counts is not None, "[P2P-Shared] the per-engine GPU counts are required to plan transfers"
        assert len(engine_cell_ids) == len(rollout_engines) == len(engine_gpu_counts), (
            f"[P2P-Shared] {len(engine_cell_ids)} cell ids and {len(engine_gpu_counts)} GPU counts "
            f"for {len(rollout_engines)} rollout engines; the per-engine metadata must describe the same engines"
        )

        self.disconnect()
        self.rollout_engines = rollout_engines
        self.inference_cell_health = InferenceCellHealth(engine_cell_ids)

        planned_targets = self.transfer_plan.plan_p2p(engine_gpu_counts)

        if planned_targets and self._transfer_engine is None:
            # Create ONE transfer engine for all engine ranks
            self._transfer_engine = create_transfer_engine()

        targets_by_cell_id: dict[str, dict[int, RemoteWeightInfo]] = {cell_id: {} for cell_id in engine_cell_ids}
        targets_grouped_by_engine_rank: dict[int, list] = {}

        if planned_targets:
            self.group_name = f"miles-p2p_{self.transfer_plan._gathered_dp_rank}"
            query = query_remote_weight_infos(
                rollout_engines, planned_targets, request_timeout=self.args.update_weight_engine_request_timeout
            )
            self.remote_weight_infos_by_session_id = query.remote_weight_infos_by_session_id
            self.session_id_to_server_args = query.session_id_to_server_args
            targets_to_session_id = query.targets_to_session_id
            for engine_ind, error in query.failures_by_engine_ind.items():
                self.inference_cell_health.mark_errored(engine_cell_ids[engine_ind], error)

            for target in planned_targets:
                if self.inference_cell_health.is_errored(engine_cell_ids[target.engine_ind]):
                    continue
                targets_grouped_by_engine_rank.setdefault(target.engine_rank, []).append(target)
                session_id = targets_to_session_id[(target.engine_ind, target.engine_rank)]
                cell_targets = targets_by_cell_id[engine_cell_ids[target.engine_ind]]
                assert target.engine_rank not in cell_targets
                cell_targets[target.engine_rank] = RemoteWeightInfo(
                    session_id, self.remote_weight_infos_by_session_id[session_id][0]
                )

        self._cell_updaters_by_cell_id = {
            cell_id: P2PInferenceCellUpdater(
                cell_id=cell_id,
                transfer_engine=self._transfer_engine,
                transfer_manager=self.transfer_manager,
                health=self.inference_cell_health,
                targets_by_engine_rank=targets_by_cell_id[cell_id],
            )
            for cell_id in engine_cell_ids
        }

        for engine_rank, rank_targets in targets_grouped_by_engine_rank.items():
            rank_session_ids = [targets_to_session_id[(t.engine_ind, t.engine_rank)] for t in rank_targets]
            self._assert_one_weight_representation(engine_rank=engine_rank, session_ids=rank_session_ids)
            model_replica = self._ensure_cpu_replica(
                parallelism_info=self.remote_weight_infos_by_session_id[rank_session_ids[0]][1],
                server_args=self.session_id_to_server_args[rank_session_ids[0]],
            )

            rank_cell_updaters = [
                self._cell_updaters_by_cell_id[engine_cell_ids[target.engine_ind]] for target in rank_targets
            ]

            self._transfer_engine_meta_list.append(
                TransferEngineMeta(
                    engine_rank=engine_rank, model_replica=model_replica, cell_updaters=rank_cell_updaters
                )
            )

        self.is_sender = bool(self._transfer_engine_meta_list)

    def disconnect(self) -> None:
        self._drain_pending_writes()
        self._transfer_engine_meta_list = []
        self._cell_updaters_by_cell_id = {}
        self.inference_cell_health = InferenceCellHealth()
        self.remote_weight_infos_by_session_id = {}
        self.session_id_to_server_args = {}
        self.rollout_engines = []
        self.is_sender = False
        self._model_param_stager = ModelParamStager()

    def _drain_pending_writes(self) -> None:
        for cell_updater in self._cell_updaters_by_cell_id.values():
            cell_updater.wait_for_pending_writes()
            cell_updater.dispose()
            self._unfinished_writes += cell_updater.take_unfinished_writes()
        if self._unfinished_writes:
            logger.error(
                f"[P2P-Shared] {len(self._unfinished_writes)} p2p writes of this trainer rank never finished; "
                f"their source buffers stay registered for the lifetime of this actor"
            )

    def _assert_one_weight_representation(self, engine_rank: int, session_ids: list[str]) -> None:
        keys_by_session = {
            session_id: _weight_representation_key(
                self.remote_weight_infos_by_session_id[session_id][1], self.session_id_to_server_args[session_id]
            )
            for session_id in session_ids
        }
        assert len(set(keys_by_session.values())) == 1, (
            f"[P2P-Shared] The targets of engine rank {engine_rank} hold different weight representations and "
            f"cannot share one CPU replica: {keys_by_session}"
        )

    def _ensure_cpu_replica(self, parallelism_info: dict, server_args: ServerArgs) -> torch.nn.Module:
        representation_key = _weight_representation_key(parallelism_info, server_args)
        if (cached := self._replicas_by_representation.get(representation_key)) is not None:
            return cached

        first_engine_rank = not self._shared_params_dict
        model_replica = _create_cpu_replica(
            RankParallelismConfig.from_dict(parallelism_info),
            self.args.hf_checkpoint,
            server_args,
            shared_params_dict=self._shared_params_dict,
            first_engine_rank=first_engine_rank,
        )
        if first_engine_rank:
            self._shared_params_dict = dict(model_replica.named_parameters())
            self._shared_param_mapper = ParameterMapper.from_model(model_replica)
        self._replicas_by_representation[representation_key] = model_replica
        return model_replica


def _weight_representation_key(parallelism_info: dict, server_args: ServerArgs) -> str:
    sharding = {name: value for name, value in parallelism_info.items() if name not in _PLACEMENT_PARALLELISM_FIELDS}
    return json.dumps(
        {"sharding": sharding, "rl_quant_profile": server_args.rl_quant_profile}, sort_keys=True, default=str
    )


def _create_cpu_replica(
    parallelism_config: RankParallelismConfig,
    model_path: str,
    server_args: ServerArgs,
    shared_params_dict: dict[str, torch.Tensor],
    first_engine_rank: bool = False,
) -> torch.nn.Module:
    """Create a CPU model replica that loads the right shard and skips post_load_weights."""
    load_config = LoadConfig(
        load_format="dummy",
        model_loader_extra_config=None,
        rl_quant_profile=server_args.rl_quant_profile,
    )
    server_args_module.set_global_server_args_for_scheduler(server_args)
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)

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

    if first_engine_rank:
        for param in model.parameters():
            param.data = param.data.pin_memory()
    else:
        for name, param in model.named_parameters():
            assert name in shared_params_dict, f"[P2P-Shared] Parameter {name} not found in shared buffers"
            shared = shared_params_dict[name]
            assert param.shape == shared.shape and param.dtype == shared.dtype, (
                f"[P2P-Shared] Parameter {name} cannot alias the shared buffer: "
                f"replica {tuple(param.shape)}/{param.dtype} vs shared {tuple(shared.shape)}/{shared.dtype}"
            )
            param.data = shared

    return model
