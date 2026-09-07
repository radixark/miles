import logging
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence

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

from .p2p_transfer_utils import (
    P2PTransferManager,
    RemoteTransferPlan,
    RemoteWeightInfo,
    TransferEngineMeta,
    create_transfer_engine,
    query_remote_weight_infos,
    register_cpu_memory,
)

logger = logging.getLogger(__name__)


class UpdateWeightP2P(WeightTransferProtocol):
    """P2P weight transfer over the updater's bucketed all-gather + HF conversion,
    and a single set of shared CPU pinned buffers for P2P writes.

    Compute transfer_ready_params once (same for all engine ranks)
    For each engine rank:
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
                futures = [
                    self.transfer_manager.submit(
                        self._do_p2p_write_one_session,
                        remote_session,
                        transfer_ready_params,
                    )
                    for remote_session in meta.remote_weight_infos
                ]

                if i != last_idx:
                    # Non-last engine rank needs to be fully written to target before next update can happen.
                    for f in futures:
                        f.result()

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

        self.is_sender = self.transfer_plan._gathered_dp_rank < self.transfer_plan._rollout_num_gpus

        if self.is_sender:
            self.group_name = f"miles-p2p_{self.transfer_plan._gathered_dp_rank}"
            targets = self.transfer_plan.plan_p2p()
            (
                self.remote_weight_infos_by_session_id,
                targets_to_session_id,
                self.session_id_to_server_args,
            ) = query_remote_weight_infos(rollout_engines, targets)

            targets_grouped_by_engine_rank: dict[int, list] = {}
            for target in targets:
                targets_grouped_by_engine_rank.setdefault(target.engine_rank, []).append(target)

            # Create ONE transfer engine for all engine ranks
            self._transfer_engine = create_transfer_engine()
            self._shared_params_dict: dict[str, torch.Tensor] = {}
            self._shared_param_mapper: ParameterMapper | None = None
            # in self._transfer_engine_meta_list: tuple of
            # - single CPU replica shared among all sessions
            # - related remote weight info
            self._transfer_engine_meta_list: list[TransferEngineMeta] = []
            first_engine_rank = True
            for rank_targets in targets_grouped_by_engine_rank.values():
                first_target = rank_targets[0]
                session_id = targets_to_session_id[(first_target.engine_ind, first_target.engine_rank)]
                parallelism_config = RankParallelismConfig.from_dict(
                    self.remote_weight_infos_by_session_id[session_id][1]
                )
                server_args = self.session_id_to_server_args[session_id]

                model_replica = _create_cpu_replica(
                    parallelism_config,
                    self.args.hf_checkpoint,
                    server_args,
                    shared_params_dict=self._shared_params_dict,
                    first_engine_rank=first_engine_rank,
                )
                if first_engine_rank:
                    self._shared_params_dict = dict(model_replica.named_parameters())
                    self._shared_param_mapper = ParameterMapper.from_model(model_replica)
                    first_engine_rank = False

                remote_infos = [
                    RemoteWeightInfo(
                        targets_to_session_id[(t.engine_ind, t.engine_rank)],
                        self.remote_weight_infos_by_session_id[targets_to_session_id[(t.engine_ind, t.engine_rank)]][
                            0
                        ],
                    )
                    for t in rank_targets
                ]

                self._transfer_engine_meta_list.append(
                    TransferEngineMeta(model_replica=model_replica, remote_weight_infos=remote_infos)
                )

    def _do_p2p_write_one_session(self, remote_session: RemoteWeightInfo, names: list[str]) -> None:
        """P2P write from shared CPU pinned buffers to a single remote session.

        Used by the parallelized submission path where each session within an
        engine rank is submitted as a separate task to P2PTransferManager.
        """
        source_ptrs, source_lens = [], []
        valid_names = []

        for name in names:
            cpu_reg = self._weight_memory_registry.get(name)
            assert cpu_reg, f"the _weight_memory_registry of {name} failed"

            data_ptr, numel, ele_size = cpu_reg
            source_ptrs.append(data_ptr)
            source_lens.append(numel * ele_size)
            valid_names.append(name)

        if not source_ptrs:
            return

        session_id = remote_session.session_id
        target_ptrs = []
        for name in valid_names:
            if name in remote_session.weights_info:
                target_ptrs.append(remote_session.weights_info[name].address)

        assert len(target_ptrs) == len(source_ptrs), (
            f"[P2P-Shared] Pointer count mismatch for session {session_id}, "
            f"source: {len(source_ptrs)}, target: {len(target_ptrs)}"
        )

        ret = self._transfer_engine.batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)
        if ret < 0:
            raise RuntimeError(f"[P2P-Shared] Transfer failed for session {session_id}, error: {ret}")


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
            param.data = shared_params_dict[name]

    return model
