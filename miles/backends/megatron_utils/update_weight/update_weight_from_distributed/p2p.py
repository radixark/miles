import copy
import dataclasses
import fcntl
import logging
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import torch
import torch.distributed as dist
from ray.actor import ActorHandle
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
from tqdm import tqdm

from miles.utils.distributed_utils import get_gloo_group

from .mixin import DistBucketedWeightUpdateMixin


from .p2p_transfer_utils import (
    P2PTransferManager,
    RemoteTransferPlan,
    RemoteWeightInfo,
    create_transfer_engine,
    query_remote_weight_infos,
    register_cpu_memory,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _DraftTransferState:
    """Per-draft-worker transfer state, mirroring the target-model attributes
    on UpdateWeightP2P. The draft (MTP/nextn) model registers separately on the
    engine side; without updating it, speculative acceptance decays to zero as
    the target trains away from the frozen draft."""

    mapper: ParameterMapper
    params_dict: dict[str, torch.Tensor]
    meta_list: list[tuple[torch.nn.Module, list[RemoteWeightInfo]]]
    nextn_prefix: str
    memory_registry: dict | None = None
    staged: dict[str, list[tuple[str, torch.Tensor]]] = dataclasses.field(default_factory=dict)
    pending: dict[str, int] = dataclasses.field(default_factory=dict)


class UpdateWeightP2P(DistBucketedWeightUpdateMixin):
    """P2P weight transfer using DistBucketedWeightUpdateMixin for bucketed all-gather + HF conversion,
    and a single set of shared CPU pinned buffers for P2P writes.

    Compute transfer_ready_params once (same for all engine ranks)
    For each engine rank:
        load_weights(shared buffer) → P2P write
        where the last rank's write is submitted to a background thread
    wait_transfers() at finish to collect all background writes
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        is_lora: bool = False,
    ) -> None:
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None
        self.rollout_engines: Sequence[ActorHandle] | None = None
        self._connection_stale: bool = False
        assert not is_lora, "LoRA weight sync is not supported for p2p (RDMA) weight transfer."
        self.is_lora = False

        self.transfer_plan = RemoteTransferPlan(args, model)
        self.global_rank = dist.get_rank(group=get_gloo_group())
        self._model_registered = False
        self._tensor_update_pending: dict[str, int] = {}

        self._staged_tensors: dict[str, list[tuple[str, torch.Tensor]]] = {}
        self._draft_state: _DraftTransferState | None = None
        self.transfer_manager = P2PTransferManager(
            num_workers=getattr(args, "p2p_transfer_num_workers", 4),
            transfer_timeout=getattr(args, "p2p_transfer_timeout", 30.0),
        )

    @property
    def _is_source(self):
        """Whether this training rank is a source that sends weights to rollout.

        In P2P mode, all training GPUs sharing the same PP rank hold a complete
        weight replica after TP/EP all-gather. Each source rank transfers its
        weights to exactly one rollout rank in a 1-to-1 fashion.

        Key quantities:
          - senders:   _gathered_dp_size  = world_size // pp_size
          - receivers: _rollout_num_gpus

        Case 1: senders <= receivers
          Every training rank is a source (all are needed to cover the rollout ranks).

        Case 2: senders > receivers
          Only the first `_rollout_num_gpus` training ranks (by gathered_dp_rank)
          are sources; the rest are idle during transfer.

        """
        return self.transfer_plan._gathered_dp_rank < self.transfer_plan._rollout_num_gpus

    def _gather_and_update_expert_weights(self, update_bucket_weight_func, pbar=None):
        """Wait for all background P2P writes to complete here."""
        super()._gather_and_update_expert_weights(update_bucket_weight_func, pbar)
        if not self._is_source:
            return
        self.transfer_manager.wait_transfers()
        assert len(self._tensor_update_pending) == 0 and len(self._staged_tensors) == 0, (
            f"Some tensors were not transferred during P2P weight update. "
            f"Pending: {self._tensor_update_pending}, Staged: {self._staged_tensors}"
        )
        if self._draft_state is not None:
            assert len(self._draft_state.pending) == 0 and len(self._draft_state.staged) == 0, (
                f"Some draft tensors were not transferred during P2P weight update. "
                f"Pending: {self._draft_state.pending}, Staged: {self._draft_state.staged}"
            )

    def _pause_and_prepare_engines(self):
        """Register shared CPU pinned memory with P2P on first call."""
        super()._pause_and_prepare_engines()
        if not self._is_source:
            return

        if not self._model_registered:
            self._weight_memory_registry = register_cpu_memory(self._shared_params_dict, self._transfer_engine)
            if self._draft_state is not None:
                self._draft_state.memory_registry = register_cpu_memory(
                    self._draft_state.params_dict, self._transfer_engine
                )
        self._model_registered = True

    def _update_weight_implementation(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """Stage incoming tensors; when all shards for a param are collected,
        load into shared buffer and P2P-write per engine rank.

        Only calls load_weights() with complete accumulated tensors, preventing
        partial writes that would corrupt the shared buffer when different engine
        ranks have different EP expert-to-local mappings.
        """
        if not self._is_source or not converted_named_tensors:
            return

        if self._draft_state is not None:
            prefix = self._draft_state.nextn_prefix
            draft_tensors = [(n, t) for n, t in converted_named_tensors if n.startswith(prefix)]
            base_tensors = [(n, t) for n, t in converted_named_tensors if not n.startswith(prefix)]
        else:
            draft_tensors, base_tensors = [], converted_named_tensors

        transfer_ready_params, ready_hf_tensors = self._get_transfer_ready_params(base_tensors)
        self._stage_and_write(
            transfer_ready_params,
            ready_hf_tensors,
            self._transfer_engine_meta_list,
            self._weight_memory_registry,
        )

        if draft_tensors:
            ready_params, ready_tensors = self._get_transfer_ready_params(
                draft_tensors,
                mapper=self._draft_state.mapper,
                params_dict=self._draft_state.params_dict,
                staged=self._draft_state.staged,
                pending=self._draft_state.pending,
            )
            self._stage_and_write(
                ready_params,
                ready_tensors,
                self._draft_state.meta_list,
                self._draft_state.memory_registry,
            )

        converted_named_tensors.clear()

    def _stage_and_write(
        self,
        transfer_ready_params: list[str],
        ready_hf_tensors: list[tuple[str, torch.Tensor]],
        meta_list: list[tuple[torch.nn.Module, list[RemoteWeightInfo]]],
        memory_registry: dict,
    ) -> None:
        if not transfer_ready_params or not ready_hf_tensors:
            return
        last_idx = len(meta_list) - 1
        for i, (model_replica, remote_weight_infos) in enumerate(meta_list):
            # All shards of a stacked param must arrive in one call: the loader's
            # fusion caches (e.g. cached_a_proj for fused_qkv_a_proj_with_mqa) are
            # local to do_load_weights, so splitting the call drops the param.
            try:
                model_replica.load_weights(ready_hf_tensors)
            except Exception as e:
                raise RuntimeError(
                    f"[P2P-Shared] staging failed for {ready_hf_tensors[0][0]} "
                    f"(+{len(ready_hf_tensors) - 1} more). p2p needs a quant finalize that keeps "
                    f"parameters loadable; fp8-block qualifies, an unquantized MoE under the "
                    f"flashinfer trtllm runner does not (it swizzles experts to a 4D layout)."
                ) from e

            is_last = i == last_idx
            if is_last:
                # Last engine rank: fire-and-forget all sessions to background,
                # as the weight will no longer be overwritten
                for remote_session in remote_weight_infos:
                    self.transfer_manager.submit(
                        self._do_p2p_write_one_session,
                        remote_session,
                        transfer_ready_params,
                        memory_registry,
                    )
            else:
                # Non-last engine rank needs to be fully written to target before next update can happen.
                futures = [
                    self.transfer_manager.submit_returning_future(
                        self._do_p2p_write_one_session,
                        remote_session,
                        transfer_ready_params,
                        memory_registry,
                    )
                    for remote_session in remote_weight_infos
                ]
                for f in futures:
                    f.result()

    # TODO: avoid dup code during yueming's refactor (temp write this to avoid introducing potentially conflicting base class)
    def is_rollout_engines_fresh(self) -> bool:
        return self.rollout_engines is not None and not self._connection_stale

    def mark_engine_connection_stale(self) -> None:
        self._connection_stale = True

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """The ``connect_rollout_engines`` here will:

        - Create a transfer plan that maps each training rank to its target
          rollout rank(s) based on GPU counts and parallelism configuration.
        - Query remote rollout engines for their weight memory registration
          info (addresses and sizes for RDMA writes).
        - Query remote parallelism config and construct a local CPU model
          replica that mirrors the target's sharding layout, enabling correct
          weight format conversion before transfer.
        """
        self.rollout_engines = rollout_engines
        self._connection_stale = False
        self.rollout_engine_lock = rollout_engine_lock

        if self._is_source:
            self._group_name = f"miles-p2p_{self.transfer_plan._gathered_dp_rank}"
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
            self._transfer_engine_meta_list: list[tuple[torch.nn.Module, list[RemoteWeightInfo]]] = []
            first_engine_rank = True
            for rank_targets in targets_grouped_by_engine_rank.values():
                first_target = rank_targets[0]
                session_id = targets_to_session_id[(first_target.engine_ind, first_target.engine_rank)]
                parallelism_config = RankParallelismConfig.from_dict(
                    self.remote_weight_infos_by_session_id[session_id][1]
                )
                server_args = self.session_id_to_server_args[session_id]

                model_replica = self._create_cpu_replica(
                    parallelism_config,
                    self.args.hf_checkpoint,
                    server_args,
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

                self._transfer_engine_meta_list.append((model_replica, remote_infos))

            first_server_args = next(iter(self.session_id_to_server_args.values()))
            # The draft consumes only model.layers.{num_hidden_layers}.* names (its
            # embed_tokens and shared_head.head are shared with the target), and the
            # converter emits those only when the MTP layer is trained. A serving-only
            # MTP setup keeps that layer frozen, so building a draft replica there
            # would register memory that never receives a byte.
            if first_server_args.speculative_algorithm and self.args.enable_mtp_training:
                self._connect_draft_sessions(rollout_engines, targets_grouped_by_engine_rank)

    def _connect_draft_sessions(self, rollout_engines, targets_grouped_by_engine_rank) -> None:
        """Mirror the target-model session setup for the speculative draft
        (MTP/nextn) worker, which registers separately. The converter emits the
        trained nextn layer under model.layers.{num_hidden_layers}; without
        these sessions the draft would serve frozen initial weights and
        speculative acceptance decays to zero as the target trains."""
        targets = [t for rank_targets in targets_grouped_by_engine_rank.values() for t in rank_targets]
        (
            draft_infos_by_session,
            draft_targets_to_session_id,
            draft_session_to_server_args,
        ) = query_remote_weight_infos(rollout_engines, targets, worker="draft")

        draft_replica = None
        draft_meta: list[tuple[torch.nn.Module, list[RemoteWeightInfo]]] = []
        for rank_targets in targets_grouped_by_engine_rank.values():
            first_target = rank_targets[0]
            session_id = draft_targets_to_session_id[(first_target.engine_ind, first_target.engine_rank)]
            if draft_replica is None:
                parallelism_config = RankParallelismConfig.from_dict(draft_infos_by_session[session_id][1])
                draft_replica = self._create_cpu_replica(
                    parallelism_config,
                    self.args.hf_checkpoint,
                    draft_session_to_server_args[session_id],
                    first_engine_rank=True,
                    is_draft=True,
                )
                draft_params = dict(draft_replica.named_parameters())
                draft_mapper = ParameterMapper.from_model(draft_replica)
            remote_infos = [
                RemoteWeightInfo(
                    draft_targets_to_session_id[(t.engine_ind, t.engine_rank)],
                    draft_infos_by_session[draft_targets_to_session_id[(t.engine_ind, t.engine_rank)]][0],
                )
                for t in rank_targets
            ]
            draft_meta.append((draft_replica, remote_infos))

        self._draft_state = _DraftTransferState(
            mapper=draft_mapper,
            params_dict=draft_params,
            meta_list=draft_meta,
            nextn_prefix=f"model.layers.{draft_replica.config.num_hidden_layers}.",
        )

    def _create_cpu_replica(
        self,
        parallelism_config: RankParallelismConfig,
        model_path: str,
        server_args: ServerArgs,
        first_engine_rank: bool = False,
        is_draft: bool = False,
    ) -> torch.nn.Module:
        """Create a CPU model replica that loads the right shard and skips post_load_weights."""
        load_config = LoadConfig(
            load_format="dummy",
            model_loader_extra_config=None,
            rl_quant_profile=server_args.rl_quant_profile,
        )
        server_args = copy.copy(server_args)
        server_args.nnodes = 1
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
        original_initialize_model = model_loader_module._initialize_model

        def _initialize_model_defer_finalize(*args, **kwargs):
            built = original_initialize_model(*args, **kwargs)
            for _, module in built.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading = lambda layer: None
            return built

        model_loader_module.post_load_weights = lambda *args, **kwargs: None
        model_loader_module._initialize_model = _initialize_model_defer_finalize
        try:
            with ParallelismContext(parallelism_config):
                model = get_model(
                    model_config=ModelConfig(model_path, is_draft_model=is_draft),
                    load_config=load_config,
                    device_config=DeviceConfig(device="cpu"),
                )
        finally:
            model_loader_module.post_load_weights = original_post_load_weights
            model_loader_module._initialize_model = original_initialize_model

        # Run the real quant finalize the loader skipped, module by module on the
        # trainer GPU (the transforms are CUDA-only), mirroring the loader loop.
        # This leaves the replica in the exact post-finalize layout the engine
        # registers, for any quantization method. The node-local lock serializes
        # this phase and the pinning below across co-located actors: their
        # combined transient host-memory churn on top of the resident replicas
        # OOM-kills the node when run concurrently.
        with open("/dev/shm/miles_p2p_replica_finalize.lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is None:
                    continue
                if "process_weights_after_loading" in quant_method.__dict__:
                    del quant_method.process_weights_after_loading
                if hasattr(module, "is_weights_quantized") and module.is_weights_quantized():
                    continue
                module.to("cuda")
                quant_method.process_weights_after_loading(module)
                module.to("cpu")
                torch.cuda.empty_cache()

            if first_engine_rank:
                for param in model.parameters():
                    param.data = param.data.pin_memory()

        # Also patch the instance method for subsequent load_weights() calls
        # (deepseek_weight_loader.py:342 calls self.post_load_weights() at the end).
        if hasattr(model, "post_load_weights"):
            model.post_load_weights = lambda *args, **kwargs: None

        if not first_engine_rank:
            for name, param in model.named_parameters():
                assert name in self._shared_params_dict, f"[P2P-Shared] Parameter {name} not found in shared buffers"
                param.data = self._shared_params_dict[name]

        return model

    def _get_transfer_ready_params(
        self,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        *,
        mapper: ParameterMapper | None = None,
        params_dict: dict[str, torch.Tensor] | None = None,
        staged: dict[str, list[tuple[str, torch.Tensor]]] | None = None,
        pending: dict[str, int] | None = None,
    ) -> tuple[list[str], list[tuple[str, torch.Tensor]]]:
        """Determine which sglang params have all shards present, returning their accumulated tensors.

        Some parameters are trained separately on the training side but fused into a
        single tensor on the rollout side (e.g., Q/K/V projections are separate in
        Megatron but merged into one qkv_proj in sglang). This function stages
        incoming HF tensors in self._staged_tensors until all shards for a
        sglang param are collected. Only returns tensors for fully-ready params,
        preventing partial load_weights() calls that would corrupt the shared buffer.

        Return:
            transfer_ready_params: tensors' names for the ones ready to be transferred.
            ready_hf_tensor: corresponding complete tensors ready to be transferred.
        """
        transfer_ready_params = []
        mapper = mapper if mapper is not None else self._shared_param_mapper
        params_dict = params_dict if params_dict is not None else self._shared_params_dict
        staged = staged if staged is not None else self._staged_tensors
        pending = pending if pending is not None else self._tensor_update_pending

        for name, tensor in converted_named_tensors:
            # map the tensor name of huggingface to the one of sglang.
            mapped_result = mapper.map(name)
            mapped, num_shards, num_experts = (
                mapped_result.sglang_name,
                mapped_result.num_shards,
                mapped_result.num_local_experts,
            )
            if mapped not in params_dict:
                logger.warning(f"Parameter {mapped} not found in shared model replica.")
                continue

            if num_experts is not None and num_experts > 0:
                total_expected = num_experts * num_shards
            else:
                total_expected = num_shards

            staged.setdefault(mapped, []).append((name, tensor))

            if total_expected == 1:
                transfer_ready_params.append(mapped)
            else:
                if mapped not in pending:
                    pending[mapped] = total_expected - 1
                else:
                    pending[mapped] -= 1
                if pending[mapped] == 0:
                    transfer_ready_params.append(mapped)

        ready_hf_tensors: list[tuple[str, torch.Tensor]] = []
        for param_name in transfer_ready_params:
            ready_hf_tensors.extend(staged.pop(param_name, []))
            pending.pop(param_name, None)

        return transfer_ready_params, ready_hf_tensors

    def _do_p2p_write_one_session(
        self, remote_session: RemoteWeightInfo, names: list[str], memory_registry: dict | None = None
    ) -> None:
        """P2P write from shared CPU pinned buffers to a single remote session.

        Used by the parallelized submission path where each session within an
        engine rank is submitted as a separate task to P2PTransferManager.
        """
        source_ptrs, source_lens = [], []
        valid_names = []

        for name in names:
            cpu_reg = (memory_registry if memory_registry is not None else self._weight_memory_registry).get(name)
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
                target_ptrs.append(remote_session.weights_info[name][0])

        missing = [n for n in valid_names if n not in remote_session.weights_info]
        assert len(target_ptrs) == len(source_ptrs), (
            f"[P2P-Shared] Pointer count mismatch for session {session_id}, "
            f"source: {len(source_ptrs)}, target: {len(target_ptrs)}, "
            f"missing_on_remote[:8]: {missing[:8]}"
        )
        for name, slen in zip(valid_names, source_lens, strict=True):
            _, r_numel, r_ele = remote_session.weights_info[name]
            assert (
                r_numel * r_ele == slen
            ), f"[P2P-Shared] Length mismatch for {name}: local {slen}, remote {r_numel * r_ele}"

        ret = self._transfer_engine.batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)
        if ret < 0:
            raise RuntimeError(f"[P2P-Shared] Transfer failed for session {session_id}, error: {ret}")
