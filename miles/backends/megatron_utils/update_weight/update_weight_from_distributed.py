import socket
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from miles.utils.distributed_utils import get_gloo_group, init_process_group

from ..megatron_to_hf import convert_to_hf
from .common import all_gather_param, named_params_and_buffers


class UpdateWeightFromDistributed:
    """
    Update distributed engines via NCCL. Each PP rank: group "miles-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.

    When PP>1, source ranks synchronize per-bucket via all_gather_object so that
    global rank 0 dispatches Ray calls for ALL PP sources before any of them
    broadcast. This prevents deadlocks from engines joining NCCL groups in
    inconsistent order.
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
        """
        Initialize. Groups created in connect_rollout_engines.
        """
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None
        # Leader rank dispatches Ray calls and waits for engine completion.
        self._is_leader = dist.get_rank() == 0
        # Ray ObjectRefs tracking engine-side weight update completion,
        # accumulated per phase and drained in _finish_phase_sync.
        self._pending_engine_refs = []

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Create NCCL "miles-pp_{pp_rank}" if PP source (DP=TP=0).
        """
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        assert not self._is_leader or self._is_pp_src_rank, "Leader rank must be a source rank"
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        if self._is_pp_src_rank:
            self._group_name = f"miles-pp_{pp_rank}"

        if self._is_pp_src_rank:
            if self._model_update_groups is not None:
                disconnect_rollout_engines_from_distributed(
                    self.args, self._group_name, self._model_update_groups, self.rollout_engines
                )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args, self._group_name, rollout_engines
            )

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        Pause → flush → non-expert (TP) → expert (EP) → continue. Progress on PP source.
        """
        self.weight_version += 1

        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

            # int4/fp4 pre_process
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        converted_named_tensors = []
        # non expert params
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None

        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                continue
            buffer_size = self._update_weight_from_distributed(
                name, param, converted_named_tensors, buffer_size, pbar=pbar
            )

        if converted_named_tensors:
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)

        if self._is_pp_src_rank:
            self._finish_phase_sync()

        buffer_size = 0
        named_tensors = []
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." not in name:
                continue
            buffer_size = self._update_expert_weight_from_distributed(
                name, param, named_tensors, buffer_size, pbar=pbar
            )

        if named_tensors:
            self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)

        if self._is_pp_src_rank:
            self._finish_phase_sync()

        if dist.get_rank() == 0:
            # int4/fp4 post_process, mxfp8 post-process (swizzle MoE scales).
            if self.quantization_config and self.quantization_config["quant_method"] in [
                "compressed-tensors",
                "mxfp8",
            ]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])

    def _update_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
    ) -> int | None:
        """
        Non-expert: gather TP → rm pad → HF → buffer (flush if full). All gather, PP source buffers.
        Returns updated bytes on source, None on non-source.
        """
        param = all_gather_param(self.args, name, param)
        if not self._is_pp_src_rank:
            return

        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > self.args.update_weight_buffer_size:
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)
            buffer_size = 0
        converted_named_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        buffer_size += param_size
        return buffer_size

    def _update_expert_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
    ) -> int:
        """
        Expert: gather TP → rm pad → buffer. EP gather + HF deferred. Threshold × EP size.
        """
        param = all_gather_param(self.args, name, param)

        param_size = param.numel() * param.element_size()
        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
            self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)
            buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _update_expert_bucket_weights_from_distributed(
        self, named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        Gather EP → HF → broadcast. Clears buffer.
        """
        names = [name for name, _ in named_tensors]
        all_names = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for names in all_names:
            assert len(named_tensors) == len(names), f"mismatch names length: {len(named_tensors)} != {len(names)}"

        all_gathered_params = [[] for _ in range(mpu.get_expert_model_parallel_world_size())]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            handles.append(handle)
            for ep_rank, names in enumerate(all_names):
                all_gathered_params[ep_rank].append((names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self._is_pp_src_rank:
            return

        all_gathered_params = sum(all_gathered_params, [])
        converted_hf_tensors = []
        for name, param in all_gathered_params:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)

        self._update_bucket_weights_from_distributed(converted_hf_tensors, pbar)

    def _update_bucket_weights_from_distributed(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        Sync PP sources via all_gather_object (no-op for PP=1), rank 0 dispatches
        Ray calls for ALL PP sources, then each source broadcasts on its own
        NCCL group concurrently.
        """
        # Gather tensor metadata across PP sources; leader schedules engine updates
        self._gather_meta_and_schedule_engines(converted_named_tensors)
        # NCCL broadcast on own group
        handles = []
        for _, param in converted_named_tensors:
            handles.append(dist.broadcast(param.data, 0, group=self._model_update_groups, async_op=True))
        for handle in handles:
            handle.wait()

        converted_named_tensors.clear()
        pbar.update(1)

    def _gather_meta_and_schedule_engines(self, converted_named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Gather tensor metadata from all PP sources to leader; leader schedules engine updates."""
        pp_group = mpu.get_pipeline_model_parallel_group()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        my_meta = (
            self._group_name,
            [name for name, _ in converted_named_tensors],
            [param.dtype for _, param in converted_named_tensors],
            [tuple(param.shape) for _, param in converted_named_tensors],
        )
        per_pp_meta = [None] * pp_size
        dist.all_gather_object(per_pp_meta, my_meta, group=pp_group)
        if self._is_leader:
            self._dispatch_from_gathered_meta(per_pp_meta)

    def _dispatch_from_gathered_meta(self, per_pp_meta: list) -> None:
        """Dispatch Ray weight-update calls for every PP source that has data."""
        for group_name, names, dtypes, shapes in per_pp_meta:
            if names:
                refs = [
                    engine.update_weights_from_distributed.remote(
                        names=names,
                        dtypes=dtypes,
                        shapes=shapes,
                        group_name=group_name,
                        weight_version=str(self.weight_version),
                    )
                    for engine in self.rollout_engines
                ]
                self._pending_engine_refs.extend(refs)

    def _finish_phase_sync(self) -> None:
        """
        After a transfer phase, sources that finished early participate in empty
        all_gather_object rounds so sources still broadcasting can synchronize.
        Exits once every PP source sends empty metadata. Then rank 0 waits for
        all accumulated engine refs before proceeding.
        """
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        if pp_size > 1:
            pp_group = mpu.get_pipeline_model_parallel_group()
            while True:
                empty_meta = (self._group_name, [], [], [])
                per_pp_meta = [None] * pp_size
                dist.all_gather_object(per_pp_meta, empty_meta, group=pp_group)
                if all(not meta[1] for meta in per_pp_meta):
                    break
                if self._is_leader:
                    self._dispatch_from_gathered_meta(per_pp_meta)

        if self._is_leader and self._pending_engine_refs:
            ray.get(self._pending_engine_refs)
        self._pending_engine_refs = []


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int] | None = None,
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the NCCL group.
    """
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(engine_gpu_counts) + 1

    refs = []
    rank_cursor = 1
    for i, engine in enumerate(rollout_engines):
        refs.append(
            engine.init_weights_update_group.remote(
                master_address,
                master_port,
                rank_cursor,
                world_size,
                group_name,
                backend="nccl",
            )
        )
        rank_cursor += engine_gpu_counts[i]
    model_update_groups = init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    ray.get(refs)
    return model_update_groups


def disconnect_rollout_engines_from_distributed(args, group_name, model_update_groups, rollout_engines):
    """
    Destroy NCCL on training and engines.
    """
    refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
    dist.destroy_process_group(model_update_groups)
    ray.get(refs)


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → engines).
    """
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
        )
        for engine in rollout_engines
    ]

    handles = []
    for _, param in converted_named_tensors:
        handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return refs


def post_process_weights(
    restore_weights_before_load: bool,
    post_process_quantization: bool,
    rollout_engines: Sequence[ActorHandle],
):
    """
    Trigger post-process for int4/fp4 quantization on all rollout engines.
    """
    ray.get(
        [
            engine.post_process_weights.remote(
                restore_weights_before_load=restore_weights_before_load,
                post_process_quantization=post_process_quantization,
            )
            for engine in rollout_engines
        ]
    )
