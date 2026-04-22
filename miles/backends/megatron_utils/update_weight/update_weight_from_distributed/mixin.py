import logging
from argparse import Namespace
from collections.abc import Callable, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from tqdm import tqdm

from miles.utils.distributed_utils import get_gloo_group
from miles.utils.timer import timer

from ...lora_utils import LORA_ADAPTER_NAME, _is_adapter_param_name, build_lora_sync_config, is_lora_weight_name
from ...megatron_to_hf import convert_to_hf
from ...sglang import FlattenedTensorBucket, MultiprocessingSerializer
from ..common import (
    _check_weight_sync_results,
    all_gather_param,
    collect_named_tensors_for_weight_transfer,
    post_process_weights,
)
from ..hf_weight_iterator_base import HfWeightIteratorBase

logger = logging.getLogger(__name__)


class DistBucketedWeightUpdateMixin:
    """Mixin providing bucketed TP/EP all-gather, HF format conversion, pre-process/post-process
        and the weight updating pipeline.

    Requires the consuming class to set:
        self.args: Namespace with update_weight_buffer_size (as the bucket size).
        self.model: Sequence[torch.nn.Module] (Megatron model chunks).
        self.model_name: str (for HF conversion).
        self.quantization_config: dict | None.
        self._is_source: bool (whether it's the rank broadcasting weights after `all_gather`).
        self.weight_version: int.
        self.rollout_engines: Sequence[ActorHandle]. engines of rollout side.
        self._group_name: str. Identifier shown in the tqdm progress bar.
        self._update_weight_implementation(converted_named_tensors, pbar) -> None
            Transfer a bucket of HF-format ``(name, tensor)`` pairs to rollout
            engines (via NCCL broadcast, p2p write, etc.).
    """

    def _init_lora(
        self,
        *,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        model_name: str,
        quantization_config: dict | None,
        is_lora: bool,
    ) -> None:
        """Initialize LoRA-specific state. Call from subclass ``__init__``."""
        self.is_lora = is_lora
        if self.is_lora:
            self._lora_config = build_lora_sync_config(args)
            self._lora_loaded = False
            self._lora_base_synced = False
            self._hf_weight_iterator = HfWeightIteratorBase.create(
                args=args,
                model=model,
                model_name=model_name,
                quantization_config=quantization_config,
                is_lora=True,
            )

    def _gather_and_update_non_expert_weights(
        self,
        update_bucket_weight_func: Callable[[list[tuple[str, torch.Tensor]], tqdm | None], None],
        pbar: tqdm | None = None,
    ) -> None:
        """
        Bucketed TP all-gather + HF conversion for non-expert parameters.
        Non-expert: gather TP → rm pad → HF → buffer (flush if full). All gather, PP source buffers.
        After `all_gather`, update weights/buffer_size on source, do nothing on non-source.
        """

        buffer_size = 0
        converted_named_tensors: list[tuple[str, torch.Tensor]] = []

        for name, param in collect_named_tensors_for_weight_transfer(self.args, self.model, is_expert=False):
            # Skip LoRA adapter parameters; they are synced separately.
            if _is_adapter_param_name(name):
                continue
            # Strip ".to_wrap." introduced by LoRA adapter wrapping so that
            # downstream name-based checks (partition stride, HF conversion) work.
            name = name.replace(".to_wrap.", ".")
            param = all_gather_param(self.args, name, param)
            if not self._is_source:
                continue

            param_size = param.numel() * param.element_size()
            if buffer_size + param_size > self.args.update_weight_buffer_size:
                update_bucket_weight_func(converted_named_tensors, pbar)
                converted_named_tensors = []
                buffer_size = 0

            converted_named_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
            buffer_size += param_size

        if converted_named_tensors:
            update_bucket_weight_func(converted_named_tensors, pbar)

    def _gather_and_update_expert_weights(
        self,
        update_bucket_weight_func: Callable[[list[tuple[str, torch.Tensor]], tqdm | None], None],
        pbar: tqdm | None = None,
    ) -> None:
        """
        Bucketed TP + EP all-gather + HF conversion for expert parameters.
        Expert: gather TP → rm pad → buffer. EP gather + HF deferred. Threshold × EP size.
        """
        buffer_size = 0
        named_tensors: list[tuple[str, torch.Tensor]] = []

        for name, param in collect_named_tensors_for_weight_transfer(self.args, self.model, is_expert=True):
            if _is_adapter_param_name(name):
                continue
            # Strip ".to_wrap." from LoRA-wrapped names before all_gather
            name = name.replace(".to_wrap.", ".")
            param = all_gather_param(self.args, name, param)
            param_size = param.numel() * param.element_size()
            if (
                buffer_size + param_size
            ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size and named_tensors:
                self._update_expert_bucket_weights(named_tensors, update_bucket_weight_func, pbar)
                named_tensors = []
                buffer_size = 0

            named_tensors.append((name, param))
            buffer_size += param_size

        if named_tensors:
            self._update_expert_bucket_weights(named_tensors, update_bucket_weight_func, pbar)

    def _update_expert_bucket_weights(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        update_bucket_weight_func: Callable[[list[tuple[str, torch.Tensor]], tqdm | None], None],
        pbar: tqdm | None,
    ) -> None:
        """
        Gather EP → HF → update weights. Clears buffer.
        """
        names = [name for name, _ in named_tensors]
        all_names: list[list[str] | None] = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for ep_names in all_names:
            assert len(named_tensors) == len(
                ep_names
            ), f"mismatch names length: {len(named_tensors)} != {len(ep_names)}"

        all_gathered_params: list[list[tuple[str, torch.Tensor]]] = [
            [] for _ in range(mpu.get_expert_model_parallel_world_size())
        ]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            handles.append(handle)
            for ep_rank, ep_names in enumerate(all_names):
                all_gathered_params[ep_rank].append((ep_names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self._is_source:
            return

        flat_gathered = sum(all_gathered_params, [])

        converted_hf_tensors: list[tuple[str, torch.Tensor]] = []
        for name, param in flat_gathered:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)

        update_bucket_weight_func(converted_hf_tensors, pbar)

    def _pause_and_prepare_engines(self) -> None:
        """Pause rollout engines, flush cache, and run pre-process if needed."""
        if dist.get_rank() == 0:
            mode = self.args.pause_generation_mode
            ray.get([engine.pause_generation.remote(mode=mode) for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

            # int4/fp4 pre_process
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    rollout_engines=self.rollout_engines,
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                )

    def _finalize_and_resume_engines(self, post_load_weights: bool = False) -> None:
        """Run post-process if needed and resume rollout engines."""
        if dist.get_rank() == 0:
            # post_process_quantization is related to the process_weights_after_loading
            # in the sglang rollout side, which should always be invoked after weight
            # updating.
            post_process_weights(
                rollout_engines=self.rollout_engines,
                restore_weights_before_load=False,
                post_process_quantization=True,
                post_load_weights=post_load_weights,
            )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])

    @torch.no_grad()
    def update_weights(self) -> None:
        """Orchestrate the full weight-update lifecycle.

        Non-LoRA: pause → base non-expert (TP) → base expert (EP) → resume.
        LoRA: pause → base weights (first iteration only) → LoRA adapter
        (every iteration) → resume.
        """
        self.weight_version += 1

        self._pause_and_prepare_engines()
        dist.barrier(group=get_gloo_group())

        with timer("update_weights_implementation"):
            # Base weights: skip after first round when LoRA is enabled (frozen base).
            if not (getattr(self, "is_lora", False) and self._lora_base_synced):
                pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_source else None

                self._gather_and_update_non_expert_weights(self._update_weight_implementation, pbar)
                dist.barrier(group=get_gloo_group())
                self._gather_and_update_expert_weights(self._update_weight_implementation, pbar)
                dist.barrier(group=get_gloo_group())

            # LoRA adapter weights: every iteration.
            if getattr(self, "is_lora", False):
                self._sync_lora_weights()
                dist.barrier(group=get_gloo_group())
                if not self._lora_base_synced:
                    self._lora_base_synced = True

        with timer("finalize_and_resume_engines"):
            self._finalize_and_resume_engines()
            dist.barrier(group=get_gloo_group())

    def _sync_lora_weights(self) -> None:
        """Sync LoRA adapter weights to all rollout engines via Ray RPC.

        All TP ranks iterate the bridge (required for internal TP collectives),
        but only the source rank (DP=0, TP=0) serializes and sends.
        """
        lora_sync_chunk_count = 0
        all_refs: list[ObjectRef] = []

        # Unload previous adapter before loading new weights (source rank only).
        if self._is_source and self._lora_loaded:
            ray.get(
                [engine.unload_lora_adapter.remote(lora_name=LORA_ADAPTER_NAME) for engine in self.rollout_engines]
            )

        # All ranks must iterate the bridge for TP collective participation.
        # megatron_local_weights arg is unused for LoRA (bridge reads from model).
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks({}, weight_type="lora"):
            lora_sync_chunk_count += 1

            if not self._is_source:
                continue

            if not any(is_lora_weight_name(n) for n, _ in hf_named_tensors):
                raise RuntimeError(
                    "LoRA weight sync failed: chunk contains no LoRA weights "
                    "(no lora_A/lora_B names found). Check weight iterator."
                )

            # Serialize via FlattenedTensorBucket (same format as colocate path).
            bucket = FlattenedTensorBucket(named_tensors=hf_named_tensors)
            serialized = MultiprocessingSerializer.serialize(
                {
                    "flattened_tensor": bucket.get_flattened_tensor(),
                    "metadata": bucket.get_metadata(),
                },
                output_str=True,
            )

            # Send to all rollout engines via Ray RPC.
            for engine in self.rollout_engines:
                all_refs.append(
                    engine.load_lora_adapter_from_tensors.remote(
                        lora_name=LORA_ADAPTER_NAME,
                        config_dict=self._lora_config,
                        serialized_tensors=serialized,
                        load_format="flattened_bucket",
                    )
                )

        if lora_sync_chunk_count == 0:
            raise RuntimeError(
                "LoRA weight sync failed: the weight iterator produced zero chunks. "
                "No adapter weights were sent to the rollout engine. This usually means "
                "the Megatron-Bridge or SGLang version is incompatible."
            )

        if all_refs:
            _check_weight_sync_results(ray.get(all_refs), is_lora=True)

        if self._is_source:
            self._lora_loaded = True
