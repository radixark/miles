import logging
from argparse import Namespace
from collections.abc import Callable, Sequence

import ray
import torch
import torch.distributed as dist
from tqdm import tqdm

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.timer import timer

from ...lora_utils import LORA_ADAPTER_NAME, _is_adapter_param_name, build_lora_sync_config, is_lora_weight_name
from ...megatron_to_hf import convert_to_hf
from ..common import (
    all_gather_param,
    begin_weight_update,
    collect_named_tensors_for_weight_transfer,
    end_weight_update,
    get_atomic_update_groups,
    get_named_value_update_units,
)
from ..hf_weight_iterator_base import HfWeightIteratorBase

logger = logging.getLogger(__name__)


def _is_expert_update_unit(update_unit: list[tuple[str, torch.Tensor]]) -> bool:
    assert update_unit, "Update unit must contain at least one param"
    name, _tensor = update_unit[0]
    return ".experts." in name


class DistBucketedWeightUpdateMixin:
    """Mixin providing bucketed TP/EP all-gather, HF format conversion, pre-process/post-process
        and the weight updating pipeline.

    Requires the consuming class to set:
        self.args: Namespace with update_weight_buffer_size (as the bucket size).
        self.model: Sequence[torch.nn.Module] (Megatron model chunks).
        self.model_name: str (for HF conversion).
        self.quantization_config: dict | None.
        self._is_source: bool (whether it's the rank broadcasting weights after `all_gather`).
        self._is_lora_source: bool (the single rank holding the full adapter; for LoRA sync).
        self.weight_version: int.
        self.rollout_engines: Sequence[ActorHandle]. engines of rollout side.
        self._group_name: str. Identifier shown in the tqdm progress bar.
        self._update_weight_implementation(converted_named_tensors, pbar) -> None
            Transfer a bucket of HF-format ``(name, tensor)`` pairs to rollout
            engines (via NCCL broadcast, p2p write, etc.).
        self._update_lora_weight_implementation(named_tensors) -> None
            Transfer the full LoRA adapter (HF-format ``(name, tensor)`` pairs) to
            rollout engines. Only required when ``is_lora``; the
            unload-before-reload is handled by ``_update_lora_weights``.
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
            # Distributed LoRA sync requires the bridge iterator
            assert args.megatron_to_hf_mode == "bridge", (
                "LoRA weight sync over distributed engines requires "
                f"--megatron-to-hf-mode bridge (got {args.megatron_to_hf_mode!r})."
            )
            # The bridge exports adapters per local (PP-stage) model, so a single
            # source rank holds the complete adapter only at PP=1. With PP>1 each
            # stage would broadcast a partial adapter, so reject it explicitly.
            assert args.pipeline_model_parallel_size == 1, (
                "LoRA weight sync over distributed engines requires "
                f"--pipeline-model-parallel-size 1 (got {args.pipeline_model_parallel_size})."
            )
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

        for update_unit in self._get_weight_transfer_update_units(is_expert=False):
            gathered_params, unit_size = self._all_gather_update_unit(update_unit)

            if not self._is_source:
                continue

            if buffer_size + unit_size > self.args.update_weight_buffer_size and converted_named_tensors:
                update_bucket_weight_func(converted_named_tensors, pbar)
                converted_named_tensors = []
                buffer_size = 0

            for name, param in gathered_params:
                converted_named_tensors += convert_to_hf(
                    self.args, self.model_name, name, param, self.quantization_config
                )
            buffer_size += unit_size

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

        for update_unit in self._get_weight_transfer_update_units(is_expert=True):
            gathered_params, unit_size = self._all_gather_update_unit(update_unit)

            if (
                buffer_size + unit_size
            ) * get_parallel_state().ep.size > self.args.update_weight_buffer_size and named_tensors:
                self._update_expert_bucket_weights(named_tensors, update_bucket_weight_func, pbar)
                named_tensors = []
                buffer_size = 0

            named_tensors.extend(gathered_params)
            buffer_size += unit_size

        if named_tensors:
            self._update_expert_bucket_weights(named_tensors, update_bucket_weight_func, pbar)

    def _all_gather_update_unit(
        self, update_unit: list[tuple[str, torch.Tensor]]
    ) -> tuple[list[tuple[str, torch.Tensor]], int]:
        gathered_params = []
        unit_size = 0
        for name, param in update_unit:
            param = all_gather_param(self.args, name, param)
            gathered_params.append((name, param))
            unit_size += param.numel() * param.element_size()
        return gathered_params, unit_size

    def _get_weight_transfer_update_units(self, is_expert: bool) -> list[list[tuple[str, torch.Tensor]]]:
        named_tensors = list(collect_named_tensors_for_weight_transfer(self.args, self.model, is_expert=None))
        named_tensors = [
            (name.replace(".to_wrap.", "."), tensor)
            for name, tensor in named_tensors
            if not _is_adapter_param_name(name)
        ]
        atomic_update_groups = get_atomic_update_groups(self.args, self.model_name)
        update_units = get_named_value_update_units(named_tensors, atomic_update_groups)
        for unit in update_units:
            assert len({".experts." in name for name, _tensor in unit}) == 1, [name for name, _tensor in unit]
        return [unit for unit in update_units if _is_expert_update_unit(unit) == is_expert]

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
        all_names: list[list[str] | None] = [None] * get_parallel_state().ep.size
        dist.all_gather_object(all_names, names, group=get_parallel_state().ep.group)

        for ep_names in all_names:
            assert len(named_tensors) == len(
                ep_names
            ), f"mismatch names length: {len(named_tensors)} != {len(ep_names)}"

        all_gathered_params: list[list[tuple[str, torch.Tensor]]] = [[] for _ in range(get_parallel_state().ep.size)]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(get_parallel_state().ep.size)
            ]
            handle = dist.all_gather(params, param.data, group=get_parallel_state().ep.group, async_op=True)
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

    def _update_lora_weights(self) -> None:
        """Orchestrate the LoRA adapter update; delegate transmit to the subclass.

        Mirrors the base path's split: this method owns the transport-agnostic
        steps (bridge iteration, validation, source gating, and the
        unload-before-reload), and hands the gathered adapter to
        ``self._update_lora_weight_implementation`` — broadcast (NCCL) or p2p
        provide their own.

        All TP ranks iterate the bridge (required for internal TP collectives),
        but only the source rank transmits.
        """
        # All ranks must iterate the bridge for TP collective participation.
        # {} weights: bridge exports adapters directly from self.model and ignores
        # this dict (bridge-only is enforced in _init_lora).
        accumulated_named_tensors: list[tuple[str, torch.Tensor]] = []
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks({}, weight_type="lora"):
            accumulated_named_tensors.extend(hf_named_tensors)

        if not accumulated_named_tensors:
            raise RuntimeError(
                "LoRA weight sync failed: the weight iterator produced zero chunks. "
                "No adapter weights were sent to the rollout engine. This usually means "
                "the Megatron-Bridge or SGLang version is incompatible."
            )

        if not self._is_lora_source:
            return

        if not any(is_lora_weight_name(n) for n, _ in accumulated_named_tensors):
            raise RuntimeError(
                "LoRA weight sync failed: chunk contains no LoRA weights "
                "(no lora_A/lora_B names found). Check weight iterator."
            )

        if self._lora_loaded:
            ray.get(
                [engine.unload_lora_adapter.remote(lora_name=LORA_ADAPTER_NAME) for engine in self.rollout_engines]
            )
        self._update_lora_weight_implementation(accumulated_named_tensors)
        self._lora_loaded = True

    def _pause_and_prepare_engines(self) -> None:
        """Pause rollout engines, flush cache, and open the weight-update session."""
        if dist.get_rank() == 0:
            mode = self.args.pause_generation_mode
            ray.get([engine.pause_generation.remote(mode=mode) for engine in self.rollout_engines])
            if mode not in ("in_place"):
                ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

            begin_weight_update(self.rollout_engines)

    def _finalize_and_resume_engines(self) -> None:
        """Close the weight-update session and resume rollout engines."""
        if dist.get_rank() == 0:
            end_weight_update(self.rollout_engines)
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])

    @torch.no_grad()
    def update_weights(self) -> None:
        """Orchestrate the full weight-update lifecycle.

        Pause → flush → non-expert (TP) → expert (EP) → continue.
        Progress is showed on the rank `_is_source`.

        - `_pause_and_prepare_engines`: pause rollout engines, flush caches,
             run pre-process.
        - `_gather_and_update_non_expert_weights`
        - `_gather_and_update_expert_weights`
        - `_finalize_and_resume_engines`: run post-process, resume rollout
            generation.

        Full: pause → base non-expert (TP) → base expert (EP) → resume.
        LoRA: pause → base weights (first iteration only) → LoRA adapter
        (every iteration) → resume.
        """
        self.weight_version += 1

        self._pause_and_prepare_engines()
        dist.barrier(group=get_gloo_group())

        with timer("update_weights_implementation"):
            # Base weight sync model:
            #   full-param RL: base weights change every step -> always sync.
            #   LoRA RL: base is frozen -> only sync once, on the first iteration.
            if not (self.is_lora and self._lora_base_synced):
                pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_source else None

                self._gather_and_update_non_expert_weights(self._update_weight_implementation, pbar)
                dist.barrier(group=get_gloo_group())
                self._gather_and_update_expert_weights(self._update_weight_implementation, pbar)
                dist.barrier(group=get_gloo_group())

            # LoRA adapter weights: every iteration.
            if self.is_lora:
                self._update_lora_weights()
                dist.barrier(group=get_gloo_group())
                if not self._lora_base_synced:
                    self._lora_base_synced = True

        with timer("finalize_and_resume_engines"):
            self._finalize_and_resume_engines()
            dist.barrier(group=get_gloo_group())
