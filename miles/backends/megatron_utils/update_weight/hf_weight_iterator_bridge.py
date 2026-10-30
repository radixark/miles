import dataclasses
import inspect
import itertools
import logging
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.lora.slots import pad_lora_to_rank, slice_lora_to_rank
from miles.backends.megatron_utils.update_weight.hf_weight_iterator import (
    MegatronHfWeightIteratorBase,
    _iter_mm_tower_units,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils import megatron_bridge_utils
from miles.utils.hf_utils.weight_mapping import get_param_name_remap_for_checkpoint
from miles.utils.lora.utils import AdapterSpec, is_lora_weight_name

from ..megatron_to_hf import postprocess_hf_param
from ..megatron_to_hf.processors import quantize_params
from ..misc_utils import strip_param_name_prefix

logger = logging.getLogger(__name__)


class HfWeightIteratorBridge(MegatronHfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        bridge_checkpoint = _select_bridge_checkpoint(self.args)
        self._bridge = megatron_bridge_utils.load_auto_bridge(bridge_checkpoint)
        # Bridge may export official DSV4 checkpoint names, e.g. layers.0.attn.wq_a.weight,
        # while SGLang's model uses model.layers.0.self_attn.wq_a.weight.
        # Resolve the mapping once so postprocessing, quantization, and bucketing
        # all use SGLang's model namespace.
        self._remap_hf_name = _load_checkpoint_name_remap(bridge_checkpoint)
        self._export_dtypes = None

    def _iter_hf_param_units(self, weights, *, materialize):
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in weights.items()}
        # Bridge mappings for the hyper-connection alphas and the hc head read
        # sibling parameters through the live module, which colocate sleep() has
        # paused. Point those (tiny) params at resident copies for the export.
        swapped_params = []
        for vp_stage, model_chunk in enumerate(self.model):
            for param_name, param in model_chunk.named_parameters():
                stripped = strip_param_name_prefix(param_name)
                if "hyper_connection.alpha" not in stripped and "hc_head" not in stripped:
                    continue
                source = renamed_megatron_local_weights.get(f"vp_stages.{vp_stage}.{stripped}")
                if source is None:
                    continue
                swapped_params.append((param, param.data))
                param.data = source.detach().to(device="cuda", dtype=param.dtype)
        with megatron_bridge_utils.patch_megatron_model(self.model):
            conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            # Explicit per-parameter dtypes disable checkpoint requantization
            # while preserving BF16 weights and FP32-only parameters on all PP ranks.
            if "weight_dtype" in inspect.signature(self._bridge.export_hf_weights).parameters:
                if self._export_dtypes is None:
                    self._export_dtypes = _gather_export_dtypes(conversion_tasks)
            conversion_tasks = _process_conversion_tasks(
                conversion_tasks, renamed_megatron_local_weights, weight_dtypes=self._export_dtypes
            )
            named_weights = self._bridge.export_hf_weights(
                self.model,
                cpu=False,
                conversion_tasks=conversion_tasks,
                merge_adapter_weights=False,
                **self._source_name_kwargs(self._bridge.export_hf_weights),
            )

            # Apply postprocess + quantization (when targeting a quantized rollout,
            # e.g. FP8 sglang): base weights are quantized to match the rollout's
            # storage format so update_weights_from_tensor lands real weight + scale
            # pairs.
            if not materialize:
                # The export's internal TP collectives must still run on every rank.
                for _ in named_weights:
                    pass
                return

            named_weights = self._postprocess_and_quantize(named_weights, "base")
            # Group by the (tuple of) source names so quantize's weight + scales land in one unit.
            for _megatron_name, group in itertools.groupby(named_weights, key=lambda item: item[2]):
                unit = [(h, w) for h, w, _m in group if not is_lora_weight_name(h)]
                if unit:
                    yield unit
        for param, original_data in swapped_params:
            param.data = original_data
        yield from _iter_mm_tower_units(self.args, materialize=materialize)

    def _export_pp_local_lora(self, adapter):
        if adapter is None:
            return self._export_current_adapter()

        from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

        with expose_adapter_slot(self.model, adapter.slot):
            named_tensors = self._export_current_adapter()
        return [(h, slice_lora_to_rank(h, w, adapter.rank)) for h, w in named_tensors]

    @torch.no_grad()
    def load_adapter(self, adapter: AdapterSpec, hf_tensors) -> None:
        """Inverse of ``_export_pp_local_lora``: write an adapter's HF tensors into its slot.

        The export layout (TP/EP shards, fused q/k/v and gate/up splits, per-expert names,
        the vocab-split lm_head) is traced instead of restated: exporting the slot while its
        parameters hold world-unique element ids yields, at every HF position, the id of the
        slot element the export copied there. ``hf_tensors`` (a safetensors handle) must hold
        exactly the tensors the slot exports, at ``adapter.rank``. Collective."""
        from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

        with expose_adapter_slot(self.model, adapter.slot):
            owned_params = [
                (module, attr, param)
                for chunk in self.model
                for module_name, module in chunk.named_modules()
                for attr, param in module.named_parameters(recurse=False)
                if ".adapter." in f"{module_name}.{attr}"
            ]
            params = [param for _, _, param in owned_params]
            values = torch.zeros(
                sum(param.numel() for param in params), dtype=params[0].dtype, device=params[0].device
            )
            with _element_ids(owned_params) as first_id:
                _scatter_traced_adapter(
                    values,
                    self._iter_current_adapter(),
                    hf_tensors,
                    adapter=adapter,
                    max_rank=self.args.lora_rank,
                    first_id=first_id,
                )
            for param, param_values in zip(params, values.split([param.numel() for param in params]), strict=True):
                param.copy_(param_values.view_as(param))

    def _export_current_adapter(self) -> list:
        return list(self._iter_current_adapter())

    def _iter_current_adapter(self):
        """The exposed adapter's HF-named tensors, streamed one adapter projection at a time."""
        with megatron_bridge_utils.patch_megatron_model(self.model):
            named_weights = self._bridge.export_adapter_weights(
                self.model,
                cpu=False,
                show_progress=False,
                **self._source_name_kwargs(self._bridge.export_adapter_weights),
            )
            for h, w, _m in self._postprocess_and_quantize(named_weights, "lora"):
                if is_lora_weight_name(h):
                    yield h, w

    @staticmethod
    def _source_name_kwargs(export_fn) -> dict:
        """Request source Megatron names via ``with_megatron_names`` when the bridge accepts it (main only)."""
        if "with_megatron_names" in inspect.signature(export_fn).parameters:
            return {"with_megatron_names": True}
        return {}

    @staticmethod
    def _source_names(item) -> tuple:
        """Normalize the third field to a tuple of 0/1/N source Megatron names (a single str on the bridge branch)."""
        source = item[2] if len(item) > 2 else None
        if source is None:
            return ()
        if isinstance(source, str):
            return (source,)
        return tuple(source)

    def _postprocess_and_quantize(self, named_weights, weight_type: str):
        for item in named_weights:
            hf_param_name, weight = item[0], item[1]
            megatron_param_names = self._source_names(item)
            # Padding/quantization rules key on a Megatron name; packed grouped-expert tensors use the first source.
            megatron_param_name = megatron_param_names[0] if megatron_param_names else None
            hf_name = self._remap_hf_name(hf_param_name).replace(".base_layer.", ".")
            weight = postprocess_hf_param(
                args=self.args,
                megatron_param_name=megatron_param_name,
                hf_param_name=hf_name,
                param=weight,
            )
            if weight_type == "base" and self.quantization_config is not None and megatron_param_name is not None:
                # quantize_params expects the megatron name with the `module.module.`
                # prefix that the direct iterator uses; the bridge yields it without.
                # A tensor with no Megatron source (HF-only passthrough) is not a trainable weight: pass it through.
                qmegatron_name = f"module.module.{megatron_param_name}"
                for q_hf_name, q_weight in quantize_params(
                    self.args,
                    qmegatron_name,
                    [(hf_name, weight)],
                    self.quantization_config,
                    self.packed_weight_basenames,
                ):
                    yield q_hf_name, q_weight, megatron_param_names
            else:
                yield hf_name, weight, megatron_param_names


@contextmanager
def _element_ids(owned_params: list[tuple[torch.nn.Module, str, torch.nn.Parameter]]):
    """Put float64 parameters holding element ids unique across the world (exact below 2**53)
    in place of each ``(module, attr, param)`` and yield this rank's first id; the originals
    are put back on exit. The parameter objects are swapped, not their ``.data``: a dtype
    change through ``.data`` resets the parameter's gradient accumulator, which detaches the
    DDP hook that fills ``main_grad`` and leaves the slot untrainable."""
    params = [param for _, _, param in owned_params]
    assert len({param.dtype for param in params}) == 1, "slot parameters must share one dtype"
    counts = [None] * dist.get_world_size()
    dist.all_gather_object(counts, sum(param.numel() for param in params))
    assert sum(counts) < 2**53, "element ids would not be exact in float64"
    first_id = sum(counts[: dist.get_rank()])
    next_id = first_id
    for module, attr, param in owned_params:
        ids = torch.arange(next_id, next_id + param.numel(), dtype=torch.float64, device=param.device)
        traced = torch.nn.Parameter(ids.view(param.shape), requires_grad=param.requires_grad)
        # keep the Megatron parallel attributes the export may read
        traced.__dict__.update(param.__dict__)
        setattr(module, attr, traced)
        next_id += param.numel()
    try:
        yield first_id
    finally:
        for module, attr, param in owned_params:
            setattr(module, attr, param)


def _scatter_traced_adapter(values, traced, hf_tensors, *, adapter: AdapterSpec, max_rank: int, first_id: int) -> None:
    """Fill ``values`` (this rank's slot elements, flattened) from the HF tensors at the positions
    the traced export maps to them."""
    filled = torch.zeros_like(values, dtype=torch.bool)
    names, exported = set(hf_tensors.keys()), set()
    for hf_name, ids in traced:
        exported.add(hf_name)
        if hf_name not in names:
            continue
        ids = ids.to(torch.int64) - first_id
        mine = (ids >= 0) & (ids < values.numel())
        if not mine.any():
            continue
        tensor = pad_lora_to_rank(hf_name, hf_tensors.get_tensor(hf_name), adapter.rank, max_rank)
        if tensor.shape != ids.shape or tensor.dtype != values.dtype:
            raise ValueError(
                f"{hf_name}: the adapter holds {tensor.dtype} {tuple(tensor.shape)} at max rank, "
                f"slot {adapter.slot} exports {values.dtype} {tuple(ids.shape)}"
            )
        positions, tensor = ids[mine], tensor.to(values.device)[mine]
        # the export copies one element to several HF positions, e.g. a fused linear_qkv's lora_A to q/k/v
        repeated = filled[positions]
        if not torch.equal(values[positions[repeated]], tensor[repeated]):
            raise ValueError(f"{hf_name} disagrees with another HF tensor exported from the same slot parameter")
        values[positions] = tensor
        filled[positions] = True
    if exported != names:
        raise ValueError(
            f"the adapter's tensors are not slot {adapter.slot}'s layout: "
            f"missing {sorted(exported - names)[:3]}, unexpected {sorted(names - exported)[:3]}"
        )


def _load_checkpoint_name_remap(checkpoint):
    """Resolve export names from the checkpoint's architecture and tensor namespace."""
    path = Path(checkpoint)
    if not (path / "config.json").is_file() or not _has_safetensors(path):
        logger.warning(
            "Checkpoint %s has no local config or safetensors weights/index; preserving Bridge export names.",
            checkpoint,
        )
        return lambda name: name
    return get_param_name_remap_for_checkpoint(path)


def _has_safetensors(path: Path) -> bool:
    return (path / "model.safetensors.index.json").is_file() or any(path.glob("*.safetensors"))


def _select_bridge_checkpoint(args):
    """Use the HF trainer seed for export mappings when one is available."""
    if args.ref_load is not None and _has_safetensors(Path(args.ref_load)):
        return args.ref_load
    return args.hf_checkpoint


def _gather_export_dtypes(conversion_tasks):
    """Share owner parameter dtypes with PP receivers that have no local tensor."""
    local_dtypes = {
        task.global_param_name: (
            torch.float32
            if getattr(
                task.param_weight, "keep_in_fp32", False
            )  # config-access-exempt: Megatron annotates only parameters requiring float32 preservation
            else task.param_weight.dtype
        )
        for task in conversion_tasks
        if task is not None and task.param_weight is not None
    }
    pp = get_parallel_state().pp
    gathered = [local_dtypes]
    if pp.size > 1:
        gathered = [None] * pp.size
        dist.all_gather_object(gathered, local_dtypes, group=pp.gloo_group or pp.group)
    dtypes = {}
    for rank_dtypes in gathered:
        for name, dtype in rank_dtypes.items():
            if name in dtypes and dtypes[name] != dtype:
                raise ValueError(f"Inconsistent export dtype for {name}: {dtypes[name]} vs {dtype}")
            dtypes[name] = dtype
    return dtypes


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict, *, weight_dtypes=None):
    def _handle_one(task):
        if task is None:
            # no HF mapping (e.g. Gemma-4 post_shared_expert_layernorm)
            return task
        if weight_dtypes is not None:
            task = dataclasses.replace(task, weight_dtype=weight_dtypes[task.global_param_name])
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        if weight_dict_key not in new_weight_dict:
            # buffer-like params (Gemma-4 layer_scalar/scale) aren't in optimizer state; keep as-is
            return task
        new_param_weight = new_weight_dict[weight_dict_key]
        new_param_weight = new_param_weight.cuda()
        return dataclasses.replace(task, param_weight=new_param_weight)

    return _MapWithLen(_handle_one, vanilla_conversion_tasks)


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)
