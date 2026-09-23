import dataclasses
import inspect
import itertools
import json
import os

from miles.backends.megatron_utils.update_weight.hf_weight_iterator import (
    MegatronHfWeightIteratorBase,
    _iter_mm_tower_units,
)
from miles.utils import megatron_bridge_utils
from miles.utils.lora import is_lora_weight_name

from ..megatron_to_hf import postprocess_hf_param
from ..megatron_to_hf.processors import quantize_params
from ..misc_utils import strip_param_name_prefix


class HfWeightIteratorBridge(MegatronHfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)

        if (
            self.quantization_config is not None
            and self.quantization_config.get("quant_method") == "compressed-tensors"
        ):
            quantized_basenames = _load_quantized_param_basenames(self.args.hf_checkpoint)
            if quantized_basenames is not None:
                # Quantize exactly the params the checkpoint stores packed; the
                # published ignore list of multimodal checkpoints (e.g.
                # Kimi-K2.5 VL) omits vision_tower/mm_projector, so it cannot
                # be trusted as the sole quantization criterion.
                self.quantization_config = {
                    **self.quantization_config,
                    "_miles_quantized_basenames": quantized_basenames,
                }

    def _iter_hf_param_units(self, weights, *, materialize):
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)
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
        yield from _iter_mm_tower_units(self.args, materialize=materialize)

    def _export_pp_local_lora(self, adapter):
        if adapter is None:
            return self._export_current_adapter()

        from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

        from ..lora.slots import slice_lora_to_rank

        with expose_adapter_slot(self.model, adapter.slot):
            named_tensors = self._export_current_adapter()
        return [(h, slice_lora_to_rank(h, w, adapter.rank)) for h, w in named_tensors]

    def _export_current_adapter(self) -> list:
        with megatron_bridge_utils.patch_megatron_model(self.model):
            named_weights = self._bridge.export_adapter_weights(
                self.model,
                cpu=False,
                show_progress=False,
                **self._source_name_kwargs(self._bridge.export_adapter_weights),
            )
            named_weights = self._postprocess_and_quantize(named_weights, "lora")
            return [(h, w) for h, w, _m in named_weights if is_lora_weight_name(h)]

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
            hf_name = hf_param_name.replace(".base_layer.", ".")
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
                    self.args, qmegatron_name, [(hf_name, weight)], self.quantization_config
                ):
                    yield q_hf_name, q_weight, megatron_param_names
            else:
                yield hf_name, weight, megatron_param_names


def _load_quantized_param_basenames(hf_checkpoint):
    """Base names of params stored packed (`<base>.weight_packed`) in the checkpoint, or None if unknown."""
    index_path = os.path.join(hf_checkpoint, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return None
    with open(index_path) as f:
        names = json.load(f)["weight_map"]
    return {n.removesuffix(".weight_packed") for n in names if n.endswith(".weight_packed")}


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    def _handle_one(task):
        if task is None:
            # no HF mapping (e.g. Gemma-4 post_shared_expert_layernorm)
            return task
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
