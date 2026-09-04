import dataclasses
import itertools

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
            # One unit per megatron param: quantize emits weight + scales
            # consecutively, so grouping by source name keeps them together.
            for _megatron_name, group in itertools.groupby(named_weights, key=lambda item: item[2]):
                unit = [(h, w) for h, w, _m in group if not is_lora_weight_name(h)]
                if unit:
                    yield unit
        yield from _iter_mm_tower_units(self.args, materialize=materialize)

    def _export_pp_local_lora(self, adapter, weights):
        if adapter is None:
            return self._export_current_adapter()

        from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

        from ..multi_lora_utils import slice_lora_to_rank

        with expose_adapter_slot(self.model, adapter.slot):
            named_tensors = self._export_current_adapter()
        return [(h, slice_lora_to_rank(h, w, adapter.config.rank)) for h, w in named_tensors]

    def _export_current_adapter(self) -> list:
        with megatron_bridge_utils.patch_megatron_model(self.model):
            named_weights = self._bridge.export_adapter_weights(self.model, cpu=False, show_progress=False)
            named_weights = self._postprocess_and_quantize(named_weights, "lora")
            return [(h, w) for h, w, _m in named_weights if is_lora_weight_name(h)]

    def _postprocess_and_quantize(self, named_weights, weight_type: str):
        for hf_param_name, weight, megatron_param_name in named_weights:
            hf_name = hf_param_name.replace(".base_layer.", ".")
            weight = postprocess_hf_param(
                args=self.args,
                megatron_param_name=megatron_param_name,
                hf_param_name=hf_name,
                param=weight,
            )
            if weight_type == "base" and self.quantization_config is not None:
                # quantize_params expects the megatron name with the `module.module.`
                # prefix that the direct iterator uses; the bridge yields it without.
                qmegatron_name = f"module.module.{megatron_param_name}"
                for q_hf_name, q_weight in quantize_params(
                    self.args, qmegatron_name, [(hf_name, weight)], self.quantization_config
                ):
                    yield q_hf_name, q_weight, megatron_param_name
            else:
                yield hf_name, weight, megatron_param_name


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
