import dataclasses
import json
import os

from miles.backends.megatron_utils.lora_utils import is_lora_weight_name
from miles.utils import megatron_bridge_utils
from miles.utils.iter_utils import chunk_named_params_by_size

from ..megatron_to_hf import postprocess_hf_param
from ..megatron_to_hf.processors import quantize_params
from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorBridge(HfWeightIteratorBase):
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

    def get_hf_weight_chunks(self, megatron_local_weights, weight_type: str = "base"):
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            if weight_type == "lora":
                named_weights = self._bridge.export_adapter_weights(
                    self.model,
                    cpu=False,
                    show_progress=False,
                )
            elif weight_type == "base":
                conversion_tasks = self._bridge.get_conversion_tasks(self.model)
                conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)
                named_weights = self._bridge.export_hf_weights(
                    self.model,
                    cpu=False,
                    conversion_tasks=conversion_tasks,
                    merge_adapter_weights=False,
                )

            # Apply postprocess + quantization (when targeting a quantized rollout,
            # e.g. FP8 sglang). Base weights are quantized to match the rollout's
            # storage format so update_weights_from_tensor lands real weight + scale
            # pairs; LoRA adapters are passed through unchanged.
            named_weights = self._postprocess_and_quantize(named_weights, weight_type)

            if weight_type == "base":
                named_weights = ((n, t) for n, t in named_weights if not is_lora_weight_name(n))
            elif weight_type == "lora":
                named_weights = ((n, t) for n, t in named_weights if is_lora_weight_name(n))

            yield from chunk_named_params_by_size(named_weights, chunk_size=self.args.update_weight_buffer_size)

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
                yield from quantize_params(self.args, qmegatron_name, [(hf_name, weight)], self.quantization_config)
            else:
                yield hf_name, weight


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
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        assert (
            weight_dict_key in new_weight_dict
        ), f"{weight_dict_key=} not in new_weight_dict ({task.vp_stage=}, {task.param_name=}, {list(new_weight_dict)=})"

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
