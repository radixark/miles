import dataclasses

import torch

from miles.backends.megatron_utils.lora_utils import is_lora_weight_name
from miles.utils import megatron_bridge_utils
from miles.utils.iter_utils import chunk_named_params_by_size

from ..megatron_to_hf import postprocess_hf_param, postprocess_q_lora_converted_tensors
from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        import miles_plugins.megatron_bridge  # noqa: F401

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)

    def get_hf_weight_chunks(self, megatron_local_weights, weight_type: str = "base"):
        # TODO: support quantization (e.g. modify megatron-bridge to provide megatron param name)
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
                )

            # TODO: verify if postprocess_hf_param is needed for LoRA weights
            def _postprocessed_weights():
                cached_q_lora_tensors = {}
                for named_weight in named_weights:
                    if len(named_weight) == 3:
                        hf_param_name, weight, megatron_param_name = named_weight
                    elif len(named_weight) == 2:
                        hf_param_name, weight = named_weight
                        megatron_param_name = hf_param_name
                    else:
                        raise ValueError(f"Unexpected bridge weight tuple: {named_weight!r}")

                    processed_weight = postprocess_hf_param(
                        args=self.args,
                        megatron_param_name=megatron_param_name,
                        hf_param_name=hf_param_name,
                        param=weight,
                    )
                    converted_named_params = postprocess_q_lora_converted_tensors(
                        args=self.args,
                        converted_named_tensors=[(hf_param_name, processed_weight)],
                        cached_tensors=cached_q_lora_tensors,
                    )
                    for converted_name, converted_weight in converted_named_params:
                        if weight_type == "base" and is_lora_weight_name(converted_name):
                            continue
                        if weight_type == "lora" and not is_lora_weight_name(converted_name):
                            continue
                        if torch.cuda.is_available():
                            target_device = torch.device("cuda", torch.cuda.current_device())
                            if converted_weight.device != target_device:
                                converted_weight = converted_weight.to(target_device, non_blocking=True)
                        yield converted_name, converted_weight

            yield from chunk_named_params_by_size(
                _postprocessed_weights(), chunk_size=self.args.update_weight_buffer_size
            )


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
