import dataclasses

from miles.utils.iter_utils import chunk_named_params_by_size
from miles.utils import megatron_bridge_utils
from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint)
        with megatron_bridge_utils.patch_megatron_model(self.model):
            self._vanilla_conversion_tasks = self._bridge.build_conversion_tasks(self._bridge.hf_pretrained, self.model)

    def get_hf_weight_chunks(self, megatron_local_weights):
        # TODO support quantization (e.g. modify megatron-bridge to provide megatron param name)
        conversion_tasks = _change_conversion_tasks_weights(self._vanilla_conversion_tasks, megatron_local_weights)
        vanilla_named_weights = self._bridge.export_hf_weights(
            self.model, cpu=False, conversion_tasks=conversion_tasks
        )
        yield from chunk_named_params_by_size(vanilla_named_weights, chunk_size=self.args.update_weight_buffer_size)


def _change_conversion_tasks_weights(vanilla_conversion_tasks, new_weight_dict):
    def _handle_one(task):
        if task.param_weight is None:
            return task

        assert task.param_name in new_weight_dict, f"{task.param_name=} not in new_weight_dict"
        new_param_weight = new_weight_dict[task.param_name]

        return dataclasses.replace(task, param_weight=new_param_weight)

    return [_handle_one(task) for task in vanilla_conversion_tasks]
