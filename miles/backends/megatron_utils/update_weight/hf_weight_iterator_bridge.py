from miles.utils.iter_utils import chunk_named_params_by_size
from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        self._bridge = AutoBridge.from_hf_pretrained(TODO_what)
        self._vanilla_conversion_tasks = self._bridge.build_conversion_tasks(TODO_hf_config, self.model)

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

        return TODO

    return [_handle_one(task) for task in vanilla_conversion_tasks]
