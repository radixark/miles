import torch
from miles.backends.megatron_utils.update_weight.common import named_params_and_buffers


class WeightChangeChecker:
    def __init__(self, model):
        TODO

    def finalize(self):
        TODO

    @staticmethod
    def _snapshot(args, model):
        return {
            name: _compute_hash(param)
            for name, param in named_params_and_buffers(args, model, convert_to_global_name=False)
        }


def _compute_hash(x: torch.Tensor):
    return TODO
