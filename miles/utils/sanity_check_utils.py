import torch
from miles.backends.megatron_utils.update_weight.common import named_params_and_buffers


class WeightChangeChecker:
    def __init__(self, args, model):
        self._args = args
        self._model = model
        self._initial_state = self._snapshot(self._args, self._model)

    def finalize(self):
        initial_state = self._initial_state
        final_state = self._snapshot(self._args, self._model)
        assert set(initial_state.keys()) == set(final_state.keys())

        bad_names = [name for name in initial_state if initial_state[name] == final_state[name]]
        assert len(bad_names) == 0, f"These tensors are not changed after training: {bad_names}"

    @staticmethod
    def _snapshot(args, model):
        return {
            name: _compute_hash(param)
            for name, param in named_params_and_buffers(args, model, convert_to_global_name=False)
        }


def _compute_hash(x: torch.Tensor):
    return TODO
