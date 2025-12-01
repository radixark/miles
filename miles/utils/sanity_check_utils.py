import logging

from miles.backends.megatron_utils.update_weight.common import named_params_and_buffers

from miles.utils.tensor_backper import compute_hash_tensor

logger = logging.getLogger(__name__)


class WeightChangeChecker:
    def __init__(self, args, model):
        self._args = args
        self._model = model
        self._initial_state = self._snapshot(self._args, self._model)

    def finalize(self):
        initial_state = self._initial_state
        final_state = self._snapshot(self._args, self._model)
        assert set(initial_state.keys()) == set(final_state.keys())
        print(f"{final_state=}")

        all_tensor_names = list(initial_state)
        unchanged_tensor_names = [name for name in all_tensor_names if initial_state[name] == final_state[name]]
        changed_tensor_names = sorted(list(set(all_tensor_names) - set(unchanged_tensor_names)))

        assert len(unchanged_tensor_names) == 0, f"{unchanged_tensor_names=} {changed_tensor_names=}"

        logger.info(f"WeightChangeChecker passed ({len(unchanged_tensor_names)=}, {len(changed_tensor_names)=})")

    @staticmethod
    def _snapshot(args, model):
        return {
            name: compute_hash_tensor(param)
            for name, param in named_params_and_buffers(args, model, convert_to_global_name=False)
        }
