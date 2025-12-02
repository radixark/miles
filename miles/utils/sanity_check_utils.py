import logging
from collections.abc import Callable, Mapping

import torch

from miles.utils.tensor_backper import compute_hash_tensor

logger = logging.getLogger(__name__)


class WeightChangeChecker:
    def __init__(self, weights_getter: Callable[[], Mapping[str, torch.Tensor]]):
        self.weights_getter = weights_getter
        self.last_state = None

    def step(self):
        curr_state = _snapshot(self.weights_getter())
        if self.last_state is not None:
            _check(self.last_state, curr_state)
        self.last_state = curr_state


def _snapshot(weights_dict: Mapping[str, torch.Tensor]):
    return {name: compute_hash_tensor(param) for name, param in weights_dict.items()}


def _check(state_a: dict[str, int], state_b: dict[str, int]):
    assert set(state_a.keys()) == set(state_b.keys())
    all_tensor_names = list(state_a)

    unchanged_tensor_names = [name for name in all_tensor_names if state_a[name] == state_b[name]]
    changed_tensor_names = sorted(list(set(all_tensor_names) - set(unchanged_tensor_names)))

    # TODO
    # TODO
    # TODO
    # assert (
    #     len(unchanged_tensor_names) == 0
    # ), f"{unchanged_tensor_names=} {changed_tensor_names=} {state_a=} {state_b=}"
    # logger.info(f"WeightChangeChecker passed ({len(unchanged_tensor_names)=}, {len(changed_tensor_names)=})")
    print(f"hi {unchanged_tensor_names=} {changed_tensor_names=} {state_a=} {state_b=}")
