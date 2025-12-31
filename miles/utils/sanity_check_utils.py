import hashlib
import logging
from collections.abc import Callable, Mapping

import torch

from sglang.srt.debug_utils.dumper import get_tensor_info

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
    return {name: _compute_hash_tensor_slow(param) for name, param in weights_dict.items()}


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


def _compute_hash_tensor_slow(x: torch.Tensor):
    # TODO
    info = get_tensor_info(x)
    x = x.contiguous()
    x = x.view(-1)
    x = x.view(torch.uint8)
    np_array = x.cpu().numpy()
    byte_string = np_array.tobytes()
    hash_object = hashlib.md5(byte_string)
    # TODO
    # TODO
    # TODO
    return hash_object.hexdigest() + ";" + info
