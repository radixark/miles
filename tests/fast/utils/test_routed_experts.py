from types import SimpleNamespace

import numpy as np
import pytest
from sglang.srt.state_capturer.routed_experts_codec import RoutedExpertsEncoder

from miles.rollout.generate_utils.generate_endpoint_utils import _merge_routed_experts, request_routed_experts_patch
from miles.utils.routed_experts import decode_routed_experts
from miles.utils.types import Sample

_ENCODER = RoutedExpertsEncoder()


def _routing(num_tokens, num_layers=3, topk=4, num_experts=128, dtype=np.uint8, seed=0):
    rng = np.random.default_rng(seed)
    rows = np.argsort(rng.random((num_tokens, num_layers, num_experts)), axis=-1)[..., :topk]
    return np.sort(rows, axis=-1).astype(dtype)


def _meta(rows):
    return {"routed_experts": _ENCODER.encode(rows)}


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32])
def test_decode_keeps_the_captured_dtype(dtype):
    rows = _routing(9, dtype=dtype)
    out = decode_routed_experts(_meta(rows), 9, 3)
    assert out.dtype == dtype
    np.testing.assert_array_equal(out, rows)


def test_decode_drops_only_an_allowed_stop_edge_tail_row():
    rows = _routing(6)
    np.testing.assert_array_equal(decode_routed_experts(_meta(rows), 5, 3, allow_extra_tail_row=True), rows[:5])
    with pytest.raises(ValueError, match="do not match"):
        decode_routed_experts(_meta(rows), 5, 3)


def test_decode_rejects_wrong_layer_count():
    with pytest.raises(ValueError, match="do not match"):
        decode_routed_experts(_meta(_routing(4, num_layers=2)), 4, 3)


def test_decode_missing_payload_and_empty():
    assert decode_routed_experts({}, 4, 3) is None
    assert decode_routed_experts(_meta(_routing(0)), 0, 3).shape == (0, 3, 4)


def test_decode_rejects_all_zero_payload():
    with pytest.raises(AssertionError, match="all zeros"):
        decode_routed_experts(_meta(np.zeros((2, 3, 4), np.uint8)), 2, 3)


def _args(pause_generation_mode="in_place"):
    return SimpleNamespace(num_layers=3, pause_generation_mode=pause_generation_mode)


def test_additional_r3_requests_and_appends_only_new_rows():
    rows = _routing(10)
    sample = Sample(tokens=list(range(7)), rollout_routed_experts=rows[:6])
    payload = {"input_ids": list(range(8)), "return_routed_experts": True}
    request_routed_experts_patch(_args(), payload, sample)
    assert payload["routed_experts_start_len"] == 6

    sample.tokens = list(range(11))  # input (8) + 3 generated tokens -> 10 rows
    merged = _merge_routed_experts(_args(), sample, payload, {"meta_info": _meta(rows[6:])})
    np.testing.assert_array_equal(merged, rows)


def test_additional_r3_off_when_a_retract_may_recompute_the_prefix():
    sample = Sample(tokens=list(range(7)), rollout_routed_experts=_routing(6))
    payload = {"input_ids": list(range(8)), "return_routed_experts": True}
    request_routed_experts_patch(_args("retract"), payload, sample)
    assert "routed_experts_start_len" not in payload


def test_additional_r3_off_on_first_turn_and_without_replay():
    payload = {"input_ids": [1, 2], "return_routed_experts": True}
    request_routed_experts_patch(_args(), payload, Sample(tokens=[1, 2]))
    assert "routed_experts_start_len" not in payload
    payload = {"input_ids": [1, 2], "return_routed_experts": False}
    request_routed_experts_patch(_args(), payload, Sample(tokens=[1, 2], rollout_routed_experts=_routing(1)))
    assert "routed_experts_start_len" not in payload
