from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-b-cpu", labels=["replay"])

from types import SimpleNamespace

import numpy as np
import pybase64

from miles.rollout.generate_utils.generate_endpoint_utils import get_routed_experts_from_response


def _encode_int32(values: np.ndarray) -> str:
    return pybase64.b64encode(values.astype(np.int32).tobytes()).decode("ascii")


def test_reshapes_by_the_layer_count_sglang_reports():
    # 2 tokens x 3 MoE layers x top-2 from a 6-layer Nemotron-H.
    args = SimpleNamespace(num_layers=6)
    values = np.arange(2 * 3 * 2, dtype=np.int32)
    output = {"meta_info": {"routed_experts": _encode_int32(values), "routed_experts_num_layers": 3}}

    decoded = get_routed_experts_from_response(args, output, num_tokens=2)

    np.testing.assert_array_equal(decoded, values.reshape(2, 3, 2))


def test_sglang_without_the_layer_count_returns_every_layer():
    args = SimpleNamespace(num_layers=3)
    values = np.arange(2 * 3 * 2, dtype=np.int32)
    output = {"meta_info": {"routed_experts": _encode_int32(values)}}

    decoded = get_routed_experts_from_response(args, output, num_tokens=2)

    np.testing.assert_array_equal(decoded, values.reshape(2, 3, 2))
