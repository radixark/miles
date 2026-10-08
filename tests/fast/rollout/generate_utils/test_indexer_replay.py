from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import numpy as np
import pybase64
import pytest

from miles.rollout.generate_utils.generate_endpoint_utils import (
    get_indexer_topk_from_response,
    get_routed_experts_from_response,
)
from miles.rollout.generate_utils.output_store import ReplayOutputs
from miles.utils.types import Sample


def _encode_int32(values: np.ndarray) -> str:
    return pybase64.b64encode(values.astype(np.int32).tobytes()).decode("ascii")


def test_get_indexer_topk_from_response_decodes_using_meta_info_num_layers():
    args = SimpleNamespace()
    sample = Sample(tokens=[1, 2, 3])
    values = np.arange(2 * 2 * 3, dtype=np.int32)
    output = {
        "meta_info": {
            "indexer_topk": _encode_int32(values),
            "indexer_topk_num_layers": 2,
        }
    }

    decoded = get_indexer_topk_from_response(args, output, sample)

    np.testing.assert_array_equal(decoded, values.reshape(2, 2, 3))


def test_get_indexer_topk_from_response_returns_none_when_absent():
    args = SimpleNamespace()
    sample = Sample(tokens=[1, 2, 3])
    output = {"meta_info": {}}

    assert get_indexer_topk_from_response(args, output, sample) is None


def test_get_indexer_topk_from_response_rejects_missing_num_layers():
    args = SimpleNamespace()
    sample = Sample(tokens=[1, 2, 3])
    values = np.arange(2 * 2 * 3, dtype=np.int32)
    output = {"meta_info": {"indexer_topk": _encode_int32(values)}}

    with pytest.raises(AssertionError, match="indexer_topk_num_layers"):
        get_indexer_topk_from_response(args, output, sample)


def test_an_output_store_response_takes_the_layer_count_from_the_array_shape():
    """Store responses carry no indexer_topk_num_layers; the stream check must still run."""
    sample = Sample(tokens=[1, 2, 3])
    indexer_topk = np.arange(2 * 2 * 3, dtype=np.int32).reshape(2, 2, 3)
    replay = ReplayOutputs(indexer_topk=indexer_topk)

    decoded = get_indexer_topk_from_response(
        SimpleNamespace(rollout_indexer_topk_num_streams=2), {"meta_info": {}}, sample, replay=replay
    )

    np.testing.assert_array_equal(decoded, indexer_topk)
    with pytest.raises(AssertionError, match="2 streams but the model has 3"):
        get_indexer_topk_from_response(
            SimpleNamespace(rollout_indexer_topk_num_streams=3), {"meta_info": {}}, sample, replay=replay
        )


def test_an_output_store_array_must_cover_every_token_but_the_last():
    replay = ReplayOutputs(indexer_topk=np.zeros((3, 2, 3), dtype=np.int32))

    with pytest.raises(ValueError, match="expected \\(2, 2, topk\\)"):
        get_indexer_topk_from_response(SimpleNamespace(), {"meta_info": {}}, Sample(tokens=[1, 2, 3]), replay=replay)


@pytest.mark.parametrize(
    ("routed_experts", "error", "match"),
    [
        (np.zeros((2, 4, 2), dtype=np.int32), ValueError, "expected \\(2, 3, topk\\)"),
        (np.zeros((2, 3, 2), dtype=np.int32), AssertionError, "all zeros"),
    ],
)
def test_output_store_routed_experts_get_the_inline_checks(routed_experts, error, match):
    with pytest.raises(error, match=match):
        get_routed_experts_from_response(
            SimpleNamespace(num_layers=3), {"meta_info": {}}, 2, replay=ReplayOutputs(routed_experts=routed_experts)
        )
