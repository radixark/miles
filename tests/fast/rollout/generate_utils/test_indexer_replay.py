from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import numpy as np
import pybase64
import pytest

from miles.rollout.generate_utils.generate_endpoint_utils import get_indexer_topk_from_response
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
