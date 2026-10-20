from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["replay"])

import numpy as np
import pybase64
import pytest

from miles.utils.replay_index import ReplayDataError, ReplayIndex


def _encode_int32(values: np.ndarray) -> str:
    return pybase64.b64encode(values.astype(np.int32).tobytes()).decode("ascii")


def _parse(meta_info: dict, *, num_streams: int | None = None) -> ReplayIndex:
    return ReplayIndex.from_meta_info(
        meta_info,
        via_store=False,
        routed_experts_rows=2,
        num_layers=None,
        indexer_topk_rows=2,
        indexer_num_streams=num_streams,
        return_sampling_mask=False,
        support_logprobs=False,
        num_output_tokens=1,
        aborted=False,
    )


def test_indexer_topk_decodes_using_meta_info_num_layers():
    values = np.arange(2 * 2 * 3, dtype=np.int32)
    index = _parse({"indexer_topk": _encode_int32(values), "indexer_topk_num_layers": 2})

    np.testing.assert_array_equal(index.indexer_topk.materialize_rows({}), values.reshape(2, 2, 3))


def test_indexer_topk_is_none_when_absent():
    assert _parse({}).indexer_topk is None


def test_indexer_topk_rejects_missing_num_layers():
    values = np.arange(2 * 2 * 3, dtype=np.int32)
    with pytest.raises(ReplayDataError, match="indexer_topk_num_layers"):
        _parse({"indexer_topk": _encode_int32(values)})


def test_indexer_topk_rejects_stream_count_mismatch():
    values = np.arange(2 * 2 * 3, dtype=np.int32)
    with pytest.raises(ReplayDataError, match="2 streams but the model has 3"):
        _parse({"indexer_topk": _encode_int32(values), "indexer_topk_num_layers": 2}, num_streams=3)
