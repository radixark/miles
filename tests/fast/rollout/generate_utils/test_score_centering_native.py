"""Exercise score-centering metadata through native generation."""

import asyncio

import numpy as np
from tests.fast.fixtures.score_centering_fixtures import _args, _meta

from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload, update_sample_from_response
from miles.rollout.generate_utils.score_centering import validate_score_centering_sample
from miles.utils.types import Sample


def test_native_generate_producer_appends_each_call() -> None:
    args = _args(
        rollout_max_response_len=20,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        sglang_speculative_algorithm=None,
    )
    payload, status = compute_request_payload(args, [0, 1], {})
    assert status is None and payload["top_logprobs_num"] == 3 and payload["return_logprob"]
    sample = Sample()
    for output, probabilities in (([2, 3], [0.5, 0.25]), ([4, 5], [0.55, 0.2])):
        asyncio.run(
            update_sample_from_response(
                args, sample, payload, {"text": str(output), "meta_info": _meta(output, probabilities)}, True
            )
        )
    assert sample.tokens == [0, 1, 2, 3, 4, 5]
    np.testing.assert_array_equal(sample.rollout_topk_token_ids[:, 0], [2, 2, 4, 4])
    validate_score_centering_sample(sample, 3)


def test_greedy_evaluation_does_not_collect_training_candidates() -> None:
    args = _args(
        rollout_max_response_len=20,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        sglang_speculative_algorithm=None,
    )
    payload, _ = compute_request_payload(args, [0, 1], {"temperature": 0.0}, evaluation=True)
    assert payload["sampling_params"]["temperature"] == 0.0 and "top_logprobs_num" not in payload
    meta = _meta([2, 3], [0.5, 0.25])
    del meta["output_top_logprobs"]
    sample = Sample()
    asyncio.run(update_sample_from_response(args, sample, payload, {"text": "eval", "meta_info": meta}))
    assert sample.rollout_topk_token_ids is None
