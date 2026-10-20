"""Exercise score-centering metadata through native generation."""

import asyncio
import math

import numpy as np
import pytest
from tests.fast.fixtures.score_centering_fixtures import _args, _meta

from miles.backends.training_utils.data.replay import materialize_replay_columns
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload, update_sample_from_response
from miles.rollout.generate_utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sample
from miles.utils.types import Sample


def test_native_generate_producer_appends_each_call() -> None:
    args = _args(
        rollout_max_response_len=20,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_output_store=False,
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
    validate_rollout_topk_logprobs_sample(sample, 3)


@pytest.mark.parametrize("candidate_count", [0, 3])
def test_filtered_native_generation_records_post_filter_distribution(candidate_count: int) -> None:
    args = _args(
        rollout_top_logprobs_num=candidate_count,
        rollout_top_p=0.6,
        rollout_top_k=3,
        rollout_sampling_logprobs_mode="support",
        use_sampling_support_replay=True,
        rollout_max_response_len=20,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_output_store=False,
        sglang_speculative_algorithm=None,
    )
    payload, status = compute_request_payload(args, [0, 1], {"temperature": 0.7, "top_p": 0.6, "top_k": 3})
    assert status is None
    assert payload["return_sampling_mask"]
    assert payload.get("sampling_logprobs_mode") == ("support" if candidate_count else None)
    assert "top_logprobs_num" not in payload
    meta = {
        "output_token_logprobs": [(math.log(0.4), 2, None)],
        "output_token_sampling_mask": [[3, 2]],
        "output_token_sampling_logprobs": (
            [[math.log(3 / 7), math.log(4 / 7)]] if candidate_count else [math.log(4 / 7)]
        ),
        "finish_reason": {"type": "stop"},
    }
    sample = Sample(index=0, group_index=0, reward=1.0, status=Sample.Status.COMPLETED)
    asyncio.run(update_sample_from_response(args, sample, payload, {"text": "2", "meta_info": meta}, True))
    # The rollout keeps the sampled tokens' post-filter log-probs; the replay assembly derives the candidates.
    assert sample.rollout_topk_token_ids is None
    assert sample.rollout_log_probs == pytest.approx([math.log(4 / 7)])
    data = convert_samples_to_train_data(args, [sample], {}, None, None)
    materialize_replay_columns(args, data)
    if candidate_count:
        # Score centering shares the ordered support IDs instead of a dense candidate array.
        assert data["rollout_topk_lengths"][0].tolist() == [2]
        np.testing.assert_allclose(np.exp(data["rollout_topk_log_probs"][0][0, :2]), [3 / 7, 4 / 7], rtol=1e-6)
    else:
        assert "rollout_topk_token_ids" not in data and "rollout_topk_log_probs" not in data
    np.testing.assert_allclose(data["rollout_log_probs"][0], [math.log(4 / 7)], rtol=1e-6)


def test_greedy_evaluation_does_not_collect_training_candidates() -> None:
    args = _args(
        rollout_max_response_len=20,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_output_store=False,
        sglang_speculative_algorithm=None,
    )
    payload, _ = compute_request_payload(args, [0, 1], {"temperature": 0.0}, evaluation=True)
    assert payload["sampling_params"]["temperature"] == 0.0 and "top_logprobs_num" not in payload
    meta = _meta([2, 3], [0.5, 0.25])
    del meta["output_top_logprobs"]
    sample = Sample()
    asyncio.run(update_sample_from_response(args, sample, payload, {"text": "eval", "meta_info": meta}))
    assert sample.rollout_topk_token_ids is None
