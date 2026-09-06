from copy import deepcopy

import pytest

from miles.rollout.session.samples.codec import (
    COMPUTED_FIELDS_V2,
    decode_samples_and_merge_input_sample,
    encode_samples,
)
from miles.rollout.session.v2.postprocessor_hub.non_positive_attempts import non_positive_attempts
from miles.utils.types import Sample


def _sample(node_id: int = 1, token_count: int = 12) -> Sample:
    return Sample(
        tokens=list(range(token_count)),
        response_length=token_count - 2,
        loss_mask=([1, 1, 1, 0, 0, 1, 1, 1, 1, 1])[: token_count - 2],
        status=Sample.Status.COMPLETED,
        metadata={"leaf": {"node_id": node_id, "path_node_ids": [0, node_id]}},
    )


def _metadata(response_ids: list[str]) -> dict:
    return {
        "agent": {"reward": 1.0, "non_positive_advantage_response_ids": response_ids},
        "tree": {
            "nodes": [
                {"id": 0, "response_id": "bad", "completion_span": [2, 5]},
                {"id": 1, "response_id": "recovery", "completion_span": [7, 12]},
                {"id": 2, "response_id": "other-leaf", "completion_span": [7, 12]},
                {"id": 3, "response_id": "dropped", "completion_span": [7, 20]},
            ]
        },
    }


def test_maps_failed_ancestor_including_eos_without_changing_context_or_reward() -> None:
    sample = _sample()
    original = deepcopy(sample)
    metadata = _metadata(["bad", "dropped"])
    metadata["agent"]["non_positive_advantage_spans"] = [[0, 10]]  # untrusted coordinates
    assert non_positive_attempts([sample], metadata) == [sample]
    assert sample.metadata["non_positive_advantage_spans"] == [[0, 3]]
    assert sample.tokens == original.tokens
    assert sample.loss_mask == original.loss_mask
    assert sample.status == original.status
    assert sample.reward == 1.0


def test_wire_round_trip_preserves_response_spans() -> None:
    sample = non_positive_attempts([_sample()], _metadata(["bad"]))[0]
    payload = encode_samples([sample], {}, None, fields=COMPUTED_FIELDS_V2)
    reply = decode_samples_and_merge_input_sample(payload, Sample(index=7), fields=COMPUTED_FIELDS_V2)
    assert reply.samples[0].metadata["non_positive_advantage_spans"] == [[0, 3]]
    assert reply.samples[0].loss_mask == sample.loss_mask
    assert reply.samples[0].reward == sample.reward


def test_preserves_exactly_once_masking_of_shared_failed_completion() -> None:
    first, second = _sample(), _sample(node_id=2)
    non_positive_attempts([second, first], _metadata(["bad"]))
    assert first.loss_mask[:3] == [1, 1, 1]
    assert second.loss_mask[:3] == [0, 0, 0]
    assert first.metadata["non_positive_advantage_spans"] == [[0, 3]]
    assert second.metadata["non_positive_advantage_spans"] == [[0, 3]]


@pytest.mark.parametrize("token_count, expected", [(3, [[0, 1]]), (6, [[0, 3]]), (10, [[0, 3], [5, 8]])])
def test_spans_clip_to_retained_tokens(token_count: int, expected: list[list[int]]) -> None:
    sample = non_positive_attempts([_sample(token_count=token_count)], _metadata(["bad", "recovery"]))[0]
    assert sample.metadata["non_positive_advantage_spans"] == expected


@pytest.mark.parametrize("response_ids", [None, "bad", [None], [""], [17], ["unknown"]])
def test_incompatible_or_unmappable_metadata_fails(response_ids: object) -> None:
    metadata = _metadata([])
    metadata["agent"]["non_positive_advantage_response_ids"] = response_ids
    with pytest.raises(ValueError):
        non_positive_attempts([_sample()], metadata)


def test_duplicate_server_response_ids_fail() -> None:
    metadata = _metadata(["bad"])
    metadata["tree"]["nodes"][1]["response_id"] = "bad"
    with pytest.raises(ValueError, match="unique"):
        non_positive_attempts([_sample()], metadata)


def test_no_invalid_attempts_emit_explicit_empty_spans() -> None:
    sample = non_positive_attempts([_sample()], _metadata([]))[0]
    assert sample.metadata["non_positive_advantage_spans"] == []
