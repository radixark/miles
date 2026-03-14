"""Tests for OpenAI endpoint sample reconstruction."""

from unittest.mock import MagicMock

from miles.rollout.generate_utils.openai_endpoint_utils import compute_samples_from_openai_records
from miles.router.session.sessions import SessionRecord
from miles.utils.types import Sample


def _mock_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.decode.side_effect = lambda token_ids: "".join(f"[{token_id}]" for token_id in token_ids)
    return tokenizer


def _make_input_sample() -> Sample:
    return Sample(
        group_index=0,
        index=0,
        prompt="test prompt",
        tokens=[],
        response="",
        response_length=0,
        status=Sample.Status.PENDING,
        label="test",
        reward=1.0,
    )


def _make_record(
    *,
    input_token_ids: list[int],
    output_token_ids: list[int],
    output_log_probs: list[float],
    finish_reason: str = "stop",
    request_input_ids: list[int] | None = None,
) -> SessionRecord:
    return SessionRecord(
        timestamp=0.0,
        method="POST",
        path="/v1/chat/completions",
        status_code=200,
        request={
            "messages": [{"role": "user", "content": "hello"}],
            **({"input_ids": request_input_ids} if request_input_ids is not None else {}),
        },
        response={
            "choices": [
                {
                    "input_token_ids": input_token_ids,
                    "message": {"role": "assistant", "content": "response"},
                    "finish_reason": finish_reason,
                    "logprobs": {
                        "content": [
                            {"token_id": token_id, "logprob": log_prob, "token": f"t{token_id}"}
                            for token_id, log_prob in zip(output_token_ids, output_log_probs, strict=True)
                        ]
                    },
                }
            ]
        },
    )


class TestComputeSamplesFromOpenAIRecords:
    def test_single_record_builds_sample_from_choice_input_token_ids(self) -> None:
        samples = compute_samples_from_openai_records(
            _make_input_sample(),
            [
                _make_record(
                    input_token_ids=[1, 2, 3],
                    output_token_ids=[10, 11],
                    output_log_probs=[-0.5, -0.6],
                )
            ],
            _mock_tokenizer(),
        )

        assert len(samples) == 1
        sample = samples[0]
        assert sample.tokens == [1, 2, 3, 10, 11]
        assert sample.rollout_log_probs == [-0.5, -0.6]
        assert sample.response == "[10][11]"
        assert sample.response_length == 2
        assert sample.loss_mask == [1, 1]
        assert sample.status == Sample.Status.COMPLETED

    def test_matching_request_input_ids_are_accepted(self) -> None:
        samples = compute_samples_from_openai_records(
            _make_input_sample(),
            [
                _make_record(
                    input_token_ids=[4, 5],
                    output_token_ids=[6],
                    output_log_probs=[-0.1],
                    request_input_ids=[4, 5],
                )
            ],
            _mock_tokenizer(),
        )

        assert samples[0].tokens == [4, 5, 6]

    def test_finish_reason_length_marks_sample_truncated(self) -> None:
        samples = compute_samples_from_openai_records(
            _make_input_sample(),
            [
                _make_record(
                    input_token_ids=[7, 8],
                    output_token_ids=[9],
                    output_log_probs=[-0.2],
                    finish_reason="length",
                )
            ],
            _mock_tokenizer(),
        )

        assert samples[0].status == Sample.Status.TRUNCATED
