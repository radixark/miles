"""Session configuration must distinguish training calls from evaluation."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from tests.fast.fixtures.score_centering_fixtures import _args, _meta, _Tokenizer
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.generate_utils.score_centering import validate_score_centering_sample
from miles.rollout.session.core import SessionCore
from miles.rollout.session.errors import MessageValidationError
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.request_args import prepare_chat_request
from miles.rollout.session.samples.merge import compute_samples_from_openai_records
from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2.core import SessionCoreV2
from miles.rollout.session.v2.session_state import SessionRegistryV2


@pytest.mark.parametrize(
    "registry_type,core_type", [(SessionRegistry, SessionCore), (SessionRegistryV2, SessionCoreV2)]
)
def test_score_centering_training_and_evaluation_sessions(registry_type: type, core_type: type) -> None:
    config = make_session_server_config(loss_type="score_centering", rollout_temperature=0.7)
    tokenizer = SimpleNamespace(
        create_comparator=lambda: None,
        chat_template_kwargs={},
        resolve_request_args=lambda request, **kwargs: request,
    )
    registry = registry_type(tokenizer=None, tito_tokenizer=tokenizer)
    core = core_type(None, registry, config)
    for evaluation in (False, True):
        response = asyncio.run(core.create_session(evaluation=evaluation))
        session = registry.get_session(json.loads(response.body)["session_id"])
        assert session.evaluation is evaluation
        if evaluation:
            request = prepare_chat_request(
                {"temperature": 0}, tokenizer, config=config, turn_args=None, evaluation=session.evaluation
            ).body
            assert request["temperature"] == 0 and "top_logprobs" not in request
        else:
            request = prepare_chat_request(
                {}, tokenizer, config=config, turn_args=None, evaluation=session.evaluation
            ).body
            assert request["top_logprobs"] == 128 and request["temperature"] == 0.7
            with pytest.raises(MessageValidationError, match="temperature"):
                prepare_chat_request(
                    {"temperature": 0}, tokenizer, config=config, turn_args=None, evaluation=session.evaluation
                )


def test_session_producer_trims_candidates_with_tito_tokens() -> None:
    records = []
    for prompt, output, probabilities in (([0, 1], [2, 3], [0.5, 0.25]), ([0, 1, 2, 6], [4, 5], [0.55, 0.2])):
        records.append(
            SessionRecord(
                timestamp=2.0,
                request_timestamp=1.0,
                method="POST",
                path="v1/chat/completions",
                status_code=200,
                request={"input_ids": prompt, "top_logprobs": 3},
                response={"choices": [{"meta_info": _meta(output, probabilities), "finish_reason": "stop"}]},
            )
        )
    samples = compute_samples_from_openai_records(
        _args(save_debug_trajectory_data=None, sglang_speculative_algorithm=None),
        records,
        _Tokenizer(),
        accumulated_token_ids=[0, 1, 2, 6, 4, 5],
        max_trim_tokens=1,
    )
    assert samples[0].response_length == 1
    assert samples[0].rollout_topk_token_ids.shape == (1, 3)
    merged = merge_samples(samples, _Tokenizer())
    assert merged.loss_mask == [1, 0, 1, 1]
    validate_score_centering_sample(merged, 3)


@pytest.mark.parametrize("override", [{"temperature": 0.8}, {"top_p": 0.9}, {"min_p": 0.1}, {"logit_bias": {"1": 2}}])
def test_final_model_sampling_rules_are_validated(override: dict) -> None:
    config = make_session_server_config(loss_type="score_centering", rollout_temperature=0.7)
    tokenizer = SimpleNamespace(resolve_request_args=lambda request, **kwargs: request | override)
    client_args = {"messages": [{"role": "user", "content": "Solve the problem."}]}
    with pytest.raises(MessageValidationError, match="Score centering"):
        prepare_chat_request(client_args, tokenizer, config=config, turn_args=None)
    assert client_args == {"messages": [{"role": "user", "content": "Solve the problem."}]}


def test_nullable_sampling_fields_receive_explicit_defaults() -> None:
    config = make_session_server_config(loss_type="score_centering", rollout_temperature=0.7)
    tokenizer = SimpleNamespace(resolve_request_args=lambda request, **kwargs: request)
    prepared = prepare_chat_request(
        {"temperature": None, "top_p": None, "top_k": None, "min_p": None},
        tokenizer,
        config=config,
        turn_args=None,
    )
    assert {key: prepared.body[key] for key in ("temperature", "top_p", "top_k", "min_p")} == {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
    }
