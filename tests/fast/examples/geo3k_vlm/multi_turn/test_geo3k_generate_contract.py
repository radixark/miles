"""Exit paths of the geo3k multi-turn rollout, with the environment and the engine call replaced.

    turn k:  request ─▶ response ─▶ [a] ─▶ env.step ─▶ [b] ─▶ append observation ─▶ [c] ─▶ turn k+1
        [a] token space: expanded prompt ≥ --max-seq-len → roll the turn back (turn ≥ 1; a first prompt
            past the limit keeps no response); len(tokens) > limit → trim
            both paths: finish length → TRUNCATED; budget ≤ 0 → TRUNCATED
        [b] token space only: last turn → COMPLETED; budget − len(obs) ≤ 0 → TRUNCATED;
            otherwise checkpoint + append
        [c] legacy only (the processor expands the observation before the append, so the budget is exact):
            budget ≤ 0 → TRUNCATED; last turn → COMPLETED

    legacy      test_generation_uses_caller_state_and_closes_environment_on_every_return   budget 0 / finish length
    token space test_a_token_space_turn_records_the_engines_routed_experts                  request asks for them;
                                                                                             rows = served prompt + response − 1
                test_a_later_token_space_turn_appends_only_the_rows_after_the_held_ones   request starts at the held
                                                                                             rows; they stay, new ones append
                test_a_response_past_max_seq_len_is_trimmed_to_it                         [a] trim
                test_an_observation_whose_media_leaves_no_room_is_rolled_back             [a] roll back, two limits
                TestObservationStaysOutOfAnEndingSample                                   [b]
                    test_the_last_turn_ends_before_its_observation
                    test_an_observation_with_no_room_for_a_response_ends_the_sample_truncated
                    test_an_observation_that_leaves_room_is_appended_and_answered
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pybase64
import pytest
from examples.geo3k_vlm.multi_turn import rollout

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import adopt_expanded_prompt_ids
from miles.utils.types import Sample


def _state(max_seq_len: int | None = None, token_space: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        args=SimpleNamespace(
            partial_rollout=False, multimodal_token_space_processing=token_space, max_seq_len=max_seq_len
        ),
        tokenizer=SimpleNamespace(decode=Mock(return_value="answer")),
        processor=None,
    )


async def _run(
    monkeypatch: pytest.MonkeyPatch, *, env, state, sample, budget: int, max_turns: int, infer, env_step=None
):
    """Drive generate() with the resources, start state and engine call replaced; returns its output."""
    monkeypatch.setattr(
        rollout, "_initialize_resources", lambda args, sample: (env, None, {"max_turns": max_turns}, "url")
    )
    monkeypatch.setattr(rollout, "_prepare_start_state", Mock(return_value=(None, [], budget, [])))
    monkeypatch.setattr(rollout, "_run_inference_step", infer)
    if env_step is not None:
        monkeypatch.setattr(rollout, "_process_env_step", env_step)
    return await rollout.generate(
        GenerateFnInput(state=state, sample=sample, sampling_params={"max_new_tokens": 5}, evaluation=False)
    )


@pytest.mark.parametrize("token_space", [True, False])
async def test_a_token_space_turn_records_the_engines_routed_experts(
    monkeypatch: pytest.MonkeyPatch, token_space: bool
) -> None:
    """A token-space request keeps the routed experts of the prompt the engine served plus the response,
    so replay (R3) sees one row per token of the expanded sequence; the legacy request leaves them alone."""
    layers, topk, response = 2, 3, [7, 8]
    sample = Sample(prompt="", tokens=[1, -101, 2])
    served_prompt = [1, 5, 5, 5, 5, 2]  # the placeholder became a 4-token image run
    rows = np.arange((len(served_prompt) + len(response) - 1) * layers * topk, dtype=np.int32)
    output = {
        "text": "answer",
        "prompt_token_ids": served_prompt,
        "meta_info": {
            "output_token_logprobs": [(-0.5, t, None) for t in response],
            "finish_reason": {"type": "stop"},
            "prompt_tokens": len(served_prompt),
            "completion_tokens": len(response),
            "routed_experts": pybase64.b64encode(rows.tobytes()).decode(),
        },
    }
    post = AsyncMock(return_value=output)
    monkeypatch.setattr(rollout, "post", post)

    await rollout._run_inference_step(
        "url",
        sample,
        {},
        None,
        0 if token_space else None,
        args=SimpleNamespace(num_layers=layers, use_rollout_routing_replay=True),
    )

    payload = post.call_args.args[1]
    if token_space:
        assert payload["return_routed_experts"] is True
        assert payload["routed_experts_start_len"] == 0
        assert sample.tokens == served_prompt
        assert sample.rollout_routed_experts.shape == (len(served_prompt) + len(response) - 1, layers, topk)
    else:
        assert "return_routed_experts" not in payload
        assert sample.tokens == [1, -101, 2]
        assert sample.rollout_routed_experts is None


async def test_a_request_aborted_in_the_queue_keeps_the_rows_already_held(monkeypatch: pytest.MonkeyPatch) -> None:
    """SGLang answers a request aborted before it ran with a synthesized meta_info: no prompt_tokens, no routed
    experts. Reading them anyway would raise inside the end-of-rollout abort and lose the sample's earlier rows."""
    held = np.full((4, 2, 3), 9, dtype=np.int32)
    sample = Sample(prompt="", tokens=[1, 5, 2, 7, 8, 3, 3], rollout_routed_experts=held)
    output = {
        "text": "",
        "prompt_token_ids": list(sample.tokens),
        "meta_info": {"finish_reason": {"type": "abort"}, "completion_tokens": 0},
    }
    monkeypatch.setattr(rollout, "post", AsyncMock(return_value=output))

    await rollout._run_inference_step(
        "url", sample, {}, None, 5, args=SimpleNamespace(num_layers=2, use_rollout_routing_replay=True)
    )

    np.testing.assert_array_equal(sample.rollout_routed_experts, held)


async def test_a_later_token_space_turn_appends_only_the_rows_after_the_held_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later turn keeps the routed experts the earlier requests recorded and asks the engine only for the rows
    after them: the request's prefill would route bf16 ties differently from the decode that produced the
    earlier tokens' logprobs, and replay (R3) must use the routing those logprobs came from."""
    layers, topk = 2, 3
    held = np.full((4, layers, topk), 9, dtype=np.int32)  # turn 0: served prompt [1, 5, 2] + response [7, 8] → 4 rows
    sample = Sample(prompt="", tokens=[1, 5, 2, 7, 8, 3, 3], rollout_routed_experts=held)  # + a 2-token observation
    response = [6]
    new_rows = np.arange((len(sample.tokens) + len(response) - 1 - len(held)) * layers * topk, dtype=np.int32)
    output = {
        "text": "answer",
        "prompt_token_ids": list(sample.tokens),
        "meta_info": {
            "output_token_logprobs": [(-0.5, t, None) for t in response],
            "finish_reason": {"type": "stop"},
            "prompt_tokens": len(sample.tokens),
            "completion_tokens": len(response),
            "routed_experts": pybase64.b64encode(new_rows.tobytes()).decode(),
        },
    }
    post = AsyncMock(return_value=output)
    monkeypatch.setattr(rollout, "post", post)

    await rollout._run_inference_step(
        "url", sample, {}, None, 5, args=SimpleNamespace(num_layers=layers, use_rollout_routing_replay=True)
    )

    assert post.call_args.args[1]["routed_experts_start_len"] == len(held)
    assert sample.rollout_routed_experts.shape == (len(sample.tokens) + len(response) - 1, layers, topk)
    np.testing.assert_array_equal(sample.rollout_routed_experts[: len(held)], held)
    np.testing.assert_array_equal(sample.rollout_routed_experts[len(held) :], new_rows.reshape(-1, layers, topk))


@pytest.mark.parametrize("budget", [0, 1])
async def test_generation_uses_caller_state_and_closes_environment_on_every_return(
    monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    """Both exhausted and generated responses retain the caller's sample and close its environment."""
    env = SimpleNamespace(reset=Mock(), close=Mock())
    tokenizer = SimpleNamespace(decode=Mock(return_value="answer"))
    state = SimpleNamespace(
        args=SimpleNamespace(partial_rollout=False, multimodal_token_space_processing=False, max_seq_len=None),
        tokenizer=tokenizer,
        processor=None,
    )
    sample = Sample(prompt="question", tokens=[10], loss_mask=[], rollout_log_probs=[])
    sampling = {"max_new_tokens": 5}
    prepare = Mock(return_value=(None, [], budget, []))
    infer = AsyncMock(return_value=("answer", [20], [-0.2], "length"))
    monkeypatch.setattr(rollout, "_initialize_resources", lambda args, sample: (env, None, {"max_turns": 1}, "url"))
    monkeypatch.setattr(rollout, "_prepare_start_state", prepare)
    monkeypatch.setattr(rollout, "_run_inference_step", infer)

    output = await rollout.generate(
        GenerateFnInput(state=state, sample=sample, sampling_params=sampling, evaluation=False)
    )

    assert isinstance(output, GenerateFnOutput)
    assert output.samples is sample
    assert prepare.call_args.args[1] is state
    assert prepare.call_args.args[4] is not sampling
    assert sampling == {"max_new_tokens": 5}
    env.reset.assert_called_once_with()
    env.close.assert_called_once_with()
    assert sample.status == Sample.Status.TRUNCATED
    if budget:
        assert infer.call_args.args[1] is sample
        assert infer.call_args.args[-1] is None
        assert sample.tokens == [10, 20]
        assert sample.loss_mask == [1]
        assert sample.rollout_log_probs == [-0.2]
        assert sample.response == "answer"
        assert sample.response_length == 1
    else:
        infer.assert_not_called()
        assert sample.tokens == [10]


class TestObservationStaysOutOfAnEndingSample:
    async def test_the_last_turn_ends_before_its_observation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env = SimpleNamespace(reset=Mock(), close=Mock())
        sample = Sample(prompt="question", tokens=[10], loss_mask=[], rollout_log_probs=[])
        env_step = Mock(return_value=([30, 31], [], None, None, False))

        await _run(
            monkeypatch,
            env=env,
            state=_state(token_space=True),
            sample=sample,
            budget=10,
            max_turns=1,
            infer=AsyncMock(return_value=("answer", [20], [-0.2], "stop")),
            env_step=env_step,
        )

        env_step.assert_called_once()
        assert sample.tokens == [10, 20]
        assert sample.loss_mask == [1]
        assert sample.status == Sample.Status.COMPLETED

    async def test_an_observation_with_no_room_for_a_response_ends_the_sample_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # budget 3 - response 1 = 2 tokens left; the observation [30, 31] fills them with nothing to generate into.
        env = SimpleNamespace(reset=Mock(), close=Mock())
        sample = Sample(prompt="question", tokens=[10], loss_mask=[], rollout_log_probs=[])
        infer = AsyncMock(return_value=("answer", [20], [-0.2], "stop"))

        await _run(
            monkeypatch,
            env=env,
            state=_state(token_space=True),
            sample=sample,
            budget=3,
            max_turns=4,
            infer=infer,
            env_step=Mock(return_value=([30, 31], [], None, None, False)),
        )

        infer.assert_called_once()
        assert sample.tokens == [10, 20]
        assert sample.status == Sample.Status.TRUNCATED

    async def test_an_observation_that_leaves_room_is_appended_and_answered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # budget 4 - response 1 = 3 tokens left; the observation [30, 31] leaves one, so a second request goes out.
        env = SimpleNamespace(reset=Mock(), close=Mock())
        sample = Sample(prompt="question", tokens=[10], loss_mask=[], rollout_log_probs=[])
        infer = AsyncMock(side_effect=[("answer", [20], [-0.2], "stop"), ("again", [40], [-0.4], "stop")])

        await _run(
            monkeypatch,
            env=env,
            state=_state(token_space=True),
            sample=sample,
            budget=4,
            max_turns=4,
            infer=infer,
            env_step=Mock(return_value=([30, 31], [], None, None, False)),
        )

        assert infer.call_count == 2
        assert infer.call_args_list[1].args[2]["max_new_tokens"] == 1
        assert sample.tokens == [10, 20, 30, 31, 40]
        assert sample.loss_mask == [1, 0, 0, 1]
        assert sample.status == Sample.Status.TRUNCATED


async def test_a_response_past_max_seq_len_is_trimmed_to_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """max_new_tokens = 3 was budgeted from [10]; the engine expands the prompt to [10, 11, 12] and answers 3 tokens,
    so the sample runs 2 tokens past --max-seq-len 4 and loses the last two output tokens."""
    env = SimpleNamespace(reset=Mock(), close=Mock())
    sample = Sample(prompt="question", tokens=[10], loss_mask=[], rollout_log_probs=[])

    async def infer(url, sample, sampling_params, image_data, mm_token_expansion_start_len, *, args):
        assert sampling_params["max_new_tokens"] == 3
        sample.tokens = [10, 11, 12]
        return "answer", [20, 21, 22], [-0.1, -0.2, -0.3], "length"

    await _run(
        monkeypatch,
        env=env,
        state=_state(max_seq_len=4, token_space=True),
        sample=sample,
        budget=3,
        max_turns=4,
        infer=infer,
    )

    assert sample.tokens == [10, 11, 12, 20]
    assert sample.loss_mask == [1]
    assert sample.rollout_log_probs == [-0.1]
    assert sample.response_length == 1
    assert sample.status == Sample.Status.TRUNCATED


@pytest.mark.parametrize("max_seq_len", [8, 10], ids=["expansion-overruns", "expansion-fills-exactly"])
async def test_an_observation_whose_media_leaves_no_room_is_rolled_back(
    monkeypatch: pytest.MonkeyPatch, max_seq_len: int
) -> None:
    """Turn 1 answers [20]; the observation [30, 31] with one image leaves room, so it is appended. The engine then
    expands the image by 6 tokens, to 10 prompt tokens, and answers [40]. With --max-seq-len 8 the expanded prompt
    overruns the limit; with 10 it fills it exactly. Either way no response token fits after the observation, so
    the observation, its image and the answer are dropped and the sample ends at turn 1. Token space only: in the
    legacy path the processor expands the observation before it is appended, and the budget stays exact."""
    env = SimpleNamespace(reset=Mock(), close=Mock())
    sample = Sample(
        prompt="question", tokens=[10], loss_mask=[], rollout_log_probs=[], multimodal_inputs={"images": ["a.png"]}
    )
    answers = iter([([20], [-0.2]), ([40], [-0.4])])

    async def infer(url, sample, sampling_params, image_data, mm_token_expansion_start_len, *, args):
        if sample.tokens == [10, 20, 30, 31]:
            expanded = {"prompt_token_ids": [10, 20, 30, 9, 9, 9, 9, 9, 9, 31]}
            adopt_expanded_prompt_ids(sample, {"return_prompt_token_ids": True}, expanded)
        tokens, log_probs = next(answers)
        return "answer", tokens, log_probs, "stop"

    await _run(
        monkeypatch,
        env=env,
        state=_state(max_seq_len=max_seq_len, token_space=True),
        sample=sample,
        budget=max_seq_len - 1,
        max_turns=4,
        infer=infer,
        env_step=Mock(return_value=([30, 31], ["b.png"], {"images": ["b.png"]}, None, False)),
    )

    assert sample.tokens == [10, 20]
    assert sample.loss_mask == [1]
    assert sample.rollout_log_probs == [-0.2]
    assert sample.response_length == 1
    assert sample.multimodal_inputs == {"images": ["a.png"]}
    assert sample.status == Sample.Status.TRUNCATED
