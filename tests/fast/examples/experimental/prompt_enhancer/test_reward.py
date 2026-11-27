"""`prompt_enhancer_rm` against a fake diffusion engine reached through its router entry, and a fake scorer server."""

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
from examples.experimental.prompt_enhancer import reward
from examples.experimental.prompt_enhancer.reward import (
    ASSISTANT_PREFIX,
    ORI_PROMPT_KEY,
    REWARD_KEY,
    format_reward,
    length_reward,
    log_rollout_data,
    prompt_enhancer_rm,
    t2i_prompt,
)

from miles.utils.retry_utils import NonRetryableError
from miles.utils.types import Sample

from .conftest import png

_ANSWER_20 = " ".join(["word"] * 20)
_GOOD = f" plan </think>\n<answer>{_ANSWER_20}</answer>"


@pytest.fixture(autouse=True)
def _fresh_clients(monkeypatch):
    monkeypatch.setattr(reward, "_clients", {})
    monkeypatch.setattr(reward, "_RETRY_DELAY_SECS", 0.01)


@pytest.mark.parametrize(
    "response, expected",
    [
        (_GOOD, 1.0),
        (_GOOD + "\n", 1.0),
        # RePrompt's pattern has no DOTALL, so reasoning that spans lines fails the format.
        (f" plan\nmore </think>\n<answer>{_ANSWER_20}</answer>", -1.0),
        (f" plan </think><answer>{_ANSWER_20}</answer>", -1.0),
        (f"{_GOOD} trailing", -1.0),
    ],
)
def test_format_needs_one_line_of_reasoning_then_the_answer(response, expected):
    assert format_reward(ASSISTANT_PREFIX + response) == expected


@pytest.mark.parametrize("words, expected", [(15, -1.0), (16, 1.0), (76, 1.0), (77, -1.0)])
def test_length_rewards_answers_of_16_to_76_words(words, expected):
    assert length_reward(f"<think> x </think>\n<answer>{' '.join(['w'] * words)}</answer>") == expected


def test_no_answer_scores_zero_length_and_renders_a_photo():
    completion = ASSISTANT_PREFIX + " still thinking"
    assert length_reward(completion) == 0.0
    assert t2i_prompt(completion) == "a photo"
    assert t2i_prompt("<answer>first</answer><answer>second</answer>") == "first"


def _args(engine, scorer, **overrides) -> SimpleNamespace:
    values = dict(
        rollout_skip_special_tokens=True,
        sglang_model_routers={"policy": ("127.0.0.1", 1), "generator": (engine.host, engine.port)},
        pe_generator_model="generator",
        pe_width=512,
        pe_height=256,
        pe_num_inference_steps=9,
        pe_guidance_scale=0.0,
        pe_seed_per_sample=False,
        pe_scorer_urls=scorer.url,
    )
    return SimpleNamespace(**(values | overrides))


def _sample(response: str = _GOOD, *, index: int = 0, group_index: int = 0, prompt: str = "a cat") -> Sample:
    return Sample(
        prompt="p", response=response, index=index, group_index=group_index, metadata={ORI_PROMPT_KEY: prompt}
    )


def _rewards(args, samples: list[Sample]) -> list[float | None]:
    async def run():
        return await asyncio.gather(*(prompt_enhancer_rm(args, s) for s in samples))

    return asyncio.run(run())


def test_the_rewrite_is_rendered_and_the_image_is_scored_against_the_original_prompt(engine, scorer):
    sample = _sample()

    (value,) = _rewards(_args(engine, scorer), [sample])

    (request,) = engine.requests
    assert request["prompt"] == _ANSWER_20
    assert (request["width"], request["height"], request["num_inference_steps"]) == (512, 256, 9)
    assert request["guidance_scale"] == 0.0
    (scored,) = scorer.requests
    assert scored["prompt"] == "a cat"
    assert base64.b64decode(scored["image_b64"]) == png(512, 256)
    assert value == pytest.approx(0.5 * 0.8 / 2.0 + 1.0 + 1.0)
    parts = sample.metadata[REWARD_KEY]
    assert (parts["image_reward"], parts["format"], parts["length"]) == (0.8, 1.0, 1.0)
    assert parts["t2i_prompt"] == _ANSWER_20 and parts["seed"] == request["seed"]


def test_unset_sampling_fields_are_left_to_the_model(engine, scorer):
    _rewards(_args(engine, scorer, pe_num_inference_steps=None, pe_guidance_scale=None), [_sample()])

    (request,) = engine.requests
    assert "num_inference_steps" not in request and "guidance_scale" not in request


def test_a_group_shares_one_seed_that_changes_with_the_draw(engine, scorer):
    group = [_sample(index=i, group_index=5) for i in range(4)]
    redraw = _sample(index=4, group_index=6)
    other_prompt = _sample(index=5, group_index=5, prompt="a dog")

    args = _args(engine, scorer)
    # One at a time: concurrent requests reach the fake engine in no fixed order.
    for sample in [*group, redraw, other_prompt]:
        _rewards(args, [sample])

    seeds = [r["seed"] for r in engine.requests]
    assert len(set(seeds[:4])) == 1
    assert len(set(seeds[3:])) == 3
    assert all(0 <= s < 2**31 for s in seeds)


def test_the_parity_flag_draws_one_seed_per_sample(engine, scorer):
    group = [_sample(index=i, group_index=5) for i in range(4)]

    _rewards(_args(engine, scorer, pe_seed_per_sample=True), group)

    assert len({r["seed"] for r in engine.requests}) == 4


def test_the_same_draw_gets_the_same_seed_in_another_process(engine, scorer):
    _rewards(_args(engine, scorer), [_sample(group_index=7)])

    # blake2b over the draw and the prompt, so the seed does not depend on PYTHONHASHSEED.
    assert engine.requests[0]["seed"] == reward.image_seed("a cat", 7)
    assert reward.image_seed("a cat", 7) == 846558865


def test_a_generator_that_keeps_failing_drops_the_sample(engine, scorer):
    engine.status = 503

    assert _rewards(_args(engine, scorer), [_sample()]) == [None]
    assert len(engine.requests) == 4  # the first attempt and three retries
    assert scorer.requests == []


def test_a_request_the_generator_refuses_stops_the_run(engine, scorer):
    engine.status = 400

    with pytest.raises(NonRetryableError, match="refused the request with 400"):
        _rewards(_args(engine, scorer), [_sample()])


def test_a_scorer_that_keeps_failing_drops_the_sample(engine, scorer):
    scorer.status = 500

    assert _rewards(_args(engine, scorer), [_sample()]) == [None]
    assert len(scorer.requests) == reward._SCORER_ATTEMPTS


def test_an_unknown_generator_model_stops_the_run(engine, scorer):
    with pytest.raises(NonRetryableError, match="no router for model 'painter'"):
        _rewards(_args(engine, scorer, pe_generator_model="painter"), [_sample()])


def test_the_stop_token_must_be_stripped(engine, scorer):
    with pytest.raises(NonRetryableError, match="rollout-skip-special-tokens"):
        _rewards(_args(engine, scorer, rollout_skip_special_tokens=False), [_sample()])


def test_the_log_hook_logs_the_terms_and_writes_each_scored_sample(monkeypatch, tmp_path, engine, scorer):
    logged = []
    monkeypatch.setattr(reward.tracking, "log", lambda args, metrics, step_key: logged.append(metrics))
    sample = _sample(group_index=3)
    (sample.reward,) = _rewards(_args(engine, scorer), [sample])
    unscored = _sample(index=1, group_index=3)
    log_args = SimpleNamespace(pe_sample_log=str(tmp_path), reward_key=None, wandb_always_use_train_step=False)

    assert log_rollout_data(7, log_args, [sample, unscored], {}, 0.0) is False

    (metrics,) = logged
    assert metrics["rollout/step"] == 7 and metrics["rollout/pe/image_reward"] == 0.8
    assert metrics["rollout/pe/reward"] == pytest.approx(2.2)
    assert metrics["rollout/pe/unscored"] == 1.0 and metrics["rollout/pe/render_secs_p95"] >= 0.0
    (row,) = [json.loads(line) for line in (tmp_path / "rollout_00007.jsonl").read_text().splitlines()]
    assert row["prompt"] == "a cat" and row["group"] == 3
    assert row["completion"].startswith(ASSISTANT_PREFIX) and row["format"] == 1.0
