import asyncio
from argparse import ArgumentParser
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_samples_grouped

from miles.ray.rollout.train_data_conversion import _post_process_rewards, convert_samples_to_train_data
from miles.rollout import sglang_rollout
from miles.rollout.base_types import GenerateFnOutput
from miles.rollout.filter_hub.common_filters import apply_preput_filters
from miles.rollout.inference_rollout import inference_rollout_common
from miles.utils.arguments import _validate_remax_args, get_miles_extra_args_provider
from miles.utils.lifecycle import TrajectoryLifecycle
from miles.utils.types import Sample


@pytest.fixture(params=[sglang_rollout, inference_rollout_common], ids=["legacy", "class"])
def rollout_harness(request, monkeypatch):
    module = request.param
    args = make_args(
        advantage_estimator="remax",
        partial_rollout=False,
        group_rm=False,
        custom_generate_function_path=None,
        sglang_router_policy="consistent_hashing",
        sglang_enable_deterministic_inference=True,
        rollout_seed=17,
    )
    calls = []
    state = SimpleNamespace(
        args=args,
        aborted=False,
        semaphore=asyncio.Semaphore(1),
        generate_fn_semaphore=asyncio.Semaphore(1),
        dp_rank_context=nullcontext,
        group_sampling_seeds=[17, 18],
        baseline_status=Sample.Status.COMPLETED,
        baseline_rewards=[3.0, 0.5],
    )

    async def generate(_args, sample, sampling_params):
        assert sample.reward is None
        assert not sample.response
        calls.append((sample, sampling_params.copy()))
        sample.response = "greedy" if sampling_params["temperature"] == 0 else "sampled"
        sample.response_length = 2
        sample.tokens = [10, 11, 12]
        sample.status = state.baseline_status if sample.metadata.get("remax_baseline") else Sample.Status.COMPLETED
        return sample

    async def generate_fn(input):
        return GenerateFnOutput(samples=await generate(input.args, input.sample, input.sampling_params))

    async def reward(_args, sample):
        score = state.baseline_rewards[sample.group_index] if sample.metadata.get("remax_baseline") else sample.label
        return {"score": score} if args.reward_key else score

    state.generate_function = generate_fn
    monkeypatch.setattr(module, "async_rm", reward)
    monkeypatch.setattr(TrajectoryLifecycle(), "sink", None)
    if module is sglang_rollout:
        monkeypatch.setattr(module, "GenerateState", lambda _args: state)
        monkeypatch.setattr(module, "generate", generate)
    context = args if module is sglang_rollout else state

    async def run_group(group, sampling_params, evaluation=False):
        return await module.generate_and_rm_group(context, group, sampling_params, evaluation=evaluation)

    return args, state, calls, run_group


def pending_groups(group_size):
    samples = make_samples_grouped(n_groups=2, group_size=group_size)
    for i, sample in enumerate(samples):
        sample.reset_for_retry()
        sample.label = [1.0, 5.0][i % group_size]
        sample.metadata = {"dataset": "math"}
    return [samples[:group_size], samples[group_size:]]


@pytest.mark.parametrize("group_size", [1, 2])
@pytest.mark.parametrize("reward_key", [None, "score"])
async def test_greedy_baseline_is_per_prompt_and_excluded_from_training(rollout_harness, group_size, reward_key):
    args, state, calls, run_group = rollout_harness
    args.n_samples_per_prompt = group_size
    args.rollout_batch_size = 2
    args.reward_key = reward_key
    args.grpo_std_normalization = True
    params = dict(temperature=0.8, top_p=0.9, top_k=20, max_new_tokens=7, stop=["END"])
    groups = pending_groups(group_size)
    outputs = await asyncio.gather(*(run_group(group, params) for group in groups))

    baselines = [(sample, p) for sample, p in calls if sample.metadata.get("remax_baseline")]
    assert len(baselines) == 2
    assert len(calls) == 2 * (group_size + 1)
    assert params["temperature"] == 0.8
    for baseline, baseline_params in baselines:
        assert baseline_params == {**params, "temperature": 0.0}
        assert baseline.response == "greedy"
        assert baseline.remove_sample
        assert baseline.index < 0
        assert baseline.metadata["dataset"] == "math"
        assert all(baseline.routing_key != sample.routing_key for group in outputs for sample in group)

    for group_index, output in enumerate(outputs):
        assert len(output) == group_size
        assert apply_preput_filters(args, None, output).keep
        assert [s.metadata["remax_baseline_reward"] for s in output] == [[3.0, 0.5][group_index]] * group_size
        assert all(not s.remove_sample and not s.metadata.get("remax_baseline") for s in output)
    training_calls = [p for s, p in calls if not s.metadata.get("remax_baseline")]
    assert all(p["temperature"] == 0.8 for p in training_calls)
    assert sorted(p["sampling_seed"] for p in training_calls) == sorted([17 + i for i in range(group_size)] * 2)
    train_data = convert_samples_to_train_data(args, [s for g in outputs for s in g], {}, None, None)
    expected = [-2.0, 0.5] if group_size == 1 else [-2.0, 2.0, 0.5, 4.5]
    assert train_data["rewards"] == expected
    assert train_data["raw_reward"] == [1.0, 5.0][:group_size] * 2
    assert train_data["loss_masks"] == [[1, 1]] * (2 * group_size)
    assert train_data["sample_indices"] == list(range(2 * group_size))


@pytest.mark.parametrize("evaluation,estimator", [(True, "remax"), (False, "grpo")])
async def test_evaluation_and_other_estimators_do_not_generate_baselines(rollout_harness, evaluation, estimator):
    args, _, calls, run_group = rollout_harness
    args.advantage_estimator = estimator
    output = await run_group(pending_groups(1)[0], {"temperature": 0.8}, evaluation=evaluation)
    assert len(calls) == len(output) == 1
    assert "remax_baseline_reward" not in output[0].metadata


@pytest.mark.parametrize("status", [Sample.Status.ABORTED, Sample.Status.FAILED])
async def test_failed_greedy_baseline_rejects_prompt_group(rollout_harness, status):
    args, state, calls, run_group = rollout_harness
    state.baseline_status = status
    output = await run_group(pending_groups(2)[0], {"temperature": 0.8})
    assert len(calls) == 1
    assert len(output) == 2
    assert not apply_preput_filters(args, None, output).keep


@pytest.mark.parametrize("reward_key", [None, "score"])
async def test_missing_greedy_reward_rejects_prompt_group(rollout_harness, reward_key):
    args, state, _, run_group = rollout_harness
    args.reward_key = reward_key
    state.baseline_rewards[0] = None
    output = await run_group(pending_groups(1)[0], {"temperature": 0.8})
    assert not apply_preput_filters(args, None, output).keep


async def test_truncated_greedy_response_can_supply_a_baseline(rollout_harness):
    args, state, _, run_group = rollout_harness
    state.baseline_status = Sample.Status.TRUNCATED
    output = await run_group(pending_groups(1)[0], {"temperature": 0.8})
    assert apply_preput_filters(args, None, output).keep
    assert output[0].metadata["remax_baseline_reward"] == 3.0


@pytest.mark.parametrize("rewards_normalization", [False, True])
def test_remax_subtraction_does_not_depend_on_reward_normalization_flags(rewards_normalization):
    args = make_args(
        advantage_estimator="remax", rewards_normalization=rewards_normalization, grpo_std_normalization=True
    )
    samples = make_samples_grouped(n_groups=1, group_size=2, rewards=[1.0, 5.0])
    for sample in samples:
        sample.metadata["remax_baseline_reward"] = 3.0
    raw, adjusted = _post_process_rewards(args, samples[::-1], None)
    assert raw == [5.0, 1.0]
    assert adjusted == [2.0, -2.0]


def test_remax_requires_baseline_metadata():
    args = make_args(advantage_estimator="remax")
    with pytest.raises(ValueError, match="remax_baseline_reward"):
        _post_process_rewards(args, make_samples_grouped(1, 1), None)


@pytest.mark.parametrize(
    "overrides",
    [
        {"partial_rollout": True},
        {"multi_lora": True},
        {"fully_async": True},
        {"group_rm": True},
        {"normalize_advantages": True},
        {"rollout_submission_granularity": "sample"},
        {"custom_reward_post_process_path": "custom.reward"},
        {"custom_convert_samples_to_train_data_path": "custom.convert"},
    ],
)
def test_remax_rejects_incompatible_options(overrides):
    with pytest.raises(ValueError, match="remax"):
        _validate_remax_args(make_args(advantage_estimator="remax", **overrides))


def test_remax_accepts_default_options_and_singleton_groups():
    _validate_remax_args(make_args(advantage_estimator="remax", n_samples_per_prompt=1))


def test_cli_accepts_remax_with_singleton_groups():
    parser = ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(["--advantage-estimator", "remax", "--rollout-batch-size", "2"])
    assert args.advantage_estimator == "remax"
    assert args.n_samples_per_prompt == 1
    _validate_remax_args(args)
