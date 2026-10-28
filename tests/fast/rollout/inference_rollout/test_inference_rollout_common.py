from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config
from tests.fast.rollout.inference_rollout.conftest import (
    StampRecordingGenerate,
    make_eval_args,
    make_eval_prompt_dataset_cache,
    make_state,
)

import miles.rollout.inference_rollout.inference_rollout_common as inference_rollout_common
from miles.rollout.base_types import (
    GenerateFnInput,
    GenerateFnOutput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnTrainInput,
)
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn, generate_and_rm_group
from miles.utils.types import Sample


def test_inference_rollout_fn_exposes_the_constructor_input(monkeypatch) -> None:
    """The base class promises constructor_input, so the framework's own subclasses must set it too."""
    monkeypatch.setattr(inference_rollout_common, "GenerateState", lambda args: MagicMock())
    constructor_input = RolloutFnConstructorInput(args=MagicMock(), data_source=MagicMock())

    fn = InferenceRolloutFn(constructor_input)

    assert fn.constructor_input is constructor_input


async def _echo_generate(input: GenerateFnInput) -> GenerateFnOutput:
    return GenerateFnOutput(samples=input.sample)


class TestInferenceRolloutFnKvCacheNamespace:
    def _make_fn(self, monkeypatch, generate_function=_echo_generate, *, partition: bool = True):
        captured: dict[str, str | None] = {}

        async def fake_generate_rollout_async(state, rollout_id, get_samples, *, kv_cache_namespace=None):
            captured["kv_cache_namespace"] = kv_cache_namespace
            return SimpleNamespace(), []

        args = SimpleNamespace(**make_eval_args(namespaced_radix_cache=partition))
        monkeypatch.setattr(
            inference_rollout_common, "GenerateState", lambda args: make_state(generate_function, args=args)
        )
        monkeypatch.setattr(
            "miles.rollout.inference_rollout.inference_rollout_train.generate_rollout_async",
            fake_generate_rollout_async,
        )
        fn = InferenceRolloutFn(RolloutFnConstructorInput(args=args, data_source=MagicMock()))
        fn.eval_prompt_dataset_cache.update(make_eval_prompt_dataset_cache(args))
        return fn, captured

    async def test_a_train_call_hands_its_own_namespace_to_the_generation_loop(self, monkeypatch):
        """The train branch names the call by the policy and rollout id it carries."""
        fn, captured = self._make_fn(monkeypatch)

        await fn(RolloutFnTrainInput(rollout_id=5, trainer_model_id="solver"))

        assert captured["kv_cache_namespace"] == "train:solver:5"

    async def test_shared_state_eval_samples_are_stamped_with_the_eval_namespace(self, monkeypatch):
        """Shared-engine eval stamps every sample of the call under the eval namespace of that rollout id."""
        recorder = StampRecordingGenerate()
        fn, _ = self._make_fn(monkeypatch, recorder)

        await fn(RolloutFnEvalInput(rollout_id=4, invocation_id="eval-call"))

        assert recorder.take_stamps() == {"eval:-:4:eval-call"}

    async def test_fleet_state_eval_stamps_the_samples_of_the_fleet_call(self, monkeypatch):
        """Eval on a dedicated fleet stamps the samples that fleet generates, under the eval namespace."""
        recorder = StampRecordingGenerate()
        fn, captured = self._make_fn(monkeypatch)
        fleet_state = make_state(recorder, args=fn.state.args)

        await fn(
            RolloutFnEvalInput(rollout_id=4, generate_state=fleet_state, weight_version="0", invocation_id="eval-call")
        )

        assert recorder.take_stamps() == {"eval:-:4:eval-call"}
        assert captured == {}

    async def test_the_partition_being_off_names_no_namespace_for_a_train_call(self, monkeypatch):
        """With --no-namespaced-radix-cache a train call names no namespace at all."""
        fn, captured = self._make_fn(monkeypatch, partition=False)

        await fn(RolloutFnTrainInput(rollout_id=5, trainer_model_id="solver"))

        assert captured["kv_cache_namespace"] is None

    async def test_the_partition_being_off_leaves_eval_samples_unstamped(self, monkeypatch):
        """With the partition off no eval sample is stamped, so no request carries an extra_key."""
        recorder = StampRecordingGenerate()
        fn, _ = self._make_fn(monkeypatch, recorder, partition=False)

        await fn(RolloutFnEvalInput(rollout_id=4, invocation_id="eval-call"))

        assert recorder.take_stamps() == {None}


class TestSkipGroupReward:
    """The producer may skip the reward of a group whose generated samples already doom it."""

    GROUP_SIZE = 3

    @staticmethod
    async def _complete(input: GenerateFnInput) -> GenerateFnOutput:
        input.sample.status = Sample.Status.COMPLETED
        return GenerateFnOutput(samples=input.sample)

    def _state(self, *, group_rm: bool):
        args = SimpleNamespace(
            partial_rollout=False,
            mask_offpolicy_in_partial_rollout=False,
            group_rm=group_rm,
            sglang_router_policy="round_robin",
            sglang=make_sglang_config(enable_deterministic_inference=False),
        )
        return make_state(self._complete, args=args)

    def _group(self) -> list[Sample]:
        return [Sample(group_index=1, index=10 + i) for i in range(self.GROUP_SIZE)]

    async def test_each_reward_call_sees_the_outputs_generated_so_far(self, monkeypatch) -> None:
        """Once the samples generated so far doom the group, the rest of its reward calls are skipped."""
        scored: list[int] = []

        async def fake_async_rm(args, sample, **kwargs):
            scored.append(sample.index)
            return 1.0

        monkeypatch.setattr(inference_rollout_common, "async_rm", fake_async_rm)
        seen_sizes: list[int] = []

        def skip_from_the_second_output(generated) -> bool:
            seen_sizes.append(len(generated))
            return len(generated) >= 2

        group = await generate_and_rm_group(
            self._state(group_rm=False),
            self._group(),
            sampling_params={},
            skip_group_reward_fn=skip_from_the_second_output,
        )

        assert sorted(seen_sizes) == [1, 2, 3]
        assert len(scored) == 1
        assert sorted(sample.reward is None for sample in group) == [False, True, True]

    @pytest.mark.parametrize(argnames="skip", argvalues=[True, False])
    async def test_a_group_rm_call_is_asked_once_with_the_whole_group(self, monkeypatch, skip: bool) -> None:
        batched_calls: list[int] = []

        async def fake_batched_async_rm(args, samples, inplace_set_reward_field=False):
            batched_calls.append(len(samples))

        monkeypatch.setattr(inference_rollout_common, "batched_async_rm", fake_batched_async_rm)
        asked: list[int] = []

        def skip_group(generated) -> bool:
            asked.append(len(generated))
            return skip

        await generate_and_rm_group(
            self._state(group_rm=True), self._group(), sampling_params={}, skip_group_reward_fn=skip_group
        )

        assert asked == [self.GROUP_SIZE]
        assert batched_calls == ([] if skip else [self.GROUP_SIZE])
