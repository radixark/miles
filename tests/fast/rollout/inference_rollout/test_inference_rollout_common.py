from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tests.fast.rollout.inference_rollout.conftest import (
    StampRecordingGenerate,
    make_eval_args,
    make_eval_prompt_dataset_cache,
    make_state,
)
from tests.fast.utils.output_store.test_dropped import sample_owning

import miles.rollout.inference_rollout.inference_rollout_common as inference_rollout_common
from miles.rollout.base_types import (
    GenerateFnInput,
    GenerateFnOutput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnTrainInput,
)
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.output_store import dropped
from miles.utils.output_store.dropped import take_dropped_refs
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


class TestFailedGenerationHandsOverItsOutputStoreRefs:
    """A sample task that raises reaches no drop site, so the runners record what it held."""

    @pytest.fixture(autouse=True)
    def _empty_dropped(self, monkeypatch):
        monkeypatch.setattr(dropped, "_dropped", {})

    async def test_a_failed_sibling_hands_over_what_the_group_filled_in_place_or_returned(self):
        async def generate(input: GenerateFnInput) -> GenerateFnOutput:
            if input.sample.index == 0:
                input.sample.rollout_store_plan = sample_owning("in-place").rollout_store_plan
                return GenerateFnOutput(samples=input.sample)
            if input.sample.index == 1:
                # Like a session: new samples, which only the task result holds.
                return GenerateFnOutput(samples=[sample_owning("returned")])
            raise RuntimeError("generate failed")

        group = [Sample(index=index) for index in range(3)]
        with pytest.raises(RuntimeError, match="generate failed"):
            await inference_rollout_common.generate_and_rm_group(make_state(generate), group, sampling_params={})

        assert sorted(take_dropped_refs()) == ["in-place", "returned"]

    async def test_a_failed_group_reward_hands_over_the_group(self, monkeypatch):
        async def generate(input: GenerateFnInput) -> GenerateFnOutput:
            return GenerateFnOutput(samples=[sample_owning(f"returned-{input.sample.index}")])

        async def failing_rm(*args, **kwargs):
            raise RuntimeError("reward failed")

        monkeypatch.setattr(inference_rollout_common, "batched_async_rm", failing_rm)
        group = [Sample(index=index) for index in range(2)]
        with pytest.raises(RuntimeError, match="reward failed"):
            await inference_rollout_common.generate_and_rm_group(make_state(generate), group, sampling_params={})

        assert sorted(take_dropped_refs()) == ["returned-0", "returned-1"]

    async def test_samples_whose_reward_fails_hand_over_their_refs(self, monkeypatch):
        async def generate(input: GenerateFnInput) -> GenerateFnOutput:
            return GenerateFnOutput(samples=[sample_owning("returned")])

        async def failing_rm(*args, **kwargs):
            raise RuntimeError("reward failed")

        monkeypatch.setattr(inference_rollout_common, "batched_async_rm", failing_rm)
        args = SimpleNamespace(
            partial_rollout=False,
            mask_offpolicy_in_partial_rollout=False,
            group_rm=False,
            sglang_router_policy="round_robin",
        )
        with pytest.raises(RuntimeError, match="reward failed"):
            await inference_rollout_common.generate_and_rm(
                make_state(generate, args=args), Sample(index=0), sampling_params={}
            )

        assert list(take_dropped_refs()) == ["returned"]
