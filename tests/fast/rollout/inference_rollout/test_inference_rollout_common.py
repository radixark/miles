from types import SimpleNamespace
from unittest.mock import MagicMock

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
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn


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

        await fn(RolloutFnEvalInput(rollout_id=4))

        assert recorder.take_stamps() == {"eval:-:4"}

    async def test_fleet_state_eval_stamps_the_samples_of_the_fleet_call(self, monkeypatch):
        """Eval on a dedicated fleet stamps the samples that fleet generates, under the eval namespace."""
        recorder = StampRecordingGenerate()
        fn, captured = self._make_fn(monkeypatch)
        fleet_state = make_state(recorder, args=fn.state.args)

        await fn(RolloutFnEvalInput(rollout_id=4, generate_state=fleet_state, weight_version="0"))

        assert recorder.take_stamps() == {"eval:-:4"}
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

        await fn(RolloutFnEvalInput(rollout_id=4))

        assert recorder.take_stamps() == {None}
