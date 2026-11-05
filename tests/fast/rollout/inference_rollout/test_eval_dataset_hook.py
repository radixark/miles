import pytest
from tests.fast.rollout.inference_rollout.conftest import make_eval_args, make_eval_prompt_dataset_cache, make_state
from tests.fast.rollout.test_sglang_rollout import _make_generate_args

from miles.rollout import sglang_rollout
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.inference_rollout.inference_rollout_eval import eval_rollout_single_dataset
from miles.utils.args.component_rollout import InferenceRuntimeMutState
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.function_registry import function_registry
from miles.utils.types import Sample


def _prepare_eval(monkeypatch: pytest.MonkeyPatch, **arg_overrides):
    """Args, dataset, prompt cache and state for both evaluation engines, the dataset naming its own generator."""

    async def fallback(input: GenerateFnInput) -> GenerateFnOutput:
        raise AssertionError("Dataset generator must override the global generator")

    args = _make_generate_args()
    vars(args).update(make_eval_args())
    vars(args).update(arg_overrides)
    args.sglang.base_args["dp_size"] = 1
    args.inference_runtime_mut_state = InferenceRuntimeMutState(engine_count=1, gpu_count=1)
    [dataset] = args.eval_datasets
    dataset.custom_generate_function_path = CustomFunctionConfig(path="test-dataset-generator")
    cache = make_eval_prompt_dataset_cache(args)
    state = make_state(fallback, args=args)
    monkeypatch.setattr(sglang_rollout.SingletonMeta, "_instances", {})
    monkeypatch.setattr(sglang_rollout, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(sglang_rollout, "load_processor", lambda *args, **kwargs: None)
    monkeypatch.setattr(sglang_rollout, "EVAL_PROMPT_DATASET", cache)
    return args, dataset, cache, state


class TestEvalDatasetHook:
    @pytest.mark.parametrize("legacy", [False, True])
    async def test_dataset_hook_path_reaches_the_real_generation_dispatch(
        self, monkeypatch: pytest.MonkeyPatch, legacy: bool
    ) -> None:
        """Both evaluation engines unwrap typed dataset hooks before dispatching every generated sample."""
        captured: list[Sample] = []

        async def generate(input: GenerateFnInput) -> GenerateFnOutput:
            captured.append(input.sample)
            input.sample.response = "answer"
            input.sample.status = Sample.Status.COMPLETED
            input.sample.reward = 1.0
            return GenerateFnOutput(samples=input.sample)

        args, dataset, cache, state = _prepare_eval(monkeypatch)

        with function_registry.temporary("test-dataset-generator", generate):
            if legacy:
                result = await sglang_rollout.eval_rollout_single_dataset(args, 0, dataset)
            else:
                result = await eval_rollout_single_dataset(state, dataset, cache)

        assert len(captured) == 2
        assert [sample.generate_function_path for sample in captured] == ["test-dataset-generator"] * 2
        assert result[dataset.name]["rewards"] == [1.0, 1.0]
        assert [sample.index for sample in result[dataset.name]["samples"]] == [0, 1]
        assert next(iter(cache.values())).samples[0].generate_function_path is None

    @pytest.mark.parametrize("legacy", [False, True])
    async def test_aborted_sample_keeps_none_reward_under_a_reward_key(
        self, monkeypatch: pytest.MonkeyPatch, legacy: bool
    ) -> None:
        """An aborted sample skips the reward model, so a reward key must not index into its None reward."""

        async def generate(input: GenerateFnInput) -> GenerateFnOutput:
            if input.sample.index == 0:
                input.sample.status = Sample.Status.ABORTED
            else:
                input.sample.status = Sample.Status.COMPLETED
                input.sample.reward = {"score": 0.5}
            return GenerateFnOutput(samples=input.sample)

        args, dataset, cache, state = _prepare_eval(monkeypatch, eval_reward_key="score")

        with function_registry.temporary("test-dataset-generator", generate):
            if legacy:
                result = await sglang_rollout.eval_rollout_single_dataset(args, 0, dataset)
            else:
                result = await eval_rollout_single_dataset(state, dataset, cache)

        assert result[dataset.name]["rewards"] == [None, 0.5]
