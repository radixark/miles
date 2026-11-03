import pytest
from tests.fast.fixtures.args_fixtures import sglang_router_args
from tests.fast.rollout.inference_rollout.conftest import make_eval_args, make_eval_prompt_dataset_cache, make_state
from tests.fast.rollout.test_sglang_rollout import _make_generate_args

from miles.rollout import sglang_rollout
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.inference_rollout.inference_rollout_eval import eval_rollout_single_dataset
from miles.utils.args.component_rollout import InferenceRuntimeMutState
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.function_registry import function_registry
from miles.utils.types import Sample


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

        async def fallback(input: GenerateFnInput) -> GenerateFnOutput:
            raise AssertionError("Dataset generator must override the global generator")

        args = _make_generate_args()
        vars(args).update(make_eval_args())
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


class TestEvalPromptRoutingKey:
    @pytest.mark.parametrize("legacy", [False, True])
    async def test_the_samples_of_an_eval_prompt_share_one_routing_key(
        self, monkeypatch: pytest.MonkeyPatch, legacy: bool
    ) -> None:
        """Both evaluation engines key the n samples of a prompt together when the router places keys by load."""
        captured: list[Sample] = []

        async def generate(input: GenerateFnInput) -> GenerateFnOutput:
            captured.append(input.sample)
            input.sample.response = "answer"
            input.sample.status = Sample.Status.COMPLETED
            input.sample.reward = 1.0
            return GenerateFnOutput(samples=input.sample)

        args = _make_generate_args()
        vars(args).update(make_eval_args() | sglang_router_args("manual"))
        args.router_args["assignment_mode"] = "min_load"
        args.sglang.base_args["dp_size"] = 1
        args.inference_runtime_mut_state = InferenceRuntimeMutState(engine_count=1, gpu_count=1)
        [dataset] = args.eval_datasets
        dataset.custom_generate_function_path = CustomFunctionConfig(path="test-group-key-generator")
        cache = make_eval_prompt_dataset_cache(args)
        state = make_state(generate, args=args)
        monkeypatch.setattr(sglang_rollout.SingletonMeta, "_instances", {})
        monkeypatch.setattr(sglang_rollout, "load_tokenizer", lambda *args, **kwargs: object())
        monkeypatch.setattr(sglang_rollout, "load_processor", lambda *args, **kwargs: None)
        monkeypatch.setattr(sglang_rollout, "EVAL_PROMPT_DATASET", cache)

        with function_registry.temporary("test-group-key-generator", generate):
            if legacy:
                await sglang_rollout.eval_rollout_single_dataset(args, 0, dataset)
            else:
                await eval_rollout_single_dataset(state, dataset, cache)

        assert len(captured) == 2
        keys = {sample.routing_key for sample in captured}
        assert len(keys) == 1 and None not in keys
        assert next(iter(cache.values())).samples[0].routing_key is None
