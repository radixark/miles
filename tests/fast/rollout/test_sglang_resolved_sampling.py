import pytest
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config
from tests.fast.rollout.inference_rollout.conftest import make_eval_args, make_eval_prompt_dataset_cache, make_state
from tests.fast.rollout.test_sglang_rollout import _make_generate_args

from miles.rollout import sglang_rollout
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.inference_rollout.inference_rollout_common import generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import eval_rollout_single_dataset
from miles.utils.args.component_rollout import InferenceRuntimeMutState
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.function_registry import function_registry
from miles.utils.types import Sample


class TestResolvedDeterministicSampling:
    @pytest.mark.parametrize("legacy", [False, True])
    @pytest.mark.parametrize("evaluation", [False, True])
    @pytest.mark.parametrize("enabled", [False, True])
    async def test_group_and_dataset_sampling_follow_resolved_determinism_without_flat_fields(
        self, monkeypatch: pytest.MonkeyPatch, legacy: bool, evaluation: bool, enabled: bool
    ) -> None:
        """Both generation engines derive independent per-sample seeds from resolved SGLang settings."""
        seeds: list[int | None] = []

        async def generate(input: GenerateFnInput) -> GenerateFnOutput:
            seeds.append(input.sampling_params.get("sampling_seed"))
            input.sample.response = "answer"
            input.sample.status = Sample.Status.COMPLETED
            input.sample.reward = 1.0
            return GenerateFnOutput(samples=input.sample)

        args = _make_generate_args()
        vars(args).update(make_eval_args())
        args.sglang = make_sglang_config(enable_deterministic_inference=enabled, dp_size=1)
        vars(args).pop("sglang_enable_deterministic_inference", None)
        args.rollout_seed = 17
        args.inference_runtime_mut_state = InferenceRuntimeMutState(engine_count=1, gpu_count=1)
        args.custom_generate_function_path = CustomFunctionConfig(path="test-generation")
        args.n_samples_per_prompt = 2
        state = make_state(generate, args=args)
        cache = make_eval_prompt_dataset_cache(args)
        sampling = {"temperature": 0.5}
        monkeypatch.setattr(sglang_rollout.SingletonMeta, "_instances", {})
        monkeypatch.setattr(sglang_rollout, "load_tokenizer", lambda *args, **kwargs: object())
        monkeypatch.setattr(sglang_rollout, "load_processor", lambda *args, **kwargs: None)
        monkeypatch.setattr(sglang_rollout, "EVAL_PROMPT_DATASET", cache)

        with function_registry.temporary("test-generation", generate):
            if evaluation:
                [dataset] = args.eval_datasets
                if legacy:
                    await sglang_rollout.eval_rollout_single_dataset(args, 0, dataset)
                else:
                    await eval_rollout_single_dataset(state, dataset, cache)
            else:
                samples = [Sample(index=index, prompt="prompt") for index in range(2)]
                if legacy:
                    await sglang_rollout.generate_and_rm_group(args, samples, sampling, evaluation=True)
                else:
                    await generate_and_rm_group(state, samples, sampling, evaluation=True)

        assert seeds == ([17, 18] if enabled else [None, None])
        assert sampling == {"temperature": 0.5}

    def test_legacy_state_sizes_dp_balancing_from_resolved_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Legacy generation state derives its DP lanes and deterministic group seeds from resolved ownership."""
        args = _make_generate_args()
        args.sglang = make_sglang_config(enable_deterministic_inference=True, dp_size=3)
        args.rollout_seed = 17
        args.n_samples_per_prompt = 2
        vars(args).pop("sglang_dp_size", None)
        vars(args).pop("sglang_enable_deterministic_inference", None)
        monkeypatch.setattr(sglang_rollout.SingletonMeta, "_instances", {})
        monkeypatch.setattr(sglang_rollout, "load_tokenizer", lambda *args, **kwargs: object())
        monkeypatch.setattr(sglang_rollout, "load_processor", lambda *args, **kwargs: None)

        state = sglang_rollout.GenerateState(args)

        assert state.dp_counts == [0, 0, 0]
        assert state.group_sampling_seeds == [17, 18]
