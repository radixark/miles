import pytest
from tests.fixtures.rollout_integration import DEFAULT_DATA_ROWS, IntegrationEnvConfig

from miles.rollout.base_types import (
    GenerateFnInput,
    GenerateFnOutput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnTrainInput,
)
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.modular_rollout.compatibility import call_rollout_function, load_rollout_function
from miles.utils.misc import function_registry
from miles.utils.types import Sample


def _expected_sample(*, group_index: int | None) -> Sample:
    return Sample(
        group_index=group_index,
        index=0,
        prompt="What is 1+7?",
        tokens=[3838, 374, 220, 16, 10, 22, 30, 59, 79075, 90, 23, 92],
        multimodal_inputs=None,
        multimodal_train_inputs=None,
        response="\\boxed{8}",
        response_length=5,
        label="8",
        reward=1,
        loss_mask=None,
        weight_versions=[],
        rollout_log_probs=[-0.0, -0.0078125, -0.015625, -0.0234375, -0.03125],
        rollout_routed_experts=None,
        remove_sample=False,
        status=Sample.Status.COMPLETED,
        metadata={},
        train_metadata=None,
        non_generation_time=0.0,
        spec_info=Sample.SpecInfo(
            spec_accept_token_num=0, spec_draft_token_num=0, spec_verify_ct=0, completion_token_num=0
        ),
        prefix_cache_info=Sample.PrefixCacheInfo(cached_tokens=0, total_prompt_tokens=7),
    )


_MODULAR_ROLLOUT_BASE_ARGV = [
    "--rollout-function-path",
    "miles.rollout.modular_rollout.orchestration_train.SimpleTrainRolloutFn",
    "--eval-function-path",
    "miles.rollout.modular_rollout.orchestration_eval.SimpleEvalRolloutFn",
    "--custom-generate-function-path",
    "miles.rollout.modular_rollout.inference_wrapper.generate",
]

def _load_and_call_train(args, data_source):
    fn = load_rollout_function(
        RolloutFnConstructorInput(args=args, data_source=data_source),
        args.rollout_function_path,
    )
    return call_rollout_function(fn, RolloutFnTrainInput(rollout_id=0))


class TestSimpleRolloutFnIntegration:
    _VARIANTS = [
        pytest.param(
            IntegrationEnvConfig(
                extra_argv=[
                    "--rollout-function-path",
                    "miles.rollout.sglang_rollout.generate_rollout",
                    "--eval-function-path",
                    "miles.rollout.sglang_rollout.generate_rollout",
                    "--custom-generate-function-path",
                    "miles.rollout.sglang_rollout.generate",
                ]
            ),
            id="old_rollout_old_generate",
        ),
        pytest.param(
            IntegrationEnvConfig(
                extra_argv=[
                    "--rollout-function-path",
                    "miles.rollout.modular_rollout.orchestration_train.SimpleTrainRolloutFn",
                    "--eval-function-path",
                    "miles.rollout.modular_rollout.orchestration_eval.SimpleEvalRolloutFn",
                    "--custom-generate-function-path",
                    "miles.rollout.sglang_rollout.generate",
                ]
            ),
            id="new_rollout_old_generate",
        ),
        pytest.param(
            IntegrationEnvConfig(extra_argv=_MODULAR_ROLLOUT_BASE_ARGV),
            id="new_rollout_new_generate",
        ),
    ]

    @pytest.mark.parametrize("rollout_integration_env", _VARIANTS, indirect=True)
    def test_train(self, rollout_integration_env):
        env = rollout_integration_env
        out = _load_and_call_train(env.args, env.data_source)

        assert len(out.samples) == env.args.rollout_batch_size
        group = out.samples[0]
        assert len(group) == env.args.n_samples_per_prompt
        assert group[0] == _expected_sample(group_index=0)

    @pytest.mark.parametrize("rollout_integration_env", _VARIANTS, indirect=True)
    def test_eval(self, rollout_integration_env):
        env = rollout_integration_env
        fn = load_rollout_function(
            RolloutFnConstructorInput(args=env.args, data_source=env.data_source), env.args.eval_function_path
        )
        out = call_rollout_function(fn, RolloutFnEvalInput(rollout_id=0))

        assert "toy" in out.data
        rewards = out.data["toy"]["rewards"]
        samples = out.data["toy"]["samples"]
        assert len(rewards) == len(samples) == env.args.n_samples_per_eval_prompt
        assert rewards[0] == 1
        assert samples[0] == _expected_sample(group_index=None)


_MULTI_DATA_ROWS = [
    {"input": "What is 1+7?", "label": "8"},
    {"input": "What is 1+8?", "label": "9"},
    {"input": "What is 1+9?", "label": "wrong"},
    {"input": "What is 1+6?", "label": "7"},
]


def _config(extra_argv: list[str], data_rows: list[dict] | None = None, latency: float = 0.0):
    return IntegrationEnvConfig(
        extra_argv=_MODULAR_ROLLOUT_BASE_ARGV + extra_argv,
        data_rows=data_rows,
        latency=latency,
    )


class TestSemaphoreIntegration:
    _DATA_ROWS = [{"input": f"What is 1+{i}?", "label": str(1 + i)} for i in range(10)]

    @pytest.mark.parametrize(
        "rollout_integration_env,expected_range",
        [
            pytest.param(
                _config(
                    ["--sglang-server-concurrency", "1", "--rollout-batch-size", "4", "--n-samples-per-prompt", "2"],
                    data_rows=_DATA_ROWS,
                    latency=0.05,
                ),
                (1, 1),
                id="limit_1",
            ),
            pytest.param(
                _config(
                    ["--sglang-server-concurrency", "999", "--rollout-batch-size", "4", "--n-samples-per-prompt", "2"],
                    data_rows=_DATA_ROWS,
                    latency=0.05,
                ),
                (2, 999),
                id="no_limit",
            ),
        ],
        indirect=["rollout_integration_env"],
    )
    def test_max_concurrent(self, rollout_integration_env, expected_range):
        env = rollout_integration_env
        _load_and_call_train(env.args, env.data_source)
        min_expected, max_expected = expected_range
        assert min_expected <= env.mock_server.max_concurrent <= max_expected


class TestDeterministicInferenceIntegration:
    @pytest.mark.parametrize(
        "rollout_integration_env,expected_seeds",
        [
            pytest.param(
                _config(
                    [
                        "--sglang-enable-deterministic-inference",
                        "--rollout-seed",
                        "42",
                        "--n-samples-per-prompt",
                        "3",
                        "--rollout-batch-size",
                        "1",
                    ]
                ),
                {42, 43, 44},
                id="enabled",
            ),
            pytest.param(
                _config(["--n-samples-per-prompt", "2", "--rollout-batch-size", "1"]),
                {None},
                id="disabled",
            ),
        ],
        indirect=["rollout_integration_env"],
    )
    def test_sampling_seeds(self, rollout_integration_env, expected_seeds):
        env = rollout_integration_env
        _load_and_call_train(env.args, env.data_source)

        seeds = {req.get("sampling_params", {}).get("sampling_seed") for req in env.mock_server.request_log}
        assert seeds == expected_seeds


class TestGroupRMIntegration:
    @pytest.mark.parametrize(
        "rollout_integration_env",
        [
            pytest.param(
                _config(["--group-rm", "--n-samples-per-prompt", "2", "--rollout-batch-size", "1"]),
                id="group_rm_enabled",
            ),
        ],
        indirect=True,
    )
    def test_group_rm_rewards_set(self, rollout_integration_env):
        env = rollout_integration_env
        out = _load_and_call_train(env.args, env.data_source)

        assert len(out.samples) == env.args.rollout_batch_size
        for group in out.samples:
            for sample in group:
                assert sample.reward is not None


def _filter_by_reward(args, samples, **kwargs):
    reward = samples[0].reward if not isinstance(samples[0], list) else samples[0][0].reward
    if reward == 1:
        return DynamicFilterOutput(keep=True)
    return DynamicFilterOutput(keep=False, reason="reward_zero")


class TestOverSamplingIntegration:
    @pytest.mark.parametrize(
        "rollout_integration_env",
        [
            pytest.param(
                _config(
                    [
                        "--over-sampling-batch-size",
                        "2",
                        "--rollout-batch-size",
                        "2",
                        "--dynamic-sampling-filter-path",
                        "test:filter_by_reward",
                    ],
                    data_rows=[
                        {"input": "What is 1+7?", "label": "8"},
                        {"input": "What is 1+8?", "label": "9"},
                        {"input": "What is 1+9?", "label": "10"},
                    ],
                ),
                id="over_sampling_with_filter",
            ),
        ],
        indirect=True,
    )
    def test_over_sampling_with_dynamic_filter(self, rollout_integration_env):
        env = rollout_integration_env
        with function_registry.temporary("test:filter_by_reward", _filter_by_reward):
            out = _load_and_call_train(env.args, env.data_source)

            assert len(out.samples) == env.args.rollout_batch_size
            for group in out.samples:
                assert group[0].reward == 1


class TestDynamicFilterIntegration:
    @pytest.mark.parametrize(
        "rollout_integration_env",
        [
            pytest.param(
                _config(
                    [
                        "--rollout-batch-size",
                        "2",
                        "--dynamic-sampling-filter-path",
                        "test:filter_by_reward",
                    ],
                    data_rows=_MULTI_DATA_ROWS,
                ),
                id="dynamic_filter",
            ),
        ],
        indirect=True,
    )
    def test_dynamic_filter_only_keeps_correct(self, rollout_integration_env):
        env = rollout_integration_env
        with function_registry.temporary("test:filter_by_reward", _filter_by_reward):
            out = _load_and_call_train(env.args, env.data_source)

            assert len(out.samples) == env.args.rollout_batch_size
            for group in out.samples:
                assert group[0].reward == 1


class TestSampleFilterAndAllSamplesProcessIntegration:
    @pytest.mark.parametrize(
        "rollout_integration_env",
        [
            pytest.param(
                _config(
                    [
                        "--rollout-batch-size",
                        "2",
                        "--dynamic-sampling-filter-path",
                        "test:filter_by_reward",
                        "--rollout-sample-filter-path",
                        "test:sample_filter",
                        "--rollout-all-samples-process-path",
                        "test:all_samples_process",
                    ],
                    data_rows=_MULTI_DATA_ROWS,
                ),
                id="sample_filter_vs_all_samples",
            ),
        ],
        indirect=True,
    )
    def test_sample_filter_only_sees_unfiltered(self, rollout_integration_env):
        env = rollout_integration_env
        sample_filter_log = {"called": False, "data_len": None, "rewards": None}
        all_samples_log = {"called": False, "all_samples_len": None, "has_data_source": False}

        def sample_filter(args, data):
            sample_filter_log["called"] = True
            sample_filter_log["data_len"] = len(data)
            sample_filter_log["rewards"] = [g[0][0].reward if isinstance(g[0], list) else g[0].reward for g in data]

        def all_samples_process(args, all_samples, data_source):
            all_samples_log["called"] = True
            all_samples_log["all_samples_len"] = len(all_samples)
            all_samples_log["has_data_source"] = data_source is not None

        with (
            function_registry.temporary("test:filter_by_reward", _filter_by_reward),
            function_registry.temporary("test:sample_filter", sample_filter),
            function_registry.temporary("test:all_samples_process", all_samples_process),
        ):
            _load_and_call_train(env.args, env.data_source)

            assert sample_filter_log["called"]
            assert sample_filter_log["data_len"] == env.args.rollout_batch_size
            assert all(r == 1 for r in sample_filter_log["rewards"])

            assert all_samples_log["called"]
            assert all_samples_log["all_samples_len"] >= env.args.rollout_batch_size
            assert all_samples_log["has_data_source"]


async def _multi_sample_generate(input: GenerateFnInput) -> GenerateFnOutput:
    sample = input.sample
    s1 = Sample(
        prompt=sample.prompt,
        response="\\boxed{8}",
        response_length=5,
        tokens=sample.tokens + [59, 79075, 90, 23, 92],
        label=sample.label,
        reward=None,
        status=Sample.Status.COMPLETED,
    )
    s2 = Sample(
        prompt=sample.prompt,
        response="\\boxed{8}",
        response_length=5,
        tokens=sample.tokens + [59, 79075, 90, 23, 92],
        label=sample.label,
        reward=0.5,
        status=Sample.Status.COMPLETED,
    )
    return GenerateFnOutput(samples=[s1, s2])


class TestMultiSampleOutputIntegration:
    @pytest.mark.parametrize(
        "rollout_integration_env",
        [
            pytest.param(
                IntegrationEnvConfig(
                    extra_argv=_MODULAR_ROLLOUT_BASE_ARGV[:4]
                    + [
                        "--custom-generate-function-path",
                        "test:multi_sample_generate",
                        "--rollout-batch-size",
                        "1",
                        "--n-samples-per-prompt",
                        "1",
                    ],
                    data_rows=DEFAULT_DATA_ROWS,
                ),
                id="multi_sample_output",
            ),
        ],
        indirect=True,
    )
    def test_multi_sample_output_preserves_existing_reward(self, rollout_integration_env):
        env = rollout_integration_env
        with function_registry.temporary("test:multi_sample_generate", _multi_sample_generate):
            out = _load_and_call_train(env.args, env.data_source)

            assert len(out.samples) == env.args.rollout_batch_size
            group = out.samples[0]
            assert isinstance(group[0], list)
            samples = group[0]
            assert len(samples) == 2
            assert samples[0].reward == 1
            assert samples[1].reward == 0.5
