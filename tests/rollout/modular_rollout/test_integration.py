import pytest
from tests.fixtures.rollout_integration import IntegrationEnvConfig
from tests.rollout.modular_rollout import mock_hooks

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnTrainInput
from miles.rollout.modular_rollout.compatibility import call_rollout_function, load_rollout_function
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


_ROLLOUT_ARGV_VARIANTS = [
    pytest.param(
        [
            "--rollout-function-path",
            "miles.rollout.sglang_rollout.generate_rollout",
            "--eval-function-path",
            "miles.rollout.sglang_rollout.generate_rollout",
            "--custom-generate-function-path",
            "miles.rollout.sglang_rollout.generate",
        ],
        id="old_rollout_old_generate",
    ),
    pytest.param(
        [
            "--rollout-function-path",
            "miles.rollout.modular_rollout.orchestration_train.SimpleTrainRolloutFn",
            "--eval-function-path",
            "miles.rollout.modular_rollout.orchestration_eval.SimpleEvalRolloutFn",
            "--custom-generate-function-path",
            "miles.rollout.sglang_rollout.generate",
        ],
        id="new_rollout_old_generate",
    ),
    pytest.param(
        [
            "--rollout-function-path",
            "miles.rollout.modular_rollout.orchestration_train.SimpleTrainRolloutFn",
            "--eval-function-path",
            "miles.rollout.modular_rollout.orchestration_eval.SimpleEvalRolloutFn",
            "--custom-generate-function-path",
            "miles.rollout.modular_rollout.inference_wrapper.generate",
        ],
        id="new_rollout_new_generate",
    ),
]


def _load_and_call_train(args, data_source):
    fn = load_rollout_function(
        RolloutFnConstructorInput(args=args, data_source=data_source),
        args.rollout_function_path,
    )
    return call_rollout_function(fn, RolloutFnTrainInput(rollout_id=0))


@pytest.mark.parametrize("rollout_integration_env", _ROLLOUT_ARGV_VARIANTS, indirect=True)
def test_simple_train_rollout_fn_integration(rollout_integration_env):
    args, data_source, _ = rollout_integration_env
    out = _load_and_call_train(args, data_source)

    assert len(out.samples) == args.rollout_batch_size
    group = out.samples[0]
    assert len(group) == args.n_samples_per_prompt
    assert group[0] == _expected_sample(group_index=0)


@pytest.mark.parametrize("rollout_integration_env", _ROLLOUT_ARGV_VARIANTS, indirect=True)
def test_simple_eval_rollout_fn_integration(rollout_integration_env):
    args, data_source, _ = rollout_integration_env
    fn = load_rollout_function(RolloutFnConstructorInput(args=args, data_source=data_source), args.eval_function_path)
    out = call_rollout_function(fn, RolloutFnEvalInput(rollout_id=0))

    assert "toy" in out.data
    rewards = out.data["toy"]["rewards"]
    samples = out.data["toy"]["samples"]
    assert len(rewards) == len(samples) == args.n_samples_per_eval_prompt
    assert rewards[0] == 1
    assert samples[0] == _expected_sample(group_index=None)


_DEFAULT_DATA_ROWS = [{"input": "What is 1+7?", "label": "8"}]

_MODULAR_ROLLOUT_BASE_ARGV = [
    "--rollout-function-path",
    "miles.rollout.modular_rollout.orchestration_train.SimpleTrainRolloutFn",
    "--eval-function-path",
    "miles.rollout.modular_rollout.orchestration_eval.SimpleEvalRolloutFn",
    "--custom-generate-function-path",
    "miles.rollout.modular_rollout.inference_wrapper.generate",
]

_MULTI_DATA_ROWS = [
    {"input": "What is 1+7?", "label": "8"},
    {"input": "What is 1+8?", "label": "9"},
    {"input": "What is 1+9?", "label": "wrong"},
    {"input": "What is 1+6?", "label": "7"},
]


def _config(extra_argv: list[str], data_rows: list[dict] | None = None, latency: float = 0.0):
    return IntegrationEnvConfig(
        extra_argv=_MODULAR_ROLLOUT_BASE_ARGV + extra_argv,
        data_rows=data_rows or _DEFAULT_DATA_ROWS,
        latency=latency,
    )


class TestSemaphoreIntegration:
    @pytest.mark.parametrize(
        "rollout_integration_env",
        [
            pytest.param(
                _config(
                    ["--sglang-server-concurrency", "1", "--rollout-batch-size", "4", "--n-samples-per-prompt", "2"],
                    data_rows=[{"input": f"What is 1+{i}?", "label": str(1 + i)} for i in range(10)],
                    latency=0.05,
                ),
                id="semaphore_limit_1",
            ),
        ],
        indirect=True,
    )
    def test_max_concurrent_respects_semaphore(self, rollout_integration_env):
        args, data_source, mock_server = rollout_integration_env
        _load_and_call_train(args, data_source)
        assert mock_server.max_concurrent <= args.sglang_server_concurrency


class TestDeterministicInferenceIntegration:
    @pytest.mark.parametrize(
        "rollout_integration_env",
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
                id="deterministic_enabled",
            ),
        ],
        indirect=True,
    )
    def test_sampling_seeds_set_correctly(self, rollout_integration_env):
        args, data_source, mock_server = rollout_integration_env
        _load_and_call_train(args, data_source)

        seeds = [req.get("sampling_params", {}).get("sampling_seed") for req in mock_server.request_log]
        assert set(seeds) == {42, 43, 44}

    @pytest.mark.parametrize(
        "rollout_integration_env",
        [
            pytest.param(
                _config(["--n-samples-per-prompt", "2", "--rollout-batch-size", "1"]),
                id="deterministic_disabled",
            ),
        ],
        indirect=True,
    )
    def test_no_sampling_seeds_when_disabled(self, rollout_integration_env):
        args, data_source, mock_server = rollout_integration_env
        _load_and_call_train(args, data_source)

        seeds = [req.get("sampling_params", {}).get("sampling_seed") for req in mock_server.request_log]
        assert all(seed is None for seed in seeds)


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
        args, data_source, _ = rollout_integration_env
        out = _load_and_call_train(args, data_source)

        assert len(out.samples) == args.rollout_batch_size
        for group in out.samples:
            for sample in group:
                assert sample.reward is not None


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
                        "tests.rollout.modular_rollout.mock_hooks.filter_by_reward",
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
        args, data_source, _ = rollout_integration_env
        out = _load_and_call_train(args, data_source)

        assert len(out.samples) == args.rollout_batch_size
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
                        "tests.rollout.modular_rollout.mock_hooks.filter_by_reward",
                    ],
                    data_rows=_MULTI_DATA_ROWS,
                ),
                id="dynamic_filter",
            ),
        ],
        indirect=True,
    )
    def test_dynamic_filter_only_keeps_correct(self, rollout_integration_env):
        args, data_source, _ = rollout_integration_env
        out = _load_and_call_train(args, data_source)

        assert len(out.samples) == args.rollout_batch_size
        for group in out.samples:
            assert group[0].reward == 1


_SAMPLE_FILTER_ARGV = [
    "--rollout-batch-size",
    "2",
    "--dynamic-sampling-filter-path",
    "tests.rollout.modular_rollout.mock_hooks.filter_by_reward",
    "--rollout-sample-filter-path",
    "tests.rollout.modular_rollout.mock_hooks.sample_filter_hook",
    "--rollout-all-samples-process-path",
    "tests.rollout.modular_rollout.mock_hooks.all_samples_process_hook",
]


class TestSampleFilterAndAllSamplesProcessIntegration:
    @pytest.mark.parametrize(
        "rollout_integration_env",
        [pytest.param(_config(_SAMPLE_FILTER_ARGV, data_rows=_MULTI_DATA_ROWS), id="sample_filter_vs_all_samples")],
        indirect=True,
    )
    def test_sample_filter_only_sees_unfiltered(self, rollout_integration_env):
        mock_hooks.reset_sample_filter_call_log()
        mock_hooks.reset_all_samples_process_call_log()

        args, data_source, _ = rollout_integration_env
        _load_and_call_train(args, data_source)

        assert mock_hooks.sample_filter_call_log["called"]
        assert mock_hooks.sample_filter_call_log["data_len"] == args.rollout_batch_size
        assert all(r == 1 for r in mock_hooks.sample_filter_call_log["rewards"])

    @pytest.mark.parametrize(
        "rollout_integration_env",
        [pytest.param(_config(_SAMPLE_FILTER_ARGV, data_rows=_MULTI_DATA_ROWS), id="all_samples_sees_filtered")],
        indirect=True,
    )
    def test_all_samples_process_sees_filtered(self, rollout_integration_env):
        mock_hooks.reset_sample_filter_call_log()
        mock_hooks.reset_all_samples_process_call_log()

        args, data_source, _ = rollout_integration_env
        _load_and_call_train(args, data_source)

        assert mock_hooks.all_samples_process_call_log["called"]
        assert mock_hooks.all_samples_process_call_log["all_samples_len"] >= args.rollout_batch_size
        assert mock_hooks.all_samples_process_call_log["has_data_source"]
        assert all(r == 1 for r in mock_hooks.sample_filter_call_log["rewards"])


class TestMultiSampleOutputIntegration:
    @pytest.mark.parametrize(
        "rollout_integration_env",
        [
            pytest.param(
                IntegrationEnvConfig(
                    extra_argv=_MODULAR_ROLLOUT_BASE_ARGV[:4]
                    + [
                        "--custom-generate-function-path",
                        "tests.rollout.modular_rollout.mock_hooks.multi_sample_generate",
                        "--rollout-batch-size",
                        "1",
                        "--n-samples-per-prompt",
                        "1",
                    ],
                    data_rows=_DEFAULT_DATA_ROWS,
                ),
                id="multi_sample_output",
            ),
        ],
        indirect=True,
    )
    def test_multi_sample_output_preserves_existing_reward(self, rollout_integration_env):
        args, data_source, _ = rollout_integration_env
        out = _load_and_call_train(args, data_source)

        assert len(out.samples) == args.rollout_batch_size
        group = out.samples[0]
        assert isinstance(group[0], list)
        samples = group[0]
        assert len(samples) == 2
        assert samples[0].reward == 1
        assert samples[1].reward == 0.5
