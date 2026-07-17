from tests.ci.ci_register import register_cuda_ci

# `no_limit` asserts max_concurrent >= 2 with --sglang-server-concurrency=999
# and 50ms per-request latency. On CPU CI runners the request dispatch loop
# serializes faster than the latency window, so observed max drops to 1 and
# the assertion fails. Pinned to GPU until the assertion is rewritten to be
# scheduler-independent.
register_cuda_ci(
    est_time=60,
    suite="stage-b-2-gpu-h200",
    labels=[],
    disabled="FIXME: re-enable after shared HTTP client concurrency is reset between cases.",
)

import pytest

from tests.fast.rollout.inference_rollout.integration.utils import integration_env_config, load_and_call_train

_DATA_ROWS = [{"input": f"What is 1+{i}?", "label": str(1 + i)} for i in range(10)]
_BASE_ARGV = ["--rollout-batch-size", "4", "--n-samples-per-prompt", "2"]


@pytest.mark.parametrize(
    "rollout_env,expected_range",
    [
        pytest.param(
            integration_env_config(
                ["--sglang-server-concurrency", "1"] + _BASE_ARGV, data_rows=_DATA_ROWS, latency=0.05
            ),
            (1, 1),
            id="limit_1",
        ),
        pytest.param(
            integration_env_config(
                ["--sglang-server-concurrency", "999"] + _BASE_ARGV, data_rows=_DATA_ROWS, latency=0.05
            ),
            (2, 999),
            id="no_limit",
        ),
    ],
    indirect=["rollout_env"],
)
def test_max_concurrent(rollout_env, expected_range):
    env = rollout_env
    load_and_call_train(env.args, env.data_source)
    min_expected, max_expected = expected_range
    assert min_expected <= env.mock_server.max_concurrent <= max_expected


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
