from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="stage-b-cpu", labels=[])

"""PR #18 of radixark/sgl-router-for-miles: forward ``sampling_params.custom_params`` to HTTP workers.

miles forwards the caller's ``sampling_params`` dict as-is (``compute_request_payload``), so a custom logit
processor's arguments travel inside it. Before #18 the router's typed schema dropped ``custom_params`` on both the
regular and the PD path. miles does not send a top-level ``custom_logit_processor``; processor behaviour on the
worker is out of scope here.
"""

import pytest
from tests.fast.backends.sglang_utils.sgl_router_e2e.harness import (
    PROMPT,
    SAMPLING_PARAMS,
    assert_baseline_generation,
    router_generate_env,
    single_sample,
)
from tests.fast.fixtures.generation_fixtures import make_sample, run_generate

CUSTOM_PARAMS = {"token_ids": [42, 43]}


@pytest.fixture(scope="module", params=["regular", "pd"])
def env(request, router_log_dir):
    with router_generate_env(mode=request.param, log_dir=router_log_dir) as env:
        yield env


def _generate(env, variant: str, sampling_params: dict):
    env.reset()
    result = run_generate(env.generate_env(), make_sample(prompt=PROMPT), sampling_params, variant=variant)
    return single_sample(result)


def test_custom_params_reach_every_worker(env, variant):
    """Discriminating case: the exact ``custom_params`` value must arrive at each worker behind the router."""
    sample = _generate(env, variant, {**SAMPLING_PARAMS, "custom_params": CUSTOM_PARAMS})

    for side, request in env.last_requests().items():
        sampling_params = request["sampling_params"]
        assert sampling_params.get("custom_params") == CUSTOM_PARAMS, (side, sampling_params)
        assert sampling_params["max_new_tokens"] == SAMPLING_PARAMS["max_new_tokens"], side
        # The PD router re-serializes typed fields through f32 (0.7 -> 0.699999988...); custom_params is an
        # opaque JSON value and must arrive byte-exact, so only the float is compared approximately.
        assert sampling_params["temperature"] == pytest.approx(SAMPLING_PARAMS["temperature"], rel=1e-6), side
    assert_baseline_generation(sample)


def test_omitted_custom_params_stay_absent(env, variant):
    sample = _generate(env, variant, dict(SAMPLING_PARAMS))

    for side, request in env.last_requests().items():
        assert "custom_params" not in request["sampling_params"], side
    assert_baseline_generation(sample)


def test_null_custom_params_are_not_forwarded(env, variant):
    """Wire contract: ``null`` is treated as unset (``skip_serializing_if = Option::is_none``)."""
    sample = _generate(env, variant, {**SAMPLING_PARAMS, "custom_params": None})

    for side, request in env.last_requests().items():
        assert "custom_params" not in request["sampling_params"], side
    assert_baseline_generation(sample)
