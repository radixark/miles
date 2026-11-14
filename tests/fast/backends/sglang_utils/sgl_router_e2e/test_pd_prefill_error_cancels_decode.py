from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=[])

"""PR #17 of radixark/sgl-router-for-miles: cancel decode as soon as prefill returns an error status.

A prefill worker can send HTTP 500 headers and then stall while writing the error body. Before #17 the router only
cancelled the decode side after that body had been read, so decode stayed connected (holding a PD bootstrap that
could never complete) for the whole stall. The client must still receive the original prefill status and message.

The requests use the payload miles builds (``compute_request_payload``) and miles' HTTP client (``post``) with a
single attempt; router retries and the circuit breaker are disabled through the same ``--router-*`` argument path
production uses, otherwise each retry would replay the stall.
"""

import time

import httpx
import pytest
from tests.fast.backends.sglang_utils.sgl_router_e2e.harness import (
    PROMPT,
    PROMPT_TOKENS,
    SAMPLING_PARAMS,
    assert_baseline_generation,
    router_generate_env,
    single_sample,
)
from tests.fast.fixtures.generation_fixtures import make_sample, run_generate

from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload
from miles.utils.async_utils import run
from miles.utils.http_utils import post
from miles.utils.test_utils.mock_sglang_pd_worker import ErrorScript, HoldScript

PREFILL_ERROR_BODY = b"original prefill failure"
BODY_DELAY_S = 1.5


@pytest.fixture(scope="module")
def pd_env(router_log_dir):
    with router_generate_env(
        mode="pd",
        log_dir=router_log_dir,
        args_kwargs={"extra_argv": ["--router-disable-retries", "--router-disable-circuit-breaker"]},
    ) as env:
        assert "--disable-retries" in env.router.argv and "--disable-circuit-breaker" in env.router.argv
        yield env


def _payload(env) -> dict:
    payload, status = compute_request_payload(env.args, PROMPT_TOKENS, SAMPLING_PARAMS)
    assert status is None
    return payload


def _wait_for(predicate, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.mark.parametrize("mode", ["await_headers", "holding_response"])
def test_decode_is_cancelled_before_the_prefill_error_body_arrives(pd_env, mode):
    """Discriminating case, one per decode state: still awaiting headers, or already holding a response."""
    pd_env.reset()
    pd_env.workers["prefill"].script.error = ErrorScript(body=PREFILL_ERROR_BODY, body_delay_s=BODY_DELAY_S)
    pd_env.workers["decode"].script.hold = HoldScript(mode=mode)
    scenario = pd_env.scenario

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        run(post(f"{pd_env.router.url}/generate", _payload(pd_env), max_retries=1))

    response = excinfo.value.response
    assert response.status_code == 500, response.text
    assert PREFILL_ERROR_BODY.decode() in response.text
    assert _wait_for(lambda: scenario.prefill_body_done_at is not None, 2.0), "prefill body never completed"
    assert scenario.prefill_headers_sent_at is not None
    assert scenario.decode_disconnected_at is not None, "decode never observed a disconnect from the router"
    since_headers = scenario.decode_disconnected_at - scenario.prefill_headers_sent_at
    # Before #17 the router only dropped decode after reading the whole error body, i.e. >= BODY_DELAY_S later.
    assert since_headers < BODY_DELAY_S / 2, f"decode was cut {since_headers:.3f}s after the prefill error headers"
    assert scenario.decode_disconnected_at < scenario.prefill_body_done_at


def test_router_still_serves_after_prefill_errors(pd_env):
    """Control: with the breaker disabled the same router keeps serving a healthy request afterwards."""
    pd_env.reset()
    result = run_generate(pd_env.generate_env(), make_sample(prompt=PROMPT), SAMPLING_PARAMS, variant="single_turn")
    assert_baseline_generation(single_sample(result))
