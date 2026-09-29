from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="stage-b-cpu", labels=[])

"""PR #12 of radixark/sgl-router-for-miles: keep prefill expert outputs when decode returns none.

Under PD disaggregation with routing replay (R3) the router merges ``meta_info.routed_experts`` from the prefill
response into the decode response. When prefill already produced the final token, decode runs no forward pass
and returns no expert field; before #12 the router then dropped the prefill rows and miles saw ``None``.

The scenario runs through miles' own generate functions against a real router fronting two scripted workers.
Not covered: the OpenAI ``sglext`` envelope when decode omits the whole object (pre-existing gap, not a miles path).
"""

import numpy as np
import pybase64
import pytest
from tests.fast.backends.sglang_utils.sgl_router_e2e.harness import (
    PROMPT,
    PROMPT_TOKENS,
    RESPONSE_TOKENS,
    SAMPLING_PARAMS,
    router_generate_env,
    single_sample,
)
from tests.fast.fixtures.generation_fixtures import make_sample, run_generate

from miles.utils.test_utils.mock_sglang_server import ProcessResult

NUM_LAYERS = 2
TOPK = 2
PROMPT_ROWS = len(PROMPT_TOKENS)  # one routing row per prompt position when the response is a single token
FULL_ROWS = len(PROMPT_TOKENS) + len(RESPONSE_TOKENS) - 1  # len(tokens) - 1 for the five-token response


def rows(start: int, count: int) -> np.ndarray:
    return np.arange(start, start + count * NUM_LAYERS * TOPK, dtype=np.int32).reshape(count, NUM_LAYERS, TOPK)


def b64(array: np.ndarray) -> str:
    return pybase64.b64encode(array.tobytes()).decode("ascii")


@pytest.fixture(scope="module")
def pd_env(router_log_dir):
    with router_generate_env(
        mode="pd",
        log_dir=router_log_dir,
        args_kwargs={"use_rollout_routing_replay": True, "num_layers": NUM_LAYERS, "moe_router_topk": TOPK},
    ) as env:
        yield env


@pytest.fixture(scope="module")
def one_token_text(pd_env) -> str:
    """A response that the mock tokenizer encodes as exactly one token (prefill produced the final token)."""
    tokenizer = pd_env.workers["decode"].tokenizer
    for text in ("8", "9", "1"):
        if len(tokenizer.encode(text, add_special_tokens=False)) == 1:
            return text
    pytest.fail("no single-token response candidate for this tokenizer")


def _set_prefill(env, routed: np.ndarray | None) -> None:
    overrides = {"input_token_logprobs": []}
    if routed is not None:
        overrides["routed_experts"] = b64(routed)
    env.workers["prefill"].script.meta_info_overrides = overrides


def _set_decode_rows(env, routed: np.ndarray) -> None:
    env.workers["decode"].script.meta_info_overrides = {"routed_experts": b64(routed)}


def _generate(env, variant: str):
    result = run_generate(env.generate_env(), make_sample(prompt=PROMPT), SAMPLING_PARAMS, variant=variant)
    return single_sample(result)


def _assert_dual_dispatch(env, *, want_routed_experts: bool) -> None:
    prefill, decode = env.workers["prefill"].request_log[-1], env.workers["decode"].request_log[-1]
    assert prefill["input_ids"] == PROMPT_TOKENS and decode["input_ids"] == PROMPT_TOKENS
    assert prefill["return_logprob"] is True and decode["return_logprob"] is True
    assert bool(prefill.get("return_routed_experts")) is want_routed_experts
    assert bool(decode.get("return_routed_experts")) is want_routed_experts
    assert "bootstrap_host" in prefill and "bootstrap_port" in prefill
    assert prefill["bootstrap_room"] == decode["bootstrap_room"]


def test_prefill_only_rows_are_kept(pd_env, variant, one_token_text):
    """Discriminating case: decode has no ``routed_experts``; miles must still receive the prefill rows."""
    pd_env.reset()
    prefill_rows = rows(0, PROMPT_ROWS)
    _set_prefill(pd_env, prefill_rows)
    one_token = ProcessResult(text=one_token_text, finish_reason="stop")
    pd_env.workers["prefill"].process_fn = lambda _prompt: one_token
    pd_env.workers["decode"].process_fn = lambda _prompt: one_token

    sample = _generate(pd_env, variant)

    _assert_dual_dispatch(pd_env, want_routed_experts=True)
    assert sample.response == one_token_text
    assert len(sample.tokens) == len(PROMPT_TOKENS) + 1
    assert sample.rollout_routed_experts is not None, "router dropped the prefill-only expert rows"
    assert sample.rollout_routed_experts.shape == (PROMPT_ROWS, NUM_LAYERS, TOPK)
    np.testing.assert_array_equal(sample.rollout_routed_experts, prefill_rows)


def test_both_sides_present_replaces_prompt_rows(pd_env, variant):
    """Control (pre-#12 behaviour): prefill rows replace decode's prompt prefix, decode's response rows stay."""
    pd_env.reset()
    prefill_rows = rows(0, PROMPT_ROWS)
    decode_tail = rows(1000, FULL_ROWS - PROMPT_ROWS)
    decode_rows = np.concatenate([np.full((PROMPT_ROWS, NUM_LAYERS, TOPK), -1, dtype=np.int32), decode_tail])
    _set_prefill(pd_env, prefill_rows)
    _set_decode_rows(pd_env, decode_rows)

    sample = _generate(pd_env, variant)

    _assert_dual_dispatch(pd_env, want_routed_experts=True)
    assert sample.tokens == PROMPT_TOKENS + RESPONSE_TOKENS
    np.testing.assert_array_equal(sample.rollout_routed_experts, np.concatenate([prefill_rows, decode_tail]))
    assert not (sample.rollout_routed_experts < 0).any()


def test_flag_off_leaves_decode_rows_untouched(pd_env, variant):
    """Control: without ``--use-rollout-routing-replay`` the router does not merge and miles keeps decode's rows."""
    pd_env.reset()
    decode_rows = np.concatenate(
        [np.full((PROMPT_ROWS, NUM_LAYERS, TOPK), -1, dtype=np.int32), rows(1000, FULL_ROWS - PROMPT_ROWS)]
    )
    _set_prefill(pd_env, rows(0, PROMPT_ROWS))
    _set_decode_rows(pd_env, decode_rows)
    pd_env.args.use_rollout_routing_replay = False
    try:
        sample = _generate(pd_env, variant)
    finally:
        pd_env.args.use_rollout_routing_replay = True

    _assert_dual_dispatch(pd_env, want_routed_experts=False)
    np.testing.assert_array_equal(sample.rollout_routed_experts, decode_rows)
    assert (sample.rollout_routed_experts[:PROMPT_ROWS] == -1).all()


def test_prefill_without_rows_passes_decode_through(pd_env, variant):
    """Control: no prefill expert data, so decode's rows arrive verbatim and nothing raises."""
    pd_env.reset()
    decode_rows = rows(0, FULL_ROWS)
    _set_prefill(pd_env, None)
    _set_decode_rows(pd_env, decode_rows)

    sample = _generate(pd_env, variant)

    _assert_dual_dispatch(pd_env, want_routed_experts=True)
    np.testing.assert_array_equal(sample.rollout_routed_experts, decode_rows)
