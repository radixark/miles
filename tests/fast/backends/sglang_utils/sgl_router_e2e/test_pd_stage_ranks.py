from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=[])

"""PR #16 of radixark/sgl-router-for-miles: caller-selected prefill and decode ranks on HTTP ``/generate``.

The router accepts ``routed_prefill_dp_rank`` / ``routed_decode_dp_rank`` (falling back to the legacy
``routed_dp_rank`` / ``data_parallel_rank``), sends each stage its own execution rank as ``routed_dp_rank`` and the
resolved prefill rank to both stages as ``disagg_prefill_dp_rank`` (the KV-transfer hint SGLang's prefill side also
validates). Stage fields never reach the workers.

miles has no caller-side support for stage ranks yet, so this is a router contract check driven through miles'
request payload and HTTP client. Retract/resume replay on the SGLang worker side is out of scope.
"""

import pytest
from tests.fast.backends.sglang_utils.sgl_router_e2e.harness import PROMPT_TOKENS, SAMPLING_PARAMS, router_generate_env

from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload
from miles.utils.async_utils import run
from miles.utils.http_utils import post

# name -> (injected fields, (prefill.routed_dp_rank, decode.routed_dp_rank, prefill.hint, decode.hint))
CASES = {
    "separate": ({"routed_prefill_dp_rank": 3, "routed_decode_dp_rank": 37}, (3, 37, 3, 3)),
    "prefill-override": ({"routed_prefill_dp_rank": 3, "routed_dp_rank": 5}, (3, 5, 3, 3)),
    "decode-override": ({"routed_decode_dp_rank": 37, "routed_dp_rank": 5}, (5, 37, 5, 5)),
    "legacy": ({"routed_dp_rank": 5}, (5, 5, 5, 5)),
    "alias": ({"data_parallel_rank": 5}, (5, 5, 5, 5)),
    "precedence": ({"routed_dp_rank": 5, "data_parallel_rank": 6}, (5, 5, 5, 5)),
    "null-stages": (
        {"routed_prefill_dp_rank": None, "routed_decode_dp_rank": None, "routed_dp_rank": 5},
        (5, 5, 5, 5),
    ),
    "no-rank": ({}, (None, None, None, None)),
    "hint-only": ({"disagg_prefill_dp_rank": 9}, (None, None, 9, 9)),
    "prefill-only": ({"routed_prefill_dp_rank": 3}, (3, None, 3, 3)),
    "decode-only": ({"routed_decode_dp_rank": 37}, (None, 37, None, None)),
    "batch": ({"routed_prefill_dp_rank": 3, "routed_decode_dp_rank": 37}, (3, 37, 3, 3)),
    "explicit-prefill-hint": (
        {"routed_prefill_dp_rank": 3, "routed_decode_dp_rank": 37, "disagg_prefill_dp_rank": 9},
        (3, 37, 3, 3),
    ),
}
STAGE_ONLY_FIELDS = ("routed_prefill_dp_rank", "routed_decode_dp_rank")


@pytest.fixture(scope="module")
def pd_env(router_log_dir):
    with router_generate_env(mode="pd", log_dir=router_log_dir) as env:
        yield env


@pytest.mark.parametrize("name", list(CASES))
def test_stage_ranks_reach_each_worker(pd_env, name):
    injected, expected = CASES[name]
    pd_env.reset()
    payload, status = compute_request_payload(pd_env.args, PROMPT_TOKENS, SAMPLING_PARAMS)
    assert status is None
    payload.update(injected)
    if name == "batch":
        payload["input_ids"] = [PROMPT_TOKENS, PROMPT_TOKENS[:3]]

    run(post(f"{pd_env.router.url}/generate", payload, max_retries=1))

    prefill, decode = pd_env.workers["prefill"].request_log[-1], pd_env.workers["decode"].request_log[-1]
    got = (
        prefill.get("routed_dp_rank"),
        decode.get("routed_dp_rank"),
        prefill.get("disagg_prefill_dp_rank"),
        decode.get("disagg_prefill_dp_rank"),
    )
    assert got == expected, {"prefill": prefill, "decode": decode}
    for side, request in (("prefill", prefill), ("decode", decode)):
        assert not any(field in request for field in STAGE_ONLY_FIELDS), side
        assert "data_parallel_rank" not in request, side
        assert request["input_ids"] == payload["input_ids"], side
    assert prefill["bootstrap_room"] == decode["bootstrap_room"]
