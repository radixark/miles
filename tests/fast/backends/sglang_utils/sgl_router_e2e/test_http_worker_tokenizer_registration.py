from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="stage-b-cpu", labels=[])

"""PR #13 of radixark/sgl-router-for-miles: use configured tokenizer paths for HTTP workers.

An HTTP worker advertises ``model_path``/``tokenizer_path`` that exist only on its own host (a pod-local model
volume in miles deployments). Before #13 the router queued a tokenizer load for that path and failed; when the
router had an explicit ``--tokenizer-path``/``--model-path`` the worker label still took precedence, so lookups by
the worker's model name returned 400. Now HTTP workers use the router's configured path under the worker's model
name, or skip registration when none is configured.

The router log level is raised to ``info`` so the skip message is observable; everything else is the production
argument path.
"""

import json
import re
import time

import httpx
import pytest
from tests.fast.backends.sglang_utils.sgl_router_e2e.harness import (
    PROMPT,
    SAMPLING_PARAMS,
    assert_baseline_generation,
    list_workers,
    router_generate_env,
    single_sample,
)
from tests.fast.fixtures.generation_fixtures import make_sample, run_generate

BACKEND_ONLY_PATH = "/backend-only/models/chat-model"
SERVED_MODEL_NAME = "chat-model"
WORKER_KWARGS = {
    "served_model_name": SERVED_MODEL_NAME,
    "advertised_model_path": BACKEND_ONLY_PATH,
    "advertised_tokenizer_path": BACKEND_ONLY_PATH,
}
SKIP_LINE = f"Skipping automatic tokenizer registration for HTTP model {SERVED_MODEL_NAME}"


def _env(router_log_dir, extra_router_args: dict):
    return router_generate_env(
        mode="regular",
        log_dir=router_log_dir,
        worker_kwargs=WORKER_KWARGS,
        extra_router_args={"log_level": "info", **extra_router_args},
    )


def _tokenizers_text(router) -> str:
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{router.url}/v1/tokenizers")
    assert response.status_code == 200, response.text
    return json.dumps(response.json())


def _wait_until(predicate, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return predicate()


def test_backend_only_paths_do_not_break_http_worker_registration(router_log_dir):
    """Discriminating case: no router tokenizer configured, worker paths unreachable -> no load attempt, healthy."""
    with _env(router_log_dir, {}) as env:
        worker = env.workers["regular"]
        assert any(w["url"] == worker.url and w["is_healthy"] for w in list_workers(env.router))
        assert _wait_until(lambda: SKIP_LINE in env.router.log_text(), 10.0), env.router.log_text()[-4000:]
        assert BACKEND_ONLY_PATH not in _tokenizers_text(env.router)
        for line in env.router.log_text().splitlines():
            if BACKEND_ONLY_PATH in line:
                assert not re.search(r"error|fail", line, re.IGNORECASE), line

        result = run_generate(env.generate_env(), make_sample(prompt=PROMPT), SAMPLING_PARAMS, variant="single_turn")
        assert_baseline_generation(single_sample(result))


@pytest.mark.parametrize("config_key", ["tokenizer_path", "model_path"])
def test_explicit_router_tokenizer_is_registered_under_the_worker_model_name(
    router_log_dir, tiny_tokenizer_dir, config_key
):
    """Discriminating case: the router-configured tokenizer must stay reachable by the worker's model name."""
    with _env(router_log_dir, {config_key: str(tiny_tokenizer_dir)}) as env:
        assert _wait_until(lambda: SERVED_MODEL_NAME in _tokenizers_text(env.router), 15.0), _tokenizers_text(
            env.router
        )
        with httpx.Client(timeout=10.0) as client:
            tokenize = client.post(
                f"{env.router.url}/v1/tokenize", json={"model": SERVED_MODEL_NAME, "prompt": "hello world"}
            )
            assert tokenize.status_code == 200, tokenize.text
            assert tokenize.json()["tokens"] == [1, 2]

            detokenize = client.post(
                f"{env.router.url}/v1/detokenize", json={"model": SERVED_MODEL_NAME, "tokens": [1, 2]}
            )
            assert detokenize.status_code == 200, detokenize.text
            assert "hello world" in json.dumps(detokenize.json())

            by_path = client.post(
                f"{env.router.url}/v1/tokenize", json={"model": str(tiny_tokenizer_dir), "prompt": "hello world"}
            )
            assert by_path.status_code == 200, by_path.text

        result = run_generate(env.generate_env(), make_sample(prompt=PROMPT), SAMPLING_PARAMS, variant="single_turn")
        assert_baseline_generation(single_sample(result))
