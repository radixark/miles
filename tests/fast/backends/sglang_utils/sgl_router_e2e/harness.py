"""Launch a real sgl-model-gateway the way miles does and front it with scripted mock workers.

The router argv comes from ``compute_sglang_router_args`` + ``router_args_to_argv`` (the path
``miles/ray/specs/inference.py`` uses) and workers register through ``SGLangRouterApiClient.add_worker`` (the path
``miles/ray/rollout/server_cell.py`` uses). Tests then drive miles' own generate functions at the router URL.
"""

from __future__ import annotations

import atexit
import functools
import os
import subprocess
import sys
import time
from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.fast.fixtures.generation_fixtures import GenerateEnv, make_args

from miles.backends.sglang_utils.router_args_utils import compute_sglang_router_args, router_args_to_argv
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient, use_legacy_router_api
from miles.utils.async_utils import run
from miles.utils.http_utils import find_available_port
from miles.utils.misc import SingletonMeta
from miles.utils.test_utils.mock_sglang_pd_worker import PDScenario, ScriptedSGLangWorker
from miles.utils.test_utils.mock_sglang_server import ProcessFn, default_process_fn
from miles.utils.workers.process_utils import terminate_process_tree

ROUTER_READY_TIMEOUT_S = 60.0
WORKER_HEALTHY_TIMEOUT_S = 30.0
REGISTER_RETRY_TIMEOUT_S = 15.0
PREFILL_BOOTSTRAP_PORT = 8998

# Strings that only exist in a sgl-router-for-miles build that contains the merged PRs (#13, #16, #15).
FORK_RUST_MARKERS = {
    "pr13_http_tokenizer_skip": b"Skipping automatic tokenizer registration for HTTP model",
    "pr16_routed_prefill_dp_rank": b"routed_prefill_dp_rank",
}
FORK_PYTHON_MARKERS = {"pr15_mini_lb_merge_prefill_json": b"_merge_prefill_json"}
REQUIRE_FORK_ROUTER_ENV = "MILES_TEST_REQUIRE_FORK_ROUTER"
# Run against whatever router is installed even without the fork markers (used to prove the tests fail on the
# pre-PR base build; never set in CI).
FORCE_FORK_ROUTER_TESTS_ENV = "MILES_TEST_FORCE_FORK_ROUTER_TESTS"


@functools.lru_cache(maxsize=1)
def fork_router_status() -> tuple[bool, dict[str, bool], str]:
    """(ready, markers, description) for the installed ``sglang_router`` package."""
    import sglang_router

    package_dir = Path(sglang_router.__file__).resolve().parent
    so_bytes = b"".join(p.read_bytes() for p in sorted(package_dir.glob("sglang_router_rs*.so")))
    mini_lb = package_dir / "mini_lb.py"
    mini_lb_bytes = mini_lb.read_bytes() if mini_lb.exists() else b""
    markers = {name: needle in so_bytes for name, needle in FORK_RUST_MARKERS.items()}
    markers.update({name: needle in mini_lb_bytes for name, needle in FORK_PYTHON_MARKERS.items()})
    version = getattr(sglang_router, "__version__", "?")
    description = f"sglang_router {version} at {package_dir}"
    return all(markers.values()), markers, description


@dataclass
class RouterProcess:
    host: str
    port: int
    prometheus_port: int
    argv: list[str]
    process: subprocess.Popen
    log_path: Path

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def log_text(self) -> str:
        return self.log_path.read_text(errors="replace") if self.log_path.exists() else ""

    def is_running(self) -> bool:
        return self.process.poll() is None

    def stop(self) -> None:
        if self.is_running():
            terminate_process_tree(self.process)


def launch_router_for_test(
    args: Namespace,
    *,
    has_pd_disaggregation: bool,
    port: int,
    log_dir: Path,
    extra_router_args: dict[str, Any] | None = None,
) -> RouterProcess:
    prometheus_port = find_available_port(21000)
    router_args = compute_sglang_router_args(
        args,
        host="127.0.0.1",
        port=port,
        prometheus_port=prometheus_port,
        has_pd_disaggregation=has_pd_disaggregation,
    )
    if extra_router_args:
        router_args.update(extra_router_args)
    argv = [sys.executable, "-m", "sglang_router.launch_router", *router_args_to_argv(router_args)]

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"router-{port}.log"
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            argv,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    router = RouterProcess(
        host="127.0.0.1", port=port, prometheus_port=prometheus_port, argv=argv, process=process, log_path=log_path
    )
    atexit.register(router.stop)
    try:
        _wait_router_ready(router)
    except Exception:
        router.stop()
        raise
    return router


def _wait_router_ready(router: RouterProcess) -> None:
    deadline = time.monotonic() + ROUTER_READY_TIMEOUT_S
    started = time.monotonic()
    last_error = "no attempt"
    with httpx.Client(timeout=2.0) as client:
        while time.monotonic() < deadline:
            if not router.is_running():
                raise RuntimeError(
                    f"router exited with {router.process.returncode}; log:\n{router.log_text()[-4000:]}"
                )
            try:
                response = client.get(f"{router.url}/health")
                # Prefer a real 200; after a grace period any HTTP answer counts as "accepting connections".
                if response.status_code == 200 or time.monotonic() - started > 10.0:
                    return
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as e:
                last_error = repr(e)
            time.sleep(0.1)
    raise TimeoutError(f"router not ready after {ROUTER_READY_TIMEOUT_S}s: {last_error}\n{router.log_text()[-4000:]}")


def register_worker(
    router: RouterProcess,
    worker: ScriptedSGLangWorker,
    *,
    args: Namespace,
    worker_type: str,
    bootstrap_port: int | None = None,
) -> None:
    use_legacy_api = use_legacy_router_api(args)
    assert not use_legacy_api, "these tests target the /workers API of the Rust router"

    async def _add() -> None:
        await SGLangRouterApiClient(router_url=router.url).add_worker(
            worker.url, worker_type=worker_type, use_legacy_api=use_legacy_api, bootstrap_port=bootstrap_port
        )

    # The control plane can answer /health slightly before its job queue accepts registrations.
    deadline = time.monotonic() + REGISTER_RETRY_TIMEOUT_S
    while True:
        try:
            run(_add())
            break
        except httpx.HTTPStatusError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)
    _wait_worker_healthy(router, worker.url)


def _wait_worker_healthy(router: RouterProcess, worker_url: str) -> None:
    deadline = time.monotonic() + WORKER_HEALTHY_TIMEOUT_S
    seen: Any = None
    with httpx.Client(timeout=5.0) as client:
        while time.monotonic() < deadline:
            response = client.get(f"{router.url}/workers")
            if response.status_code == 200:
                seen = response.json()
                for entry in seen.get("workers", []):
                    if entry.get("url") == worker_url and entry.get("is_healthy"):
                        return
            time.sleep(0.1)
    raise TimeoutError(
        f"worker {worker_url} not healthy after {WORKER_HEALTHY_TIMEOUT_S}s; /workers={seen}\n{router.log_text()[-4000:]}"
    )


def list_workers(router: RouterProcess) -> list[dict[str, Any]]:
    with httpx.Client(timeout=5.0) as client:
        return client.get(f"{router.url}/workers").json()["workers"]


@dataclass
class RouterGenerateEnv:
    args: Namespace
    router: RouterProcess
    workers: dict[str, ScriptedSGLangWorker]
    scenario: PDScenario
    mode: str

    def generate_env(self, capture: str | None = None) -> GenerateEnv:
        """Adapter for ``generation_fixtures.run_generate``; ``capture`` names the worker whose request log it reads."""
        if capture is None:
            capture = "decode" if self.mode == "pd" else "regular"
        return GenerateEnv(args=self.args, mock_server=self.workers[capture])

    def reset(self) -> None:
        """Start a new staged request: fresh scenario object, plain scripts, no handler from a previous case alive."""
        self.wait_idle()
        self.scenario = PDScenario()
        for worker in self.workers.values():
            worker.scenario = self.scenario
            worker.script.reset()
            worker.request_log.clear()
            worker.flush_cache_calls.clear()
            worker.process_fn = default_process_fn

    def wait_idle(self, timeout_s: float = 10.0) -> None:
        deadline = time.monotonic() + timeout_s
        while any(worker.inflight for worker in self.workers.values()):
            if time.monotonic() >= deadline:
                raise TimeoutError({side: worker.inflight for side, worker in self.workers.items()})
            time.sleep(0.01)

    def last_requests(self) -> dict[str, dict]:
        return {side: worker.request_log[-1] for side, worker in self.workers.items()}


@contextmanager
def router_generate_env(
    *,
    mode: str,
    log_dir: Path,
    variant: str = "single_turn",
    args_kwargs: dict[str, Any] | None = None,
    extra_router_args: dict[str, Any] | None = None,
    worker_kwargs: dict[str, Any] | None = None,
    process_fn: ProcessFn = default_process_fn,
) -> Iterator[RouterGenerateEnv]:
    """Start mock worker(s) behind a freshly launched router and build miles args that point at the router.

    ``mode`` is ``"regular"`` (one worker) or ``"pd"`` (prefill + decode). ``variant`` only selects the argv
    defaults; ``run_generate(..., variant=...)`` chooses the generate function at call time, so one env serves both
    ``old_sglang_rollout`` and ``single_turn``.
    """
    assert mode in ("regular", "pd"), mode
    SingletonMeta.clear_all_instances()
    port = find_available_port(20000)
    args = make_args(variant=variant, router_port=port, **(args_kwargs or {}))

    scenario = PDScenario()
    workers: dict[str, ScriptedSGLangWorker] = {}
    router: RouterProcess | None = None
    try:
        for side in ["prefill", "decode"] if mode == "pd" else ["regular"]:
            worker = ScriptedSGLangWorker(side=side, scenario=scenario, process_fn=process_fn, **(worker_kwargs or {}))
            worker.start()
            workers[side] = worker
        router = launch_router_for_test(
            args, has_pd_disaggregation=(mode == "pd"), port=port, log_dir=log_dir, extra_router_args=extra_router_args
        )
        for side, worker in workers.items():
            register_worker(
                router,
                worker,
                args=args,
                worker_type=side,
                bootstrap_port=PREFILL_BOOTSTRAP_PORT if side == "prefill" else None,
            )
        yield RouterGenerateEnv(args=args, router=router, workers=workers, scenario=scenario, mode=mode)
    finally:
        if router is not None:
            router.stop()
        for worker in workers.values():
            worker.stop()
        SingletonMeta.clear_all_instances()


# Baseline generation used by every test file here (same prompt/response as tests/fast/rollout/generate_hub).
PROMPT = "What is 1+7?"
PROMPT_TOKENS = [3838, 374, 220, 16, 10, 22, 30]
RESPONSE_TEXT = "\\boxed{8}"
RESPONSE_TOKENS = [59, 79075, 90, 23, 92]
RESPONSE_LOG_PROBS = [-i / 128 for i in range(len(RESPONSE_TOKENS))]
SAMPLING_PARAMS = {"max_new_tokens": 16, "temperature": 0.7}
GENERATE_VARIANTS = ["old_sglang_rollout", "single_turn"]


def single_sample(result):
    """``run_generate`` returns one Sample or a one-element list depending on the variant."""
    return result.sample[0] if isinstance(result.sample, list) else result.sample


def assert_baseline_generation(sample) -> None:
    assert sample.tokens == PROMPT_TOKENS + RESPONSE_TOKENS
    assert sample.response == RESPONSE_TEXT
    assert sample.response_length == len(RESPONSE_TOKENS)
    assert sample.rollout_log_probs == pytest.approx(RESPONSE_LOG_PROBS)
