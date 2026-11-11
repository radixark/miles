import argparse
import asyncio
import concurrent.futures
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from examples.experimental.agent_sandbox import local_start
from tests.fast.fixtures.hook_consumer_fixtures import _hook
from tests.fast.rollout.generate_hub.test_agentic_v2 import _generate_input, _patch_agent, _Tracer
from tests.fast.utils.output_store.test_dropped import sample_owning

from miles.backends.megatron_utils.megatron_config import MegatronConfig
from miles.rollout.agentic import per_call_worker
from miles.rollout.agentic.agent_function import (
    InfraAbort,
    call_agent_function,
    cancel_calls,
    resolve_worker,
    rollout_abort_scope,
)
from miles.rollout.agentic.remote import SECRET_ENV_VAR
from miles.rollout.agentic.rollout_limits import UNSHARED_ENV_VAR
from miles.rollout.agentic.worker import create_app
from miles.rollout.generate_hub import agentic_tool_call
from miles.rollout.inference_rollout import inference_rollout_train
from miles.rollout.session.samples.codec import SamplesReply
from miles.utils.arguments import get_miles_extra_args_provider
from miles.utils.function_registry import function_registry
from miles.utils.misc import call_agent_abort_hook
from miles.utils.output_store import dropped
from miles.utils.output_store.dropped import take_dropped_refs
from miles.utils.retry_utils import NonRetryableError
from miles.utils.test_utils import agent_function_stubs as stubs
from miles.utils.types import Sample

from .test_worker import _rollout_args, _serve_in_thread, _wait_for

START_PATH = "test_per_call_worker.start"


def _until(condition, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.05)
    return condition()


def _worker_app(sandboxes: SimpleNamespace) -> web.Application:
    app = create_app()

    @web.middleware
    async def gate(request, handler):
        sandboxes.requests.append((request.method, request.path))
        if request.path == "/health" and not sandboxes.healthy.is_set():
            raise web.HTTPServiceUnavailable()
        return await handler(request)

    app.middlewares.append(gate)
    return app


@pytest.fixture
def sandboxes():
    state = SimpleNamespace(
        started=[], closed=[], requests=[], servers=[], healthy=threading.Event(), creating=threading.Event()
    )
    state.healthy.set()
    state.creating.set()

    def start(task_id, tasks_dir):
        assert state.creating.wait(30)
        url, close = _serve_in_thread(_worker_app(state))
        state.servers.append(close)
        state.started.append((task_id, tasks_dir))
        return (lambda: state.closed.append(task_id)), url

    with function_registry.temporary(START_PATH, start):
        yield state
    for close in state.servers:
        close()


def _on_started_worker(fn, worker=START_PATH, **kwargs):
    return asyncio.run(call_agent_function(fn, mode="worker", worker=worker, **kwargs))


def test_the_call_runs_on_the_worker_its_start_function_started_which_is_then_closed(sandboxes):
    seen = _on_started_worker(
        stubs.read_env,
        extra_env={"SANDBOX_TEST_FORWARDED": "the caller's"},
        names=["SANDBOX_TEST_FORWARDED"],
        metadata={"instance_id": "task-7"},
    )

    assert seen == {"SANDBOX_TEST_FORWARDED": "the caller's"}
    assert sandboxes.started == [("task-7", "")]
    assert sandboxes.closed == ["task-7"]
    assert ("POST", "/calls") in sandboxes.requests


def test_an_infra_abort_keeps_its_cause_and_the_sandbox_still_closes(sandboxes):
    with pytest.raises(InfraAbort) as raised:
        _on_started_worker(stubs.abort_for_infra)

    assert raised.value.exit_status == "SandboxUnavailable"
    assert sandboxes.closed == ["abort_for_infra"]


def test_a_sandbox_that_cannot_be_started_discards_the_sample():
    def start(task_id, tasks_dir):
        raise RuntimeError("500: snapshot not ready")

    with function_registry.temporary("test_per_call_worker.failing", start), pytest.raises(InfraAbort) as raised:
        _on_started_worker(stubs.report_pid, worker="test_per_call_worker.failing")

    assert raised.value.exit_status == "SandboxWorkerUnavailable"


def test_a_worker_that_never_turns_healthy_discards_the_sample_and_closes_its_sandbox(sandboxes, monkeypatch):
    sandboxes.healthy.clear()
    monkeypatch.setattr(per_call_worker, "_HEALTH_DEADLINE_S", 0.5)

    with pytest.raises(InfraAbort) as raised:
        _on_started_worker(stubs.report_pid)

    assert raised.value.exit_status == "SandboxWorkerUnavailable"
    assert sandboxes.closed == ["report_pid"]
    assert ("POST", "/calls") not in sandboxes.requests


def test_cancelling_the_caller_runs_the_agent_cleanup_and_closes_the_sandbox(sandboxes, tmp_path):
    pid_file, cleaned_up = tmp_path / "pid", tmp_path / "cleaned_up"

    async def cancel_once_running():
        task = asyncio.create_task(
            call_agent_function(
                stubs.wait_until_cancelled,
                mode="worker",
                worker=START_PATH,
                pid_file=str(pid_file),
                cleaned_up=str(cleaned_up),
            )
        )
        while not pid_file.exists():
            await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_once_running())

    assert sandboxes.closed == ["wait_until_cancelled"]
    assert _wait_for(cleaned_up)


@pytest.mark.parametrize("phase", ["create", "health"])
def test_a_rollout_abort_while_the_sandbox_boots_stops_the_call_before_it_starts(sandboxes, phase):
    args = _rollout_args()
    held = sandboxes.creating if phase == "create" else sandboxes.healthy
    held.clear()

    async def abort_while_booting():
        task = asyncio.create_task(
            call_agent_function(
                stubs.report_pid,
                mode="worker",
                worker=START_PATH,
                abort_scope=rollout_abort_scope(args),
            )
        )
        await asyncio.sleep(0.5)
        await call_agent_abort_hook(args)
        held.set()
        with pytest.raises(InfraAbort) as raised:
            await task
        return raised.value

    abort = asyncio.run(abort_while_booting())

    assert abort.exit_status == "Cancelled"
    assert ("POST", "/calls") not in sandboxes.requests
    assert _until(lambda: sandboxes.closed == ["report_pid"])


def test_a_rollout_abort_leaves_another_rollouts_booting_call_to_run(sandboxes):
    sandboxes.healthy.clear()

    async def abort_another_rollout():
        task = asyncio.create_task(
            call_agent_function(
                stubs.report_pid,
                mode="worker",
                worker=START_PATH,
                abort_scope=rollout_abort_scope(_rollout_args(router_port=30001)),
            )
        )
        await asyncio.sleep(0.5)
        await call_agent_abort_hook(_rollout_args(router_port=30000))
        sandboxes.healthy.set()
        return await task

    assert asyncio.run(abort_another_rollout())["pid"]
    assert sandboxes.closed == ["report_pid"]


def test_a_call_that_starts_after_an_abort_runs(sandboxes):
    asyncio.run(cancel_calls())

    assert _on_started_worker(stubs.report_pid)["pid"]


@pytest.mark.parametrize(
    "stub, kwargs",
    [
        ("hold_semaphore", {"name": "create", "limit": 2, "hold_s": 0.0}),
        ("hold_lock", {"name": "build", "hold_s": 0.0}),
    ],
)
def test_rollout_wide_limits_refuse_rather_than_limit_one_sandbox(sandboxes, stub, kwargs):
    with pytest.raises(RuntimeError, match="cannot be shared from a per-call worker agent"):
        _on_started_worker(getattr(stubs, stub), **kwargs)


def test_the_agent_sees_that_its_limits_are_unshared(sandboxes):
    assert _on_started_worker(stubs.read_env, names=[UNSHARED_ENV_VAR]) == {UNSHARED_ENV_VAR: "per-call worker"}


def test_a_fixed_workers_agent_keeps_its_shared_limits(sandboxes):
    url, close = _serve_in_thread(create_app())
    try:
        assert _on_started_worker(stubs.read_env, worker=url, names=[UNSHARED_ENV_VAR]) == {UNSHARED_ENV_VAR: None}
    finally:
        close()
    assert sandboxes.started == []


async def _not_a_start_function(task_id, tasks_dir):
    return None


@pytest.mark.parametrize(
    "path",
    [
        "no_such_module.start",
        "miles.utils.test_utils.agent_function_stubs.no_such_start",
        "no_dots",
        "worker.example.com:11100",
        "tests.fast.rollout.agentic.test_per_call_worker.START_PATH",
        "tests.fast.rollout.agentic.test_per_call_worker._not_a_start_function",
    ],
)
def test_a_value_that_is_neither_a_url_nor_a_start_function_is_refused(path):
    with pytest.raises(NonRetryableError, match="--custom-agent-function-worker"):
        resolve_worker(path)


def test_a_url_is_a_fixed_worker_and_a_path_a_start_function():
    assert resolve_worker("https://worker:11100") == "https://worker:11100"
    assert resolve_worker("examples.experimental.agent_sandbox.local_start.start") is local_start.start


def _train_args(monkeypatch, *flags: str) -> argparse.Namespace:
    argv = [
        "--custom-generate-function-path=miles.rollout.generate_hub.agentic_tool_call.generate",
        "--custom-agent-function-mode=worker",
        *flags,
    ]
    monkeypatch.setattr(sys, "argv", ["train.py", *argv])
    parser = argparse.ArgumentParser(allow_abbrev=False)
    MegatronConfig.add_arguments(parser)
    get_miles_extra_args_provider()(parser)
    return parser.parse_args([*argv, "--num-rollout", "1", "--rollout-batch-size", "64"])


def test_a_bad_start_path_stops_the_trainers_argument_parse(monkeypatch):
    with pytest.raises(SystemExit):
        _train_args(monkeypatch, "--custom-agent-function-worker=no_such_module.start")


@pytest.mark.parametrize("worker", ["http://worker:11100", "examples.experimental.agent_sandbox.local_start.start"])
def test_the_trainers_argument_parse_keeps_a_good_worker_as_given(monkeypatch, worker):
    args = _train_args(monkeypatch, f"--custom-agent-function-worker={worker}")

    assert args.custom_agent_function_worker == worker


def _parsed(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    agentic_tool_call.generate.add_arguments(parser)
    args = parser.parse_args(list(argv))
    args.sglang_router_ip, args.sglang_router_port = "127.0.0.1", 30000
    args.custom_agent_function_path = _hook("test_per_call_worker_harness.agent.run")
    return args


def test_a_started_workers_agent_gets_the_providers_and_the_harness_env(sandboxes, monkeypatch):
    monkeypatch.setenv("E2B_API_URL", "https://sandbox-service.example")
    monkeypatch.setenv("TEST_HARNESS_DIR", "/scratch/tasks")
    monkeypatch.setenv("TEST_OTHER", "not forwarded")
    args = _parsed("--custom-agent-function-mode=worker", f"--custom-agent-function-worker={START_PATH}")

    with function_registry.temporary("test_per_call_worker_harness.agent.WORKER_ENV", ("TEST_HARNESS_*",)):
        options = agentic_tool_call._worker_options(args)

    assert options["worker"] == START_PATH
    names = ["E2B_API_URL", "TEST_HARNESS_DIR", "TEST_OTHER"]
    for name in names:
        monkeypatch.delenv(name)
    seen = asyncio.run(call_agent_function(stubs.read_env, mode="worker", **options, names=names))
    assert seen == {
        "E2B_API_URL": "https://sandbox-service.example",
        "TEST_HARNESS_DIR": "/scratch/tasks",
        "TEST_OTHER": None,
    }


def test_worker_mode_without_a_worker_is_still_refused():
    with pytest.raises(NonRetryableError, match="--custom-agent-function-worker"):
        agentic_tool_call._worker_options(_parsed("--custom-agent-function-mode=worker"))


def test_an_abort_while_the_close_waits_for_a_thread_still_closes_the_sandbox(sandboxes):
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    occupied, release = [], threading.Event()
    sandboxes.healthy.clear()

    def occupy():
        occupied.append(threading.current_thread())
        release.wait()

    async def abort_during_close():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(pool)
        task = asyncio.create_task(
            call_agent_function(stubs.report_pid, mode="worker", worker=START_PATH, abort_scope="scope")
        )
        while not sandboxes.started:
            await asyncio.sleep(0.01)
        blockers = [loop.run_in_executor(None, occupy) for _ in range(2)]
        while len(occupied) < 2:
            await asyncio.sleep(0.01)
        sandboxes.healthy.set()
        while pool._work_queue.qsize() == 0:
            await asyncio.sleep(0.05)
        await asyncio.wait_for(cancel_calls("scope"), timeout=60)
        with pytest.raises(InfraAbort):
            await asyncio.wait_for(task, timeout=20)
        release.set()
        await asyncio.gather(*blockers)

    try:
        asyncio.run(abort_during_close())
        assert _until(lambda: sandboxes.closed == ["report_pid"], timeout_s=5)
    finally:
        release.set()


def _generate_on_started_worker(monkeypatch, instance_id: str):
    collected = sample_owning(f"collected-{instance_id}")
    collected.status = Sample.Status.COMPLETED
    _patch_agent(monkeypatch, _Tracer(SamplesReply(samples=[collected], session_metadata={}, empty_reason=None)))
    monkeypatch.setattr(agentic_tool_call, "load_function", lambda path: stubs.report_pid)
    generate_input = _generate_input(
        custom_agent_function_path=None,
        custom_agent_function_mode="worker",
        custom_agent_function_worker=START_PATH,
    )
    generate_input.sample.metadata["instance_id"] = instance_id
    return generate_input


async def test_generate_starts_a_worker_for_each_call_and_closes_it_after(sandboxes, monkeypatch):
    inputs = [_generate_on_started_worker(monkeypatch, f"task-{i}") for i in range(2)]

    outputs = await asyncio.gather(*(agentic_tool_call.generate(i) for i in inputs))

    assert [o.samples[0].status for o in outputs] == [Sample.Status.COMPLETED] * 2
    assert sorted(sandboxes.started) == [("task-0", ""), ("task-1", "")]
    assert sorted(sandboxes.closed) == ["task-0", "task-1"]
    assert sandboxes.requests.count(("POST", "/calls")) == 2


@pytest.mark.parametrize("phase", ["create", "health"])
async def test_a_rollout_abort_during_boot_closes_the_worker_and_discards_the_sample(sandboxes, monkeypatch, phase):
    monkeypatch.setattr(dropped, "_dropped", {})
    generate_input = _generate_on_started_worker(monkeypatch, "task-3")
    logged: list = []

    async def no_engines(args):
        return []

    async def group():
        return (await agentic_tool_call.generate(generate_input)).samples

    monkeypatch.setattr(inference_rollout_train, "get_worker_urls", no_engines)
    monkeypatch.setattr(
        inference_rollout_train.SampleOwnershipRecorder, "log_dropped_samples", lambda **kw: logged.append(kw)
    )
    held = sandboxes.creating if phase == "create" else sandboxes.healthy
    held.clear()
    pending = {asyncio.create_task(group())}
    while phase == "health" and not sandboxes.started:
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.5)

    returned = await asyncio.wait_for(
        inference_rollout_train.abort(generate_input.state, pending, rollout_id=0), timeout=30
    )
    held.set()

    assert returned == []
    [drop] = logged
    [aborted] = drop["samples"]
    assert aborted.metadata["exit_status"] == "Cancelled"
    assert list(take_dropped_refs()) == ["collected-task-3"]
    assert ("POST", "/calls") not in sandboxes.requests
    assert await asyncio.to_thread(_until, lambda: sandboxes.closed == ["task-3"])


def test_a_process_whose_start_was_aborted_while_queued_for_a_thread_still_exits():
    code = textwrap.dedent(
        """
        import asyncio, concurrent.futures, threading
        from miles.rollout.agentic.agent_function import InfraAbort, call_agent_function, cancel_calls
        from miles.utils.test_utils import agent_function_stubs as stubs

        async def main():
            loop = asyncio.get_running_loop()
            loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))
            release = threading.Event()
            occupied = loop.run_in_executor(None, release.wait)
            call = asyncio.create_task(
                call_agent_function(
                    stubs.report_pid,
                    mode="worker",
                    worker="examples.experimental.agent_sandbox.local_start.start",
                    abort_scope="scope",
                )
            )
            await asyncio.sleep(0.5)
            await cancel_calls("scope")
            try:
                await call
            except InfraAbort as abort:
                print(abort.exit_status)
            release.set()
            await occupied

        asyncio.run(main())
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["Cancelled"]


def test_a_started_worker_that_holds_the_rollouts_secret_is_reached_with_it(monkeypatch):
    monkeypatch.setenv(SECRET_ENV_VAR, "per-call-secret")
    monkeypatch.setattr(per_call_worker, "_HEALTH_DEADLINE_S", 15.0)

    reported = _on_started_worker(stubs.report_pid, worker="examples.experimental.agent_sandbox.local_start.start")

    assert reported["pid"]
