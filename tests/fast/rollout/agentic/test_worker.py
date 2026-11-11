import asyncio
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from aiohttp import web

from miles.rollout.agentic import agent_function, remote
from miles.rollout.agentic.agent_function import InfraAbort, call_agent_function, cancel_calls, has_calls
from miles.rollout.agentic.credentials import provider_env_names
from miles.rollout.agentic.rollout_limits import current_scope
from miles.rollout.agentic.worker import create_app
from miles.rollout.inference_rollout import inference_rollout_train
from miles.utils.misc import call_agent_abort_hook
from miles.utils.retry_utils import NonRetryableError
from miles.utils.test_utils import agent_function_stubs as stubs


def _serve_in_thread(app: web.Application):
    loop = asyncio.new_event_loop()
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", 0)
    loop.run_until_complete(site.start())
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    def close() -> None:
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(timeout=60)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)

    return url, close


@pytest.fixture
def serve():
    closers = []

    def start(app: web.Application | None = None, **worker_options) -> str:
        url, close = _serve_in_thread(app or create_app(**worker_options))
        closers.append(close)
        return url

    yield start
    for close in closers:
        close()


def _rollout_args(router_port: int = 30000, **overrides):
    return SimpleNamespace(
        custom_agent_function_path=None, sglang_router_ip="127.0.0.1", sglang_router_port=router_port, **overrides
    )


def _on_worker(url, fn, extra_env=None, on_child_reports=None, **kwargs):
    return asyncio.run(
        call_agent_function(
            fn, mode="worker", worker=url, extra_env=extra_env, on_child_reports=on_child_reports, **kwargs
        )
    )


def _start_waiting_call(url, tmp_path, name="call", **kwargs):
    pid_file, cleaned_up = tmp_path / f"{name}.pid", tmp_path / f"{name}.cleaned_up"
    task = asyncio.create_task(
        call_agent_function(
            stubs.wait_until_cancelled,
            mode="worker",
            worker=url,
            pid_file=str(pid_file),
            cleaned_up=str(cleaned_up),
            **kwargs,
        )
    )
    return task, pid_file, cleaned_up


async def _until_exists(path: Path) -> None:
    while not path.exists():
        await asyncio.sleep(0.1)


def _wait_for(path: Path, timeout_s: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    return path.exists()


def _submit(url: str, call_id: str, fn, **kwargs) -> None:
    body = {"call_id": call_id, "fn": remote.function_path(fn), "kwargs": kwargs, "env": {}, "scope": call_id}
    httpx.post(f"{url}/calls", json=body).raise_for_status()


def test_the_call_runs_in_a_child_process_on_the_worker(serve):
    reported = _on_worker(serve(), stubs.report_pid)

    assert reported["pid"] != os.getpid()


def test_the_child_gets_the_forwarded_env_except_what_the_worker_keeps(serve, monkeypatch):
    monkeypatch.setenv("WORKER_TEST_KEPT", "the worker's own")
    url = serve(keep_env=("WORKER_TEST_KEPT",))

    seen = _on_worker(
        url,
        stubs.read_env,
        extra_env={"WORKER_TEST_KEPT": "the caller's", "WORKER_TEST_FORWARDED": "the caller's"},
        names=["WORKER_TEST_KEPT", "WORKER_TEST_FORWARDED"],
    )

    assert seen == {"WORKER_TEST_KEPT": "the worker's own", "WORKER_TEST_FORWARDED": "the caller's"}


def test_the_childs_rollout_limits_are_the_callers_and_no_other_hosts(serve):
    url = serve()

    scopes = {_on_worker(url, stubs.report_limits_scope) for _ in range(2)}

    assert len(scopes) == 1
    (scope,) = scopes
    assert scope.startswith(f"{current_scope()}-") and scope != current_scope()


def test_reports_reach_the_caller_in_order(serve):
    received: list = []

    result = _on_worker(
        serve(), stubs.report_then_return, on_child_reports=received.extend, items=[{"n": 1}, {"n": 2}]
    )

    assert result == "done"
    assert received == [{"n": 1}, {"n": 2}]


def test_an_infra_abort_keeps_its_cause_across_the_worker(serve):
    with pytest.raises(InfraAbort) as raised:
        _on_worker(serve(), stubs.abort_for_infra)

    assert raised.value.exit_status == "SandboxUnavailable"


def test_an_agent_error_arrives_with_its_traceback(serve):
    with pytest.raises(RuntimeError, match="ValueError: agent failed on purpose") as raised:
        _on_worker(serve(), stubs.fail)

    assert "agent_function_stubs.py" in str(raised.value.__cause__)


def test_cancelling_the_caller_runs_the_agent_cleanup_on_the_worker(serve, tmp_path):
    url = serve()

    async def cancel_once_running():
        task, pid_file, cleaned_up = _start_waiting_call(url, tmp_path)
        await _until_exists(pid_file)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return cleaned_up

    assert _wait_for(asyncio.run(cancel_once_running()))


def test_a_rollout_abort_cancels_the_calls_running_on_workers(serve, tmp_path):
    url = serve()

    async def abort_while_running():
        task, pid_file, cleaned_up = _start_waiting_call(url, tmp_path, abort_scope="127.0.0.1:30000")
        await _until_exists(pid_file)
        await call_agent_abort_hook(_rollout_args())
        with pytest.raises(InfraAbort) as raised:
            await asyncio.wait_for(task, timeout=60)
        return raised.value.exit_status, cleaned_up, has_calls()

    exit_status, cleaned_up, calls_left = asyncio.run(abort_while_running())

    assert exit_status == "Cancelled"
    assert not calls_left
    assert _wait_for(cleaned_up)


def _recording_cancels(cancelled: list[str]) -> web.Application:
    app = create_app()

    @web.middleware
    async def record(request, handler):
        if request.method == "DELETE":
            cancelled.append(request.match_info["call_id"])
        return await handler(request)

    app.middlewares.append(record)
    return app


def test_the_abort_hook_returns_only_after_the_worker_got_the_cancel(serve, tmp_path):
    cancelled: list[str] = []
    url = serve(app=_recording_cancels(cancelled))

    async def abort_while_running():
        task, pid_file, _ = _start_waiting_call(url, tmp_path, abort_scope="127.0.0.1:30000")
        await _until_exists(pid_file)
        await call_agent_abort_hook(_rollout_args())
        seen_when_the_hook_returned = list(cancelled)
        with pytest.raises(InfraAbort):
            await asyncio.wait_for(task, timeout=60)
        return seen_when_the_hook_returned

    assert len(asyncio.run(abort_while_running())) == 1


def test_a_cancel_that_hangs_holds_the_abort_hook_only_up_to_the_bound(serve, tmp_path, monkeypatch):
    monkeypatch.setattr(agent_function, "_ABORT_WAIT_S", 0.5)
    release = asyncio.Event()

    async def hang(client, call_id):
        await release.wait()

    monkeypatch.setattr(remote, "_cancel", hang)
    url = serve(orphan_timeout_s=2.0)

    async def abort_with_a_hung_cancel():
        task, pid_file, _ = _start_waiting_call(url, tmp_path, abort_scope="127.0.0.1:30000")
        await _until_exists(pid_file)
        await asyncio.wait_for(call_agent_abort_hook(_rollout_args()), timeout=10)
        still_cancelling = not task.done()
        release.set()
        with pytest.raises(InfraAbort):
            await asyncio.wait_for(task, timeout=10)
        return still_cancelling

    assert asyncio.run(abort_with_a_hung_cancel())


def test_cancelling_the_caller_of_an_aborted_call_stays_a_cancel(serve, tmp_path, monkeypatch):
    monkeypatch.setattr(agent_function, "_ABORT_WAIT_S", 0.1)
    release = asyncio.Event()

    async def hang(client, call_id):
        await release.wait()

    monkeypatch.setattr(remote, "_cancel", hang)
    url = serve(orphan_timeout_s=2.0)

    async def cancel_during_the_abort():
        task, pid_file, _ = _start_waiting_call(url, tmp_path)
        await _until_exists(pid_file)
        await cancel_calls()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_during_the_abort())


def test_a_cancel_no_rollout_abort_asked_for_stays_worker_cancelled(serve):
    async def accept(request):
        return web.json_response({"call_id": "x"})

    async def cancelled_by_the_worker(request):
        return web.json_response({"state": "done", "reports": [], "next": 0, "outcome": {"kind": "cancelled"}})

    app = web.Application()
    app.add_routes([web.post("/calls", accept), web.get("/calls/{call_id}", cancelled_by_the_worker)])

    with pytest.raises(InfraAbort) as raised:
        _on_worker(serve(app=app), stubs.report_pid)

    assert raised.value.exit_status == "WorkerCancelled"


def test_a_rollout_abort_leaves_another_rollouts_calls_running(serve, tmp_path):
    url = serve()

    async def abort_training():
        train, train_pid, train_cleaned = _start_waiting_call(url, tmp_path, "train", abort_scope="127.0.0.1:30000")
        evaluation, eval_pid, eval_cleaned = _start_waiting_call(url, tmp_path, "eval", abort_scope="127.0.0.1:31000")
        await _until_exists(train_pid)
        await _until_exists(eval_pid)
        await call_agent_abort_hook(_rollout_args(router_port=30000))
        with pytest.raises(InfraAbort):
            await asyncio.wait_for(train, timeout=60)
        await asyncio.sleep(2)
        still_running = not evaluation.done()
        assert await cancel_calls() == 1
        with pytest.raises(InfraAbort):
            await asyncio.wait_for(evaluation, timeout=60)
        return still_running, train_cleaned, eval_cleaned

    still_running, train_cleaned, eval_cleaned = asyncio.run(abort_training())

    assert still_running
    assert _wait_for(train_cleaned) and _wait_for(eval_cleaned)


def test_the_rollout_abort_entrypoint_cancels_an_agent_waiting_on_its_environment(serve, tmp_path, monkeypatch):
    async def no_engines(args):
        return []

    monkeypatch.setattr(inference_rollout_train, "get_worker_urls", no_engines)
    monkeypatch.setattr(inference_rollout_train.SampleOwnershipRecorder, "log_dropped_samples", lambda **kw: None)
    monkeypatch.setattr(inference_rollout_train, "record_dropped_samples", lambda group: None)
    url = serve()

    async def abort_mid_episode():
        task, pid_file, cleaned_up = _start_waiting_call(url, tmp_path, abort_scope="127.0.0.1:30000")

        async def group():
            try:
                await task
            except InfraAbort as exc:
                return [SimpleNamespace(exit_status=exc.exit_status)]

        pending = {asyncio.create_task(group())}
        await _until_exists(pid_file)
        state = SimpleNamespace(args=_rollout_args(partial_rollout=False), aborted=False)
        await asyncio.wait_for(inference_rollout_train.abort(state, pending, rollout_id=0), timeout=60)
        return cleaned_up

    assert _wait_for(asyncio.run(abort_mid_episode()))


def test_a_cancel_that_overtakes_its_submit_keeps_the_call_from_starting(serve, tmp_path):
    url = serve()
    pid_file, cleaned_up = tmp_path / "pid", tmp_path / "cleaned_up"

    httpx.delete(f"{url}/calls/late").raise_for_status()
    _submit(url, "late", stubs.wait_until_cancelled, pid_file=str(pid_file), cleaned_up=str(cleaned_up))
    state = httpx.get(f"{url}/calls/late", params={"wait": 5}).json()

    assert state["state"] == "done" and state["outcome"] == {"kind": "cancelled"}
    time.sleep(2)
    assert not pid_file.exists()
    assert httpx.get(f"{url}/health").json()["running"] == 0


def test_an_abort_during_the_submit_keeps_the_agent_from_starting(serve, tmp_path):
    app = create_app()

    @web.middleware
    async def slow_submit(request, handler):
        if request.method == "POST":
            await asyncio.sleep(2)
        return await handler(request)

    app.middlewares.append(slow_submit)
    url = serve(app=app)

    async def abort_during_submit():
        task, pid_file, _ = _start_waiting_call(url, tmp_path)
        await asyncio.sleep(0.5)
        assert await cancel_calls() == 1
        with pytest.raises(InfraAbort) as raised:
            await asyncio.wait_for(task, timeout=60)
        return raised.value.exit_status, pid_file

    exit_status, pid_file = asyncio.run(abort_during_submit())

    assert exit_status == "Cancelled"
    time.sleep(2)
    assert not pid_file.exists()


def test_an_aborted_call_whose_cancel_is_lost_stops_polling_and_is_reaped(serve, tmp_path, monkeypatch):
    monkeypatch.setattr("miles.rollout.agentic.worker._REAP_INTERVAL_S", 0.2)
    url = serve(orphan_timeout_s=2.0)

    async def lose_the_cancel(client, call_id):
        return None

    monkeypatch.setattr(remote, "_cancel", lose_the_cancel)

    async def abort_with_a_lost_cancel():
        task, pid_file, cleaned_up = _start_waiting_call(url, tmp_path)
        await _until_exists(pid_file)
        await cancel_calls()
        with pytest.raises(InfraAbort) as raised:
            await asyncio.wait_for(task, timeout=30)
        return raised.value.exit_status, cleaned_up

    exit_status, cleaned_up = asyncio.run(abort_with_a_lost_cancel())

    assert exit_status == "Cancelled"
    assert _wait_for(cleaned_up, timeout_s=30)


def test_secrets_are_not_forwarded_by_value(monkeypatch):
    monkeypatch.setenv("TEST_HARNESS_ADMIN_SECRET", "s3cret")
    monkeypatch.setenv("TEST_HARNESS_ADMIN_SECRET_FILE", "/secrets/admin")
    monkeypatch.setenv("E2B_API_KEY", "e2b_value")
    monkeypatch.setenv("TEST_HARNESS_TASKS_DIR", "/tasks")
    monkeypatch.setenv("AGENT_MAX_OUTPUT_TOKENS", "2048")
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-value")
    monkeypatch.setenv("MODAL_PROFILE", "team")

    env = remote.forwarded_env(["TEST_HARNESS_*", "E2B_*", "AGENT_*", "MODAL_*"])

    assert env["TEST_HARNESS_ADMIN_SECRET_FILE"] == "/secrets/admin"
    assert env["TEST_HARNESS_TASKS_DIR"] == "/tasks"
    assert env["AGENT_MAX_OUTPUT_TOKENS"] == "2048"
    assert "TEST_HARNESS_ADMIN_SECRET" not in env
    assert "E2B_API_KEY" not in env
    assert "MODAL_TOKEN_ID" not in env
    assert env["MODAL_PROFILE"] == "team"


def test_the_provider_names_carry_every_providers_endpoint_but_no_key():
    for name in ("E2B_API_URL", "E2B_SANDBOX_URL", "E2B_DOMAIN", "E2B_API_KEY_FILE", "MODAL_PROFILE"):
        assert name in provider_env_names()
    assert "E2B_API_KEY" not in provider_env_names()


def test_a_worker_with_a_secret_refuses_a_caller_without_it(serve, caplog):
    url = serve(secret="worker-s3cret")

    with pytest.raises(InfraAbort) as raised:
        _on_worker(url, stubs.report_pid)

    assert raised.value.exit_status == "WorkerUnauthorized"
    assert httpx.get(f"{url}/health").status_code == 401
    assert "worker-s3cret" not in str(raised.value) and "worker-s3cret" not in caplog.text


def test_a_worker_with_a_secret_refuses_a_wrong_one(serve, monkeypatch):
    url = serve(secret="worker-s3cret")
    monkeypatch.setenv("MILES_AGENT_WORKER_SECRET", "guessed")

    with pytest.raises(InfraAbort) as raised:
        _on_worker(url, stubs.report_pid)

    assert raised.value.exit_status == "WorkerUnauthorized"


def test_a_caller_and_worker_sharing_the_secret_by_env_run_the_call(serve, monkeypatch):
    monkeypatch.setenv("MILES_AGENT_WORKER_SECRET", "worker-s3cret")
    url = serve()
    monkeypatch.delenv("MILES_AGENT_WORKER_SECRET")
    assert httpx.get(f"{url}/health").status_code == 401
    monkeypatch.setenv("MILES_AGENT_WORKER_SECRET", "worker-s3cret")

    reported = _on_worker(url, stubs.report_pid)

    assert reported["pid"] != os.getpid()


def test_a_caller_and_worker_sharing_the_secret_by_file_run_the_call(serve, monkeypatch, tmp_path):
    secret_file = tmp_path / "worker_secret"
    secret_file.write_text("worker-s3cret\n")
    monkeypatch.setenv("MILES_AGENT_WORKER_SECRET_FILE", str(secret_file))
    url = serve()

    reported = _on_worker(url, stubs.report_pid)

    assert reported["pid"] != os.getpid()
    assert httpx.get(f"{url}/health").status_code == 401


def test_an_agent_on_a_worker_with_a_secret_cannot_read_it(serve, monkeypatch, tmp_path):
    secret_file = tmp_path / "worker_secret"
    secret_file.write_text("worker-s3cret")
    monkeypatch.setenv("MILES_AGENT_WORKER_SECRET", "worker-s3cret")
    monkeypatch.setenv("MILES_AGENT_WORKER_SECRET_FILE", str(secret_file))
    names = ["MILES_AGENT_WORKER_SECRET", "MILES_AGENT_WORKER_SECRET_FILE"]

    seen = _on_worker(
        serve(), stubs.read_env, extra_env={"MILES_AGENT_WORKER_SECRET_FILE": "/forwarded"}, names=names
    )

    assert seen == {name: None for name in names}


def test_the_worker_secret_is_never_forwarded_to_an_agent_by_value(monkeypatch):
    monkeypatch.setenv("MILES_AGENT_WORKER_SECRET", "worker-s3cret")
    monkeypatch.setenv("MILES_AGENT_WORKER_SECRET_FILE", "/secrets/worker")

    env = remote.forwarded_env(["MILES_AGENT_WORKER_*"])

    assert env == {"MILES_AGENT_WORKER_SECRET_FILE": "/secrets/worker"}


def test_a_call_whose_caller_vanished_is_cancelled(serve, tmp_path, monkeypatch):
    monkeypatch.setattr("miles.rollout.agentic.worker._REAP_INTERVAL_S", 0.2)
    url = serve(orphan_timeout_s=1.0)
    pid_file, cleaned_up = tmp_path / "pid", tmp_path / "cleaned_up"

    _submit(url, "orphan", stubs.wait_until_cancelled, pid_file=str(pid_file), cleaned_up=str(cleaned_up))

    assert _wait_for(pid_file)
    assert _wait_for(cleaned_up, timeout_s=30)


def test_a_resubmitted_call_runs_once(serve):
    url = serve()

    for _ in range(2):
        _submit(url, "same", stubs.report_pid)
    done = httpx.get(f"{url}/calls/same", params={"wait": 20}).json()

    assert done["state"] == "done"
    assert done["outcome"]["kind"] == "ok"
    assert httpx.get(f"{url}/health").json()["running"] == 0


def test_a_worker_that_lost_the_call_discards_the_sample(serve):
    async def accept(request):
        return web.json_response({"call_id": "x"})

    forgetful = web.Application()
    forgetful.add_routes([web.post("/calls", accept)])

    with pytest.raises(InfraAbort) as raised:
        _on_worker(serve(app=forgetful), stubs.report_pid)

    assert raised.value.exit_status == "WorkerLost"


def test_an_unreachable_worker_discards_the_sample(monkeypatch):
    monkeypatch.setattr(remote, "_REQUEST_ATTEMPTS", 1)

    with pytest.raises(InfraAbort) as raised:
        _on_worker("http://127.0.0.1:9", stubs.report_pid)

    assert raised.value.exit_status == "WorkerUnreachable"


def test_a_kwarg_the_protocol_cannot_carry_fails_before_the_call(serve):
    with pytest.raises(TypeError):
        _on_worker(serve(), stubs.report_pid, payload=object())


@pytest.mark.parametrize("worker", [None, "", "not_a_url.start"])
def test_worker_mode_without_a_worker_url_is_refused(worker):
    with pytest.raises(NonRetryableError, match="--custom-agent-function-worker"):
        asyncio.run(call_agent_function(stubs.report_pid, mode="worker", worker=worker))


def test_the_worker_side_imports_without_torch_sglang_or_ray():
    code = textwrap.dedent(
        """
        import sys, types

        def find_spec(name, path=None, target=None):
            if name.split(".")[0] in ("torch", "sglang", "ray"):
                raise ImportError("blocked: " + name)

        sys.meta_path.insert(0, types.SimpleNamespace(find_spec=find_spec))
        import miles.rollout.agentic.worker
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[4])
