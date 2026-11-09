import asyncio
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.fast.rollout.agentic.test_agent_function import _is_running, _wait_until_gone

from miles.rollout.agentic import agent_function
from miles.rollout.agentic.agent_function import InfraAbort, call_agent_function, cancel_calls, has_calls
from miles.utils.misc import call_agent_abort_hook
from miles.utils.test_utils import agent_function_stubs as stubs

TRAINING = SimpleNamespace(custom_agent_function_path=None, sglang_router_ip="127.0.0.1", sglang_router_port=30000)
TRAINING_SCOPE = "127.0.0.1:30000"
EVAL_SCOPE = "127.0.0.1:31000"


def _start_call(fn, mode, tmp_path, name="call", scope=TRAINING_SCOPE, **kwargs):
    pid_file, cleaned_up = tmp_path / f"{name}.pid", tmp_path / f"{name}.cleaned_up"
    task = asyncio.create_task(
        call_agent_function(
            fn, mode=mode, abort_scope=scope, pid_file=str(pid_file), cleaned_up=str(cleaned_up), **kwargs
        )
    )
    return task, pid_file, cleaned_up


async def _until_exists(path: Path) -> None:
    while not path.exists():
        await asyncio.sleep(0.05)


@pytest.mark.parametrize("mode", ["subproc", "inline"])
@pytest.mark.parametrize(
    "agent",
    [stubs.wait_until_cancelled, stubs.return_when_cancelled, stubs.fail_when_cancelled],
    ids=lambda a: a.__name__,
)
def test_a_rollout_abort_cancels_the_call_and_discards_it_however_the_agent_ends(tmp_path, mode, agent):
    async def abort_while_running():
        task, pid_file, cleaned_up = _start_call(agent, mode, tmp_path)
        await asyncio.wait_for(_until_exists(pid_file), timeout=60)
        await call_agent_abort_hook(TRAINING)
        with pytest.raises(InfraAbort) as raised:
            await asyncio.wait_for(task, timeout=10)
        pid = int(pid_file.read_text())
        if mode == "subproc":
            assert await _wait_until_gone(pid, timeout_s=30)
        return raised.value.exit_status, cleaned_up, has_calls()

    exit_status, cleaned_up, calls_left = asyncio.run(abort_while_running())

    assert exit_status == "Cancelled"
    assert cleaned_up.exists()
    assert not calls_left


def test_the_abort_hook_does_not_wait_for_a_childs_cleanup(tmp_path):
    async def abort_a_slow_cleanup():
        task, pid_file, cleaned_up = _start_call(stubs.clean_up_slowly, "subproc", tmp_path, cleanup_s=5.0)
        await asyncio.wait_for(_until_exists(pid_file), timeout=60)
        start = time.monotonic()
        await call_agent_abort_hook(TRAINING)
        with pytest.raises(InfraAbort):
            await asyncio.wait_for(task, timeout=10)
        elapsed = time.monotonic() - start
        cleaned_when_the_call_ended = cleaned_up.exists()
        await asyncio.wait_for(_until_exists(cleaned_up), timeout=30)
        return elapsed, cleaned_when_the_call_ended

    elapsed, cleaned_when_the_call_ended = asyncio.run(abort_a_slow_cleanup())

    assert elapsed < 2.0
    assert not cleaned_when_the_call_ended


def test_an_agent_that_ignores_the_abort_holds_nothing_and_dies_at_the_grace(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_function, "_CANCEL_GRACE_S", 3.0)

    async def abort_an_ignoring_agent():
        task, pid_file, _ = _start_call(stubs.ignore_cancellation, "subproc", tmp_path)
        await asyncio.wait_for(_until_exists(pid_file), timeout=60)
        start = time.monotonic()
        await call_agent_abort_hook(TRAINING)
        with pytest.raises(InfraAbort):
            await asyncio.wait_for(task, timeout=10)
        elapsed = time.monotonic() - start
        pid = int(pid_file.read_text())
        return elapsed, _is_running(pid), await _wait_until_gone(pid, timeout_s=30)

    elapsed, running_when_the_call_ended, gone = asyncio.run(abort_an_ignoring_agent())

    assert elapsed < 2.0
    assert running_when_the_call_ended
    assert gone


@pytest.mark.parametrize("mode", ["subproc", "inline"])
def test_a_rollout_abort_leaves_an_evaluations_calls_running(tmp_path, mode):
    async def abort_training():
        train, train_pid, _ = _start_call(stubs.wait_until_cancelled, mode, tmp_path, "train")
        evaluation, eval_pid, _ = _start_call(stubs.wait_until_cancelled, mode, tmp_path, "eval", scope=EVAL_SCOPE)
        await asyncio.wait_for(asyncio.gather(_until_exists(train_pid), _until_exists(eval_pid)), timeout=60)
        await call_agent_abort_hook(TRAINING)
        with pytest.raises(InfraAbort):
            await asyncio.wait_for(train, timeout=10)
        await asyncio.sleep(1)
        still_running = not evaluation.done()
        cancelled_by_the_eval_abort = await cancel_calls(EVAL_SCOPE)
        with pytest.raises(InfraAbort):
            await asyncio.wait_for(evaluation, timeout=10)
        return still_running, cancelled_by_the_eval_abort

    assert asyncio.run(abort_training()) == (True, 1)


def test_a_finished_call_returns_its_value_and_leaves_no_entry(tmp_path):
    async def run_one():
        result = await call_agent_function(stubs.report_pid, mode="inline", abort_scope=TRAINING_SCOPE)
        return result["pid"], has_calls(), await cancel_calls()

    assert asyncio.run(run_one()) == (os.getpid(), False, 0)


@pytest.mark.parametrize("agent", [stubs.wait_until_cancelled, stubs.return_when_cancelled], ids=lambda a: a.__name__)
def test_cancelling_the_caller_of_an_inline_call_is_not_an_abort(tmp_path, agent):
    async def cancel_the_caller():
        task, pid_file, cleaned_up = _start_call(agent, "inline", tmp_path)
        await asyncio.wait_for(_until_exists(pid_file), timeout=60)
        task.cancel()
        outcome = (await asyncio.gather(task, return_exceptions=True))[0]
        return outcome, cleaned_up

    outcome, cleaned_up = asyncio.run(cancel_the_caller())

    assert not isinstance(outcome, InfraAbort)
    assert cleaned_up.exists()
