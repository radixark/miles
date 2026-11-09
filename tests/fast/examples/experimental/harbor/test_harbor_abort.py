import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from miles.rollout.agentic.agent_function import InfraAbort, call_agent_function  # noqa: E402
from miles.utils.misc import call_agent_abort_hook  # noqa: E402

from .blocking_environment import run_trial  # noqa: E402

TRAINING = SimpleNamespace(custom_agent_function_path=None, sglang_router_ip="127.0.0.1", sglang_router_port=30000)


def _write_task(tasks_dir: Path) -> None:
    task = tasks_dir / "task-1"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM scratch\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\n")
    (task / "instruction.md").write_text("Do nothing.\n")
    (task / "task.toml").write_text('version = "1.0"\n')


def test_a_rollout_abort_stops_a_real_harbor_trials_environment_under_subproc(tmp_path, monkeypatch):
    _write_task(tmp_path)
    markers = tmp_path / "markers"
    markers.mkdir()
    monkeypatch.setenv("BLOCKING_ENV_MARKERS", str(markers))
    monkeypatch.setenv("HARBOR_TASKS_DIR", str(tmp_path))
    monkeypatch.setenv("HARBOR_TRIALS_DIR", str(tmp_path / "trials"))
    monkeypatch.setenv("HARBOR_ENV_TYPE", "")

    async def abort_mid_trial():
        call = asyncio.create_task(
            call_agent_function(
                run_trial,
                mode="subproc",
                abort_scope="127.0.0.1:30000",
                base_url="http://127.0.0.1:1/sessions/s1",
                prompt="do nothing",
                request_kwargs={},
                metadata={"instance_id": "task-1", "agent_name": "nop"},
            )
        )
        while not (markers / "verifier_running").exists():
            assert not call.done(), call
            await asyncio.sleep(0.1)
        await call_agent_abort_hook(TRAINING)
        with pytest.raises(InfraAbort) as raised:
            await asyncio.wait_for(call, timeout=10)
        while not (markers / "stopped_delete_True").exists():
            await asyncio.sleep(0.1)
        return raised.value.exit_status

    assert asyncio.run(asyncio.wait_for(abort_mid_trial(), timeout=120)) == "Cancelled"
