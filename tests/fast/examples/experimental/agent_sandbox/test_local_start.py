import asyncio
import os

import httpx
import pytest
from examples.experimental.agent_sandbox import local_start
from tests.fast.rollout.agentic.test_worker import _wait_for

from miles.rollout.agentic.agent_function import call_agent_function
from miles.utils.function_registry import function_registry
from miles.utils.test_utils import agent_function_stubs as stubs

START_PATH = "test_local_start.start"


@pytest.fixture
def started_urls():
    urls: list[str] = []

    def start(task_id, tasks_dir):
        close, url = local_start.start(task_id, tasks_dir)
        urls.append(url)
        return close, url

    with function_registry.temporary(START_PATH, start):
        yield urls


def _is_down(url: str) -> bool:
    try:
        httpx.get(f"{url}/health", timeout=5)
    except httpx.TransportError:
        return True
    return False


def test_each_call_runs_on_a_worker_process_of_its_own_that_stops_after(started_urls):
    urls = started_urls

    async def two_calls():
        return await asyncio.gather(
            *(call_agent_function(stubs.report_pid, mode="worker", worker=START_PATH) for _ in range(2))
        )

    reported = asyncio.run(two_calls())

    assert len({r["ppid"] for r in reported}) == 2
    assert os.getpid() not in {r["ppid"] for r in reported}
    assert len(set(urls)) == 2 and all(_is_down(url) for url in urls)


def test_cancelling_the_call_runs_the_agent_cleanup_and_stops_the_worker(started_urls, tmp_path):
    urls = started_urls
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

    assert _wait_for(cleaned_up)
    assert _is_down(urls[0])
