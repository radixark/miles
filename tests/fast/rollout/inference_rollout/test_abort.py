from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import logging
from argparse import Namespace
from types import SimpleNamespace

import httpx
import pytest

import miles.rollout.inference_rollout.inference_rollout_train as train

WORKER_URLS = ["http://10.0.0.1:20143", "http://10.0.0.2:20144"]


def _make_state() -> SimpleNamespace:
    return SimpleNamespace(args=Namespace(partial_rollout=False), aborted=False)


async def _no_pendings():
    return
    yield


@pytest.fixture
def aborting(monkeypatch: pytest.MonkeyPatch):
    posted: list[str] = []

    async def fake_post(url, _payload):
        posted.append(url)
        if url.startswith(WORKER_URLS[0]):
            raise httpx.ConnectError("All connection attempts failed")

    monkeypatch.setattr(train, "get_worker_urls", lambda _args: _ready(WORKER_URLS))
    monkeypatch.setattr(train, "post", fake_post)
    monkeypatch.setattr(train, "call_agent_abort_hook", lambda _args: _ready(None))
    monkeypatch.setattr(train, "as_completed_async", lambda _pendings: _no_pendings())
    return posted


async def _ready(value):
    return value


class TestAbortSurvivesAnEngineThatCannotAnswer:
    @pytest.mark.asyncio
    async def test_an_engine_being_healed_does_not_end_the_rollout(self, aborting: list[str]):
        """A worker restarting is unreachable by definition, and losing its requests anyway."""
        await train.abort(_make_state(), set(), rollout_id=0)

    @pytest.mark.asyncio
    async def test_the_engines_that_can_answer_are_still_told_to_abort(self, aborting: list[str]):
        """One unreachable worker must not cancel the aborts its neighbours can still receive."""
        await train.abort(_make_state(), set(), rollout_id=0)

        assert sorted(aborting) == sorted(f"{url}/abort_request" for url in WORKER_URLS)

    @pytest.mark.asyncio
    async def test_the_worker_that_could_not_be_aborted_is_named(
        self, aborting: list[str], caplog: pytest.LogCaptureFixture
    ):
        """Silently swallowing the failure would hide a fleet that is losing engines for good."""
        with caplog.at_level(logging.WARNING, logger=train.logger.name):
            await train.abort(_make_state(), set(), rollout_id=0)

        assert [record for record in caplog.records if WORKER_URLS[0] in record.getMessage()]

    @pytest.mark.asyncio
    async def test_a_router_listing_no_worker_at_all_is_not_a_failed_abort(self, monkeypatch: pytest.MonkeyPatch):
        """A router mid-heal can list nothing, and no engine to ask is not the same as none answering."""
        monkeypatch.setattr(train, "get_worker_urls", lambda _args: _ready([]))
        monkeypatch.setattr(train, "call_agent_abort_hook", lambda _args: _ready(None))
        monkeypatch.setattr(train, "as_completed_async", lambda _pendings: _no_pendings())

        await train.abort(_make_state(), set(), rollout_id=0)

    @pytest.mark.asyncio
    async def test_the_agent_integration_is_still_torn_down(self, monkeypatch: pytest.MonkeyPatch, aborting):
        """The hook runs after the aborts, so an unreachable engine must not skip it."""
        called: list[bool] = []

        async def fake_hook(_args):
            called.append(True)

        monkeypatch.setattr(train, "call_agent_abort_hook", fake_hook)

        await train.abort(_make_state(), set(), rollout_id=0)

        assert called == [True]
