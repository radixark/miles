import asyncio
import logging
from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.rollout import sglang_rollout


class TestAbort:
    async def test_abort_survives_one_unresponsive_worker_and_finishes_cleanup(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One failed worker abort still runs the agent hook and drains partial samples."""
        args = Namespace(
            partial_rollout=True,
            sglang_router_ip="router",
            sglang_router_port=30000,
            use_miles_router=True,
        )
        sample = SimpleNamespace(response="partial", metadata={})
        state = SimpleNamespace(args=args, aborted=False)
        hook_calls: list[Namespace] = []
        posted_urls: list[str] = []
        hook_finished = asyncio.Event()

        async def fake_get(url: str) -> dict[str, list[str]]:
            return {"urls": ["http://healthy", "http://unresponsive"]}

        async def fake_post(url: str, payload: dict[str, bool]) -> None:
            posted_urls.append(url)
            if "unresponsive" in url:
                raise ConnectionError("worker cannot answer")

        async def fake_agent_abort_hook(hook_args: Namespace) -> None:
            hook_calls.append(hook_args)
            hook_finished.set()

        async def finish_group() -> list[SimpleNamespace]:
            await hook_finished.wait()
            return [sample]

        state.pendings = {asyncio.create_task(finish_group())}
        await asyncio.sleep(0)
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda state_args: state)
        monkeypatch.setattr(sglang_rollout, "get", fake_get)
        monkeypatch.setattr(sglang_rollout, "post", fake_post)
        monkeypatch.setattr(sglang_rollout, "call_agent_abort_hook", fake_agent_abort_hook)

        with caplog.at_level(logging.WARNING, logger=sglang_rollout.__name__):
            aborted_groups = await sglang_rollout.abort(args, rollout_id=23)

        assert posted_urls == ["http://healthy/abort_request", "http://unresponsive/abort_request"]
        assert hook_calls == [args]
        assert aborted_groups == [[sample]]
        assert sample.metadata["start_rollout_id"] == 23
        assert "Failed to abort worker at http://unresponsive: worker cannot answer" in caplog.text
