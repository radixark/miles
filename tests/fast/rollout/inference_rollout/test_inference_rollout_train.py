import asyncio
import logging
from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.rollout.inference_rollout import inference_rollout_train
from miles.utils.output_store.plan import StorePlanError


class TestAbort:
    async def test_one_unresponsive_worker_does_not_stop_abort_cleanup(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One failed worker abort still runs the agent hook and returns pending partial groups."""
        args = Namespace(
            partial_rollout=True,
            sglang_router_ip="router",
            sglang_router_port=30000,
        )
        state = SimpleNamespace(args=args, aborted=False)
        sample = SimpleNamespace(response="partial", metadata={})
        hook_calls: list[Namespace] = []
        posted_urls: list[str] = []
        hook_finished = asyncio.Event()

        async def fake_get(url: str) -> dict[str, list[dict[str, str]]]:
            return {"workers": [{"url": "http://healthy"}, {"url": "http://unresponsive"}]}

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

        pending = asyncio.create_task(finish_group())
        await asyncio.sleep(0)
        monkeypatch.setattr(inference_rollout_train, "get", fake_get)
        monkeypatch.setattr(inference_rollout_train, "post", fake_post)
        monkeypatch.setattr(inference_rollout_train, "call_agent_abort_hook", fake_agent_abort_hook)

        with caplog.at_level(logging.WARNING, logger=inference_rollout_train.__name__):
            aborted_groups = await inference_rollout_train.abort(state, {pending}, rollout_id=17)

        assert posted_urls == ["http://healthy/abort_request", "http://unresponsive/abort_request"]
        assert hook_calls == [args]
        assert aborted_groups == [[sample]]
        assert sample.metadata["start_rollout_id"] == 17
        assert "Failed to abort worker at http://unresponsive: worker cannot answer" in caplog.text


class TestGroupErrors:
    async def test_a_non_retryable_group_error_stops_the_rollout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A broken contract fails every group the same way: raise instead of resubmitting forever."""
        args = Namespace(
            rollout_global_dataset=True,
            dynamic_sampling_filter_path=None,
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            over_sampling_batch_size=1,
        )
        submitted: list[asyncio.Task] = []

        async def failing_group() -> list:
            raise StorePlanError("the request asked for routed_experts, but the response's output store ref has none")

        def fake_submit(state, samples, callback) -> list[asyncio.Task]:
            submitted.append(asyncio.ensure_future(failing_group()))
            return submitted[-1:]

        class Scheduler:
            sample_done_callback = None

            def has_capacity(self, *, pending_groups: int, group_budget: int) -> bool:
                # Bounded, so a loop that swallows the error ends instead of resubmitting forever.
                return pending_groups == 0 and len(submitted) < 3

            def on_submit(self, samples) -> None:
                pass

            async def wait_for_progress(self, pendings):
                return await asyncio.wait(pendings)

        async def no_dumper(args) -> None:
            pass

        monkeypatch.setattr(inference_rollout_train.dumper_utils, "configure_sglang", no_dumper)
        monkeypatch.setattr(inference_rollout_train, "load_function", lambda path: None)
        monkeypatch.setattr(inference_rollout_train, "make_submission_scheduler", lambda args, default: Scheduler())
        monkeypatch.setattr(inference_rollout_train, "submit_generate_tasks", fake_submit)

        with pytest.raises(StorePlanError):
            await inference_rollout_train.generate_rollout_async(SimpleNamespace(args=args), 0, lambda n: [[None]])
        assert len(submitted) == 1
