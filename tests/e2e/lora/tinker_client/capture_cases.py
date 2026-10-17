import asyncio

import httpx
import pytest
from tinker_agent.proxy import serve_proxy
from tinker_agent.trajectory import flush_capture, read_samples, rows_to_trajectory
from tinker_cookbook.capture import capture
from tinker_cookbook.capture.store.client import StoreSink

import tinker


@pytest.mark.asyncio
async def test_capture_groups_are_isolated_and_ordered_by_call_time(capture_store, sampler_factory):
    sampler = sampler_factory()
    started = asyncio.Queue()
    results = {}

    async def complete(prompt):
        token = prompt.to_ints()[0]
        results[token] = asyncio.get_running_loop().create_future()
        started.put_nowait(token)
        await results[token]
        return await sampler.response(prompt)

    sampler.complete = complete
    tasks = []
    for group, token in [(0, 10), (0, 20), (1, 30)]:
        with capture(run_id="test", group_idx=group, traj_idx=0):
            tasks.append(
                asyncio.create_task(
                    sampler.sample_async(tinker.ModelInput.from_ints([token]), 1, tinker.SamplingParams(max_tokens=8))
                )
            )
        assert await started.get() == token
    for index, token in [(1, 20), (2, 30), (0, 10)]:
        results[token].set_result(None)
        await tasks[index]
    await flush_capture()
    first = await read_samples(capture_store.base_url, "test", group_idx=0, traj_idx=0)
    second = await read_samples(capture_store.base_url, "test", group_idx=1, traj_idx=0)
    assert [r["prompt_tokens"] for r in first] == [[10], [20]]
    assert [r["prompt_tokens"] for r in second] == [[30]]


@pytest.mark.asyncio
async def test_export_failure_prevents_training(capture_store, sampler_factory, monkeypatch):
    def fail_export(self, records, timeout=None):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(StoreSink, "export", fail_export)
    sampler = sampler_factory()
    await sampler.sample_async(tinker.ModelInput.from_ints([1]), 1, tinker.SamplingParams(max_tokens=8))
    with pytest.raises(RuntimeError, match="capture data is incomplete"):
        await flush_capture()


@pytest.mark.asyncio
async def test_reader_consumes_all_pages(capture_store, sampler_factory):
    sampler = sampler_factory()
    with capture(run_id="test", traj_idx=0):
        for token in range(1001):
            await sampler.sample_async(tinker.ModelInput.from_ints([token]), 1, tinker.SamplingParams(max_tokens=8))
    await flush_capture()
    rows = await read_samples(capture_store.base_url, "test", traj_idx=0)
    assert [r["prompt_tokens"][0] for r in rows] == list(range(1001))


@pytest.mark.asyncio
async def test_missing_logprobs_cannot_become_training_data(capture_store, sampler_factory):
    sampler = sampler_factory()

    async def complete(prompt):
        response = await sampler.response(prompt)
        response.sequences[0].logprobs = None
        return response

    sampler.complete = complete
    await sampler.sample_async(tinker.ModelInput.from_ints([1]), 1, tinker.SamplingParams(max_tokens=8))
    await flush_capture()
    with pytest.raises(AssertionError):
        rows_to_trajectory(await read_samples(capture_store.base_url, "test"), "completed")


@pytest.mark.asyncio
async def test_proxy_shutdown_waits_for_accepted_samples(capture_store, sampler_factory, renderer):
    sampler = sampler_factory()
    entered, release, closing = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def complete(prompt):
        entered.set()
        await release.wait()
        return await sampler.response(prompt)

    sampler.complete = complete
    async with httpx.AsyncClient() as http:

        async def serve():
            async with serve_proxy(renderer, sampler, model_label="policy", default_max_tokens=8) as port:
                request = asyncio.create_task(
                    http.post(
                        f"http://127.0.0.1:{port}/r/run/test/traj/0/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "hi"}]},
                    )
                )
                await entered.wait()
                closing.set()
            response = await request
            response.raise_for_status()

        task = asyncio.create_task(serve())
        await closing.wait()
        assert not task.done()
        release.set()
        await task
    await flush_capture()
    assert len(await read_samples(capture_store.base_url, "test", traj_idx=0)) == 1
