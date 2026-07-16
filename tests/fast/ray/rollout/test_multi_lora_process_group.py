"""Tests for the multi-LoRA async rollout process_group core:
keep-vs-recycle plus submission-time slot-version stamping."""

import examples.multi_lora.multi_lora_async_rollout as mod
import pytest
from examples.multi_lora.multi_lora_async_rollout import process_group

from miles.utils.types import AdapterRef, Sample


class FakeDataSource:
    def __init__(self) -> None:
        self.added: list = []

    def add_samples(self, groups) -> None:
        self.added.extend(groups)


class FakeAdapterView:
    def __init__(self, version: int) -> None:
        self.version = version


class FakeAdaptersCache:
    def __init__(self, versions: dict[str, int]) -> None:
        self.versions = versions

    def bump(self, name: str, to: int) -> None:
        self.versions[name] = to

    async def get_all(self) -> dict[str, FakeAdapterView]:
        return {name: FakeAdapterView(version) for name, version in self.versions.items()}

    async def get(self, adapter_name: str) -> FakeAdapterView | None:
        return (await self.get_all()).get(adapter_name)


def group(adapter: str = "A", slot: int = 0) -> list[Sample]:
    return [Sample(prompt="p", adapter=AdapterRef(adapter, slot))]


async def gen_completed(args, group, sampling_params):
    for s in group:
        s.status = Sample.Status.COMPLETED
    return group


@pytest.mark.asyncio
async def test_process_group_keeps_completed():
    ds = FakeDataSource()
    g = group("A")
    result = await process_group(None, g, {}, gen_completed, ds)

    assert result is g
    assert ds.added == []


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="Re-queuing aborted groups is not wired up yet (the per-adapter "
    "source is read-only); planned for a future PR. This test pins the "
    "intended end-state behavior.",
    strict=True,
)
async def test_process_group_recycles_aborted():
    async def gen(args, group, sampling_params):
        for s in group:
            s.status = Sample.Status.ABORTED
        return group

    ds = FakeDataSource()
    g = group("A")
    result = await process_group(None, g, {}, gen, ds)

    assert result is None
    assert len(ds.added) == 1


@pytest.mark.asyncio
async def test_process_group_stamps_submission_version(monkeypatch):
    """The stamp is the version live at submission (5), not completion (7)."""
    cache = FakeAdaptersCache({"A": 5})

    async def gen(args, group, sampling_params):
        cache.bump("A", 7)  # update lands mid-generation
        return await gen_completed(args, group, sampling_params)

    monkeypatch.setattr(mod, "AdaptersCache", lambda: cache)

    ds = FakeDataSource()
    g = group("A")
    result = await process_group(None, g, {}, gen, ds)

    assert result is g
    assert g[0].metadata["slot_version"] == 5


@pytest.mark.asyncio
async def test_process_group_no_adapter_skips_stamp(monkeypatch):
    class FailingCache:
        async def get(self, adapter_name):
            raise AssertionError("adapters cache should not be queried for adapter-less group")

    monkeypatch.setattr(mod, "AdaptersCache", FailingCache)

    ds = FakeDataSource()
    g = [Sample(prompt="p", adapter=None)]
    result = await process_group(None, g, {}, gen_completed, ds)

    assert result is g
    assert "slot_version" not in g[0].metadata
