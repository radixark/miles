"""Tests for the testable core of the multi-LoRA async rollout (process_group).

rid stamping lives in ``generate`` (next to ``lora_path``), not here, so these
tests cover only keep-vs-recycle.
"""

import pytest

from miles.utils.types import AdapterRef, Sample

from examples.multi_lora.multi_lora_async_rollout import process_group


class FakeDataSource:
    def __init__(self) -> None:
        self.added: list = []

    def add_samples(self, groups) -> None:
        self.added.extend(groups)


def group(adapter: str = "A", slot: int = 0) -> list[Sample]:
    return [Sample(prompt="p", adapter=AdapterRef(adapter, slot))]


@pytest.mark.asyncio
async def test_process_group_keeps_completed():
    async def gen(args, group, sampling_params):
        for s in group:
            s.status = Sample.Status.COMPLETED
        return group

    ds = FakeDataSource()
    g = group("A")
    result = await process_group(None, g, {}, gen, ds)

    assert result is g
    assert ds.added == []


@pytest.mark.asyncio
async def test_process_group_recycles_aborted():
    async def gen(args, group, sampling_params):
        for s in group:
            s.status = Sample.Status.ABORTED
        return group

    ds = FakeDataSource()
    g = group("A")
    result = await process_group(None, g, {}, gen, ds)

    assert result is None
    assert len(ds.added) == 1  # recycled back to the data source
