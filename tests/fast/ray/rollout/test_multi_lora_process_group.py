"""Pins process_group's submission-time slot-version stamping: the staleness
filter compares against the version live when the group was submitted, not
when it completed."""

import pytest

import miles.rollout.multi_lora.async_rollout as mod
from miles.rollout.multi_lora.async_rollout import process_group
from miles.utils.types import AdapterRef, Sample


class FakeDataSource:
    def __init__(self) -> None:
        self.added: list = []

    def add_samples(self, groups) -> None:
        self.added.extend(groups)


class FakeAdapterView:
    def __init__(self, version: int, registration_id: str = "reg-1") -> None:
        self.version = version
        self.registration_id = registration_id


class FakeAdaptersCache:
    def __init__(self, versions: dict[str, int]) -> None:
        self.versions = versions

    def bump(self, name: str, to: int) -> None:
        self.versions[name] = to

    async def get(self, adapter_name: str) -> FakeAdapterView | None:
        version = self.versions.get(adapter_name)
        return FakeAdapterView(version) if version is not None else None


@pytest.mark.asyncio
async def test_process_group_stamps_submission_version(monkeypatch):
    """The stamp is the version live at submission (5), not completion (7)."""
    cache = FakeAdaptersCache({"A": 5})

    async def gen(args, group, sampling_params):
        cache.bump("A", 7)  # update lands mid-generation
        for s in group:
            s.status = Sample.Status.COMPLETED
        return group

    monkeypatch.setattr(mod, "AdaptersCache", lambda: cache)

    g = [Sample(prompt="p", group_index=3, rollout_id=7, adapter=AdapterRef("A", 0))]
    result = await process_group(None, g, {}, gen, FakeDataSource())

    assert result is g
    assert g[0].metadata["slot_version"] == 5
    assert g[0].metadata["registration_id"] == "reg-1"


@pytest.mark.asyncio
async def test_process_group_propagates_root_identity_to_all_outputs(monkeypatch):
    cache = FakeAdaptersCache({})
    monkeypatch.setattr(mod, "AdaptersCache", lambda: cache)
    adapter = AdapterRef("A", 0)
    roots = [
        Sample(prompt="p0", group_index=3, rollout_id=10, adapter=adapter),
        Sample(prompt="p1", group_index=3, rollout_id=11, adapter=adapter),
    ]

    async def gen(args, group, sampling_params):
        return [
            Sample(status=Sample.Status.COMPLETED),
            [
                Sample(status=Sample.Status.COMPLETED),
                Sample(status=Sample.Status.COMPLETED),
            ],
        ]

    result = await process_group(None, roots, {}, gen, FakeDataSource())

    scalar = result[0]
    compact = result[1]
    assert (scalar.group_index, scalar.rollout_id, scalar.adapter) == (3, 10, adapter)
    assert [(sample.group_index, sample.rollout_id, sample.adapter) for sample in compact] == [
        (3, 11, adapter),
        (3, 11, adapter),
    ]


@pytest.mark.asyncio
async def test_process_group_rejects_changed_top_level_cardinality(monkeypatch):
    cache = FakeAdaptersCache({})
    monkeypatch.setattr(mod, "AdaptersCache", lambda: cache)
    roots = [Sample(prompt="p", group_index=3, rollout_id=10, adapter=AdapterRef("A", 0))]

    async def gen(args, group, sampling_params):
        return []

    with pytest.raises(ValueError, match="one output per input root"):
        await process_group(None, roots, {}, gen, FakeDataSource())


@pytest.mark.asyncio
async def test_process_group_rejects_conflicting_leaf_identity(monkeypatch):
    cache = FakeAdaptersCache({})
    monkeypatch.setattr(mod, "AdaptersCache", lambda: cache)
    roots = [Sample(prompt="p", group_index=3, rollout_id=10, adapter=AdapterRef("A", 0))]

    async def gen(args, group, sampling_params):
        group[0].group_index = 99
        group[0].status = Sample.Status.COMPLETED
        return group

    with pytest.raises(ValueError, match="group_index"):
        await process_group(None, roots, {}, gen, FakeDataSource())
