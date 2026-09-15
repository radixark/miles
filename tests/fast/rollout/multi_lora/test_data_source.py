from argparse import Namespace
from collections import deque
from typing import Any

import miles.rollout.multi_lora.data_source as data_source_module
from miles.rollout.multi_lora.data_source import MultiLoRAAsyncDataSource
from miles.utils.adapter_config import AdapterRun, AdapterRunConfig
from miles.utils.types import AdapterRef, RewardSpec, Sample


class InnerSourceFake:
    def __init__(self, groups: list[list[Sample]]) -> None:
        self.groups = deque(groups)
        self.added: list[list[Sample]] = []

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        if not self.groups:
            return []
        return [self.groups.popleft()]

    def add_samples(self, samples: list[list[Sample]]) -> None:
        self.added.extend(samples)


def make_adapter(name: str, slot: int, **config_overrides: Any) -> AdapterRun:
    config = AdapterRunConfig(data=f"/{name}", **config_overrides)
    return AdapterRun(name=name, config=config, slot=slot)


def make_data_source() -> MultiLoRAAsyncDataSource:
    return MultiLoRAAsyncDataSource(Namespace())


class TestGetSamples:
    async def test_get_samples_awaits_reconciliation_and_returns_the_next_stamped_live_group(
        self, monkeypatch: Any
    ) -> None:
        """Retrieval reconciles live adapters, skips exhaustion, and stamps the next group."""
        exhausted = make_adapter(name="exhausted", slot=1)
        live = make_adapter(
            name="live",
            slot=2,
            rm_type="custom",
            custom_rm_path="reward.py",
            metadata={"shared": "adapter", "adapter_only": 1},
        )
        group = [Sample(prompt="prompt", metadata={"shared": "sample", "sample_only": 2})]
        exhausted_source = InnerSourceFake([])
        live_source = InnerSourceFake([group])
        stale_source = InnerSourceFake([[Sample(prompt="stale")]])
        data_source = make_data_source()
        data_source.sources = {
            "stale": stale_source,
            "exhausted": exhausted_source,
            "live": live_source,
        }
        data_source.source_queue = deque(["stale", "exhausted", "live"])
        snapshot_completed = False

        async def fetch_snapshot() -> dict[str, dict[str, AdapterRun]]:
            nonlocal snapshot_completed
            snapshot_completed = True
            return {"active": {"exhausted": exhausted, "live": live}, "retiring": {}}

        monkeypatch.setattr(data_source_module, "fetch_snapshot", fetch_snapshot)

        groups = await data_source.get_samples(num_samples=1)

        assert snapshot_completed
        assert groups == [group]
        assert "stale" not in data_source.sources
        assert group[0].adapter == AdapterRef(name="live", slot=2)
        assert group[0].reward_spec == RewardSpec(rm_type="custom", custom_rm_path="reward.py")
        assert group[0].metadata == {"shared": "sample", "adapter_only": 1, "sample_only": 2}


class TestAddSamples:
    async def test_add_samples_recycles_only_groups_for_live_adapters(self, monkeypatch: Any) -> None:
        """Recycling retains only tagged groups whose adapters remain registered."""
        live = make_adapter(name="live", slot=1)
        live_source = InnerSourceFake([])
        removed_source = InnerSourceFake([])
        data_source = make_data_source()
        data_source.sources = {"live": live_source, "removed": removed_source}
        data_source.source_queue = deque(["live", "removed"])
        live_group = [Sample(prompt="live", adapter=AdapterRef(name="live", slot=1))]
        removed_group = [Sample(prompt="removed", adapter=AdapterRef(name="removed", slot=2))]
        untagged_group = [Sample(prompt="untagged")]

        async def fetch_snapshot() -> dict[str, dict[str, AdapterRun]]:
            return {"active": {"live": live}, "retiring": {}}

        monkeypatch.setattr(data_source_module, "fetch_snapshot", fetch_snapshot)

        await data_source.add_samples(samples=[live_group, removed_group, untagged_group])

        assert live_source.added == [live_group]
        assert removed_source.added == []
        assert "removed" not in data_source.sources
