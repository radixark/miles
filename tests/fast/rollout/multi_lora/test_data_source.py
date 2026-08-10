from argparse import Namespace
from collections import deque
from typing import Any

import pytest

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


class _FakeController:
    def __init__(self, snapshots) -> None:
        self.resolutions: list[tuple[str, int]] = []
        self.snapshots = snapshots

    async def snapshot(self):
        return self.snapshots.pop(0)

    async def resolve_num_step(self, name: str, dataset_length: int) -> None:
        self.resolutions.append((name, dataset_length))


class TestMultiLoRAAsyncDataSource:
    async def test_public_sampling_rotates_to_the_active_source_and_resolves_its_length(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Public sampling builds an active source, resolves its length, and returns adapter-stamped samples."""
        adapter = AdapterRun(
            name="active",
            slot=3,
            config=AdapterRunConfig(data="dataset", rollout_batch_size=1, n_samples_per_prompt=1),
        )
        snapshot = {"active": {"active": adapter}, "retiring": {}}
        controller = _FakeController([snapshot, {"active": {}, "retiring": {}}])
        args = Namespace(
            input_key="text",
            label_key=None,
            metadata_key=None,
            save=None,
            load=None,
            n_samples_per_prompt=1,
            rollout_global_dataset=True,
            hf_checkpoint="model",
            chat_template_path=None,
            dump_details=None,
            rollout_max_prompt_len=32,
            multimodal_keys=None,
            tool_key=None,
            apply_chat_template=False,
            apply_chat_template_kwargs=None,
            rollout_seed=1,
            rollout_shuffle=False,
        )

        class _Dataset:
            def __init__(self, *_args, **_kwargs) -> None:
                self.samples = [Sample(), Sample(), Sample()]

            def __len__(self) -> int:
                return len(self.samples)

        monkeypatch.setattr(data_source_module, "get_multi_lora_controller", lambda: controller)
        monkeypatch.setattr("miles.rollout.data_source.load_tokenizer", lambda *_args, **_kwargs: object())
        monkeypatch.setattr("miles.rollout.data_source.load_processor", lambda *_args, **_kwargs: object())
        monkeypatch.setattr("miles.rollout.data_source.Dataset", _Dataset)
        source = data_source_module.MultiLoRAAsyncDataSource(args)

        groups = await source.get_samples()

        assert controller.resolutions == [("active", 3)]
        assert groups[0][0].adapter.name == "active"
        assert (await source.get_samples()) == []
