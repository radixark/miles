import argparse
import logging
from pathlib import Path

import polars as pl
import pytest
import torch
from fastapi.testclient import TestClient
from tests.fast.dashboard.dummy_dump import dump_segmented_run

from miles.dashboard.args import collector_config_from_args
from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.server import make_app
from miles.dashboard.store import Meta, MetricStore, TrajectoryEvent, TrajectoryEventKind
from miles.utils.args.configs.dashboard import DashboardConfig
from miles.utils.types import SampleLineage


def _set_train_raw_reward(dump_dir: Path, values: list | None) -> None:
    for path in (dump_dir / "train_data").glob("*.pt"):
        pack = torch.load(path, weights_only=False)
        if values is None:
            pack["rollout_data"].pop("raw_reward")
        else:
            pack["rollout_data"]["raw_reward"] = values
        torch.save(pack, path)


def _write_reward_key(dump_dir: Path, reward_key: str) -> None:
    MetricStore(dump_dir / "dashboard").write_meta(Meta(run_name="run", start_ts=0.0, args={"reward_key": reward_key}))


@pytest.mark.parametrize("explicit_ids", [True, False], ids=["rollout_id", "legacy_index"])
@pytest.mark.parametrize("compact_rewards", [True, False], ids=["episode_rewards", "sample_rewards"])
def test_segmented_rollouts_keep_tokens_and_equal_episode_weight(tmp_path, explicit_ids, compact_rewards):
    samples = dump_segmented_run(tmp_path, explicit_ids=explicit_ids, compact_rewards=compact_rewards)
    reader = DumpReader(tmp_path)
    joined = reader.load_joined(0)
    assert joined.train_coverage == 1.0
    for key, row in joined.train_rows.items():
        sample = [s for s in samples if s.index == key[0]][key[1]]
        assert row.raw_reward == sample.reward
        assert row.tokens.tolist() == sample.tokens
    assert reader.summary(0).height == 8
    assert reader.episodes(0)["n_samples"].to_list() == [3, 1, 2, 2]
    assert reader.episodes(0)["episode_reward"].to_list() == [1.0, 0.0, 0.25, 0.75]
    assert reader.groups(0)["n"].to_list() == [2, 2]
    assert reader.groups(0)["reward_mean"].to_list() == [0.5, 0.5]
    assert reader.groups(0)["reward_std"].to_list() == pytest.approx([2**-0.5, 0.5 * 2**-0.5])
    assert reader.step_aggregates()["reward_mean"].to_list() == [0.5]
    assert reader.episodes(0, evaluation=True)["reward"].to_list() == [1.0, 0.0, 0.25, 0.75]

    client = TestClient(make_app(MetricStore(tmp_path / "dashboard"), reader))
    for endpoint in ("summary", "episodes", "groups", "trajectories"):
        response = client.get(f"/api/rollout/0/{endpoint}")
        assert response.status_code == 200, response.text
    assert client.get("/api/advisory").status_code == 200
    episode = reader.episodes(0).row(0, named=True)
    tokens = client.get(f"/api/rollout/0/sample/{episode['sample_index']}/tokens")
    assert tokens.status_code == 200
    assert tokens.json()["token_ids"] == samples[0].tokens


def test_episode_rewards_survive_a_train_only_compact_reward_layout(tmp_path):
    dump_segmented_run(tmp_path)
    for path in (tmp_path / "train_data").glob("*.pt"):
        pack = torch.load(path, weights_only=False)
        pack["rollout_data"]["raw_reward"] = [99.0, 88.0]
        torch.save(pack, path)
    assert DumpReader(tmp_path).episodes(0)["episode_reward"].to_list() == [1.0, 0.0, 0.25, 0.75]


def test_known_episode_reward_survives_unjoined_zero_loss_replacements(tmp_path):
    samples = dump_segmented_run(tmp_path, explicit_ids=False)
    for path in (tmp_path / "train_data").glob("*.pt"):
        pack = torch.load(path, weights_only=False)
        data = pack["rollout_data"]
        for row, index in enumerate(data["sample_indices"]):
            if index == 0:
                data["tokens"][row] = torch.tensor([0, 0])
                data["loss_masks"][row] = torch.tensor([0])
                data["response_lengths"][row] = 1
                data["total_lengths"][row] = 2
        torch.save(pack, path)
    reader = DumpReader(tmp_path)
    assert reader.episodes(0)["episode_reward"].to_list() == [1.0, 0.0, 0.25, 0.75]
    assert reader.groups(0)["reward_mean"].to_list() == [0.5, 0.5]
    assert reader.tokens(0, samples[0].index)["train_log_probs"] is None


def test_unsegmented_global_reward_overrides_survive_missing_token_rows(tmp_path):
    from tests.fast.dashboard.dummy_dump import dump_dummy_run

    dump_dummy_run(tmp_path, steps=1, dp_size=2, tp_dup=1)
    for path in (tmp_path / "train_data").glob("*.pt"):
        pack = torch.load(path, weights_only=False)
        data = pack["rollout_data"]
        data["raw_reward"] = [i + 0.25 for i in range(8)]
        count = len(data["sample_indices"])
        retained = [i for i, index in enumerate(data["sample_indices"]) if index != 0]
        for key, values in data.items():
            if key != "raw_reward" and isinstance(values, list) and len(values) == count:
                data[key] = [values[i] for i in retained]
        torch.save(pack, path)
    summary = DumpReader(tmp_path).summary(0).sort("sample_index")
    assert summary["raw_reward"].to_list() == [i + 0.25 for i in range(8)]
    assert summary["dumped_rank"][0] is None


def test_reward_key_reaches_the_reader_through_the_run_args(tmp_path):
    dump_segmented_run(tmp_path, reward_key="acc")
    _set_train_raw_reward(tmp_path, None)
    parser = argparse.ArgumentParser()
    DashboardConfig.add_arguments(parser=parser)
    args = parser.parse_args(["--use-miles-dashboard"])
    args.dump_details = str(tmp_path)
    args.wandb_group = None
    args.reward_key = "acc"
    config = collector_config_from_args(args, start_ts=0.0)
    MetricStore(Path(config.dashboard_dir)).write_meta(
        Meta(run_name=config.run_name, start_ts=0.0, args=config.args_snapshot)
    )
    assert DumpReader(tmp_path).groups(0)["reward_mean"].to_list() == [0.5, 0.5]


def test_singleton_episodes_take_the_reward_key_from_a_dict_raw_reward(tmp_path):
    def prepare(samples):
        for sample in samples:
            sample.metadata["raw_reward"] = {"acc": sample.reward, "format": 1.0}

    dump_segmented_run(tmp_path, lengths=(1, 1, 1, 1), compact_rewards=False, prepare=prepare)
    _write_reward_key(tmp_path, "acc")
    assert DumpReader(tmp_path).summary(0)["raw_reward"].to_list() == [1.0, 0.0, 0.25, 0.75]


def test_a_raw_reward_column_matching_neither_samples_nor_episodes_is_not_sliced(tmp_path, caplog):
    dump_segmented_run(tmp_path, lengths=(1, 1, 1, 1), compact_rewards=False)
    _set_train_raw_reward(tmp_path, [9.0, 1.0, 0.0, 0.25, 0.75])
    with caplog.at_level(logging.WARNING, logger="miles.dashboard.dump_reader"):
        groups = DumpReader(tmp_path).groups(0)
    assert groups["reward_mean"].to_list() == [0.5, 0.5]
    assert "matching neither" in caplog.text


def test_a_multi_sample_episode_keeps_the_per_sample_train_rewards(tmp_path):
    samples = dump_segmented_run(tmp_path, compact_rewards=False)
    _set_train_raw_reward(tmp_path, [10 * sample.reward for sample in samples])
    assert DumpReader(tmp_path).groups(0)["reward_mean"].to_list() == [5.0, 5.0]


def test_an_episode_counts_its_last_sample_reward_once(tmp_path, caplog):
    dump_segmented_run(tmp_path)
    _set_train_raw_reward(tmp_path, None)
    path = tmp_path / "rollout_data/0.pt"
    pack = torch.load(path, weights_only=False)
    for sample, reward in zip(pack["samples"][:3], [0.0, 0.0, 1.0], strict=True):
        sample["reward"] = reward
    torch.save(pack, path)
    reader = DumpReader(tmp_path)
    with caplog.at_level(logging.WARNING, logger="miles.dashboard.dump_reader"):
        assert reader.groups(0)["reward_mean"].to_list() == [0.5, 0.5]
        reader.groups(0)
    assert caplog.text.count("carry different episode_reward") == 1


@pytest.mark.parametrize(
    "metadata", [{"raw_reward": None}, {"others": {"episode_return": None}}], ids=["raw_reward", "episode_return"]
)
def test_a_null_reward_source_falls_through_to_the_next(tmp_path, metadata):
    def prepare(samples):
        for sample in samples:
            sample.metadata.update(metadata)

    dump_segmented_run(tmp_path, compact_rewards=False, prepare=prepare)
    _set_train_raw_reward(tmp_path, None)
    assert DumpReader(tmp_path).groups(0)["reward_mean"].to_list() == [0.5, 0.5]


def test_reward_keeps_the_shaped_sample_reward_beside_the_episode_reward(tmp_path):
    def prepare(samples):
        for sample in samples:
            sample.metadata["raw_reward"] = sample.reward
            sample.reward = sample.reward + 0.5

    samples = dump_segmented_run(tmp_path, compact_rewards=False, prepare=prepare)
    reader = DumpReader(tmp_path)
    assert reader.summary(0)["reward"].to_list() == [sample.reward for sample in samples]
    assert reader.groups(0, evaluation=True)["reward_mean"].to_list() == [1.0, 1.0]
    assert reader.groups(0)["reward_mean"].to_list() == [0.5, 0.5]


def test_group_response_length_is_a_per_sample_mean(tmp_path):
    dump_segmented_run(tmp_path)
    reader = DumpReader(tmp_path)
    expected = reader.summary(0).group_by("group_index").agg(pl.col("response_length").mean()).sort("group_index")
    assert reader.groups(0)["response_length_mean"].to_list() == expected["response_length"].to_list()


def test_a_rollout_id_equal_to_another_samples_index_is_a_separate_episode(tmp_path):
    def prepare(samples):
        samples[0].rollout_id = samples[1].index
        samples[1].rollout_id = None
        samples[1].reward = samples[0].reward

    dump_segmented_run(tmp_path, lengths=(1, 1, 1, 1), prepare=prepare)
    reader = DumpReader(tmp_path)
    assert reader.episodes(0).height == 4
    assert reader.groups(0)["n"].to_list() == [2, 2]


def test_segment_lanes_and_clicks_resolve_to_the_source_sample(tmp_path):
    def prepare(samples):
        for output_index, sample in enumerate(samples[:3]):
            sample.lineage = SampleLineage(source_sample_index=88, output_index=output_index, output_count=3)

    dump_segmented_run(tmp_path, prepare=prepare)
    store = MetricStore(tmp_path / "dashboard")
    for ts, kind in [(10.0, TrajectoryEventKind.ATTEMPT_START), (11.0, TrajectoryEventKind.ATTEMPT_END)]:
        store.append(TrajectoryEvent(ts, kind, 88, 0, -1, "", ""))
    store.flush()
    reader = DumpReader(tmp_path)
    client = TestClient(make_app(store, reader))
    for query in ("", "?sample_index=1"):
        lanes = client.get(f"/api/rollout/0/trajectories{query}").json()["lanes"]
        assert [lane["sample_index"] for lane in lanes] == [88]
    assert reader.summary(0)["lifecycle_index"].to_list()[:4] == [88, 88, 88, 3]
