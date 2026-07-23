import os
import time
from collections import defaultdict

import pytest
import torch
from tests.fast.dashboard.dummy_dump import dump_dummy_run

from miles.dashboard.dump_reader import DumpReader, DumpStillWriting, TrainRow
from miles.utils.types import Sample


@pytest.fixture
def run(tmp_path):
    truth = dump_dummy_run(tmp_path, steps=3, dp_size=2, tp_dup=2, with_eval=True)
    return DumpReader(tmp_path), truth


def test_rollout_ids_split_and_sorted(run):
    reader, truth = run
    ids = reader.rollout_ids()
    assert ids.train == [0, 1, 2]
    assert ids.eval == truth.eval_ids == [0]


def test_fresh_rollout_hidden_until_train_companion_exists(run):
    reader, _ = run
    fresh = reader.rollout_dir / "3.pt"
    torch.save(dict(rollout_id=3, samples=[]), fresh)  # mtime = now
    assert 3 not in reader.rollout_ids().train

    (reader.train_dir / "3_0.pt").touch()  # train companion written strictly later
    assert 3 in reader.rollout_ids().train


def test_fresh_eval_hidden_until_aged(run):
    reader, _ = run
    fresh = reader.rollout_dir / "eval_2.pt"
    torch.save(dict(rollout_id=2, samples=[]), fresh)
    assert 2 not in reader.rollout_ids().eval

    stamp = time.time() - 100
    os.utime(fresh, (stamp, stamp))
    assert 2 in reader.rollout_ids().eval


def test_load_joined_full_coverage(run):
    reader, truth = run
    for rollout_id in reader.rollout_ids().train:
        joined = reader.load_joined(rollout_id)
        assert len(joined.samples) == truth.n_samples_per_step
        assert joined.train_coverage == 1.0
        assert set(joined.train_rows) == {s.index for s in joined.samples}
        assert all(isinstance(s, Sample) for s in joined.samples)
        assert all(isinstance(r, TrainRow) for r in joined.train_rows.values())
    # Sample deserialization went through miles' own from_dict: enums restored.
    statuses = {s.status for s in reader.load_joined(0).samples}
    assert statuses <= {Sample.Status.COMPLETED, Sample.Status.TRUNCATED}


def test_join_row_matches_sample(run):
    reader, _ = run
    joined = reader.load_joined(1)
    sample_of = {s.index: s for s in joined.samples}
    for index, row in joined.train_rows.items():
        sample = sample_of[index]
        assert row.response_length == sample.response_length
        assert row.total_length == len(sample.tokens)
        assert torch.equal(row.tokens, torch.tensor(sample.tokens))
        assert torch.allclose(row.rollout_log_probs, torch.tensor(sample.rollout_log_probs))
        assert len(row.log_probs) == len(row.advantages) == len(row.loss_mask) == sample.response_length


def test_raw_reward_uses_batch_global_indexing(run):
    # Regression test: raw_reward is the one batch-global column
    # (split_train_data_by_dp ships it unpartitioned, "splited at train side");
    # with shuffled balanced partitions, indexing it by shard row silently
    # misattributes rewards across samples.
    reader, _ = run
    for rollout_id in reader.rollout_ids().train:
        joined = reader.load_joined(rollout_id)
        for sample in joined.samples:
            assert joined.train_rows[sample.index].raw_reward == sample.reward


def test_rewards_went_through_real_group_normalization(run):
    # The dummy pipeline runs the real GRPO reward post-processing: per-group
    # mean-centering makes shard-local `rewards` sum to ~0 within each group.
    reader, _ = run
    joined = reader.load_joined(0)
    group_of = {s.index: s.group_index for s in joined.samples}
    group_rewards = defaultdict(list)
    for index, row in joined.train_rows.items():
        group_rewards[group_of[index]].append(row.reward)
    for rewards in group_rewards.values():
        assert abs(sum(rewards)) < 1e-4


def test_tp_duplicates_keep_first_rank(run):
    reader, truth = run
    joined = reader.load_joined(0)
    # dp_size=2, tp_dup=2: shard 0 lives on ranks {0,1}, shard 1 on ranks {2,3};
    # dedup must keep the lowest rank of each shard.
    assert {row.rank for row in joined.train_rows.values()} == {0, 2}
    for shard_idx, indices in enumerate(truth.shard_indices[0]):
        for index in indices:
            assert joined.train_rows[index].rank == shard_idx * 2


def test_train_index_absent_from_rollout_asserts(run):
    reader, _ = run
    path = reader.train_dir / "0_0.pt"
    pack = torch.load(path, weights_only=False)
    pack["rollout_data"]["sample_indices"][0] = 9999
    torch.save(pack, path)

    with pytest.raises(AssertionError, match="9999 absent"):
        reader.load_joined(0)


def test_inconsistent_tp_duplicate_asserts(run):
    reader, _ = run
    path = reader.train_dir / "0_1.pt"  # TP duplicate of rank 0
    pack = torch.load(path, weights_only=False)
    pack["rollout_data"]["response_lengths"][0] += 1
    torch.save(pack, path)

    with pytest.raises(AssertionError, match="disagrees"):
        reader.load_joined(0)


def test_eval_join_has_no_train_rows(run):
    reader, truth = run
    joined = reader.load_joined(0, evaluation=True)
    assert joined.evaluation
    assert joined.train_rows == {}
    assert len(joined.samples) == truth.n_samples_per_step // 2


def test_still_writing_vs_corruption(run):
    reader, _ = run
    path = reader.rollout_dir / "9.pt"
    path.write_bytes(b"garbage")  # fresh mtime: a torch.save in progress
    assert 9 not in reader.rollout_ids().train
    with pytest.raises(DumpStillWriting):
        reader.load_joined(9)

    stamp = time.time() - 100
    os.utime(path, (stamp, stamp))  # old + unloadable = real corruption
    with pytest.raises(Exception) as exc_info:
        reader.load_joined(9)
    assert not isinstance(exc_info.value, DumpStillWriting)


def test_missing_rollout_file_raises(run):
    reader, _ = run
    with pytest.raises(FileNotFoundError):
        reader.load_joined(42)


def test_optional_columns_absent(tmp_path):
    dump_dummy_run(tmp_path, steps=1, with_entropy=False, with_eval=False)
    joined = DumpReader(tmp_path).load_joined(0)
    for row in joined.train_rows.values():
        assert row.entropy is None and row.ref_entropy is None
        assert row.advantages is not None  # unaffected columns still load


def test_empty_dump_dir(tmp_path):
    ids = DumpReader(tmp_path).rollout_ids()
    assert ids.train == [] and ids.eval == []


@pytest.mark.skipif(
    "MILES_DASHBOARD_REALDATA_DIR" not in os.environ,
    reason="set MILES_DASHBOARD_REALDATA_DIR to a real --dump-details dir",
)
def test_realdata_join():
    reader = DumpReader(os.environ["MILES_DASHBOARD_REALDATA_DIR"])
    ids = reader.rollout_ids()
    assert ids.train, "no rollout dumps found"
    for rollout_id in ids.train:
        joined = reader.load_joined(rollout_id)
        assert joined.train_coverage == 1.0
        row = next(iter(joined.train_rows.values()))
        assert len(row.log_probs) == row.response_length
        # raw_reward correspondence must hold on real data too
        sample_of = {s.index: s for s in joined.samples}
        for index, train_row in joined.train_rows.items():
            assert train_row.raw_reward == sample_of[index].reward
