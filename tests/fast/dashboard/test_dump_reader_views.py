import os
import statistics
import time

import pytest
from tests.fast.dashboard.dummy_dump import dump_dummy_run

from miles.dashboard.dump_reader import DumpReader

REMOVED = (3,)  # within-step positions marked remove_sample=True by the fixture


@pytest.fixture
def reader(tmp_path):
    dump_dummy_run(tmp_path, steps=2, dp_size=2, tp_dup=2, with_eval=True, remove_sample_indices=REMOVED)
    return DumpReader(tmp_path)


def _df_row(df, sample_index):
    matches = df.filter(df["sample_index"] == sample_index)
    assert matches.height == 1
    return matches.row(0, named=True)


def test_summary_matches_hand_computed(reader):
    joined = reader.load_joined(0)
    df = reader.summary(0)
    assert df.height == len(joined.samples)

    for sample in joined.samples:
        entry = _df_row(df, sample.index)
        row = joined.train_rows[sample.index]
        assert entry["group_index"] == sample.group_index
        assert entry["response_length"] == sample.response_length
        assert entry["raw_reward"] == sample.reward
        assert entry["normalized_reward"] == pytest.approx(row.reward)
        assert entry["dumped_rank"] == row.rank

        mask = row.loss_mask > 0
        if mask.any():
            expected = (row.log_probs - row.rollout_log_probs).abs()[mask]
            assert entry["mean_abs_lp_diff"] == pytest.approx(float(expected.mean()), rel=1e-5)
            assert entry["max_abs_lp_diff"] == pytest.approx(float(expected.max()), rel=1e-5)
            assert entry["mean_imp_ratio"] == pytest.approx(
                float((row.log_probs - row.rollout_log_probs).exp()[mask].mean()), rel=1e-5
            )
            assert entry["mean_entropy"] == pytest.approx(float(row.entropy[mask].mean()), rel=1e-5)


def test_summary_removed_sample_has_no_masked_stats(reader):
    df = reader.summary(0)
    entry = _df_row(df, REMOVED[0])  # step 0: Sample.index == within-step position
    assert entry["remove_sample"] is True
    for column in ("mean_entropy", "mean_abs_lp_diff", "mean_imp_ratio", "adv_mean", "return_mean"):
        assert entry[column] is None, column


def test_summary_parquet_cache_hit_and_invalidation(reader, monkeypatch):
    df = reader.summary(0)
    assert (reader.cache_dir / "rollout_0.parquet").exists()

    def boom(self, path):
        raise RuntimeError("recompute attempted")

    monkeypatch.setattr(DumpReader, "_torch_load", boom)
    reader._joined_cache.clear()
    assert reader.summary(0).equals(df)  # served from parquet, no dump loads

    stamp = time.time() + 5
    os.utime(next(reader.train_dir.glob("0_*.pt")), (stamp, stamp))
    with pytest.raises(RuntimeError, match="recompute attempted"):
        reader.summary(0)  # source changed -> cache invalidated -> recompute


def test_groups(reader):
    joined = reader.load_joined(0)
    groups_df = reader.groups(0)
    rewards_by_group = {}
    for sample in joined.samples:
        rewards_by_group.setdefault(sample.group_index, []).append(sample.reward)
    assert groups_df.height == len(rewards_by_group)

    for entry in groups_df.iter_rows(named=True):
        rewards = rewards_by_group[entry["group_index"]]
        assert entry["n"] == len(rewards)
        assert entry["reward_mean"] == pytest.approx(statistics.mean(rewards))
        assert entry["reward_std"] == pytest.approx(statistics.stdev(rewards))
        assert entry["zero_std"] == (statistics.stdev(rewards) <= 1e-12)


def test_step_aggregates(reader):
    aggregates = reader.step_aggregates()
    assert aggregates["rollout_id"].to_list() == [0, 1]
    for entry in aggregates.iter_rows(named=True):
        df = reader.summary(entry["rollout_id"])
        assert entry["n_samples"] == df.height
        assert entry["reward_mean"] == pytest.approx(df["raw_reward"].mean())
        assert entry["mean_abs_lp_diff"] is not None


def test_tokens_full_range(reader):
    joined = reader.load_joined(0)
    sample = joined.samples[0]
    payload = reader.tokens(0, sample.index)
    prompt_len = len(sample.tokens) - sample.response_length

    assert payload["total_len"] == len(sample.tokens)
    assert payload["prompt_len"] == payload["response_offset"] == prompt_len
    assert payload["token_ids"] == sample.tokens
    assert len(payload["train_log_probs"]) == sample.response_length
    assert len(payload["imp_ratio"]) == sample.response_length
    assert payload["rollout_log_probs"] == pytest.approx(sample.rollout_log_probs)
    row = joined.train_rows[sample.index]
    assert payload["lp_diff"][0] == pytest.approx(float(row.log_probs[0] - row.rollout_log_probs[0]))


def test_tokens_window_straddles_response_boundary(reader):
    sample = reader.load_joined(0).samples[0]
    prompt_len = len(sample.tokens) - sample.response_length

    payload = reader.tokens(0, sample.index, start=prompt_len - 2, end=prompt_len + 3)
    assert len(payload["token_ids"]) == 5
    assert payload["response_offset"] == 2
    assert len(payload["train_log_probs"]) == 3  # stats only cover the response overlap

    prompt_only = reader.tokens(0, sample.index, start=0, end=2)
    assert prompt_only["train_log_probs"] == []
    assert prompt_only["response_offset"] == 2

    clamped = reader.tokens(0, sample.index, start=0, end=10**6)
    assert clamped["end"] == len(sample.tokens)

    with pytest.raises(ValueError, match="empty token range"):
        reader.tokens(0, sample.index, start=5, end=5)

    with pytest.raises(KeyError, match="unknown sample_index"):
        reader.tokens(0, 10**6)


def test_tokens_decode_via_dumped_tokenizer(reader):
    payload = reader.tokens(0, 0)
    assert payload["token_text"] == [f"t{tid}" for tid in payload["token_ids"]]


def test_tokens_eval_sample_has_rollout_side_only(reader):
    payload = reader.tokens(0, 0, evaluation=True)
    assert payload["rollout_log_probs"] is not None
    for key in ("train_log_probs", "lp_diff", "imp_ratio", "entropy", "advantages", "loss_mask"):
        assert payload[key] is None, key


def test_tokens_without_entropy_or_tokenizer(tmp_path):
    dump_dummy_run(tmp_path, steps=1, with_entropy=False, with_eval=False, with_tokenizer=False)
    reader = DumpReader(tmp_path)
    payload = reader.tokens(0, 0)
    assert payload["entropy"] is None and payload["ref_entropy"] is None
    assert payload["train_log_probs"] is not None
    assert payload["token_text"] is None
    assert _df_row(reader.summary(0), 0)["mean_entropy"] is None


def test_joined_lru_eviction(tmp_path):
    dump_dummy_run(tmp_path, steps=2, with_eval=False)
    reader = DumpReader(tmp_path, tensor_lru=1)
    first = reader.joined(0)
    assert reader.joined(0) is first  # cache hit
    reader.joined(1)
    assert list(reader._joined_cache) == [(1, False)]  # 0 evicted


@pytest.mark.skipif(
    "MILES_DASHBOARD_REALDATA_DIR" not in os.environ,
    reason="set MILES_DASHBOARD_REALDATA_DIR to a real --dump-details dir",
)
def test_realdata_views(tmp_path):
    reader = DumpReader(os.environ["MILES_DASHBOARD_REALDATA_DIR"], cache_dir=tmp_path)
    df = reader.summary(0)
    assert df.height == 256
    assert df["truncated"].cast(int).sum() == 112  # measured on qwen30b-dash step 0
    assert df["mean_abs_lp_diff"].null_count() == 0
    assert df["mean_entropy"].null_count() == 0

    groups_df = reader.groups(0)
    assert groups_df.height == 32  # 256 samples / 8 per prompt

    sample_index = int(df["sample_index"][0])
    payload = reader.tokens(0, sample_index, start=0, end=64)
    assert len(payload["token_ids"]) == 64
    assert payload["token_text"] is not None

    aggregates = reader.step_aggregates()
    assert aggregates.height == 5
