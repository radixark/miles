import json
from types import SimpleNamespace

import pytest

from miles.rollout.packed_sft_rollout import PackedSFTDataSource, generate_rollout


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _args(path, **overrides):
    args = SimpleNamespace(
        prompt_data=str(path),
        rollout_seed=123,
        rollout_shuffle=False,
        n_samples_per_prompt=1,
        metadata_key="metadata",
        rollout_max_context_len=16,
        packed_sft_tokens_key="tokens",
        packed_sft_loss_mask_key="loss_mask",
        packed_sft_response_length_key="response_length",
        packed_sft_allow_empty_loss=False,
        rollout_batch_size=2,
        save=str(path.parent / "ckpt"),
        load=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_full_length_sparse_loss_mask_uses_whole_block_response_window(tmp_path):
    data = tmp_path / "packed.jsonl"
    _write_jsonl(data, [{"tokens": [10, 11, 12, 13, 14], "loss_mask": [0, 1, 0, 1, 0]}])

    source = PackedSFTDataSource(_args(data, rollout_batch_size=1))
    sample = source.get_samples(1)[0][0]

    assert sample.tokens == [10, 11, 12, 13, 14]
    assert sample.response_length == 4
    assert sample.loss_mask == [1, 0, 1, 0]
    assert sample.reward == 0
    assert sample.metadata["packed_sft_total_length"] == 5
    assert sample.metadata["packed_sft_loss_tokens"] == 2


def test_explicit_response_length_accepts_tail_loss_mask(tmp_path):
    data = tmp_path / "packed.jsonl"
    _write_jsonl(data, [{"tokens": [10, 11, 12, 13, 14], "response_length": 2, "loss_mask": [0, 1]}])

    source = PackedSFTDataSource(_args(data, rollout_batch_size=1))
    sample = source.get_samples(1)[0][0]

    assert sample.response_length == 2
    assert sample.loss_mask == [0, 1]


def test_explicit_response_length_accepts_full_loss_mask_and_crops_tail(tmp_path):
    data = tmp_path / "packed.jsonl"
    _write_jsonl(data, [{"tokens": [10, 11, 12, 13, 14], "response_length": 2, "loss_mask": [0, 0, 0, 1, 1]}])

    source = PackedSFTDataSource(_args(data, rollout_batch_size=1))
    sample = source.get_samples(1)[0][0]

    assert sample.response_length == 2
    assert sample.loss_mask == [1, 1]


def test_explicit_whole_block_response_length_is_rejected(tmp_path):
    data = tmp_path / "packed.jsonl"
    _write_jsonl(data, [{"tokens": [10, 11, 12, 13], "response_length": 4, "loss_mask": [0, 1, 0, 1]}])

    with pytest.raises(ValueError, match="equals tokens length"):
        PackedSFTDataSource(_args(data, rollout_batch_size=1))


def test_rejects_overlength_block(tmp_path):
    data = tmp_path / "packed.jsonl"
    _write_jsonl(data, [{"tokens": list(range(17)), "loss_mask": [0] * 16 + [1]}])

    with pytest.raises(ValueError, match="exceeds max_length=16"):
        PackedSFTDataSource(_args(data))


def test_rejects_empty_loss_by_default(tmp_path):
    data = tmp_path / "packed.jsonl"
    _write_jsonl(data, [{"tokens": [1, 2, 3], "loss_mask": [0, 0, 0]}])

    with pytest.raises(ValueError, match="contains no trainable token"):
        PackedSFTDataSource(_args(data))


def test_rejects_loss_starting_at_first_token(tmp_path):
    data = tmp_path / "packed.jsonl"
    _write_jsonl(data, [{"tokens": [1, 2, 3], "loss_mask": [1, 1, 1]}])

    with pytest.raises(ValueError, match="first trainable token is at position 0"):
        PackedSFTDataSource(_args(data))


def test_rejects_explicit_response_window_with_prefix_loss(tmp_path):
    data = tmp_path / "packed.jsonl"
    _write_jsonl(data, [{"tokens": [1, 2, 3, 4], "response_length": 2, "loss_mask": [0, 1, 1, 1]}])

    with pytest.raises(ValueError, match="outside the explicit"):
        PackedSFTDataSource(_args(data))


def test_generate_rollout_returns_groups_and_metrics(tmp_path):
    data = tmp_path / "packed.jsonl"
    _write_jsonl(
        data,
        [
            {"tokens": [1, 2, 3], "loss_mask": [0, 1, 1]},
            {"tokens": [4, 5], "loss_mask": [0, 1]},
        ],
    )

    args = _args(data, rollout_batch_size=2)
    source = PackedSFTDataSource(args)
    output = generate_rollout(args, rollout_id=0, data_buffer=source, evaluation=False)

    samples = output.samples if hasattr(output, "samples") else output
    assert len(samples) == 2
    assert samples[0][0].tokens == [1, 2, 3]
    if hasattr(output, "metrics"):
        assert output.metrics["packed_sft/total_tokens"] == 5
        assert output.metrics["packed_sft/loss_tokens"] == 3
