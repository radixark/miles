from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from miles.utils.debug_utils.run_megatron.worker.output import (
    _compute_logprob_entries,
    _compute_output_info,
    compute_and_save_output_info,
)


# ---------------------------------------------------------------------------
# TestComputeLogprobEntries
# ---------------------------------------------------------------------------


class TestComputeLogprobEntries:
    def test_returns_none_for_2d_logits(self) -> None:
        logits = torch.randn(2, 4)
        labels = torch.tensor([[1, 2, 3, -100]])
        position_ids = torch.arange(4).unsqueeze(0)
        assert _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids) is None

    def test_returns_none_for_scalar_vocab(self) -> None:
        logits = torch.randn(1, 4, 1)
        labels = torch.tensor([[0, 0, 0, 0]])
        position_ids = torch.arange(4).unsqueeze(0)
        assert _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids) is None

    def test_basic_shape(self) -> None:
        batch_size, seq_len, vocab_size = 2, 3, 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[1, 2, -100], [3, -100, 5]])
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        result = _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids)
        assert result is not None
        assert len(result) == batch_size
        for batch_entries in result:
            assert len(batch_entries) == seq_len

    def test_entry_keys(self) -> None:
        logits = torch.randn(1, 2, 5)
        labels = torch.tensor([[1, -100]])
        position_ids = torch.tensor([[0, 1]])

        result = _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids)
        assert result is not None
        expected_keys = {"global_position", "token_id", "logprob", "is_valid"}
        for entry in result[0]:
            assert set(entry.keys()) == expected_keys

    def test_valid_entry_values(self) -> None:
        vocab_size = 5
        logits = torch.zeros(1, 1, vocab_size)
        logits[0, 0, 2] = 10.0
        labels = torch.tensor([[2]])
        position_ids = torch.tensor([[7]])

        result = _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids)
        assert result is not None
        entry = result[0][0]
        assert entry["global_position"] == 7
        assert entry["token_id"] == 2
        assert entry["is_valid"] is True
        assert entry["logprob"] < 0  # log_softmax is always negative

    def test_ignored_entry_values(self) -> None:
        logits = torch.randn(1, 1, 5)
        labels = torch.tensor([[-100]])
        position_ids = torch.tensor([[3]])

        result = _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids)
        assert result is not None
        entry = result[0][0]
        assert entry["global_position"] == 3
        assert entry["token_id"] == -1
        assert entry["is_valid"] is False
        assert entry["logprob"] == 0.0

    def test_logprob_sums_to_one(self) -> None:
        """exp(logprob) for each valid token should be a valid probability."""
        logits = torch.randn(1, 4, 10)
        labels = torch.tensor([[0, 3, 7, 9]])
        position_ids = torch.arange(4).unsqueeze(0)

        result = _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids)
        assert result is not None
        for entry in result[0]:
            prob = torch.exp(torch.tensor(entry["logprob"]))
            assert 0.0 < prob.item() <= 1.0

    def test_perfect_prediction_high_logprob(self) -> None:
        vocab_size = 5
        logits = torch.full((1, 1, vocab_size), -100.0)
        logits[0, 0, 3] = 100.0
        labels = torch.tensor([[3]])
        position_ids = torch.tensor([[0]])

        result = _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids)
        assert result is not None
        assert result[0][0]["logprob"] > -0.01

    def test_position_ids_preserved(self) -> None:
        logits = torch.randn(1, 3, 5)
        labels = torch.tensor([[1, 2, -100]])
        position_ids = torch.tensor([[10, 20, 30]])

        result = _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids)
        assert result is not None
        assert [e["global_position"] for e in result[0]] == [10, 20, 30]


# ---------------------------------------------------------------------------
# TestComputeOutputInfo
# ---------------------------------------------------------------------------


_MOCK_MPU = "miles.utils.debug_utils.run_megatron.worker.output.mpu"
_MOCK_DIST = "miles.utils.debug_utils.run_megatron.worker.output.dist"


def _patch_distributed(
    tp_size: int = 1,
    cp_size: int = 1,
    pp_size: int = 1,
) -> Any:
    """Context manager that mocks dist.is_initialized and mpu parallel sizes."""
    def _decorator(func: Any) -> Any:
        @patch(f"{_MOCK_DIST}.is_initialized", return_value=True)
        @patch(f"{_MOCK_MPU}.get_tensor_model_parallel_world_size", return_value=tp_size)
        @patch(f"{_MOCK_MPU}.get_context_parallel_world_size", return_value=cp_size)
        @patch(f"{_MOCK_MPU}.get_pipeline_model_parallel_world_size", return_value=pp_size)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        return wrapper
    return _decorator


class TestComputeOutputInfo:
    def test_returns_none_for_invalid_logits(self) -> None:
        logits = torch.randn(1, 4, 1)
        labels = torch.tensor([[0, 0, 0, 0]])
        position_ids = torch.arange(4).unsqueeze(0)

        result = _compute_output_info(logits=logits, labels=labels, position_ids=position_ids, rank=0)
        assert result is None

    @_patch_distributed(tp_size=2, cp_size=4, pp_size=3)
    def test_payload_structure(self, *_mocks: Any) -> None:
        logits = torch.randn(1, 3, 10)
        labels = torch.tensor([[1, 2, -100]])
        position_ids = torch.arange(3).unsqueeze(0)

        result = _compute_output_info(logits=logits, labels=labels, position_ids=position_ids, rank=5)
        assert result is not None
        assert result["rank"] == 5
        assert result["tp_size"] == 2
        assert result["cp_size"] == 4
        assert result["pp_size"] == 3
        assert "logprob_entries" in result
        assert len(result["logprob_entries"]) == 1
        assert len(result["logprob_entries"][0]) == 3

    @_patch_distributed()
    def test_defaults_to_size_1(self, *_mocks: Any) -> None:
        logits = torch.randn(1, 2, 5)
        labels = torch.tensor([[1, 2]])
        position_ids = torch.arange(2).unsqueeze(0)

        result = _compute_output_info(logits=logits, labels=labels, position_ids=position_ids, rank=0)
        assert result is not None
        assert result["tp_size"] == 1
        assert result["cp_size"] == 1
        assert result["pp_size"] == 1


# ---------------------------------------------------------------------------
# TestComputeAndSaveOutputInfo
# ---------------------------------------------------------------------------


class TestComputeAndSaveOutputInfo:
    @patch(f"{_MOCK_DIST}.is_initialized", return_value=False)
    def test_writes_json_file(self, _mock: Any, tmp_path: Path) -> None:
        logits = torch.randn(1, 3, 10)
        labels = torch.tensor([[1, 2, -100]])
        position_ids = torch.arange(3).unsqueeze(0)

        compute_and_save_output_info(
            logits=logits,
            labels=labels,
            position_ids=position_ids,
            output_dir=tmp_path,
        )

        output_file = tmp_path / "rank_0.json"
        assert output_file.exists()

        payload = json.loads(output_file.read_text())
        assert payload["rank"] == 0
        assert "logprob_entries" in payload
        assert len(payload["logprob_entries"]) == 1
        assert len(payload["logprob_entries"][0]) == 3

    @patch(f"{_MOCK_DIST}.is_initialized", return_value=False)
    def test_skips_invalid_logits(self, _mock: Any, tmp_path: Path) -> None:
        logits = torch.randn(1, 4, 1)
        labels = torch.tensor([[0, 0, 0, 0]])
        position_ids = torch.arange(4).unsqueeze(0)

        compute_and_save_output_info(
            logits=logits,
            labels=labels,
            position_ids=position_ids,
            output_dir=tmp_path,
        )

        assert not (tmp_path / "rank_0.json").exists()

    @patch(f"{_MOCK_DIST}.is_initialized", return_value=False)
    def test_creates_output_dir(self, _mock: Any, tmp_path: Path) -> None:
        output_dir = tmp_path / "nested" / "dir"
        logits = torch.randn(1, 2, 5)
        labels = torch.tensor([[1, 2]])
        position_ids = torch.arange(2).unsqueeze(0)

        compute_and_save_output_info(
            logits=logits,
            labels=labels,
            position_ids=position_ids,
            output_dir=output_dir,
        )

        assert output_dir.exists()
        assert (output_dir / "rank_0.json").exists()

    @patch(f"{_MOCK_DIST}.is_initialized", return_value=False)
    def test_json_is_valid(self, _mock: Any, tmp_path: Path) -> None:
        logits = torch.randn(2, 4, 8)
        labels = torch.tensor([[1, 2, -100, 3], [4, -100, 5, 6]])
        position_ids = torch.arange(4).unsqueeze(0).expand(2, -1)

        compute_and_save_output_info(
            logits=logits,
            labels=labels,
            position_ids=position_ids,
            output_dir=tmp_path,
        )

        payload = json.loads((tmp_path / "rank_0.json").read_text())
        assert len(payload["logprob_entries"]) == 2
        assert len(payload["logprob_entries"][0]) == 4
        assert len(payload["logprob_entries"][1]) == 4

        for batch_entries in payload["logprob_entries"]:
            for entry in batch_entries:
                assert set(entry.keys()) == {"global_position", "token_id", "logprob", "is_valid"}
