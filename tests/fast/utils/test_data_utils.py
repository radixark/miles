"""Tests for witness_ids handling in the data pipeline (get_batch, split_train_data_by_dp)."""

from unittest.mock import MagicMock

import torch

from miles.utils.data_utils import split_train_data_by_dp


class TestSplitTrainDataIncludesWitnessIds:
    def test_witness_ids_split_across_dp(self) -> None:
        tokens = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10, 11]]
        witness_ids = [
            torch.tensor([0, 0, 0]),
            torch.tensor([1, 1]),
            torch.tensor([2, 2, 2, 2]),
            torch.tensor([3, 3]),
        ]

        data = {
            "tokens": tokens,
            "seq_witness_ids": witness_ids,
            "response_lengths": [1, 1, 1, 1],
            "loss_masks": [[0, 0, 1], [0, 1], [0, 0, 0, 1], [0, 1]],
        }

        args = MagicMock()
        args.balance_data = False

        result = split_train_data_by_dp(args, data, dp_size=2)

        assert len(result) == 2
        assert "seq_witness_ids" in result[0]
        assert "seq_witness_ids" in result[1]
        assert len(result[0]["seq_witness_ids"]) == 2
        assert len(result[1]["seq_witness_ids"]) == 2

    def test_no_witness_ids_when_absent(self) -> None:
        tokens = [[1, 2], [3, 4]]
        data = {
            "tokens": tokens,
            "response_lengths": [1, 1],
            "loss_masks": [[0, 1], [0, 1]],
        }

        args = MagicMock()
        args.balance_data = False

        result = split_train_data_by_dp(args, data, dp_size=1)
        assert "seq_witness_ids" not in result[0]
