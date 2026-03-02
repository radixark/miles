from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from miles.utils.debug_utils.run_megatron.worker.top_k_print import (
    _decode_token,
    _get_dist_info,
    _print_top_predictions_for_rank,
)


class TestDecodeToken:
    def test_with_tokenizer(self) -> None:
        mock_tok = MagicMock()
        mock_tok.decode.return_value = "hello"
        result = _decode_token(mock_tok, token_id=42)
        assert result == "hello"
        mock_tok.decode.assert_called_once_with([42])

    def test_without_tokenizer(self) -> None:
        result = _decode_token(None, token_id=42)
        assert result == "t42"


class TestGetDistInfo:
    @patch("miles.utils.debug_utils.run_megatron.worker.top_k_print.dist")
    def test_not_initialized(self, mock_dist: MagicMock) -> None:
        mock_dist.is_initialized.return_value = False
        rank, world_size = _get_dist_info()
        assert rank == 0
        assert world_size == 1

    @patch("miles.utils.debug_utils.run_megatron.worker.top_k_print.dist")
    def test_initialized(self, mock_dist: MagicMock) -> None:
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 2
        mock_dist.get_world_size.return_value = 8
        rank, world_size = _get_dist_info()
        assert rank == 2
        assert world_size == 8


class TestPrintTopPredictionsForRank:
    def test_smoke_single_position(self, capsys: object) -> None:
        """Basic smoke test: function runs without error."""
        logits = torch.randn(1, 3, 10)  # batch=1, seq=3, vocab=10
        input_ids = torch.tensor([[1, 2, 3]])
        mock_tok = MagicMock()
        mock_tok.decode.return_value = "x"

        _print_top_predictions_for_rank(
            logits=logits,
            input_ids=input_ids,
            top_k=3,
            tokenizer=mock_tok,
            rank=0,
        )

    def test_pad_token_skipped(self, capsys: object) -> None:
        logits = torch.randn(1, 4, 10)
        input_ids = torch.tensor([[1, 99, 2, 99]])  # 99 is pad
        mock_tok = MagicMock()
        mock_tok.decode.return_value = "x"

        _print_top_predictions_for_rank(
            logits=logits,
            input_ids=input_ids,
            top_k=2,
            tokenizer=mock_tok,
            rank=0,
            pad_token_id=99,
        )
        # decode called for non-pad positions only:
        # 2 non-pad positions × (1 input token + 2 top-k tokens) = 6 calls
        assert mock_tok.decode.call_count == 6

    def test_batch_size_gt1(self, capsys: object) -> None:
        logits = torch.randn(2, 2, 10)
        input_ids = torch.tensor([[1, 2], [3, 4]])
        mock_tok = MagicMock()
        mock_tok.decode.return_value = "x"

        _print_top_predictions_for_rank(
            logits=logits,
            input_ids=input_ids,
            top_k=2,
            tokenizer=mock_tok,
            rank=0,
        )
        # 2 batches × 2 positions × (1 input + 2 topk) = 12 decode calls
        assert mock_tok.decode.call_count == 12
