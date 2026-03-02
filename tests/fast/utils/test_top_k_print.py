from unittest.mock import MagicMock, patch

from miles.utils.debug_utils.run_megatron.worker.top_k_print import (
    _decode_token,
    _get_dist_info,
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
