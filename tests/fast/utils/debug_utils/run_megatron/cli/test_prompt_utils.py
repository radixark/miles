from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.debug_utils.run_megatron.cli.prompt_utils import (
    PromptConfig,
    _build_math_sequence,
    _resolve_raw_text,
    generate_token_ids,
    write_token_ids_to_tmpfile,
)


class TestPromptConfig:
    def test_frozen(self) -> None:
        config = PromptConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.mode = "text"  # type: ignore[misc]

    def test_defaults(self) -> None:
        config = PromptConfig()
        assert config.mode == "math"
        assert config.text is None
        assert config.file is None
        assert config.seq_length == 137
        assert config.apply_chat_template is False


class TestResolveRawText:
    def test_math_mode(self) -> None:
        text = _resolve_raw_text(PromptConfig(mode="math", seq_length=10))
        assert "1+1=2" in text

    def test_text_mode(self) -> None:
        text = _resolve_raw_text(PromptConfig(mode="text", text="hello world"))
        assert text == "hello world"

    def test_text_mode_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="--prompt-text is required"):
            _resolve_raw_text(PromptConfig(mode="text"))

    def test_file_mode(self, tmp_path: Path) -> None:
        f = tmp_path / "prompt.txt"
        f.write_text("content from file")
        text = _resolve_raw_text(PromptConfig(mode="file", file=f))
        assert text == "content from file"

    def test_file_mode_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="--prompt-file is required"):
            _resolve_raw_text(PromptConfig(mode="file"))

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown prompt mode"):
            _resolve_raw_text(PromptConfig(mode="unknown"))  # type: ignore[arg-type]


class TestBuildMathSequence:
    def test_starts_with_1_plus_1(self) -> None:
        seq = _build_math_sequence(target_char_length=100)
        assert seq.startswith("1+1=2")

    def test_reaches_target_length(self) -> None:
        target = 500
        seq = _build_math_sequence(target_char_length=target)
        assert len(seq) >= target

    def test_small_target(self) -> None:
        seq = _build_math_sequence(target_char_length=1)
        assert len(seq) > 0
        assert "1+1=2" in seq

    def test_b_wraps_after_100(self) -> None:
        seq = _build_math_sequence(target_char_length=10000)
        assert "2+1=" in seq


class TestWriteTokenIdsToTmpfile:
    def test_roundtrip_json(self) -> None:
        token_ids = [10, 20, 30, 40]
        path = write_token_ids_to_tmpfile(token_ids)
        loaded = json.loads(path.read_text())
        assert loaded == token_ids

    def test_prefix_and_suffix(self) -> None:
        path = write_token_ids_to_tmpfile([1, 2, 3])
        assert path.name.startswith("run_megatron_token_ids_")
        assert path.name.endswith(".json")


class TestGenerateTokenIds:
    def _make_mock_tokenizer(self, encoded_ids: list[int]) -> MagicMock:
        mock_tok = MagicMock()
        mock_tok.encode.return_value = encoded_ids
        mock_tok.apply_chat_template.return_value = "templated text"
        return mock_tok

    @patch("transformers.AutoTokenizer")
    def test_correct_length(self, mock_auto_tok: MagicMock) -> None:
        ids = list(range(200))
        mock_auto_tok.from_pretrained.return_value = self._make_mock_tokenizer(ids)

        result = generate_token_ids(
            prompt=PromptConfig(mode="math", seq_length=50),
            tokenizer_path=Path("/fake/tokenizer"),
        )
        assert len(result) == 50

    @patch("transformers.AutoTokenizer")
    def test_chat_template_called(self, mock_auto_tok: MagicMock) -> None:
        mock_tok = self._make_mock_tokenizer(list(range(200)))
        mock_auto_tok.from_pretrained.return_value = mock_tok

        generate_token_ids(
            prompt=PromptConfig(mode="text", text="hello", seq_length=50, apply_chat_template=True),
            tokenizer_path=Path("/fake/tokenizer"),
        )
        mock_tok.apply_chat_template.assert_called_once()

    @patch("transformers.AutoTokenizer")
    def test_math_mode_deterministic(self, mock_auto_tok: MagicMock) -> None:
        ids = list(range(200))
        mock_auto_tok.from_pretrained.return_value = self._make_mock_tokenizer(ids)

        prompt = PromptConfig(mode="math", seq_length=50)
        r1 = generate_token_ids(prompt=prompt, tokenizer_path=Path("/fake"))
        r2 = generate_token_ids(prompt=prompt, tokenizer_path=Path("/fake"))
        assert r1 == r2
