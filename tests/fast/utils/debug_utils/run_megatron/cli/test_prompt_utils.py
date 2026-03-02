from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from miles.utils.debug_utils.run_megatron.cli.prompt_utils import (
    PromptConfig,
    _build_math_sequence,
    _resolve_raw_text,
    generate_token_ids,
    write_token_ids_to_tmpfile,
)

QWEN3_MODEL_ID = "Qwen/Qwen3-0.6B"


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
    """Uses real Qwen3-0.6B tokenizer."""

    @pytest.fixture(scope="class")
    def tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(QWEN3_MODEL_ID, trust_remote_code=True)

    def test_correct_length(self, tokenizer: AutoTokenizer) -> None:
        result = generate_token_ids(
            prompt=PromptConfig(mode="math", seq_length=50),
            tokenizer_path=Path(QWEN3_MODEL_ID),
        )
        assert len(result) == 50
        assert all(isinstance(t, int) for t in result)

    def test_chat_template_changes_tokens(self, tokenizer: AutoTokenizer) -> None:
        prompt_text = "The quick brown fox jumps over the lazy dog. " * 5
        seq_length = 10

        without_template = generate_token_ids(
            prompt=PromptConfig(
                mode="text", text=prompt_text,
                seq_length=seq_length, apply_chat_template=False,
            ),
            tokenizer_path=Path(QWEN3_MODEL_ID),
        )
        with_template = generate_token_ids(
            prompt=PromptConfig(
                mode="text", text=prompt_text,
                seq_length=seq_length, apply_chat_template=True,
            ),
            tokenizer_path=Path(QWEN3_MODEL_ID),
        )
        assert without_template != with_template

    def test_math_mode_deterministic(self) -> None:
        prompt = PromptConfig(mode="math", seq_length=50)
        r1 = generate_token_ids(prompt=prompt, tokenizer_path=Path(QWEN3_MODEL_ID))
        r2 = generate_token_ids(prompt=prompt, tokenizer_path=Path(QWEN3_MODEL_ID))
        assert r1 == r2

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="less than seq_length"):
            generate_token_ids(
                prompt=PromptConfig(mode="text", text="hi", seq_length=9999),
                tokenizer_path=Path(QWEN3_MODEL_ID),
            )

    def test_text_mode_encodes_real_text(self) -> None:
        result = generate_token_ids(
            prompt=PromptConfig(mode="text", text="The quick brown fox jumps over the lazy dog " * 10, seq_length=30),
            tokenizer_path=Path(QWEN3_MODEL_ID),
        )
        assert len(result) == 30

    def test_file_mode(self, tmp_path: Path) -> None:
        f = tmp_path / "prompt.txt"
        f.write_text("Hello world, this is a test prompt. " * 20)
        result = generate_token_ids(
            prompt=PromptConfig(mode="file", file=f, seq_length=40),
            tokenizer_path=Path(QWEN3_MODEL_ID),
        )
        assert len(result) == 40
