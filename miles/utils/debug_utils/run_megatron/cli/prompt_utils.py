from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Literal


@dataclasses.dataclass(frozen=True)
class PromptConfig:
    mode: Literal["math", "file", "text"] = "math"
    text: str | None = None
    file: Path | None = None
    seq_length: int = 137
    apply_chat_template: bool = False


def generate_token_ids(
    *,
    prompt: PromptConfig,
    tokenizer_path: Path,
) -> list[int]:
    """Generate token IDs for Megatron standalone forward/backward.

    Three modes:
    - math: deterministic arithmetic sequence "1+1=2, 1+2=3, ..." tokenized and padded/truncated to seq_length
    - file: read long text from file, tokenize and truncate to seq_length
    - text: use user-provided text directly, tokenize and truncate to seq_length
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    raw_text: str = _resolve_raw_text(prompt)

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)

    if prompt.apply_chat_template:
        messages: list[dict[str, str]] = [{"role": "user", "content": raw_text}]
        raw_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )

    token_ids: list[int] = tokenizer.encode(raw_text)
    if len(token_ids) < prompt.seq_length:
        raise ValueError(
            f"Token count ({len(token_ids)}) is less than seq_length ({prompt.seq_length}). "
            f"Provide longer input text or reduce --seq-length."
        )
    token_ids = token_ids[: prompt.seq_length]

    return token_ids


def write_token_ids_to_tmpfile(token_ids: list[int]) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="run_megatron_token_ids_") as tmp:
        json.dump(token_ids, tmp)
    return Path(tmp.name)


def _resolve_raw_text(prompt: PromptConfig) -> str:
    if prompt.mode == "math":
        return _build_math_sequence(target_char_length=prompt.seq_length * 16)
    elif prompt.mode == "file":
        if prompt.file is None:
            raise ValueError("--prompt-file is required for file mode")
        return prompt.file.read_text()
    elif prompt.mode == "text":
        if prompt.text is None:
            raise ValueError("--prompt-text is required for text mode")
        return prompt.text
    else:
        raise ValueError(f"Unknown prompt mode: {prompt.mode!r}")


def _build_math_sequence(target_char_length: int) -> str:
    """Build "1+1=2, 1+2=3, ..." until reaching target_char_length characters."""
    parts: list[str] = []
    total_len: int = 0
    a: int = 1
    b: int = 1

    while total_len < target_char_length:
        segment: str = f"{a}+{b}={a + b}, "
        parts.append(segment)
        total_len += len(segment)
        b += 1
        if b > 100:
            a += 1
            b = 1

    return "".join(parts)


