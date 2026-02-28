import json
import tempfile
from pathlib import Path
from typing import Literal


def generate_token_ids(
    *,
    mode: Literal["math", "file", "text"],
    seq_length: int,
    tokenizer_path: Path,
    prompt_text: str | None = None,
    prompt_file: Path | None = None,
    apply_chat_template: bool = False,
) -> list[int]:
    """Generate token IDs for Megatron standalone forward/backward.

    Three modes:
    - math: deterministic arithmetic sequence "1+1=2, 1+2=3, ..." tokenized and padded/truncated to seq_length
    - file: read long text from file, tokenize and truncate to seq_length
    - text: use user-provided text directly, tokenize and truncate to seq_length
    """
    from transformers import AutoTokenizer

    if mode == "text" and prompt_text is None:
        raise ValueError("--prompt-text is required for text mode")
    if mode == "file" and prompt_file is None:
        raise ValueError("--prompt-file is required for file mode")

    raw_text: str = _resolve_raw_text(
        mode=mode,
        seq_length=seq_length,
        prompt_text=prompt_text,
        prompt_file=prompt_file,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)

    if apply_chat_template:
        messages: list[dict[str, str]] = [{"role": "user", "content": raw_text}]
        raw_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    token_ids: list[int] = tokenizer.encode(raw_text)
    token_ids = _pad_or_truncate(
        token_ids=token_ids,
        seq_length=seq_length,
        pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    assert len(token_ids) == seq_length, (
        f"token_ids length {len(token_ids)} != seq_length {seq_length}"
    )
    return token_ids


def write_token_ids_to_tmpfile(token_ids: list[int]) -> Path:
    tmp: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="run_megatron_token_ids_"
    )
    json.dump(token_ids, tmp)
    tmp.close()
    return Path(tmp.name)


def _resolve_raw_text(
    *,
    mode: Literal["math", "file", "text"],
    seq_length: int,
    prompt_text: str | None,
    prompt_file: Path | None,
) -> str:
    if mode == "math":
        return _build_math_sequence(target_char_length=seq_length * 8)
    elif mode == "file":
        assert prompt_file is not None
        return prompt_file.read_text()
    elif mode == "text":
        assert prompt_text is not None
        return prompt_text
    else:
        raise ValueError(f"Unknown prompt mode: {mode!r}")


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


def _pad_or_truncate(
    *,
    token_ids: list[int],
    seq_length: int,
    pad_id: int,
) -> list[int]:
    if len(token_ids) > seq_length:
        return token_ids[:seq_length]
    elif len(token_ids) < seq_length:
        return token_ids + [pad_id] * (seq_length - len(token_ids))
    return token_ids
