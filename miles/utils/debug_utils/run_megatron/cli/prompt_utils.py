import tempfile
from pathlib import Path
from typing import Literal


def write_prompt_to_tmpfile(prompt_text: str) -> Path:
    tmp: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="run_megatron_prompt_"
    )
    tmp.write(prompt_text)
    tmp.close()
    return Path(tmp.name)


def generate_prompt(
    *,
    mode: Literal["math", "file", "text"],
    seq_length: int,
    tokenizer_path: Path | None = None,
    prompt_text: str | None = None,
    prompt_file: Path | None = None,
    apply_chat_template: bool = False,
) -> str:
    """Generate a prompt string for Megatron standalone forward/backward.

    Three modes:
    - math: deterministic arithmetic sequence "1+1=2, 1+2=3, ..." padded to seq_length tokens
    - file: read long text from file, truncate to seq_length tokens
    - text: use user-provided text directly (no tokenizer needed)
    """
    if mode == "math":
        return _generate_math_prompt(
            seq_length=seq_length,
            tokenizer_path=tokenizer_path,
            apply_chat_template=apply_chat_template,
        )
    elif mode == "file":
        if prompt_file is None:
            raise ValueError("--prompt-file is required for file mode")
        return _generate_file_prompt(
            seq_length=seq_length,
            tokenizer_path=tokenizer_path,
            prompt_file=prompt_file,
            apply_chat_template=apply_chat_template,
        )
    elif mode == "text":
        if prompt_text is None:
            raise ValueError("--prompt-text is required for text mode")
        return prompt_text
    else:
        raise ValueError(f"Unknown prompt mode: {mode!r}")


def _generate_math_prompt(
    *,
    seq_length: int,
    tokenizer_path: Path | None,
    apply_chat_template: bool,
) -> str:
    """Generate "1+1=2, 1+2=3, 1+3=4, ..." arithmetic sequence, padded to fill seq_length tokens."""
    raw_text: str = _build_math_sequence(target_char_length=seq_length * 8)

    if tokenizer_path is not None:
        raw_text = _truncate_to_seq_length(
            text=raw_text,
            seq_length=seq_length,
            tokenizer_path=tokenizer_path,
            apply_chat_template=apply_chat_template,
        )

    return raw_text


def _generate_file_prompt(
    *,
    seq_length: int,
    tokenizer_path: Path | None,
    prompt_file: Path,
    apply_chat_template: bool,
) -> str:
    """Read long text from file, truncate to seq_length tokens."""
    raw_text: str = prompt_file.read_text()

    if tokenizer_path is not None:
        raw_text = _truncate_to_seq_length(
            text=raw_text,
            seq_length=seq_length,
            tokenizer_path=tokenizer_path,
            apply_chat_template=apply_chat_template,
        )

    return raw_text


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


def _truncate_to_seq_length(
    *,
    text: str,
    seq_length: int,
    tokenizer_path: Path,
    apply_chat_template: bool,
) -> str:
    """Truncate text so that the tokenized length fits within seq_length."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)

    if apply_chat_template:
        messages: list[dict[str, str]] = [{"role": "user", "content": text}]
        full_text: str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        full_text = text

    token_ids: list[int] = tokenizer.encode(full_text)

    if len(token_ids) <= seq_length:
        return text

    truncated_ids: list[int] = token_ids[:seq_length]
    return tokenizer.decode(truncated_ids, skip_special_tokens=False)
