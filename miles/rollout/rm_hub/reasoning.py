# Ordered from model-specific special tokens to generic/legacy markers so an
# explicit delimiter wins over `</think>` or `###Response` in reasoning text.
REASONING_ANSWER_DELIMITERS: tuple[str, ...] = (
    "<|open|>response<|sep|>",  # Kimi K3
    "<|channel|>final<|message|>",  # GPT-OSS (Harmony)
    "<|content_text|>",  # Inkling / Inkling 2
    "to=user<|message|>",  # Muse Spark
    "<channel|>",  # Gemma-4
    "◁/think▷",  # Kimi K2-Thinking
    "</think>",  # Qwen3, DeepSeek, GLM, MiniMax, Nemotron, MiMo, Kimi K2.5+, Step
    "###Response",  # DeepScaleR legacy template
)


def extract_after_reasoning(text: str | None) -> str | None:
    """Return text after the last occurrence of the first matching delimiter, or None if absent."""
    if not text:
        return None
    for delimiter in REASONING_ANSWER_DELIMITERS:
        if delimiter in text:
            return text.rsplit(delimiter, 1)[-1]
    return None
