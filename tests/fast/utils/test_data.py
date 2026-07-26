from miles.utils.data import filter_long_prompt
from miles.utils.types import Sample


def test_filter_long_prompt_preserves_conversation_prompts() -> None:
    samples = [Sample(prompt=[{"role": "user", "content": "How much?"}])]

    filtered = filter_long_prompt(
        samples,
        tokenizer=None,
        processor=None,
        max_length=2048,
    )

    assert filtered is samples
