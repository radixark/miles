from miles.utils.types import Sample


def merge_samples(a: Sample, b: Sample, tokenizer) -> Sample:
    _validate_samples(a, b)

    tool_response_len = len(b.tokens) - len(a.tokens) - b.response_length
    assert tool_response_len > 0, (
        f"tool_response_len must be > 0, got {tool_response_len}. "
        f"sample2.tokens length: {len(b.tokens)}, "
        f"sample1.tokens length: {len(a.tokens)}, "
        f"sample2.response_length: {b.response_length}"
    )

    tool_response_tokens = b.tokens[len(a.tokens): len(a.tokens) + tool_response_len]
    tool_response_text = tokenizer.decode(tool_response_tokens)

    return Sample(
        prompt=a.prompt,
        tokens=b.tokens,
        response=a.response + tool_response_text + b.response,
        response_length=a.response_length + tool_response_len + b.response_length,
        loss_mask=a.loss_mask + [0] * tool_response_len + b.loss_mask,
        rollout_log_probs=a.rollout_log_probs + [0.0] * tool_response_len + b.rollout_log_probs,
        status=b.status,
        label=b.label,
        reward=b.reward,
        index=a.index,
        group_index=a.group_index,
    )


def _validate_samples(sample1: Sample, sample2: Sample):
    assert sample1.prompt == sample2.prompt, (
        f"prompt mismatch: sample1.prompt={sample1.prompt}, sample2.prompt={sample2.prompt}"
    )

    assert sample2.tokens[: len(sample1.tokens)] == sample1.tokens, (
        f"sample2.tokens must start with sample1.tokens. "
        f"sample1.tokens: {sample1.tokens}, "
        f"sample2.tokens prefix: {sample2.tokens[:len(sample1.tokens)]}"
    )

    assert sample1.loss_mask is not None, "sample1.loss_mask is None"
    assert sample2.loss_mask is not None, "sample2.loss_mask is None"
    assert len(sample1.loss_mask) == sample1.response_length, (
        f"sample1.loss_mask length ({len(sample1.loss_mask)}) != "
        f"sample1.response_length ({sample1.response_length})"
    )
    assert len(sample2.loss_mask) == sample2.response_length, (
        f"sample2.loss_mask length ({len(sample2.loss_mask)}) != "
        f"sample2.response_length ({sample2.response_length})"
    )

    assert sample1.rollout_log_probs is not None, "sample1.rollout_log_probs is None"
    assert sample2.rollout_log_probs is not None, "sample2.rollout_log_probs is None"
    assert len(sample1.rollout_log_probs) == sample1.response_length, (
        f"sample1.rollout_log_probs length ({len(sample1.rollout_log_probs)}) != "
        f"sample1.response_length ({sample1.response_length})"
    )
    assert len(sample2.rollout_log_probs) == sample2.response_length, (
        f"sample2.rollout_log_probs length ({len(sample2.rollout_log_probs)}) != "
        f"sample2.response_length ({sample2.response_length})"
    )
