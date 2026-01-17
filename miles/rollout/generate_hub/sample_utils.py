from miles.utils.types import Sample


def merge_samples(a: Sample, b: Sample, tokenizer) -> Sample:
    def _merge_equal_value(field):
        x = getattr(a, field)
        y = getattr(b, field)
        assert x == y, f"{field} mismatch: a.{field}={x}, b.{field}={y}"
        return x

    a.validate()
    b.validate()
    assert a.loss_mask is not None, "a.loss_mask is None"
    assert b.loss_mask is not None, "b.loss_mask is None"
    assert a.rollout_log_probs is not None
    assert b.rollout_log_probs is not None
    assert b.tokens[: len(a.tokens)] == a.tokens, (
        f"b.tokens must start with a.tokens. "
        f"a.tokens: {a.tokens}, "
        f"b.tokens prefix: {b.tokens[:len(a.tokens)]}"
    )

    obs_len = len(b.tokens) - len(a.tokens) - b.response_length
    assert obs_len > 0, (
        f"obs_len (observation/intermediate tokens) must be > 0, got {obs_len}. "
        f"b.tokens length: {len(b.tokens)}, "
        f"a.tokens length: {len(a.tokens)}, "
        f"b.response_length: {b.response_length}"
    )

    obs_tokens = b.tokens[len(a.tokens): len(a.tokens) + obs_len]
    obs_text = tokenizer.decode(obs_tokens)

    return Sample(
        group_index=_merge_equal_value("group_index"),
        index=_merge_equal_value("index"),
        prompt=_merge_equal_value("prompt"),
        tokens=b.tokens,
        response=a.response + obs_text + b.response,
        response_length=a.response_length + obs_len + b.response_length,
        label=_merge_equal_value("label"),
        reward=_merge_equal_value("reward"),
        loss_mask=a.loss_mask + [0] * obs_len + b.loss_mask,
        rollout_log_probs=a.rollout_log_probs + [0.0] * obs_len + b.rollout_log_probs,
        status=b.status,
    )
