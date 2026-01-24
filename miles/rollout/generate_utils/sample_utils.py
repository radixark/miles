from copy import deepcopy
from dataclasses import fields

from miles.utils.types import Sample


def merge_samples(samples: list[Sample], tokenizer) -> Sample:
    acc = samples[0]
    for sample in samples[1:]:
        acc = _merge_sample_pair(acc, sample, tokenizer=tokenizer)
    return acc


def _merge_sample_pair(a: Sample, b: Sample, tokenizer) -> Sample:
    """Merge two samples generated from sibling inference engine calls."""
    a, b = deepcopy(a), deepcopy(b)

    def _merge_equal_value(field):
        x = getattr(a, field)
        y = getattr(b, field)
        assert x == y, f"{field} mismatch: a.{field}={x}, b.{field}={y}"
        return x

    def _fill_defaults(sample: Sample):
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = [0.0] * sample.response_length

    _fill_defaults(a)
    _fill_defaults(b)

    obs_len = len(b.tokens) - len(a.tokens) - b.response_length
    obs_tokens = b.tokens[len(a.tokens) : len(a.tokens) + obs_len]
    # TODO: is this acceptable?
    obs_text = tokenizer.decode(obs_tokens)

    try:
        a.validate()
        b.validate()
        assert _startswith(short=a.prompt, long=b.prompt), "b.prompt must start with a.prompt"
        assert _startswith(short=a.tokens, long=b.tokens), "b.tokens must start with a.tokens"
        assert obs_len > 0, f"obs_len must be > 0, got {obs_len}"
        if a.rollout_routed_experts is not None:
            assert a.rollout_routed_experts.shape[0] <= b.rollout_routed_experts.shape[0]
        assert a.status == Sample.Status.COMPLETED, f"a.status must be COMPLETED, got {a.status}"

        return _create_with_all_fields(
            Sample,
            group_index=_merge_equal_value("group_index"),
            index=_merge_equal_value("index"),
            prompt=b.prompt,
            tokens=b.tokens,
            multimodal_inputs=_merge_equal_value("multimodal_inputs"),
            multimodal_train_inputs=_merge_equal_value("multimodal_train_inputs"),
            response=a.response + obs_text + b.response,
            response_length=a.response_length + obs_len + b.response_length,
            label=_merge_equal_value("label"),
            reward=_merge_equal_value("reward"),
            loss_mask=a.loss_mask + [0] * obs_len + b.loss_mask,
            weight_versions=a.weight_versions + b.weight_versions,
            rollout_log_probs=a.rollout_log_probs + [0.0] * obs_len + b.rollout_log_probs,
            rollout_routed_experts=b.rollout_routed_experts,
            remove_sample=_merge_equal_value("remove_sample"),
            status=b.status,
            metadata=_merge_equal_value("metadata"),
            train_metadata=_merge_equal_value("train_metadata"),
            non_generation_time=_merge_equal_value("non_generation_time"),
            spec_info=_merge_spec_info(a.spec_info, b.spec_info),
            prefix_cache_info=_merge_prefix_cache_info(a.prefix_cache_info, b.prefix_cache_info),
        )
    except AssertionError as e:
        e.add_note(f"{a=} {b=}")
        raise


def _merge_spec_info(a: Sample.SpecInfo, b: Sample.SpecInfo) -> Sample.SpecInfo:
    def _merge_plus_value(field):
        return getattr(a, field) + getattr(b, field)

    return _create_with_all_fields(
        Sample.SpecInfo,
        spec_accept_token_num=_merge_plus_value("spec_accept_token_num"),
        spec_draft_token_num=_merge_plus_value("spec_draft_token_num"),
        spec_verify_ct=_merge_plus_value("spec_verify_ct"),
        completion_token_num=_merge_plus_value("completion_token_num"),
    )


def _merge_prefix_cache_info(a: Sample.PrefixCacheInfo, b: Sample.PrefixCacheInfo) -> Sample.PrefixCacheInfo:
    def _merge_plus_value(field):
        return getattr(a, field) + getattr(b, field)

    return _create_with_all_fields(
        Sample.PrefixCacheInfo,
        cached_tokens=_merge_plus_value("cached_tokens"),
        total_prompt_tokens=_merge_plus_value("total_prompt_tokens"),
    )


def _create_with_all_fields(cls, **kwargs):
    expected = {f.name for f in fields(cls)}
    actual = set(kwargs.keys())
    assert (
        expected == actual
    ), f"{cls.__name__} field mismatch. Missing: {expected - actual}, Extra: {actual - expected}"
    return cls(**kwargs)


def _startswith(*, short, long) -> bool:
    if isinstance(short, str) and isinstance(long, str):
        return long.startswith(short)
    if isinstance(short, list) and isinstance(long, list):
        return (len(long) >= len(short)) and (long[: len(short)] == short)
    raise NotImplementedError
