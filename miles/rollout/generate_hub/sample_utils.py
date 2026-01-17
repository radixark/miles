from copy import deepcopy
from dataclasses import fields

from miles.utils.types import Sample


def _get_field_names(cls):
    return {f.name for f in fields(cls)}


def merge_samples(a: Sample, b: Sample, tokenizer) -> Sample:
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
        assert b.prompt.startswith(a.prompt)
        assert b.tokens[: len(a.tokens)] == a.tokens
        assert obs_len > 0
    except AssertionError as e:
        e.add_note(f"{a=} {b=}")
        raise

    spec_info = _merge_spec_info(a.spec_info, b.spec_info)
    prefix_cache_info = _merge_prefix_cache_info(a.prefix_cache_info, b.prefix_cache_info)

    merged_fields = dict(
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
        remove_sample=a.remove_sample or b.remove_sample,
        status=b.status,
        metadata=_merge_equal_value("metadata"),
        train_metadata=_merge_equal_value("train_metadata"),
        non_generation_time=a.non_generation_time + b.non_generation_time,
        spec_info=spec_info,
        prefix_cache_info=prefix_cache_info,
    )

    expected_fields = _get_field_names(Sample)
    actual_fields = set(merged_fields.keys())
    assert expected_fields == actual_fields, (
        f"Field mismatch. Missing: {expected_fields - actual_fields}, Extra: {actual_fields - expected_fields}"
    )

    return Sample(**merged_fields)


def _merge_spec_info(a: Sample.SpecInfo, b: Sample.SpecInfo) -> Sample.SpecInfo:
    merged_fields = dict(
        spec_accept_token_num=a.spec_accept_token_num + b.spec_accept_token_num,
        spec_draft_token_num=a.spec_draft_token_num + b.spec_draft_token_num,
        spec_verify_ct=a.spec_verify_ct + b.spec_verify_ct,
        completion_token_num=a.completion_token_num + b.completion_token_num,
    )

    expected_fields = set(Sample.SpecInfo.__dataclass_fields__.keys())
    actual_fields = set(merged_fields.keys())
    assert expected_fields == actual_fields, (
        f"SpecInfo field mismatch. Missing: {expected_fields - actual_fields}, Extra: {actual_fields - expected_fields}"
    )

    return Sample.SpecInfo(**merged_fields)


def _merge_prefix_cache_info(a: Sample.PrefixCacheInfo, b: Sample.PrefixCacheInfo) -> Sample.PrefixCacheInfo:
    merged_fields = dict(
        cached_tokens=a.cached_tokens + b.cached_tokens,
        total_prompt_tokens=a.total_prompt_tokens + b.total_prompt_tokens,
    )

    expected_fields = set(Sample.PrefixCacheInfo.__dataclass_fields__.keys())
    actual_fields = set(merged_fields.keys())
    assert expected_fields == actual_fields, (
        f"PrefixCacheInfo field mismatch. Missing: {expected_fields - actual_fields}, Extra: {actual_fields - expected_fields}"
    )

    return Sample.PrefixCacheInfo(**merged_fields)
