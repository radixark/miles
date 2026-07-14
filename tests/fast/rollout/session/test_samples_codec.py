"""Tests for the samples wire codec: encode on the worker, overlay on the driver.

Covers the COMPUTED/TEMPLATE field partition, binary-segment round-trips,
and the overlay defaults guard (`_assert_overlay_template_defaults`).
"""

import dataclasses

import numpy as np
import pytest

from miles.rollout.session.samples import COMPUTED_FIELDS, TEMPLATE_FIELDS, decode_samples_reply, encode_samples_reply
from miles.utils.types import Sample


def _computed_sample(**overrides) -> Sample:
    """A blank-template sample carrying every computed field, as the worker produces."""
    s = Sample()
    s.tokens = [1, 2, 3, 10, 11]
    s.response = "[10][11]"
    s.response_length = 2
    s.loss_mask = [1, 1]
    s.rollout_log_probs = [-0.5, -0.1234567891234567]
    s.rollout_routed_experts = np.arange(24, dtype=np.int32).reshape(4, 3, 2)
    s.rollout_indexer_topk = None
    s.status = Sample.Status.COMPLETED
    s.weight_versions = ["w1", "w2"]
    s.prefix_cache_info = Sample.PrefixCacheInfo.from_dict({"cached_tokens": 2, "total_prompt_tokens": 3})
    for name, value in overrides.items():
        setattr(s, name, value)
    return s


class TestSamplesWireCodec:
    def test_field_partition_is_total_and_disjoint(self):
        all_fields = {f.name for f in dataclasses.fields(Sample)}
        assert set(COMPUTED_FIELDS) | set(TEMPLATE_FIELDS) == all_fields
        assert not set(COMPUTED_FIELDS) & set(TEMPLATE_FIELDS)

    def test_round_trip_overlays_computed_and_keeps_template(self):
        template = Sample(
            group_index=7,
            index=3,
            prompt=[{"role": "user", "content": "hi"}],
            label="lbl",
            reward=1.5,
            metadata={"task": "t"},
            session_id="sid",
            train_metadata={"loss": "ppo"},
        )
        payload = encode_samples_reply([_computed_sample()], {"max_trim_tokens": 1}, None)
        reply = decode_samples_reply(payload, template)

        assert reply.empty_reason is None
        assert reply.session_metadata == {"max_trim_tokens": 1}
        (out,) = reply.samples
        # computed fields overlaid, with exact types/values
        assert out.tokens == [1, 2, 3, 10, 11] and type(out.tokens) is list
        assert out.rollout_log_probs == [-0.5, -0.1234567891234567]
        assert out.loss_mask == [1, 1]
        assert out.response == "[10][11]" and out.response_length == 2
        assert out.status == Sample.Status.COMPLETED
        assert out.weight_versions == ["w1", "w2"]
        assert out.prefix_cache_info.to_dict() == {"cached_tokens": 2, "total_prompt_tokens": 3}
        assert out.rollout_routed_experts.dtype == np.int32
        assert np.array_equal(out.rollout_routed_experts, np.arange(24, dtype=np.int32).reshape(4, 3, 2))
        assert out.rollout_indexer_topk is None
        # template fields carried from the input sample, untouched
        assert out.group_index == 7 and out.index == 3
        assert out.prompt == [{"role": "user", "content": "hi"}]
        assert out.label == "lbl" and out.reward == 1.5
        assert out.metadata == {"task": "t"} and out.session_id == "sid"
        assert out.train_metadata == {"loss": "ppo"}
        # the input template itself is never mutated
        assert template.tokens == [] and template.metadata == {"task": "t"}

    def test_multi_sample_reply_keeps_per_sample_segments(self):
        a = _computed_sample()
        b = _computed_sample(
            tokens=[1, 2, 3, 10, 11, 20, 21, 30],
            rollout_routed_experts=np.arange(100, 100 + 42, dtype=np.int32).reshape(7, 3, 2),
            rollout_log_probs=[-1.0, -2.0],
        )
        reply = decode_samples_reply(encode_samples_reply([a, b], {}, None), Sample())
        out_a, out_b = reply.samples
        assert out_a.tokens == a.tokens and out_b.tokens == b.tokens
        assert np.array_equal(out_a.rollout_routed_experts, a.rollout_routed_experts)
        assert np.array_equal(out_b.rollout_routed_experts, b.rollout_routed_experts)
        assert out_b.rollout_log_probs == [-1.0, -2.0]

    def test_empty_reply_round_trips_reason_and_skips_defaults_guard(self):
        # The empty reply is decoded before the driver takes its ABORTED path, so
        # the overlay-defaults guard must not fire on it — even for an input
        # sample that would violate the guard.
        evolved = Sample(weight_versions=["stale"])
        reply = decode_samples_reply(encode_samples_reply([], {"max_trim_tokens": 0}, "no_records"), evolved)
        assert reply.samples == [] and reply.empty_reason == "no_records"

    def test_defaults_guard_rejects_evolved_template(self):
        payload = encode_samples_reply([_computed_sample()], {}, None)
        with pytest.raises(AssertionError, match="weight_versions"):
            decode_samples_reply(payload, Sample(weight_versions=["stale"]))
        with pytest.raises(AssertionError, match="teacher_log_probs"):
            decode_samples_reply(payload, Sample(teacher_log_probs=[-1.0]))
        with pytest.raises(AssertionError, match="opd_student_top_logprobs"):
            decode_samples_reply(payload, Sample(metadata={"opd_student_top_logprobs": [[[-0.1, 1]]]}))
