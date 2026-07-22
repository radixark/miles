"""Tests for the samples wire codec: encode on the worker, overlay on the driver.

Covers safetensors-tensor round-trips, malformed-payload rejection, and the
overlay defaults guard (`_assert_overlay_template_defaults`).
"""

import json

import numpy as np
import pytest
import safetensors.numpy
from safetensors import SafetensorError

from miles.rollout.session.samples.codec import decode_samples_reply, encode_samples_reply
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


def _mutated_payload(payload: bytes, mutate) -> bytes:
    """Re-pack a valid payload after `mutate(meta, tensors)` edits, for malformed-wire cases."""
    tensors = safetensors.numpy.load(payload)
    meta = json.loads(tensors.pop("_samples_meta").tobytes().decode("utf-8"))
    mutate(meta, tensors)
    meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    tensors["_samples_meta"] = np.frombuffer(meta_bytes, dtype=np.uint8)
    return safetensors.numpy.save(tensors)


class TestSamplesWireCodec:
    def test_round_trip_overlays_computed_and_keeps_template(self):
        template = Sample(
            group_index=7,
            index=3,
            prompt=[{"role": "user", "content": "hi"}],
            label="lbl",
            reward=1.5,
            metadata={"task": "t"},
            routing_key="rk",
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
        assert out.metadata == {"task": "t"} and out.routing_key == "rk"
        assert out.train_metadata == {"loss": "ppo"}
        # the input template itself is never mutated
        assert template.tokens == [] and template.metadata == {"task": "t"}

    def test_multi_sample_reply_keeps_per_sample_tensors(self):
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

    def test_safetensors_container_round_trips_non_contiguous_replay_tensors(self):
        routed = np.arange(24, dtype=np.int32).reshape(3, 4, 2).transpose(1, 0, 2)
        indexer = np.arange(48, dtype=np.int32).reshape(8, 3, 2)[::2]
        assert not routed.flags["C_CONTIGUOUS"] and not indexer.flags["C_CONTIGUOUS"]
        sample = _computed_sample(rollout_routed_experts=routed, rollout_indexer_topk=indexer)

        payload = encode_samples_reply([sample], {}, None)
        # the reply is a plain safetensors buffer: no Miles framing needed to open it
        tensors = safetensors.numpy.load(payload)
        assert set(tensors) == {
            "_samples_meta",
            "sample.0.tokens",
            "sample.0.rollout_log_probs",
            "sample.0.rollout_routed_experts",
            "sample.0.rollout_indexer_topk",
        }
        assert tensors["_samples_meta"].dtype == np.uint8 and tensors["_samples_meta"].ndim == 1

        (out,) = decode_samples_reply(payload, Sample()).samples
        assert out.tokens == sample.tokens and type(out.tokens) is list
        assert out.rollout_log_probs == sample.rollout_log_probs
        assert out.rollout_routed_experts.dtype == np.int32 and out.rollout_routed_experts.shape == (4, 3, 2)
        assert np.array_equal(out.rollout_routed_experts, routed)
        assert out.rollout_indexer_topk.dtype == np.int32 and np.array_equal(out.rollout_indexer_topk, indexer)

    def test_zero_size_tensor_is_distinct_from_none(self):
        sample = _computed_sample(
            rollout_routed_experts=np.empty((0, 3, 2), dtype=np.int32), rollout_indexer_topk=None
        )
        (out,) = decode_samples_reply(encode_samples_reply([sample], {}, None), Sample()).samples
        assert isinstance(out.rollout_routed_experts, np.ndarray) and out.rollout_routed_experts.shape == (0, 3, 2)
        assert out.rollout_indexer_topk is None

    def test_null_tokens_restore_fresh_empty_lists(self):
        # JSON null restores per-sample fresh instances, never one shared list.
        def null_out_tokens(meta, tensors):
            for index, sample_meta in enumerate(meta["samples"]):
                sample_meta["tensors"]["tokens"] = None
                del tensors[f"sample.{index}.tokens"]

        payload = _mutated_payload(
            encode_samples_reply([_computed_sample(), _computed_sample()], {}, None), null_out_tokens
        )
        out_a, out_b = decode_samples_reply(payload, Sample()).samples
        assert out_a.tokens == [] and out_b.tokens == []
        assert out_a.tokens is not out_b.tokens

    def test_encode_rejects_non_int32_replay_dtype(self):
        sample = _computed_sample(rollout_routed_experts=np.arange(24, dtype=np.int64).reshape(4, 3, 2))
        with pytest.raises(ValueError, match="rollout_routed_experts must have dtype int32"):
            encode_samples_reply([sample], {}, None)

    @pytest.mark.parametrize(
        ("build_payload", "expected_error", "match"),
        [
            pytest.param(lambda p: p[: len(p) - 100], SafetensorError, None, id="truncated-container"),
            pytest.param(lambda p: b"", SafetensorError, None, id="empty-container"),
            pytest.param(
                lambda p: safetensors.numpy.save({"sample.0.tokens": np.arange(3, dtype=np.int64)}),
                KeyError,
                "_samples_meta",
                id="missing-samples-meta",
            ),
            pytest.param(
                lambda p: safetensors.numpy.save({"_samples_meta": np.zeros(4, dtype=np.int32)}),
                ValueError,
                "rank-one uint8",
                id="meta-wrong-dtype",
            ),
            pytest.param(
                lambda p: safetensors.numpy.save({"_samples_meta": np.zeros((2, 2), dtype=np.uint8)}),
                ValueError,
                "rank-one uint8",
                id="meta-wrong-rank",
            ),
            pytest.param(
                lambda p: _mutated_payload(p, lambda meta, tensors: tensors.pop("sample.0.tokens")),
                KeyError,
                "sample.0.tokens",
                id="referenced-tensor-missing",
            ),
            pytest.param(
                lambda p: _mutated_payload(
                    p,
                    lambda meta, tensors: tensors.update(
                        {"sample.0.tokens": tensors["sample.0.tokens"].astype(np.int32)}
                    ),
                ),
                ValueError,
                "tokens must have dtype int64",
                id="wire-dtype-contract-violated",
            ),
            pytest.param(
                lambda p: _mutated_payload(
                    p,
                    lambda meta, tensors: meta["samples"][0]["tensors"].update(tokens="sample.0.rollout_log_probs"),
                ),
                ValueError,
                "references tensor",
                id="reference-name-mismatch",
            ),
            pytest.param(
                lambda p: _mutated_payload(
                    p, lambda meta, tensors: tensors.update(orphan=np.zeros(1, dtype=np.uint8))
                ),
                ValueError,
                "unreferenced tensors",
                id="unreferenced-leftover-tensor",
            ),
        ],
    )
    def test_malformed_safetensors_reply_fails_loudly(self, build_payload, expected_error, match):
        valid = encode_samples_reply([_computed_sample()], {}, None)
        with pytest.raises(expected_error, match=match):
            decode_samples_reply(build_payload(valid), Sample())
