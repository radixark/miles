"""Datum/result/sampling translation: official wire shapes <-> backend
payloads, with every v1 boundary rejection typed as UserInputError."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import pytest

from miles.ray.tinker_frontend import translation, wire
from miles.ray.tinker_frontend.translation import UserInputError


def tensor(data, dtype="float32", **kwargs):
    return {"data": data, "dtype": dtype, "shape": [len(data)], **kwargs}


def datum(tokens, targets, **channels):
    loss_fn_inputs = {"target_tokens": tensor(targets, "int64")}
    for name, values in channels.items():
        loss_fn_inputs[name] = tensor(values)
    return wire.Datum.model_validate(
        {"model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]}, "loss_fn_inputs": loss_fn_inputs}
    )


def fb_input(data, loss_fn="cross_entropy", config=None):
    return wire.ForwardBackwardInput.model_validate(
        {"data": [d.model_dump() for d in data], "loss_fn": loss_fn, "loss_fn_config": config}
    )


class TestDatumToSample:
    def test_shifted_targets_extend_the_token_sequence(self):
        sample = translation.datum_to_sample(0, datum([1, 2, 3], [2, 3, 4], weights=[0.0, 1.0, 1.0]), "cross_entropy")
        assert sample == {
            "tokens": [1, 2, 3, 4],
            "response_length": 3,
            "loss_mask": [1, 1, 1],
            "loss_weights": [0.0, 1.0, 1.0],
        }

    def test_active_position_must_be_next_token(self):
        with pytest.raises(UserInputError, match="next input"):
            translation.datum_to_sample(0, datum([1, 2, 3], [9, 3, 4], weights=[1.0, 1.0, 1.0]), "cross_entropy")

    def test_negative_token_ids_are_rejected(self):
        with pytest.raises(UserInputError, match="non-negative"):
            translation.datum_to_sample(0, datum([-1, 2, 3], [2, 3, 4], weights=[0.0, 1.0, 1.0]), "cross_entropy")
        with pytest.raises(UserInputError, match="non-negative"):
            translation.datum_to_sample(0, datum([1, 2, 3], [2, 3, -4], weights=[0.0, 1.0, 1.0]), "cross_entropy")

    def test_zero_weighted_mismatch_is_normalized_not_rejected(self):
        sample = translation.datum_to_sample(0, datum([1, 2, 3], [0, 3, 4], weights=[0.0, 1.0, 1.0]), "cross_entropy")
        assert sample["tokens"] == [1, 2, 3, 4]

    def test_importance_sampling_channels_map_to_backend_names(self):
        d = datum([1, 2], [2, 5], logprobs=[-0.5, -0.5], advantages=[0.0, 1.0])
        sample = translation.datum_to_sample(0, d, "importance_sampling")
        assert sample["rollout_log_probs"] == [-0.5, -0.5]
        assert sample["advantages"] == [0.0, 1.0]
        assert "loss_weights" not in sample

    def test_missing_required_channel_is_rejected(self):
        with pytest.raises(UserInputError, match="requires loss_fn_inputs\\['weights'\\]"):
            translation.datum_to_sample(0, datum([1, 2], [2, 3]), "cross_entropy")

    def test_sparse_csr_is_rejected(self):
        d = datum([1, 2], [2, 3], weights=[1.0, 1.0])
        d.loss_fn_inputs["target_tokens"].sparse_crow_indices = [0, 1, 2]
        with pytest.raises(UserInputError, match="sparse"):
            translation.datum_to_sample(0, d, "cross_entropy")

    def test_top_k_shaped_targets_are_rejected(self):
        d = datum([1, 2], [2, 3], weights=[1.0, 1.0])
        d.loss_fn_inputs["target_tokens"].shape = [2, 1]
        with pytest.raises(UserInputError, match="1-D"):
            translation.datum_to_sample(0, d, "cross_entropy")

    def test_non_text_chunks_are_rejected(self):
        d = datum([1, 2], [2, 3], weights=[1.0, 1.0])
        d.model_input.chunks[0].type = "image"
        with pytest.raises(UserInputError, match="text-only"):
            translation.datum_to_sample(0, d, "cross_entropy")

    def test_unknown_channels_and_length_mismatches_are_rejected(self):
        d = datum([1, 2], [2, 3], weights=[1.0, 1.0])
        d.loss_fn_inputs["mystery"] = d.loss_fn_inputs["weights"]
        with pytest.raises(UserInputError, match="unsupported loss_fn_inputs"):
            translation.datum_to_sample(0, d, "cross_entropy")
        with pytest.raises(UserInputError, match="one value per input token"):
            translation.datum_to_sample(0, datum([1, 2, 3], [2, 3], weights=[1.0, 1.0]), "cross_entropy")


class TestFbPayload:
    def test_payload_carries_samples_and_loss_spec(self):
        payload = translation.fb_input_to_payload(
            fb_input([datum([1, 2], [2, 3], weights=[1.0, 1.0])], config={"clip_low_threshold": 0.8})
        )
        assert payload["loss"] == {"loss_fn": "cross_entropy", "loss_fn_config": {"clip_low_threshold": 0.8}}
        assert len(payload["samples"]) == 1

    def test_unsupported_loss_fns_are_rejected(self):
        for loss_fn in ("cispo", "dro", "nope"):
            with pytest.raises(UserInputError, match="not supported"):
                translation.fb_input_to_payload(fb_input([datum([1, 2], [2, 3], weights=[1.0, 1.0])], loss_fn))

    def test_empty_data_is_rejected(self):
        with pytest.raises(UserInputError, match="at least one datum"):
            translation.fb_input_to_payload(fb_input([]))


class TestResults:
    def test_fb_result_uses_backend_metrics(self):
        body = translation.fb_result_to_response({"logprobs": [[-0.5, -0.25]], "metrics": {"loss:sum": 0.75}})
        assert body["metrics"] == {"loss:sum": 0.75}
        assert body["loss_fn_outputs"] == [{"logprobs": {"data": [-0.5, -0.25], "dtype": "float32", "shape": [2]}}]

    def test_forward_result_recomputes_metrics_from_the_request(self):
        payload = translation.fb_input_to_payload(fb_input([datum([1, 2, 3], [2, 3, 4], weights=[0.0, 1.0, 1.0])]))
        body = translation.fb_result_to_response({"logprobs": [[-0.5, -0.5, -0.5]]}, payload)
        assert body["metrics"]["loss:sum"] == pytest.approx(1.0)
        assert body["metrics"]["unmasked_tokens:sum"] == pytest.approx(3.0)

    def test_optim_result_projects_numeric_metrics(self):
        assert translation.optim_result_to_response({"grad_norm": 0.5, "learning_rate": 1e-4}) == {
            "type": "optim_step",
            "metrics": {"grad_norm": 0.5, "learning_rate": 1e-4},
        }


class TestSampling:
    def params(self, **kwargs):
        return wire.SamplingParams.model_validate({"max_tokens": 8, **kwargs})

    def test_params_map_to_sglang(self):
        params = translation.sampling_params_to_sglang(self.params(temperature=0.5, top_p=0.9, stop="\n"))
        assert params == {"max_new_tokens": 8, "temperature": 0.5, "top_p": 0.9, "top_k": -1, "stop": ["\n"]}

    def test_stop_token_ids(self):
        assert translation.sampling_params_to_sglang(self.params(stop=[7, 8]))["stop_token_ids"] == [7, 8]
        with pytest.raises(UserInputError, match="non-negative"):
            translation.sampling_params_to_sglang(self.params(stop=[7, -8]))

    def test_missing_max_tokens_is_rejected_and_seed_stays_out_of_base_params(self):
        with pytest.raises(UserInputError, match="max_tokens"):
            translation.sampling_params_to_sglang(wire.SamplingParams())
        assert "sampling_seed" not in translation.sampling_params_to_sglang(self.params(seed=1))

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"temperature": -1.0}, "temperature"),
            ({"temperature": float("nan")}, "temperature"),
            ({"top_p": 0.0}, "top_p"),
            ({"top_p": float("inf")}, "top_p"),
            ({"top_k": 0}, "top_k"),
            ({"seed": -(2**63) - 1}, "seed"),
            ({"seed": 2**63}, "seed"),
        ],
    )
    def test_invalid_sampling_ranges_are_rejected_locally(self, overrides, message):
        with pytest.raises(UserInputError, match=message):
            translation.sampling_params_to_sglang(self.params(**overrides))

    def test_generation_maps_tokens_logprobs_and_stop_reason(self):
        sequence = translation.generation_to_sequence(
            {
                "meta_info": {
                    "finish_reason": {"type": "length"},
                    "output_token_logprobs": [[-0.1, 11, None], [-0.2, 12, None]],
                }
            }
        )
        assert sequence == {"stop_reason": "length", "tokens": [11, 12], "logprobs": [-0.1, -0.2]}

    def test_aborted_generation_raises(self):
        with pytest.raises(RuntimeError, match="abort"):
            translation.generation_to_sequence({"meta_info": {"finish_reason": {"type": "abort"}}})

    def test_prompt_logprobs_map_per_token_with_leading_none(self):
        generation = {"meta_info": {"input_token_logprobs": [[None, 5, None], [-0.5, 6, None], [-1.25, 7, None]]}}
        assert translation.prompt_logprobs_from_generation(generation, 3) == [None, -0.5, -1.25]

    def test_prompt_logprobs_missing_from_the_engine_is_a_server_fault(self):
        with pytest.raises(RuntimeError, match="no input_token_logprobs"):
            translation.prompt_logprobs_from_generation({"meta_info": {}}, 2)

    def test_prompt_logprobs_length_mismatch_is_a_server_fault(self):
        generation = {"meta_info": {"input_token_logprobs": [[None, 5, None]]}}
        with pytest.raises(RuntimeError, match="1 prompt logprobs for 2 prompt tokens"):
            translation.prompt_logprobs_from_generation(generation, 2)

    def test_sample_response_carries_prompt_logprobs_only_when_scored(self):
        assert translation.sequences_to_sample_response([])["prompt_logprobs"] is None
        assert translation.sequences_to_sample_response([], [None, -0.5])["prompt_logprobs"] == [None, -0.5]
