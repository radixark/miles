"""JSON decoding validates datum inputs and renders SDK result shapes."""

import pytest

from miles.tinker.core.types import UserInputError
from miles.tinker.server.encoding import (
    ADAM_PARAM_DEFAULTS,
    build_datum,
    decode_command,
    decode_sample_request,
    render_result,
    tensor_data_to_list,
)


def _datum(tokens: list[int], **extra_inputs) -> dict:
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens[:-1]}]},
        "loss_fn_inputs": {"target_tokens": tokens[1:], **extra_inputs},
    }


class TestDecodeForwardBackward:
    def _decode(self, datum, forward_only=False):
        payload = {
            "model_id": "model",
            "seq_id": 1,
            "forward_only": forward_only,
            "forward_backward_input": {"data": [datum], "loss_fn": "cross_entropy"},
        }
        return decode_command("forward_backward", payload)

    def test_a_shifted_datum_becomes_one_datum(self):
        op, decoded = self._decode(_datum([1, 2, 3, 4], weights=[1.0, 1.0, 1.0]))
        assert op == "forward_backward"
        assert decoded["datums"] == [
            {"tokens": [1, 2, 3, 4], "target_len": 3, "target_tokens": [2, 3, 4], "weights": [1.0, 1.0, 1.0]}
        ]

    def test_forward_only_selects_the_op(self):
        op, _ = self._decode(_datum([1, 2, 3]), forward_only=True)
        assert op == "forward_only"

    def test_a_length_mismatch_is_rejected(self):
        datum = _datum([1, 2, 3])
        datum["loss_fn_inputs"]["target_tokens"] = [2]
        with pytest.raises(UserInputError, match="length"):
            self._decode(datum)

    def test_explicit_targets_pass_through(self):
        """rl_loop pads the prompt region with dummy zero targets; labels need not be the shifted sequence."""
        datum = _datum([1, 2, 3])
        datum["loss_fn_inputs"]["target_tokens"] = [0, 9]
        _, decoded = self._decode(datum)
        assert decoded["datums"][0]["tokens"] == [1, 2, 9]
        assert decoded["datums"][0]["target_tokens"] == [0, 9]


def test_build_datum_maps_the_wire_input_names():
    datum = build_datum([1, 2], {"target_tokens": [2, 3], "logprobs": [0.5, 0.5], "advantages": [1.0, -1.0]}, 0)
    assert datum["sampling_logprobs"] == [0.5, 0.5]
    assert datum["advantages"] == [1.0, -1.0]


class TestAdamParams:
    def test_defaults_fill_missing_keys(self):
        _, decoded = decode_command(
            "optim_step", {"model_id": "m", "seq_id": 1, "adam_params": {"learning_rate": 3e-4}}
        )
        assert decoded["adam_params"]["learning_rate"] == 3e-4
        assert set(decoded["adam_params"]) == set(ADAM_PARAM_DEFAULTS)

    def test_unknown_keys_are_rejected(self):
        """The SDK's AdamParams model is the contract; a key it lacks must not be silently dropped."""
        with pytest.raises(UserInputError, match="momentum"):
            decode_command("optim_step", {"model_id": "m", "seq_id": 1, "adam_params": {"momentum": 0.9}})


def test_an_off_model_key_is_rejected():
    with pytest.raises(UserInputError, match="bogus"):
        decode_command("save_state", {"model_id": "m", "seq_id": 1, "bogus": 1})


class TestTensorData:
    def test_dense_tensor_data(self):
        assert tensor_data_to_list({"data": [1.0, 2.0], "shape": [2]}) == [1.0, 2.0]

    def test_csr_expands_to_dense(self):
        sparse = {"shape": [4], "sparse_crow_indices": [0, 2], "sparse_col_indices": [1, 3], "data": [5.0, 7.0]}
        assert tensor_data_to_list(sparse) == [0, 5.0, 0, 7.0]


def test_decode_sample_request_defaults():
    decoded = decode_sample_request(
        {"prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]}, "sampling_params": {"max_tokens": 1}}
    )
    assert decoded["num_samples"] == 1
    assert decoded["prompt_tokens"] == [1, 2]
    assert decoded["topk_prompt_logprobs"] == 0


class TestRenderResult:
    def test_forward_backward_renders_per_datum_records(self):
        rendered = render_result({"op": "forward_backward", "outputs": [{"loss": 2.0, "logprobs": [0.1, 0.2]}]})
        assert rendered["loss_fn_output_type"] == "ArrayRecord"
        assert rendered["metrics"] == {"loss:sum": 2.0}
        assert rendered["loss_fn_outputs"][0]["logprobs"] == {"dtype": "float32", "shape": [2], "data": [0.1, 0.2]}
