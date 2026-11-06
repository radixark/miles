"""JSON decoding validates datum inputs and renders SDK result shapes."""

import pyrage
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from miles.tinker.core.types import UserInputError
from miles.tinker.server.encoding import (
    assert_installed_sdk_version,
    build_datum,
    decode_command,
    decode_sample_request,
    render_result,
    tensor_data_to_list,
    validate_create_model,
)
from tinker import types as tinker_types


def test_the_installed_sdk_is_the_pinned_one():
    assert_installed_sdk_version()


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
        _, decoded = self._decode(datum)
        assert "length" in decoded["validation_error"]

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
        assert set(decoded["adam_params"]) == set(tinker_types.AdamParams().model_dump())

    def test_unknown_keys_are_rejected(self):
        """The SDK's AdamParams model is the contract; a key it lacks must not be silently dropped."""
        _, decoded = decode_command("optim_step", {"model_id": "m", "seq_id": 1, "adam_params": {"momentum": 0.9}})
        assert "momentum" in decoded["validation_error"]

    @pytest.mark.parametrize(
        "params",
        [{"optimizer_params": {"type": "dimuon", "learning_rate": 1e-3}}, {"adam_params": {"type": "dimuon"}}],
        ids=["optimizer_params", "typed_adam_params"],
    )
    def test_a_non_adam_step_is_rejected(self, params):
        _, decoded = decode_command("optim_step", {"model_id": "m", "seq_id": 1, **params})
        assert "this gateway serves only plain Adam (AdamParams)" in decoded["validation_error"]

    def test_the_optim_params_alias_is_read(self):
        payload = {"model_id": "m", "seq_id": 1, "optim_params": {"learning_rate": 3e-4}}
        _, decoded = decode_command("optim_step", payload)
        assert decoded["adam_params"]["learning_rate"] == 3e-4


@pytest.mark.parametrize(
    "optimizer_config", [{"type": "dimuon"}, {"type": "adamw", "extra": 1}], ids=["dimuon", "adamw_with_extras"]
)
def test_a_non_adam_model_is_rejected(optimizer_config):
    payload = {"base_model": "base", "session_id": "s", "model_seq_id": 0, "optimizer_config": optimizer_config}
    with pytest.raises(UserInputError, match=r"serves only plain Adam \(AdamOptimizerConfig\)"):
        validate_create_model(payload)


def test_load_state_rejects_an_optimizer_config():
    """optimizer_config on load_weights only configures create-via-load, which this gateway does not serve."""
    payload = {"model_id": "m", "seq_id": 1, "path": "tinker://x/weights/y", "optimizer": False}
    _, decoded = decode_command("load_state", payload | {"optimizer_config": {"type": "adamw"}})
    assert "does not create models via load_weights" in decoded["validation_error"]


def test_lax_wire_values_are_read_as_the_sdk_parses_them():
    """The SDK models accept "false" for a bool; reading the raw payload would treat it as true."""
    _, saved = decode_command("save_state", {"model_id": "m", "seq_id": 1, "path": "ck", "overwrite": "false"})
    assert saved["overwrite"] is False
    payload = {"model_id": "m", "seq_id": 1, "path": "tinker://x/weights/y", "optimizer": "false"}
    _, loaded = decode_command("load_state", payload)
    assert loaded["optimizer"] is False
    sample = {"prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]}, "sampling_params": {"max_tokens": 1}}
    assert decode_sample_request(sample | {"prompt_logprobs": "false"})["prompt_logprobs"] is False
    lax_params = {"max_tokens": 1, "temperature": "0.5", "top_k": "5", "bogus": 1}
    params = decode_sample_request(sample | {"sampling_params": lax_params})["sampling_params"]
    assert (params["temperature"], params["top_k"], "bogus" in params) == (0.5, 5, False)


def test_an_off_model_key_is_rejected():
    _, decoded = decode_command("save_state", {"model_id": "m", "seq_id": 1, "bogus": 1})
    assert "bogus" in decoded["validation_error"]


class TestTensorData:
    def test_dense_tensor_data(self):
        assert tensor_data_to_list({"data": [1.0, 2.0], "shape": [2]}) == [1.0, 2.0]

    def test_csr_expands_to_dense(self):
        sparse = {"shape": [4], "sparse_crow_indices": [0, 2], "sparse_col_indices": [1, 3], "data": [5.0, 7.0]}
        assert tensor_data_to_list(sparse) == [0, 5.0, 0, 7.0]


def test_the_sdk_sample_body_decodes():
    """The SDK sends every sampling option explicitly at its default."""
    request = tinker_types.SampleRequest(
        prompt=tinker_types.ModelInput.from_ints([1, 2]),
        sampling_params=tinker_types.SamplingParams(max_tokens=1),
        record_stability_info=False,
    )
    payload = request.model_dump(exclude_none=True, mode="json")
    assert {"topk_sample_logprobs", "prompt_alt_tokens_k", "record_stability_info"} <= set(payload)
    assert decode_sample_request(payload)["prompt_tokens"] == [1, 2]


def test_decode_sample_request_defaults():
    decoded = decode_sample_request(
        {"prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]}, "sampling_params": {"max_tokens": 1}}
    )
    assert decoded["num_samples"] == 1
    assert decoded["prompt_tokens"] == [1, 2]
    assert decoded["topk_prompt_logprobs"] == 0


@pytest.mark.parametrize(
    "option",
    [
        {"topk_sample_logprobs": 2},
        {"target_prompt_logprobs": {"data": [3], "dtype": "int64", "shape": [1, 1]}},
        {"prompt_alt_tokens_k": 2},
        {"prompt_logprobs_last_n": 1, "prompt_logprobs": True},
        {"record_stability_info": True},
    ],
    ids=lambda option: next(iter(option)),
)
def test_unsupported_sample_options_are_rejected(option):
    payload = {
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
        "sampling_params": {"max_tokens": 1},
    }
    with pytest.raises(UserInputError, match=f"{next(iter(option))} is not supported"):
        decode_sample_request(payload | option)


class TestRenderResult:
    def test_forward_backward_renders_per_datum_records(self):
        rendered = render_result({"op": "forward_backward", "outputs": [{"loss": 2.0, "logprobs": [0.1, 0.2]}]})
        assert rendered["loss_fn_output_type"] == "ArrayRecord"
        assert rendered["metrics"] == {"loss:sum": 2.0}
        assert rendered["loss_fn_outputs"][0]["logprobs"] == {"dtype": "float32", "shape": [2], "data": [0.1, 0.2]}


_RECIPIENT = str(pyrage.x25519.Identity.generate().to_public())
_SSH_RECIPIENT = (
    ed25519.Ed25519PrivateKey.generate().public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH).decode()
)


@pytest.mark.parametrize(
    "recipients,error",
    [
        (["ssh-ed25519 AAA"], "not an age recipient"),
        (["age1notbech32"], "not an age recipient"),
        ([_RECIPIENT] * 11, "at most 10"),
    ],
)
def test_bad_age_recipients_are_refused_at_decode(recipients, error):
    payload = {"model_id": "m", "seq_id": 1, "path": "export", "age_encryption_recipients": recipients}
    _, decoded = decode_command("save_weights_external", payload)
    assert error in decoded["validation_error"]
    accepted = [_RECIPIENT] * 9 + [_SSH_RECIPIENT]
    _, decoded = decode_command("save_weights_external", payload | {"age_encryption_recipients": accepted})
    assert "validation_error" not in decoded and decoded["age_encryption_recipients"] == accepted
