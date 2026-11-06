"""Translate SDK JSON requests and results.

Datums encode input x and explicit target labels t as x + [t[-1]],
so each output scores logprob(t[i] | x[0..i])."""

import math

import pydantic

import tinker
from miles.tinker.core.input_validation import validate_save_options, validate_ttl_seconds
from miles.tinker.core.types import LOSS_INPUT_KEYS, UserInputError
from tinker import types as tinker_types
from tinker.types.topk_logprobs import MASK_LOGPROB

# the SDK release whose request models and semantics this gateway is written and tested against
TINKER_SDK_VERSION = "0.36.0"

# JSON-wire request models; forward_backward travels as protobuf and its datums
# are checked in build_datum instead
REQUEST_TYPES = {
    "optim_step": tinker_types.OptimStepRequest,
    "save_state": tinker_types.SaveWeightsRequest,
    "load_state": tinker_types.LoadWeightsRequest,
    "save_weights_for_sampler": tinker_types.SaveWeightsForSamplerRequest,
    "save_weights_external": tinker_types.SaveWeightsExternalRequest,
}


def validate_against_sdk(request_type, payload: dict):
    """The pinned SDK's request models are the wire contract; reject what they reject."""
    try:
        return request_type.model_validate(payload)
    except pydantic.ValidationError as error:
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        raise UserInputError(f"{location or 'request'}: {first['msg']}") from None


def assert_installed_sdk_version() -> None:
    if tinker.__version__ != TINKER_SDK_VERSION:
        raise RuntimeError(
            f"the gateway validates requests with tinker=={TINKER_SDK_VERSION}, found {tinker.__version__}; "
            f"pip install tinker=={TINKER_SDK_VERSION}"
        )


def validate_sdk_version(payload) -> None:
    """Other SDK versions send fields and semantics this gateway was not written for."""
    # compared before the strict model, which would reject a newer SDK's added fields with a less useful error
    client_version = payload.get("sdk_version") if isinstance(payload, dict) else None
    if client_version != TINKER_SDK_VERSION:
        raise UserInputError(
            f"Tinker SDK version mismatch: client is {client_version}, this gateway serves the "
            f"tinker=={TINKER_SDK_VERSION} wire schema; install that exact version"
        )
    validate_against_sdk(tinker_types.ClientConfigRequest, payload)


def validate_plain_adam(field: str, value, adam_type) -> None:
    if not isinstance(value, adam_type):
        raise UserInputError(
            f"{field} {value.model_dump()} is not supported: this gateway serves only plain Adam ({adam_type.__name__})"
        )


# the SDK's documented limit for age_encryption_recipients
MAX_AGE_RECIPIENTS = 10


def validate_age_recipients(recipients: list[str]) -> None:
    """Refuse a bad recipient before the export, not after the trainer wrote it."""
    if len(recipients) > MAX_AGE_RECIPIENTS:
        raise UserInputError(f"at most {MAX_AGE_RECIPIENTS} age recipients, got {len(recipients)}")
    if not recipients:
        return
    # imported here: only encrypted saves need it, and a gateway image installed with --no-deps may lack it
    import pyrage

    for recipient in recipients:
        try:
            # the trainer's export parses it the same way (adapter_exports.write_external_adapter)
            (pyrage.ssh if recipient.startswith("ssh-") else pyrage.x25519).Recipient.from_str(recipient)
        except pyrage.RecipientError as error:
            raise UserInputError(f"{recipient!r} is not an age recipient (age1... or ssh-...): {error}") from None


def validate_create_model(payload: dict) -> None:
    request = validate_against_sdk(tinker_types.CreateModelRequest, payload)
    validate_plain_adam("optimizer_config", request.optimizer_config, tinker_types.AdamOptimizerConfig)


def validate_sample_options(request) -> None:
    """Reject sampling options the gateway does not serve instead of sampling without them."""
    unsupported = {
        "topk_sample_logprobs": request.topk_sample_logprobs,
        "target_prompt_logprobs": request.target_prompt_logprobs is not None,
        "prompt_alt_tokens_k": request.prompt_alt_tokens_k,
        "prompt_logprobs_last_n": request.prompt_logprobs_last_n is not None,
        "record_stability_info": request.record_stability_info,
    }
    for field, requested in unsupported.items():
        if requested:
            raise UserInputError(f"{field} is not supported by this gateway")


def validate_create_sampling_session(payload: dict) -> None:
    validate_against_sdk(tinker_types.CreateSamplingSessionRequest, payload)


def decode_command(op: str, payload: dict) -> tuple[str, dict]:
    """Decode content errors into ordered failures when the envelope is identifiable."""
    assert op == "forward_backward" or op in REQUEST_TYPES, f"unhandled command op {op!r}"
    try:
        envelope = {"model_id": payload["model_id"], "seq_id": payload["seq_id"]}
    except KeyError as error:
        raise UserInputError(f"missing command envelope field {error.args[0]!r}") from None
    try:
        return _decode_command(op, payload, envelope)
    except (UserInputError, KeyError, TypeError, ValueError, IndexError) as error:
        if op == "forward_backward" and payload.get("forward_only"):
            op = "forward_only"
        return op, envelope | {"validation_error": str(error)}


def _decode_command(op: str, payload: dict, decoded: dict) -> tuple[str, dict]:
    if op == "forward_backward":
        fb_input = payload["forward_backward_input"]
        datums = [
            (model_input_tokens(datum["model_input"]), _decode_inputs(datum["loss_fn_inputs"]))
            for datum in fb_input["data"]
        ]
        decoded |= {
            "datums": [build_datum(tokens, inputs, i) for i, (tokens, inputs) in enumerate(datums)],
            "loss_fn": fb_input["loss_fn"],
            "loss_fn_config": fb_input.get("loss_fn_config") or {},
        }
        return ("forward_only" if payload.get("forward_only") else op), decoded
    request = validate_against_sdk(REQUEST_TYPES[op], payload)
    if op == "optim_step":
        validate_plain_adam("optim_params", request.optim_params, tinker_types.AdamParams)
        # model_dump fills every default so core and the executor can require every key
        return op, decoded | {"adam_params": request.optim_params.model_dump()}
    if op == "save_state":
        validate_save_options(request.ttl_seconds, request.user_metadata)
        return op, decoded | {"name": request.path, "overwrite": request.overwrite, "ttl_seconds": request.ttl_seconds}
    if op == "load_state":
        if request.optimizer_config is not None:
            raise UserInputError(
                "optimizer_config is not supported: this gateway does not create models via load_weights"
            )
        return op, decoded | {
            "path": request.path,
            "optimizer": request.optimizer,
            "weights_access_token": request.weights_access_token,
        }
    if op == "save_weights_for_sampler":
        validate_save_options(request.ttl_seconds, request.user_metadata)
        return op, decoded | {"sampler_path": request.path, "ttl_seconds": request.ttl_seconds}
    if op == "save_weights_external":
        validate_ttl_seconds(request.ttl_seconds)
        recipients = request.age_encryption_recipients or []
        validate_age_recipients(recipients)
        return op, decoded | {
            "name": request.path,
            "ttl_seconds": request.ttl_seconds,
            "age_encryption_recipients": recipients,
        }
    raise AssertionError(f"unhandled command op {op!r}")


def model_input_tokens(model_input: dict) -> list[int]:
    tokens: list[int] = []
    for chunk in model_input["chunks"]:
        if chunk.get("type") != "encoded_text":
            raise UserInputError(f"unsupported model_input chunk type: {chunk.get('type')}")
        tokens.extend(chunk["tokens"])
    return tokens


def build_datum(input_tokens: list[int], inputs: dict[str, list], index: int) -> dict:
    """One decoded datum (token list + loss_fn_inputs lists) -> internal datum."""
    unknown = set(inputs) - set(LOSS_INPUT_KEYS) - {"target_tokens"}
    if unknown:
        raise UserInputError(f"datum {index}: unknown loss_fn_inputs {sorted(unknown)}")
    for name, values in inputs.items():
        if any(isinstance(value, (list, tuple)) for value in values):
            raise UserInputError(
                f"datum {index}: loss_fn_inputs[{name!r}] must be 1-D; multi-target inputs are not supported"
            )
    targets = list(inputs["target_tokens"])
    if len(targets) != len(input_tokens):
        raise UserInputError(
            f"datum {index}: target_tokens length {len(targets)} != model_input length {len(input_tokens)}"
        )
    datum = {"tokens": input_tokens + targets[-1:], "target_len": len(targets), "target_tokens": targets}
    for wire_key, datum_key in LOSS_INPUT_KEYS.items():
        if wire_key in inputs:
            datum[datum_key] = [float(value) for value in inputs[wire_key]]
    return datum


def _decode_inputs(loss_fn_inputs: dict) -> dict[str, list]:
    return {name: tensor_data_to_list(value) for name, value in loss_fn_inputs.items()}


def tensor_data_to_list(tensor_data) -> list:
    if isinstance(tensor_data, list):
        return tensor_data
    if not isinstance(tensor_data, dict):
        raise UserInputError(f"expected TensorData, got {type(tensor_data).__name__}")
    if len(tensor_data.get("shape") or []) > 1:
        raise UserInputError("multi-target inputs are not supported; loss_fn_inputs must be 1-D")
    if tensor_data.get("sparse_crow_indices") is not None:
        return _dense_from_csr(tensor_data)
    data = tensor_data.get("data")
    if data is None:
        raise UserInputError("TensorData without data")
    return list(data)


def _dense_from_csr(tensor_data: dict) -> list:
    (length,) = tensor_data["shape"]
    if len(tensor_data["sparse_crow_indices"]) != 2:
        raise UserInputError("1-D CSR expected")
    dense = [0] * length
    for col, value in zip(tensor_data["sparse_col_indices"], tensor_data["data"], strict=True):
        dense[col] = value
    return dense


def decode_sample_request(payload: dict) -> dict:
    request = validate_against_sdk(tinker_types.SampleRequest, payload)
    validate_sample_options(request)
    return {
        "model_path": request.model_path,
        "base_model": request.base_model,
        "sampling_session_id": request.sampling_session_id,
        "seq_id": request.seq_id,
        "num_samples": request.num_samples,
        "prompt_tokens": model_input_tokens(payload["prompt"]),
        "sampling_params": request.sampling_params.model_dump(),
        "prompt_logprobs": bool(request.prompt_logprobs),
        "topk_prompt_logprobs": request.topk_prompt_logprobs,
    }


# -------- result rendering (JSON; proto_codec renders the binary forms) --------


def render_result(result: dict) -> dict:
    op = result["op"]
    if op in ("forward_backward", "forward_only"):
        outputs = result["outputs"]
        return {
            "type": "forward_backward",
            "loss_fn_output_type": "ArrayRecord",
            "loss_fn_outputs": [
                {"loss:sum": _tensor_json([output["loss"]]), "logprobs": _tensor_json(output["logprobs"])}
                for output in outputs
            ],
            "metrics": {"loss:sum": float(sum(output["loss"] for output in outputs))},
        }
    if op == "sample":
        rendered = {"type": "sample", "sequences": result["sequences"]}
        if result.get("prompt_logprobs") is not None:
            rendered["prompt_logprobs"] = [
                None if math.isnan(logprob) else logprob for logprob in result["prompt_logprobs"]
            ]
        if result.get("topk_prompt_logprobs") is not None:
            topk = result["topk_prompt_logprobs"]
            rendered["topk_prompt_logprobs"] = [
                [
                    (token_id, logprob)
                    for token_id, logprob in zip(ids, probs, strict=True)
                    if (token_id, logprob) != (0, MASK_LOGPROB)
                ]
                or None
                for ids, probs in zip(topk["token_ids"], topk["logprobs"], strict=True)
            ]
        return rendered
    if op == "create_model":
        return {"type": "create_model", "model_id": result["model_id"]}
    if op == "save_state":
        return {"type": "save_weights", "path": result["path"]}
    if op == "save_weights_for_sampler":
        rendered = {"type": "save_weights_for_sampler", "path": result["path"]}
        if "sampling_session_id" in result:
            rendered["sampling_session_id"] = result["sampling_session_id"]
        return rendered
    if op == "save_weights_external":
        return {"type": "save_weights_external", "path": result["path"], "size_bytes": result["size_bytes"]}
    if op == "load_state":
        return {"type": "load_weights"}
    if op == "optim_step":
        return {"type": "optim_step", "metrics": result["metrics"]}
    raise AssertionError(f"unrenderable result op {op!r}")


def _tensor_json(values: list[float]) -> dict:
    return {"dtype": "float32", "shape": [len(values)], "data": values}
