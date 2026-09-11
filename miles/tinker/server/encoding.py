"""Translate SDK JSON requests and results.

Datums encode input x and explicit target labels t as x + [t[-1]],
so each output scores logprob(t[i] | x[0..i])."""

from miles.tinker.core.types import LOSS_INPUT_KEYS, UserInputError

# materialized at the boundary so core and the executor can require every key
ADAM_PARAM_DEFAULTS = {
    "learning_rate": 1e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-12,
    "weight_decay": 0.0,
    "grad_clip_norm": 0.0,
}


def decode_command(op: str, payload: dict) -> tuple[str, dict]:
    """One JSON command body -> (op, internal payload)."""
    decoded = {"model_id": payload["model_id"], "seq_id": payload["seq_id"]}
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
    if op == "optim_step":
        return op, decoded | {"adam_params": materialize_adam_params(payload["adam_params"])}
    if op == "save_state":
        _reject_unsupported_save_options(payload)
        return op, decoded | {"name": payload.get("path"), "overwrite": bool(payload.get("overwrite", False))}
    if op == "load_state":
        return op, decoded | {"path": payload["path"], "optimizer": payload["optimizer"]}
    if op == "save_weights_for_sampler":
        _reject_unsupported_save_options(payload)
        return op, decoded | {"sampler_path": payload.get("path")}
    raise UserInputError(f"unknown command op {op!r}")


def _reject_unsupported_save_options(payload: dict) -> None:
    if payload.get("ttl_seconds") is not None:
        raise UserInputError("ttl_seconds is not supported: checkpoints on this gateway do not expire")
    if payload.get("user_metadata") is not None:
        raise UserInputError("user_metadata is not supported by this gateway")


def materialize_adam_params(raw: dict) -> dict:
    unknown = set(raw) - set(ADAM_PARAM_DEFAULTS)
    if unknown:
        raise UserInputError(f"unknown adam_params keys: {sorted(unknown)}")
    return {**ADAM_PARAM_DEFAULTS, **raw}


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
    targets = [int(t) for t in inputs["target_tokens"]]
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
    if tensor_data.get("sparse_crow_indices") is not None:
        return _dense_from_csr(tensor_data)
    data = tensor_data.get("data")
    if data is None:
        raise UserInputError("TensorData without data")
    return list(data)


def _dense_from_csr(tensor_data: dict) -> list:
    (length,) = tensor_data["shape"]
    assert len(tensor_data["sparse_crow_indices"]) == 2, "1-D CSR expected"
    dense = [0] * length
    for col, value in zip(tensor_data["sparse_col_indices"], tensor_data["data"], strict=True):
        dense[col] = value
    return dense


def decode_sample_request(payload: dict) -> dict:
    return {
        "model_path": payload.get("model_path"),
        "base_model": payload.get("base_model"),
        "sampling_session_id": payload.get("sampling_session_id"),
        "seq_id": payload.get("seq_id"),
        "num_samples": payload.get("num_samples", 1),
        "prompt_tokens": model_input_tokens(payload["prompt"]),
        "sampling_params": payload.get("sampling_params") or {},
        "prompt_logprobs": bool(payload.get("prompt_logprobs")),
        "topk_prompt_logprobs": payload.get("topk_prompt_logprobs", 0) or 0,
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
        for key in ("prompt_logprobs", "topk_prompt_logprobs"):
            if result.get(key) is not None:
                rendered[key] = result[key]
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
    if op == "load_state":
        return {"type": "load_weights"}
    if op == "optim_step":
        return {"type": "optim_step", "metrics": result["metrics"]}
    raise AssertionError(f"unrenderable result op {op!r}")


def _tensor_json(values: list[float]) -> dict:
    return {"dtype": "float32", "shape": [len(values)], "data": values}
