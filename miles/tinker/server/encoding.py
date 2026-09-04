"""SDK wire <-> gateway internal language, JSON half (proto_codec.py is the
binary half; both produce and consume the same internal shapes).

Decode errors are protocol violations and answer HTTP 400; size admission
happens later in core and fails the promise instead.

Sequence mapping: a datum's model_input tokens x[0..T-1] and target_tokens
t[0..T-1] must satisfy t[i] == x[i+1] for i < T-1 (standard next-token
supervision). The encoded row is x + [t[-1]] with T supervised positions, so
the trainer scores exactly logprob(t[i] | x[0..i]).
"""

from miles.tinker.core.types import UserInputError

# materialized at the boundary so core and the executor can require every key
ADAM_PARAM_DEFAULTS = {
    "learning_rate": 1e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-12,
    "weight_decay": 0.0,
    "grad_clip_norm": 0.0,
}

# wire loss_fn_inputs key -> internal row key
INPUT_ROW_KEYS = {"weights": "weights", "advantages": "advantages", "logprobs": "sampling_logprobs"}


def decode_command(kind: str, payload: dict) -> tuple[str, dict]:
    """One JSON command body -> (kind, internal payload)."""
    decoded = {"model_id": payload["model_id"], "seq_id": payload["seq_id"]}
    if kind == "forward_backward":
        fb_input = payload["forward_backward_input"]
        datums = [
            (model_input_tokens(datum["model_input"]), _decode_inputs(datum["loss_fn_inputs"]))
            for datum in fb_input["data"]
        ]
        decoded |= {
            "rows": [build_row(tokens, inputs, i) for i, (tokens, inputs) in enumerate(datums)],
            "loss_fn": fb_input["loss_fn"],
            "loss_fn_config": fb_input.get("loss_fn_config") or {},
        }
        return ("forward_only" if payload.get("forward_only") else kind), decoded
    if kind == "optim_step":
        return kind, decoded | {"adam_params": materialize_adam_params(payload["adam_params"])}
    if kind == "save_state":
        return kind, decoded | {"name": payload.get("path")}
    if kind == "load_state":
        return kind, decoded | {"path": payload["path"], "optimizer": payload["optimizer"]}
    if kind == "save_weights_for_sampler":
        return kind, decoded
    raise UserInputError(f"unknown command kind {kind!r}")


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


def build_row(input_tokens: list[int], inputs: dict[str, list], index: int) -> dict:
    """One decoded datum (token list + loss_fn_inputs lists) -> internal row."""
    targets = [int(t) for t in inputs["target_tokens"]]
    if len(targets) != len(input_tokens):
        raise UserInputError(
            f"datum {index}: target_tokens length {len(targets)} != model_input length {len(input_tokens)}"
        )
    if targets[:-1] != input_tokens[1:]:
        raise UserInputError(
            f"datum {index}: target_tokens must be model_input shifted by one (next-token supervision)"
        )
    row = {"tokens": input_tokens + targets[-1:], "target_len": len(targets)}
    for wire_key, row_key in INPUT_ROW_KEYS.items():
        if wire_key in inputs:
            row[row_key] = [float(value) for value in inputs[wire_key]]
    return row


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
        "sampling_session_id": payload.get("sampling_session_id"),
        "num_samples": payload.get("num_samples", 1),
        "prompt_tokens": model_input_tokens(payload["prompt"]),
        "sampling_params": payload.get("sampling_params") or {},
        "prompt_logprobs": bool(payload.get("prompt_logprobs")),
        "topk_prompt_logprobs": payload.get("topk_prompt_logprobs", 0) or 0,
    }


# -------- result rendering (JSON; proto_codec renders the binary forms) --------


def render_result(result: dict) -> dict:
    kind = result["kind"]
    if kind in ("forward_backward", "forward_only"):
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
    if kind == "sample":
        rendered = {"type": "sample", "sequences": result["sequences"]}
        for key in ("prompt_logprobs", "topk_prompt_logprobs"):
            if result.get(key) is not None:
                rendered[key] = result[key]
        return rendered
    if kind == "create_model":
        return {"type": "create_model", "model_id": result["model_id"]}
    if kind == "save_state":
        return {"type": "save_weights", "path": result["path"]}
    if kind == "save_weights_for_sampler":
        return {"type": "save_weights_for_sampler", "path": result["path"]}
    if kind == "load_state":
        return {"type": "load_weights"}
    if kind == "optim_step":
        return {"type": "optim_step", "metrics": result["metrics"]}
    raise AssertionError(f"unrenderable result kind {kind!r}")


def _tensor_json(values: list[float]) -> dict:
    return {"dtype": "float32", "shape": [len(values)], "data": values}
