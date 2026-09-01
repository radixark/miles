"""Decode an official ForwardBackwardInput into the existing neutral Trainer input, no second schema."""

from __future__ import annotations

from typing import Any

import numpy as np

from miles.tinker import codec
from miles.tinker.losses import validate_loss_inputs, validate_loss_spec


def _tensor_values(tensor: dict[str, Any] | list) -> list:
    """Flatten one wire TensorData (or bare list) into a python list."""
    if isinstance(tensor, list):
        return tensor
    data = tensor.get("data")
    if data is None:
        raise ValueError("TensorData without dense data is not supported")
    return np.asarray(data).reshape(-1).tolist()


def decode_forward_backward_input(wire: dict[str, Any], *, identity: Any, request_id: str) -> dict[str, Any]:
    """Official wire dict -> existing rollout_data keys + plain loss spec + ownership sideband."""
    loss_fn, loss_fn_config = validate_loss_spec(wire.get("loss_fn"), wire.get("loss_fn_config"))
    tokens: list[list[int]] = []
    total_lengths: list[int] = []
    response_lengths: list[int] = []
    loss_columns: dict[str, list[list]] = {}
    owners: list[tuple[str, int]] = []
    data = wire.get("data") or []
    if not data:
        raise ValueError("forward_backward_input.data must not be empty")
    for row_index, datum in enumerate(data):
        inputs = datum.get("loss_fn_inputs") or {}
        validate_loss_inputs(loss_fn, row_index, inputs)
        prompt_tokens = codec.prompt_tokens_from_wire(datum["model_input"])
        target_tokens = [int(t) for t in _tensor_values(inputs["target_tokens"])]
        tokens.append(prompt_tokens + target_tokens)
        total_lengths.append(len(prompt_tokens) + len(target_tokens))
        response_lengths.append(len(target_tokens))
        for name, tensor in inputs.items():
            loss_columns.setdefault(name, []).append(_tensor_values(tensor))
        owners.append((request_id, row_index))
    for name, column in loss_columns.items():
        if len(column) != len(data):
            raise ValueError(f"loss_fn_inputs[{name!r}] is missing on some datums")
    rollout_data = {
        "tokens": tokens,
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
        **{name: column for name, column in loss_columns.items() if name != "target_tokens"},
    }
    return {
        "identity": identity,
        "rollout_data": rollout_data,
        "request_loss_fn": loss_fn,
        "request_loss_fn_config": loss_fn_config or None,
        "owners": owners,
    }
