"""Pydantic models for the public Tinker JSON wire protocol.

The service intentionally keeps these models local instead of depending on the
Tinker SDK at runtime. This lets a Miles server accept multiple SDK releases
while the SDK remains an optional client-side dependency.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class WireModel(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())


class ClientConfigRequest(StrictModel):
    sdk_version: str


class CreateSessionRequest(StrictModel):
    tags: list[str]
    user_metadata: dict[str, Any] | None
    sdk_version: str
    project_id: str | None = None
    type: Literal["create_session"] = "create_session"


class SessionHeartbeatRequest(StrictModel):
    session_id: str
    type: Literal["session_heartbeat"] = "session_heartbeat"


class LoraConfig(StrictModel):
    rank: int
    seed: int | None = None
    train_unembed: bool = True
    train_mlp: bool = True
    train_attn: bool = True


class CreateModelRequest(StrictModel):
    session_id: str
    model_seq_id: int
    base_model: str
    user_metadata: dict[str, Any] | None = None
    lora_config: LoraConfig | None = None
    type: Literal["create_model"] = "create_model"


class GetInfoRequest(StrictModel):
    model_id: str
    type: Literal["get_info"] = "get_info"


class UnloadModelRequest(StrictModel):
    model_id: str
    type: Literal["unload_model"] = "unload_model"


class TensorData(StrictModel):
    data: list[int] | list[float]
    dtype: Literal["int64", "float32"]
    shape: list[int] | None = None
    sparse_crow_indices: list[int] | None = None
    sparse_col_indices: list[int] | None = None

    @model_validator(mode="after")
    def validate_sparse_fields(self) -> TensorData:
        sparse_fields = (self.sparse_crow_indices, self.sparse_col_indices)
        if (sparse_fields[0] is None) != (sparse_fields[1] is None):
            raise ValueError("sparse_crow_indices and sparse_col_indices must be set together")
        if sparse_fields[0] is not None and (self.shape is None or len(self.shape) != 2):
            raise ValueError("sparse TensorData requires a two-dimensional shape")
        return self


class ModelInputChunk(BaseModel):
    """A permissive chunk envelope.

    Encoded text is executable today. Keeping other official chunk payloads
    parseable lets the server return a typed user error from the future instead
    of failing request decoding with an opaque 422.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    tokens: list[int] | None = None


class ModelInput(StrictModel):
    chunks: list[ModelInputChunk]


class Datum(StrictModel):
    loss_fn_inputs: dict[str, TensorData]
    model_input: ModelInput


LossFnType = Literal["cross_entropy", "importance_sampling", "ppo", "cispo", "dro"]


class ForwardBackwardInput(StrictModel):
    data: list[Datum]
    loss_fn: LossFnType
    loss_fn_config: dict[str, float] | None = None


class ForwardRequest(StrictModel):
    forward_input: ForwardBackwardInput
    model_id: str
    seq_id: int | None = None


class ForwardBackwardRequest(StrictModel):
    forward_backward_input: ForwardBackwardInput
    model_id: str
    seq_id: int | None = None


class AdamParams(StrictModel):
    learning_rate: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-12
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0


class OptimStepRequest(StrictModel):
    adam_params: AdamParams
    model_id: str
    seq_id: int | None = None
    type: Literal["optim_step"] = "optim_step"


class SaveWeightsRequest(StrictModel):
    model_id: str
    path: str | None = None
    seq_id: int | None = None
    ttl_seconds: int | None = None
    overwrite: bool = False
    type: Literal["save_weights"] = "save_weights"


class LoadWeightsRequest(StrictModel):
    model_id: str
    path: str
    optimizer: bool
    seq_id: int | None = None
    weights_access_token: str | None = None
    type: Literal["load_weights"] = "load_weights"


class SaveWeightsForSamplerRequest(StrictModel):
    model_id: str
    path: str | None = None
    sampling_session_seq_id: int | None = None
    seq_id: int | None = None
    ttl_seconds: int | None = None
    type: Literal["save_weights_for_sampler"] = "save_weights_for_sampler"


class CreateSamplingSessionRequest(StrictModel):
    session_id: str
    sampling_session_seq_id: int
    base_model: str | None = None
    model_path: str | None = None
    type: Literal["create_sampling_session"] = "create_sampling_session"

    @model_validator(mode="after")
    def validate_source(self) -> CreateSamplingSessionRequest:
        if (self.base_model is None) == (self.model_path is None):
            raise ValueError("exactly one of base_model or model_path must be provided")
        return self


class SamplingParams(StrictModel):
    max_tokens: int | None = None
    seed: int | None = None
    stop: str | list[str] | list[int] | None = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0


class SampleRequest(StrictModel):
    num_samples: int = 1
    prompt: ModelInput
    sampling_params: SamplingParams
    base_model: str | None = None
    model_path: str | None = None
    sampling_session_id: str | None = None
    seq_id: int | None = None
    prompt_logprobs: bool | None = None
    topk_prompt_logprobs: int = 0
    type: Literal["sample"] = "sample"


class FutureRetrieveRequest(StrictModel):
    request_id: str
    allow_metadata_only: bool = False


class WeightsInfoRequest(StrictModel):
    tinker_path: str


class TelemetryRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class UntypedAPIFuture(WireModel):
    request_id: str
    model_id: str | None = None


class TinkerError(RuntimeError):
    """An error that should be surfaced through a failed API future."""

    def __init__(self, message: str, category: Literal["unknown", "server", "user"] = "user") -> None:
        super().__init__(message)
        self.category = category


def encoded_tokens(model_input: ModelInput) -> list[int]:
    tokens: list[int] = []
    for chunk in model_input.chunks:
        if chunk.type != "encoded_text" or chunk.tokens is None:
            raise TinkerError(
                f"Miles currently accepts encoded_text chunks only; got chunk type {chunk.type!r}",
                category="user",
            )
        tokens.extend(chunk.tokens)
    if not tokens:
        raise TinkerError("model_input must contain at least one token", category="user")
    return tokens


def tensor_numel(tensor: TensorData) -> int:
    if tensor.shape is None:
        return len(tensor.data)
    size = 1
    for dim in tensor.shape:
        if dim < 0:
            raise TinkerError(f"TensorData shape contains a negative dimension: {tensor.shape}")
        size *= dim
    return size


def dense_tensor_data(tensor: TensorData) -> tuple[list[int] | list[float], list[int]]:
    """Decode dense or CSR TensorData without requiring NumPy on the API actor."""
    shape = tensor.shape or [len(tensor.data)]
    if tensor.sparse_crow_indices is None:
        if tensor_numel(tensor) != len(tensor.data):
            raise TinkerError(
                f"TensorData data length {len(tensor.data)} does not match shape {shape}",
                category="user",
            )
        return list(tensor.data), shape

    assert tensor.sparse_col_indices is not None
    rows, cols = shape
    if rows < 0 or cols < 0:
        raise TinkerError(f"sparse TensorData shape contains a negative dimension: {shape}", category="user")
    crow = tensor.sparse_crow_indices
    col = tensor.sparse_col_indices
    if len(crow) != rows + 1 or not crow or crow[0] != 0 or crow[-1] != len(tensor.data) or any(left > right for left, right in zip(crow, crow[1:], strict=False)) or any(offset < 0 or offset > len(tensor.data) for offset in crow):
        raise TinkerError("invalid CSR row pointers", category="user")
    if len(col) != len(tensor.data):
        raise TinkerError("invalid CSR column/value lengths", category="user")
    zero: int | float = 0 if tensor.dtype == "int64" else 0.0
    dense: list[int] | list[float] = [zero] * (rows * cols)
    for row in range(rows):
        for offset in range(crow[row], crow[row + 1]):
            column = col[offset]
            if not 0 <= column < cols:
                raise TinkerError(f"CSR column {column} is out of range for shape {shape}", category="user")
            dense[row * cols + column] = tensor.data[offset]
    return dense, shape


def tensor_payload(tensor: TensorData) -> dict[str, Any]:
    data, shape = dense_tensor_data(tensor)
    return {"data": data, "dtype": tensor.dtype, "shape": shape}


def forward_payload(value: ForwardBackwardInput) -> dict[str, Any]:
    """Normalize an SDK request into a JSON-serializable worker payload."""
    data = []
    for datum in value.data:
        data.append(
            {
                "tokens": encoded_tokens(datum.model_input),
                "loss_fn_inputs": {name: tensor_payload(tensor) for name, tensor in datum.loss_fn_inputs.items()},
            }
        )
    if not data:
        raise TinkerError("forward data must contain at least one Datum", category="user")
    return {
        "data": data,
        "loss_fn": value.loss_fn,
        "loss_fn_config": value.loss_fn_config or {},
    }
