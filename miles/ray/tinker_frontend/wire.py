"""Wire models of the tinker SDK's REST protocol (server side).

Mirrors the request shapes ``tinker==0.24.1`` actually POSTs (verified from
the wheel source and captured traffic, not from documentation). Requests are
parsed permissively (``extra="ignore"``) so additive SDK fields never break
the server; everything the backend relies on is validated explicitly in the
translation layer. Responses are plain dicts built by the service — the SDK
deserializes JSON terminal results against its own pydantic models, so the
literal ``type`` discriminators below must match its expectations exactly.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict

TINKER_SDK_VERSION_PIN = "0.24.1"

# Flags returned from /api/v1/client/config. They steer the 0.24.1 SDK onto
# the pure-JSON protocol this frontend implements:
# - proto_write_fwdbwd=False keeps forward_backward on JSON (the wheel's own
#   default) and forward on the legacy JSON /api/v1/forward route;
# - fwd_via_fwdbwd must then also be False (the SDK asserts forward_only
#   requires the proto path);
# - parallel_fwdbwd_chunks=True lets the SDK post fwdbwd chunks concurrently,
#   first chunk last — exactly the out-of-order arrival the backend ledger
#   gap-buffers by design;
# - use_pyqwest_transport=False keeps the SDK on the plain httpx transport.
CLIENT_CONFIG_FLAGS = {
    "pjwt_auth_enabled": False,
    "credential_default_source": "api_key",
    "parallel_fwdbwd_chunks": True,
    "proto_write_fwdbwd": False,
    "proto_compress_fwdbwd": False,
    "fwd_via_fwdbwd": False,
    "use_pyqwest_transport": False,
    "create_model_via_load_weights": False,
    "sample_no_retries": False,
    "sample_max_concurrent_requests": 64,
}


class WireModel(BaseModel):
    # protected_namespaces cleared: the protocol is full of model_* fields.
    model_config = ConfigDict(extra="ignore", protected_namespaces=())


class CreateSessionRequest(WireModel):
    tags: list[str] = []
    user_metadata: dict[str, Any] | None = None
    sdk_version: str = ""
    project_id: str | None = None


class SessionHeartbeatRequest(WireModel):
    session_id: str


class ClientConfigRequest(WireModel):
    sdk_version: str = ""


class LoraConfig(WireModel):
    rank: int
    seed: int | None = None
    train_unembed: bool = True
    train_mlp: bool = True
    train_attn: bool = True


class CreateModelRequest(WireModel):
    session_id: str
    model_seq_id: int
    base_model: str
    user_metadata: dict[str, Any] | None = None
    lora_config: LoraConfig | None = None


class GetInfoRequest(WireModel):
    model_id: str


class UnloadModelRequest(WireModel):
    model_id: str


class TensorData(WireModel):
    data: list[int | float]
    dtype: str = "float32"
    shape: list[int] | None = None
    sparse_crow_indices: list[int] | None = None
    sparse_col_indices: list[int] | None = None


class ModelInputChunk(WireModel):
    # Non-text chunk types (image, dmel, ...) carry other fields; the type
    # tag alone is enough to reject them at the boundary.
    type: str = "encoded_text"
    tokens: list[int] = []


class ModelInput(WireModel):
    chunks: list[ModelInputChunk]


class Datum(WireModel):
    model_input: ModelInput
    loss_fn_inputs: dict[str, TensorData]


class ForwardBackwardInput(WireModel):
    data: list[Datum]
    loss_fn: str
    loss_fn_config: dict[str, float] | None = None


class ForwardBackwardRequest(WireModel):
    forward_backward_input: ForwardBackwardInput
    model_id: str
    seq_id: int | None = None


class ForwardRequest(WireModel):
    forward_input: ForwardBackwardInput
    model_id: str
    seq_id: int | None = None


class AdamParams(WireModel):
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-12
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0


class OptimStepRequest(WireModel):
    adam_params: AdamParams
    model_id: str
    seq_id: int | None = None


class SaveWeightsRequest(WireModel):
    model_id: str
    path: str | None = None
    seq_id: int | None = None
    ttl_seconds: int | None = None
    overwrite: bool = False


class LoadWeightsRequest(WireModel):
    model_id: str | None = None
    seq_id: int | None = None
    session_id: str | None = None
    model_seq_id: int | None = None
    base_model: str | None = None
    user_metadata: dict[str, Any] | None = None
    path: str
    optimizer: bool
    weights_access_token: str | None = None


class SaveWeightsForSamplerRequest(WireModel):
    model_id: str
    path: str | None = None
    sampling_session_seq_id: int | None = None
    seq_id: int | None = None
    ttl_seconds: int | None = None


class WeightsInfoRequest(WireModel):
    tinker_path: str


class CreateSamplingSessionRequest(WireModel):
    session_id: str
    sampling_session_seq_id: int
    base_model: str | None = None
    model_path: str | None = None


class SamplingParams(WireModel):
    max_tokens: int | None = None
    seed: int | None = None
    stop: str | list[str] | list[int] | None = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0


class SampleRequest(WireModel):
    num_samples: int = 1
    prompt: ModelInput
    sampling_params: SamplingParams
    base_model: str | None = None
    model_path: str | None = None
    sampling_session_id: str | None = None
    seq_id: int | None = None
    prompt_logprobs: bool | None = None
    topk_prompt_logprobs: int = 0


class FutureRetrieveRequest(WireModel):
    request_id: str
    allow_metadata_only: bool = False
    model_id: str | None = None


def untyped_future(request_id: str, model_id: str | None = None) -> dict:
    body: dict = {"request_id": request_id}
    if model_id is not None:
        body["model_id"] = model_id
    return body


def try_again(queue_state: str = "active", reason: str | None = None) -> dict:
    body: dict = {"type": "try_again", "queue_state": queue_state}
    if reason is not None:
        body["queue_state_reason"] = reason
    return body


def terminal_failure(error: str, category: str = "user") -> dict:
    # RequestErrorCategory on the SDK side accepts exactly unknown|server|user.
    if category not in ("unknown", "server", "user"):
        category = "unknown"
    return {"error": error, "category": category}
