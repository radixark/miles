"""Typed mirrors of the tinker==0.24.1 HTTP API payloads; validation only, no inference or training."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

LOSS_FNS = ("cross_entropy", "importance_sampling", "ppo", "cispo", "dro")


class WireModel(BaseModel):
    # extra="allow" keeps newer SDK fields from breaking parses; model_* field names need the empty namespace.
    model_config = ConfigDict(extra="allow", protected_namespaces=())


class TensorData(WireModel):
    """Flattened tensor; CSR fields present means `data` holds the non-zeros of a 2-D dense `shape`."""

    data: list[float]
    dtype: Literal["int64", "float32"]
    shape: list[int] | None = None
    sparse_crow_indices: list[int] | None = None
    sparse_col_indices: list[int] | None = None

    @property
    def is_sparse(self) -> bool:
        return self.sparse_crow_indices is not None


class EncodedTextChunk(WireModel):
    """The text-only backend rejects image/dmel chunks at conversion time."""

    type: Literal["encoded_text"]
    tokens: list[int]


class ModelInput(WireModel):
    chunks: list[EncodedTextChunk]

    def token_ids(self) -> list[int]:
        return [token for chunk in self.chunks for token in chunk.tokens]


class Datum(WireModel):
    model_input: ModelInput
    loss_fn_inputs: dict[str, TensorData]


class ForwardBackwardInput(WireModel):
    data: list[Datum]
    loss_fn: str
    loss_fn_config: dict[str, float] | None = None


class ForwardBackwardRequest(WireModel):
    model_id: str
    seq_id: int
    forward_backward_input: ForwardBackwardInput


class ForwardRequest(WireModel):
    """Same payload as forward_backward under the `forward_input` wrapper key."""

    model_id: str
    seq_id: int
    forward_input: ForwardBackwardInput


class SamplingParams(WireModel):
    max_tokens: int | None = None
    seed: int | None = None
    stop: str | list[str] | list[int] | None = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0


class SampleRequest(WireModel):
    sampling_session_id: str
    seq_id: int  # sampling seq ids are 0-based, unlike 1-based training ordinals
    prompt: ModelInput
    sampling_params: SamplingParams
    num_samples: int = 1
    prompt_logprobs: bool = False
    topk_prompt_logprobs: int = 0


class CreateSamplingSessionRequest(WireModel):
    session_id: str
    sampling_session_seq_id: int
    base_model: str | None = None
    model_path: str | None = None  # tinker://... when sampling published adapter weights


class SampledSequence(WireModel):
    stop_reason: Literal["length", "stop"]
    tokens: list[int]
    logprobs: list[float] | None = None


class SampleResponse(WireModel):
    type: Literal["sample"] = "sample"
    sequences: list[SampledSequence]
    prompt_logprobs: list[float | None] | None = None  # position 0 is null (no prior context)
    topk_prompt_logprobs: list | None = None
    prompt_cache_hit_tokens: int = 0


class FutureAck(WireModel):
    request_id: str
    model_id: str | None = None


class TryAgain(WireModel):
    type: Literal["try_again"] = "try_again"
    queue_state: Literal["active", "paused_capacity"] = "active"
    queue_state_reason: str | None = None


class TerminalError(WireModel):
    error: str
    category: Literal["unknown", "server", "user"]


class AdamParams(WireModel):
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-12
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0


class OptimStepRequest(WireModel):
    model_id: str
    seq_id: int
    adam_params: AdamParams
