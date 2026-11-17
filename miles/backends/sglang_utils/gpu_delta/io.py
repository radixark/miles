"""Feature-owned HTTP and IPC schemas for GPU-delta controls."""

from typing import Any, Literal

from sglang.srt.managers.io_struct import BaseReq, hook_custom_types

# The modes of the SGLang scheduler pause that apply runs; abort_all_requests aborts requests instead.
PauseMode = Literal["retract", "in_place"]


class ClearGpuDeltaStateReqInput(BaseReq, kw_only=True):
    pass


class ReleaseGpuDeltaCacheReqInput(BaseReq, kw_only=True):
    # Internal second phase, sent only after every rank has closed its resources.
    pass


class GetGpuDeltaInfoReqInput(BaseReq, kw_only=True):
    engine_id: str


class PrepareGpuDeltaReqInput(BaseReq, kw_only=True):
    session_id: str
    manifest_path: str
    manifest_sha256: str
    stream_id: str
    base_version: int
    target_version: int
    plan_digest: str
    participants: list[dict[str, Any]]
    validate_payload_hash: bool = True


class UpdateWeightsFromGpuDeltaReqInput(PrepareGpuDeltaReqInput, kw_only=True):
    release_state: bool = False
    pause_mode: PauseMode = "retract"
    abort_all_requests: bool = False
    weight_version: str | None = None


class ApplyGpuDeltaReqInput(BaseReq, kw_only=True):
    session_id: str
    pause_mode: PauseMode = "retract"


class AbortGpuDeltaReqInput(BaseReq, kw_only=True):
    session_id: str


class ResumeGpuDeltaReqInput(BaseReq, kw_only=True):
    session_id: str
    keep_pause: bool = False
    weight_version: str | None = None


class GpuDeltaReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str
    participants: list[dict[str, Any]]


# Register before tokenizer/scheduler/DP-controller receive loops start.
hook_custom_types(
    UpdateWeightsFromGpuDeltaReqInput,
    ClearGpuDeltaStateReqInput,
    ReleaseGpuDeltaCacheReqInput,
    GetGpuDeltaInfoReqInput,
    PrepareGpuDeltaReqInput,
    ApplyGpuDeltaReqInput,
    AbortGpuDeltaReqInput,
    ResumeGpuDeltaReqInput,
    GpuDeltaReqOutput,
)
