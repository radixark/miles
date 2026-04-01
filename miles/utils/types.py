from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

# Decoded tensor, or raw safetensor blob before load (e.g. one tensor per file or one combined file).
SafetensorOrTensor = torch.Tensor | bytes


@dataclass
class Sample:
    """The sample generated"""

    group_index: int | None = None
    index: int | None = None
    # reproducibility
    seed: int | None = None
    # correlation id from rollout engine (e.g. UUID string)
    request_id: str | None = None
    # prompt
    prompt: str | list[dict[str, str]] = ""
    # LLM text continuation (often set at runtime if not deserialized from dict)
    response: str = ""
    generated_output: str | SafetensorOrTensor | None = None
    label: str | None = None
    reward: float | dict[str, Any] | None = None
    rollout_log_probs: list[float] | SafetensorOrTensor | None = None
    rollout_routed_experts: list[list[int]] | None = None  # Routed experts from rollout engine
    trajectory_latents: dict[str, Any] | SafetensorOrTensor | None = None
    trajectory_timesteps: dict[str, Any] | SafetensorOrTensor | None = None
    # Nested tensor leaves: decoded ``torch.Tensor`` or safetensor ``bytes`` (keys match engine, e.g. rollout_variance_noises)
    rollout_debug_tensors: dict[str, Any] | None = None
    # Nested structure; tensor leaves are ``torch.Tensor`` or safetensor ``bytes``
    denoising_env: dict[str, Any] | None = None
    inference_time_s: float | None = None
    peak_memory_mb: float | None = None
    weight_versions: list[str] = field(default_factory=list)

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        # Indicates a recoverable or non-critical failure during generation (e.g., tool call failure,
        # external API error, parsing error). Unlike ABORTED, FAILED samples may still contain partial
        # valid output and can be retried or handled gracefully.
        FAILED = "failed"

    status: Status = Status.PENDING

    metadata: dict = field(default_factory=dict)
    # metadata used during training, e.g., what loss to use for this sample.
    train_metadata: dict | None = None

    non_generation_time: float = 0.0  # time spent in non-generation steps

    def to_dict(self):
        value = self.__dict__.copy()
        value["status"] = self.status.value
        return value

    @staticmethod
    def from_dict(data: dict):
        data = dict(data)
        data["status"] = Sample.Status(data["status"])
        field_names = set(Sample.__dataclass_fields__.keys())
        init_data = {k: v for k, v in data.items() if k in field_names}
        sample = Sample(**init_data)

        for key, value in data.items():
            if key not in field_names:
                setattr(sample, key, value)

        return sample

    def get_reward_value(self, args) -> float:
        return self.reward if not args.reward_key else self.reward[args.reward_key]

    def update_from_meta_info(self, args, meta_info: dict):
        """
        Update the sample with new information from meta_info returned by the rollout engine.
        And extract
        """

        if "weight_version" in meta_info:
            self.weight_versions.append(meta_info["weight_version"])

        match meta_info["finish_reason"]["type"]:
            case "length":
                self.status = Sample.Status.TRUNCATED
            case "abort":
                self.status = Sample.Status.ABORTED
            case "stop":
                self.status = Sample.Status.COMPLETED


@dataclass(frozen=True)
class ParamInfo:
    name: str
    dtype: torch.dtype
    shape: torch.Size
    attrs: dict
    size: int
    src_rank: int