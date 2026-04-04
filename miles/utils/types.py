from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from miles.utils.diffusion_rollout_response import decode_tensor_base64


import torch


class LazyTensor(ABC):
    """Deferred load: **base64 ``str``** matching :func:`~miles.utils.diffusion_rollout_response.tensor_to_base64`.

    Public API: :meth:`resolve` → **CPU** :class:`torch.Tensor` (idempotent: decodes at most once, then returns cache).
    """

    def __init__(self) -> None:
        self.tensor: torch.Tensor | None = None

    def resolve(self) -> torch.Tensor:
        """Materialize to a CPU tensor; repeated calls return the same tensor."""
        if self.tensor is not None:
            self.tensor = self._resolve_tensor()
        return self.tensor
    
    @abstractmethod
    def _resolve_tensor(self) -> torch.Tensor:
        raise NotImplementedError


class SafetensorsBase64LazyTensor(LazyTensor):
    """Lazy tensor from rollout wire: base64-encoded safetensors blob (default tensor key ``\"t\"`` in :func:`~miles.utils.diffusion_rollout_response.decode_tensor_base64`)."""

    def __init__(self, b64: str) -> None:
        super().__init__()
        self.b64: str | None = b64

    def _resolve_tensor(self) -> torch.Tensor:
        if not self.b64:
            raise ValueError("SafetensorsBase64LazyTensor: b64 must be a non-empty str")
        return decode_tensor_base64(self.b64).detach().cpu()

# Tensor field: either deferred safetensors+b64 or already materialized (e.g. after ``resolve()``).
RolloutTensorRef = LazyTensor | torch.Tensor

@dataclass
class RolloutDebugTensors:
    rollout_variance_noises: RolloutTensorRef | None = None
    rollout_prev_sample_means: RolloutTensorRef | None = None
    rollout_noise_std_devs: RolloutTensorRef | None = None
    rollout_model_outputs: RolloutTensorRef | None = None


@dataclass
class CondKwargs:
    txt_seq_lens: list[int] | None = None
    freqs_cis: list[RolloutTensorRef] | None = None
    img_shapes: list[list[tuple[int, int, int]]] | None = None
    encoder_hidden_states: list[RolloutTensorRef] | None = None


@dataclass
class DenoisingEnv:
    image_kwargs: Any | None = None
    pos_cond_kwargs: CondKwargs | None = None
    neg_cond_kwargs: CondKwargs | None = None
    guidance: Any | None = None


@dataclass
class DiTTrajectory:
    latent_model_inputs: RolloutTensorRef | None = None
    timesteps: RolloutTensorRef | None = None


@dataclass
class Sample:
    """The sample generated.

    Diffusion image rollout: fill from sglang-diffusion ``POST /rollout/images`` via
    :meth:`from_rollout_image_response` or :meth:`apply_rollout_image_response` (see
    :mod:`miles.utils.diffusion_rollout_response`).

    Rollout tensors (``generated_output``, trajectory, log-probs, etc.) are **per-sample** shapes
    ``[T, ...]`` as serialized by the engine (no leading batch dimension).
    """

    group_index: int | None = None
    index: int | None = None
    # correlation id from rollout engine (e.g. UUID string)
    request_id: str | None = None
    # prompt
    prompt: str = ""
    # reproducibility
    seed: int | None = None
    # Lazy: :class:`SafetensorsBase64LazyTensor` (safetensors+b64 ``str``); eager: :class:`torch.Tensor` (shape ``[T, ...]``)
    generated_output: RolloutTensorRef | None = None
    rollout_log_probs: RolloutTensorRef | None = None
    rollout_debug_tensors: RolloutDebugTensors | None = None
    denoising_env: DenoisingEnv | None = None
    dit_trajectory: DiTTrajectory | None = None

    inference_time_s: float | None = None
    peak_memory_mb: float | None = None

    reward: dict[str, Any] | None = None
    weight_versions: list[str] = field(default_factory=list)

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        ABORTED = "aborted"
        # Indicates a recoverable or non-critical failure during generation (e.g., tool call failure,
        # external API error, parsing err"""  """or). Unlike ABORTED, FAILED samples may still contain partial
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
