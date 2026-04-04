"""Parse sglang-diffusion ``POST /rollout/images`` JSON into :class:`~miles.utils.types.Sample`."""

from __future__ import annotations

import base64
from typing import Any
from safetensors.torch import load, save
import torch
from miles.utils.types import (
    CondKwargs,
    DenoisingEnv,
    DiTTrajectory,
    LazyTensor,
    RolloutDebugTensors,
    Sample,
    SafetensorsBase64LazyTensor,
)

__all__ = [
    "apply_rollout_image_response",
    "as_lazy_tensor",
    "coerce_rollout_images_http_response",
    "decode_tensor_base64",
    "tensor_to_base64",
]

# Prefer these keys for mapping dict ``rollout_log_probs`` → ``Sample.rollout_log_probs``.
_ROLLOUT_LOG_PROB_PRIMARY_KEYS = ("log_prob", "log_probs", "total", "per_step")


def decode_tensor_base64(b64: str) -> torch.Tensor:
    """Deserialize base64 to CPU tensor (same wire format as inference: safetensors ``[\"t\"]``, else ``torch.load``)."""
    raw = base64.b64decode(b64.encode("ascii") if isinstance(b64, str) else b64)
    return load(raw)["t"]

def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Encode a CPU tensor as base64 safetensors (single key ``tensor_key``, default ``t``)."""
    tensor = tensor.detach().cpu()
    raw = save({"t": tensor})
    return base64.b64encode(raw).decode("ascii")


def as_lazy_tensor(value: Any) -> LazyTensor | None:
    """If ``value`` is a base64 string from JSON, wrap as lazy tensor; else pass through (HTTP bodies never contain ``torch.Tensor``)."""
    if value is None:
        return None
    if isinstance(value, str):
        return SafetensorsBase64LazyTensor(b64=value)
    raise TypeError(f"Cannot convert {type(value)} to LazyTensor")


def _parse_cond_kwargs(data: dict[str, Any] | None) -> CondKwargs | None:
    if not data:
        return None
    return CondKwargs(
        txt_seq_lens=data.get("txt_seq_lens"),
        freqs_cis=[as_lazy_tensor(x) for x in data.get("freqs_cis", [])],
        img_shapes=data.get("img_shapes"),
        encoder_hidden_states=[as_lazy_tensor(x) for x in data.get("encoder_hidden_states", [])],
    )


def _parse_denoising_env(data: dict[str, Any] | None) -> DenoisingEnv | None:
    if not data:
        return None
    return DenoisingEnv(
        image_kwargs=data.get("image_kwargs"),
        pos_cond_kwargs=_parse_cond_kwargs(data.get("pos_cond_kwargs")),
        neg_cond_kwargs=_parse_cond_kwargs(data.get("neg_cond_kwargs")),
        guidance=data.get("guidance"),
    )


def _parse_dit_trajectory(data: dict[str, Any] | None) -> DiTTrajectory | None:
    if not data:
        return None
    return DiTTrajectory(
        latent_model_inputs=as_lazy_tensor(data.get("latent_model_inputs")),
        timesteps=as_lazy_tensor(data.get("timesteps")),
    )


def _parse_rollout_debug_tensors(data: dict[str, Any] | None) -> RolloutDebugTensors | None:
    if not data:
        return None
    return RolloutDebugTensors(
        rollout_variance_noises=as_lazy_tensor(data.get("rollout_variance_noises")),
        rollout_prev_sample_means=as_lazy_tensor(data.get("rollout_prev_sample_means")),
        rollout_noise_std_devs=as_lazy_tensor(data.get("rollout_noise_std_devs")),
        rollout_model_outputs=as_lazy_tensor(data.get("rollout_model_outputs")),
    )


def apply_rollout_image_response(sample: Sample, body: dict[str, Any]) -> None:
    """Fill ``sample`` fields from one ``RolloutImageResponse``-shaped dict (per-sample tensors, no batch dim)."""
    sample.request_id = body.get("request_id") or sample.request_id
    if "prompt" in body:
        sample.prompt = str(body["prompt"])
    if "seed" in body:
        sample.seed = int(body["seed"])

    sample.generated_output = as_lazy_tensor(body.get("generated_output"))
    sample.rollout_log_probs = as_lazy_tensor(body.get("rollout_log_probs"))
    sample.rollout_debug_tensors = _parse_rollout_debug_tensors(body.get("rollout_debug_tensors"))
    sample.denoising_env = _parse_denoising_env(body.get("denoising_env"))
    sample.dit_trajectory = _parse_dit_trajectory(body.get("dit_trajectory"))

    if "inference_time_s" in body and body["inference_time_s"] is not None:
        sample.inference_time_s = float(body["inference_time_s"])
    if "peak_memory_mb" in body and body["peak_memory_mb"] is not None:
        sample.peak_memory_mb = float(body["peak_memory_mb"])
