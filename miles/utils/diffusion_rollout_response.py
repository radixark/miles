"""Parse sglang-diffusion ``POST /rollout/generate`` JSON into :class:`~miles.utils.types.Sample`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import ray
import torch

from miles.utils.types import (
    CondKwargs,
    DenoisingEnv,
    DiTTrajectory,
    RolloutDebugTensors,
    Sample,
    decode_tensor_base64,
)

__all__ = [
    "apply_rollout_image_response",
    "RolloutImageResponseParserActor",
]

# Prefer these keys for mapping dict ``rollout_log_probs`` → ``Sample.rollout_log_probs``.
_ROLLOUT_LOG_PROB_PRIMARY_KEYS = ("log_prob", "log_probs", "total", "per_step")


def _default_deserialize_func(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, str):
        return decode_tensor_base64(value).detach().cpu()
    if isinstance(value, dict) and value.get("__tensor__"):
        return decode_tensor_base64(value["data"]).detach().cpu()
    raise TypeError(f"Cannot deserialize {type(value)}")


def _deserialize_rollout_log_probs(
    value: Any,
    *,
    deserialize_func: Callable[[Any], torch.Tensor | None],
) -> torch.Tensor | None:
    assert isinstance(value, dict) and value.get("__tensor__") is not None
    return deserialize_func(value["data"]).detach().cpu()


def _parse_cond_kwargs(
    data: dict[str, Any] | None,
    *,
    deserialize_func: Callable[[Any], torch.Tensor | None],
) -> CondKwargs | None:
    if not data:
        return None
    return CondKwargs(
        txt_seq_lens=data.get("txt_seq_lens"),
        freqs_cis=[deserialize_func(x) for x in data.get("freqs_cis", [])],
        img_shapes=data.get("img_shapes"),
        encoder_hidden_states=[
            deserialize_func(x) for x in data.get("encoder_hidden_states", [])
        ],
    )


def _parse_denoising_env(
    data: dict[str, Any] | None,
    *,
    deserialize_func: Callable[[Any], torch.Tensor | None],
) -> DenoisingEnv | None:
    if not data:
        return None
    return DenoisingEnv(
        image_kwargs=data.get("image_kwargs"),
        pos_cond_kwargs=_parse_cond_kwargs(data.get("pos_cond_kwargs"), deserialize_func=deserialize_func),
        neg_cond_kwargs=_parse_cond_kwargs(data.get("neg_cond_kwargs"), deserialize_func=deserialize_func),
        guidance=data.get("guidance"),
    )


def _parse_dit_trajectory(
    data: dict[str, Any] | None,
    *,
    deserialize_func: Callable[[Any], torch.Tensor | None],
) -> DiTTrajectory | None:
    if not data:
        return None
    return DiTTrajectory(
        latents=deserialize_func(data.get("latents")),
        timesteps=deserialize_func(data.get("timesteps")),
    )


def _parse_rollout_debug_tensors(
    data: dict[str, Any] | None,
    *,
    deserialize_func: Callable[[Any], torch.Tensor | None],
) -> RolloutDebugTensors | None:
    if not data:
        return None
    return RolloutDebugTensors(
        rollout_variance_noises=deserialize_func(data.get("rollout_variance_noises")),
        rollout_prev_sample_means=deserialize_func(data.get("rollout_prev_sample_means")),
        rollout_noise_std_devs=deserialize_func(data.get("rollout_noise_std_devs")),
        rollout_model_outputs=deserialize_func(data.get("rollout_model_outputs")),
    )


def apply_rollout_image_response(
    sample: Sample,
    body: dict[str, Any],
    *,
    deserialize_func: Callable[[Any], torch.Tensor | None] = _default_deserialize_func,
) -> Sample:
    """Fill ``sample`` fields from one ``RolloutImageResponse``-shaped dict (per-sample tensors, no batch dim)."""
    sample.request_id = body.get("request_id") or sample.request_id
    if "prompt" in body:
        sample.prompt = str(body["prompt"])
    if "seed" in body:
        sample.seed = int(body["seed"])

    sample.generated_output = deserialize_func(body.get("generated_output"))
    sample.rollout_log_probs = _deserialize_rollout_log_probs(body.get("rollout_log_probs"), deserialize_func=deserialize_func)
    sample.rollout_debug_tensors = _parse_rollout_debug_tensors(
        body.get("rollout_debug_tensors"),
        deserialize_func=deserialize_func,
    )
    sample.denoising_env = _parse_denoising_env(body.get("denoising_env"), deserialize_func=deserialize_func)
    sample.dit_trajectory = _parse_dit_trajectory(body.get("dit_trajectory"), deserialize_func=deserialize_func)

    if "inference_time_s" in body and body["inference_time_s"] is not None:
        sample.inference_time_s = float(body["inference_time_s"])
    if "peak_memory_mb" in body and body["peak_memory_mb"] is not None:
        sample.peak_memory_mb = float(body["peak_memory_mb"])
    return sample


@ray.remote(num_cpus=1)
class RolloutImageResponseParserActor:
    def apply(self, sample: Sample, body: dict[str, Any]) -> Sample:
        return apply_rollout_image_response(sample, body)
