"""SDE step with log probability for flow matching schedulers.

Adapted from flow_grpo/diffusers_patch/sd3_sde_with_logprob.py.
"""

import math
from typing import Optional, Union

import torch
from diffusers.utils.torch_utils import randn_tensor


def sde_step_with_logprob(
    scheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
):
    """Compute one reverse-SDE step and its log probability.

    Args:
        scheduler: A flow-matching scheduler with `sigmas` and `index_for_timestep`.
        model_output: Predicted velocity from DiT, shape (B, C, H, W).
        timestep: Current timestep(s), shape (B,).
        sample: Current latent, shape (B, C, H, W).
        noise_level: SDE noise scaling factor (eta).
        prev_sample: If provided, compute log prob of this sample instead of sampling.
        generator: Optional RNG for sampling.

    Returns:
        (prev_sample, log_prob, prev_sample_mean, std_dev_t)
    """
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    step_index = [scheduler.index_for_timestep(t) for t in timestep]
    prev_step_index = [s + 1 for s in step_index]
    sigma = scheduler.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = scheduler.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_max = scheduler.sigmas[1].item()
    dt = sigma_prev - sigma

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level

    prev_sample_mean = (
        sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
        - torch.log(std_dev_t * torch.sqrt(-1 * dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, log_prob, prev_sample_mean, std_dev_t
