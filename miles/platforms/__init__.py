"""The training process's platform: what differs between CUDA and TPU (torch_tpu), chosen once by MILES_ACCELERATOR.

Everything `torch.accelerator` covers (synchronize, empty_cache, memory stats, the device index) is called directly.
"""

import functools
import os

import torch


class Platform:
    """CUDA, and the base other platforms override: every method is the call main makes on CUDA."""

    device_type = "cuda"
    # torch.compile(dynamic=True) works here; a platform without it runs dynamic-shape helpers eager
    dynamic_shapes = True
    # NVML sees the devices, so each training rank is pinned to its GPU's NUMA node
    nvml_numa_affinity = True

    def bind_device_before_process_group(self, local_rank: int) -> None:
        torch.cuda.set_device(f"cuda:{local_rank}")

    def bind_device_after_process_group(self) -> None:
        pass

    def current_device(self) -> torch.device:
        return torch.device(self.device_type, torch.accelerator.current_device_index())

    @functools.cache  # noqa: B019 -- one instance per process
    def device_name(self) -> str | None:
        """The CUDA device's marketing name (for peak-FLOPs lookup); None without a CUDA device."""
        return torch.cuda.get_device_name() if torch.cuda.is_available() else None


def _select_platform() -> Platform:
    device_type = os.environ.get("MILES_ACCELERATOR", "cuda")
    if device_type == "cuda":
        return Platform()
    if device_type == "tpu":
        # imports torch_tpu, which CUDA processes never load
        from miles.platforms.tpu import TpuPlatform

        return TpuPlatform()
    raise ValueError(f"MILES_ACCELERATOR={device_type!r}; expected cuda or tpu")


current_platform = _select_platform()
# TODO: the TPU stack's remaining call-site checks become platform capabilities, then this goes
IS_TPU = current_platform.device_type == "tpu"


def torch_compile(*, dynamic: bool):
    """`torch.compile(dynamic=dynamic)` where the platform supports it; a dynamic helper runs eager elsewhere.

    Only for helpers without data-dependent control flow (`compute_approx_kl`, `compute_policy_loss`, the entropy's
    `mul_reduce`). Eager results are numerically close to the compiled ones, not bit-identical: fusion can change
    reduction order and intermediate rounding. Eager `mul_reduce` also materializes the [rows, vocab] product the
    fused kernel avoids, one more fp32 transient per log-prob chunk.
    """

    def wrap(fn):
        if dynamic and not current_platform.dynamic_shapes:
            return fn
        return torch.compile(dynamic=dynamic)(fn)

    return wrap
