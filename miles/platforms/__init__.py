"""The training process's platform: what differs between CUDA and TPU (torch_tpu), chosen once by MILES_ACCELERATOR.

Everything `torch.accelerator` covers (synchronize, empty_cache, memory stats, the device index) is called directly.
"""

import functools
import os

import torch
import torch.distributed as dist


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

    @property
    def dist_backend(self) -> str:
        """The collective backend torch pairs with this device: nccl on CUDA/ROCm, xccl on XPU."""
        return dist.get_default_backend_for_device(self.device_type)

    @functools.cache  # noqa: B019 -- one instance per process
    def device_name(self) -> str | None:
        """The CUDA device's marketing name (for peak-FLOPs lookup); None without a CUDA device."""
        return torch.cuda.get_device_name() if torch.cuda.is_available() else None


def _detect_device_type() -> str:
    device_type = os.environ.get("MILES_ACCELERATOR")
    if device_type:
        return device_type
    accelerator = torch.accelerator.current_accelerator()
    return accelerator.type if accelerator is not None else "cuda"


def _select_platform() -> Platform:
    device_type = _detect_device_type()
    if device_type == "cuda":
        return Platform()
    if device_type == "tpu":
        # imports torch_tpu, which CUDA processes never load
        from miles.platforms.tpu import TpuPlatform

        return TpuPlatform()
    if device_type == "xpu":
        from miles.platforms.xpu import XpuPlatform

        return XpuPlatform()
    raise ValueError(f"accelerator {device_type!r}; expected cuda, tpu or xpu (set MILES_ACCELERATOR to override)")


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
