"""The training process's platform: what differs between CUDA and TPU (torch_tpu), chosen once by MILES_ACCELERATOR.

Everything `torch.accelerator` covers (synchronize, empty_cache, memory stats, the device index) is called directly.
"""

import functools
import os
from collections.abc import Callable, Sequence
from typing import Any

import torch

from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST


class Platform:
    """CUDA, and the base other platforms override: every method is the call main makes on CUDA."""

    device_type = "cuda"
    # Ray resource key representing one accelerator device on this platform
    ray_accelerator_resource: str = "GPU"
    # default torch.distributed backend when --distributed-backend is not explicitly overridden
    default_distributed_backend: str = "nccl"
    # default HuggingFace attention implementation for native PyTorch backends (FSDP)
    default_attn_implementation: str = "flash_attention_2"
    # whether pinned host memory (CPU offload / host staging) is safe on this platform
    supports_pin_memory: bool = True
    # whether trainer and rollout engines can share the same accelerator devices via --colocate
    supports_colocate: bool = True
    # trainer and rollout get their own placement groups, each pinned to a slice or a sub-host device block
    separate_role_placement_groups: bool = False
    # Ray picks each bundle's device and reports it to the actor; otherwise the launcher numbers each node's devices
    ray_assigns_device_ids: bool = True
    # the torch.compile backend for model code
    compile_backend = "inductor"
    # Triton kernels (e.g. --log-probs-backend fused) run here
    triton_kernels = True
    # FlexAttention compiles to a block-sparse kernel here
    flex_attention = True
    # the process group has a rooted reduce
    rooted_reduce = True
    # the process group has PREMUL_SUM, which FSDP picks for a gradient divide factor other than the group size
    premul_sum_reduce = True
    # the --update-weight-transfer-mode values an engine on this platform can receive; None: all of them
    weight_transfer_modes: frozenset[str] | None = None
    # the engine reports the weight version each prompt's KV was computed under (--sglang-enable-prefill-weight-versions)
    prefill_weight_versions = True
    # torch.compile(dynamic=True) works here; a platform without it runs dynamic-shape helpers eager
    dynamic_shapes = True
    # NVML sees the devices, so each training rank is pinned to its GPU's NUMA node
    nvml_numa_affinity = True
    # the rollout engines' KV cache dtype when --sglang-kv-cache-dtype is left at "auto"; None keeps the engine's choice
    rollout_kv_cache_dtype: str | None = None
    # torch.cuda.memory records allocator history here (--memory-recorder torch)
    torch_memory_history = True
    # Ray runtime_env variables preventing Ray from overwriting accelerator visibility masks
    noset_visible_devices_env_vars: dict[str, str] = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}

    def bind_device_before_process_group(self, local_rank: int) -> None:
        torch.cuda.set_device(f"cuda:{local_rank}")

    def bind_device_after_process_group(self) -> None:
        pass

    def current_device(self) -> torch.device:
        return torch.device(self.device_type, torch.accelerator.current_device_index())

    def empty_cache(self) -> None:
        torch.accelerator.empty_cache()

    def check_weight_transfer(self, *, mode: str, colocate: bool) -> None:
        """Raise ValueError unless the trainer can sync weights to this platform's engines with `mode`."""
        modes = self.weight_transfer_modes
        if mode == "tpu-sync" and (modes is None or mode not in modes):
            raise ValueError(f"--update-weight-transfer-mode tpu-sync syncs to TPU engines; not on {self.device_type}")
        # colocate syncs through CUDA IPC, outside the transfer modes
        if modes is not None and (colocate or mode not in modes):
            raise ValueError(
                f"on {self.device_type} the trainer syncs weights with --update-weight-transfer-mode "
                f"{' or '.join(sorted(modes))} to disaggregated engines; got "
                f"{mode!r}{' with --colocate' if colocate else ''}"
            )
        if colocate and not self.supports_colocate:
            raise ValueError(f"--colocate is not supported on {self.device_type}")

    @functools.cache  # noqa: B019 -- one instance per process
    def device_name(self) -> str | None:
        """The CUDA device's marketing name (for peak-FLOPs lookup); None without a CUDA device."""
        return torch.cuda.get_device_name() if torch.cuda.is_available() else None

    def ray_actor_accelerator_options(self, num_devices: float, *, claim_whole_device: bool = False) -> dict[str, Any]:
        """Ray `.options(...)` kwargs requesting accelerator resources for a worker actor."""
        return {"num_gpus": num_devices}

    def create_placement_group_bundles(
        self,
        *,
        role: str,
        num_devices: int,
        devices_per_node: int,
        allocated_slices: dict[str, int],
        ray_nodes: list[dict[str, Any]],
    ) -> tuple[list[dict[str, float]], str, str | None, int]:
        """Build `(bundles, strategy, slice_label, subhost_offset)` for a Ray placement group.

        `ray_nodes` is `ray.nodes()`, fetched by the caller so platforms stay free of ray.
        """
        bundles = [{self.ray_accelerator_resource: 1, "CPU": 1} for _ in range(num_devices)]
        return bundles, "PACK", None, 0

    def local_device_id(
        self, *, worker_in_pod_index: int | None, ray_device_ids: Callable[[], Sequence[int | str]]
    ) -> int:
        """This process's local accelerator index; `ray_device_ids` returns the ids Ray assigned this actor."""
        if worker_in_pod_index is not None:
            return worker_in_pod_index
        ray_id = ray_device_ids()[0]
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES")
        if not cvd:
            return ray_id
        return cvd.split(",").index(str(ray_id))

    def worker_distributed_env_vars(
        self,
        *,
        bundle_node_ips: list[str],
        rank: int,
        local_rank: int,
        world_size: int,
        local_world_size: int,
        devices_per_node: int,
        subhost_offset: int = 0,
    ) -> dict[str, str]:
        """Platform-specific distributed runtime env vars for a training worker."""
        return {}

    def rollout_worker_env_vars(
        self,
        *,
        bundle_node_ips: list[str],
        tp_size: int,
        devices_per_node: int,
        subhost_offset: int = 0,
    ) -> dict[str, str]:
        """Platform-specific runtime env vars for a rollout engine launcher."""
        return {}


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
