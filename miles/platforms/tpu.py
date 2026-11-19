"""TPU v7x through torch_tpu."""

import math
import os
from collections.abc import Callable, Sequence
from typing import Any

from miles.utils.tpu_env import SLOT_DEVICE_ENV_VAR, expose_slot_device, pop_gke_webhook_env_vars, subhost_port_base

_TPU_GROUP_PREFIX = "tpu-group-"
_STANDALONE_SLICE_KEY = "__standalone__"
_FULL_HOST_SLICEBUILDER_PORT_BASE = 8471

# TPU v7x (Ironwood) topologies: 2 TensorCores per chip, 4 chips (8 cores) per 2x2x1 host.
_TPU_V7X_TOPOLOGY_MAP: dict[int, str] = {
    1: "1,1,1,1",
    2: "1,1,1,2",
    4: "1,2,1,2",
    8: "2,2,1,2",
    16: "2,2,2,2",
    32: "2,2,4,2",
    64: "2,4,4,2",
    128: "4,4,4,2",
}


# Must run before `import torch_tpu` so libtpu does not inherit pod-wide GKE webhook coordinates
# in sub-host (e.g. 4+4) sessions where each pool owns its own MegaScale coordinator/ports.
if os.environ.get("MILES_TPU_POP_WEBHOOK_ENV") == "1":
    pop_gke_webhook_env_vars()
    expose_slot_device()
    if tpu_log_dir := os.environ.get("TPU_LOG_DIR"):
        os.makedirs(tpu_log_dir, exist_ok=True)

import torch
import torch_tpu  # noqa: F401  registers the tpu device with torch

from miles.platforms import Platform


def _slice_sort_key(label: str) -> tuple[int, int | str]:
    suffix = label.removeprefix(_TPU_GROUP_PREFIX)
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, label)


def _discover_tpu_slices(ray_nodes: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int], list[int]]:
    """Inspect alive Ray nodes for KubeRay `tpu-group-*` slices and standalone TPU node sizes."""
    slice_capacities: dict[str, int] = {}
    slice_per_node: dict[str, int] = {}
    standalone_node_tpus: list[int] = []

    for node in ray_nodes:
        if not node.get("Alive", True):
            continue
        resources = node.get("Resources", {})
        node_tpus = int(resources.get("TPU", 0))
        if node_tpus <= 0:
            continue
        group_labels = [key for key in resources if key.startswith(_TPU_GROUP_PREFIX) and resources[key] > 0]
        if group_labels:
            for label in group_labels:
                slice_capacities[label] = slice_capacities.get(label, 0) + node_tpus
                slice_per_node[label] = max(slice_per_node.get(label, 0), node_tpus)
        else:
            standalone_node_tpus.append(node_tpus)

    return slice_capacities, slice_per_node, standalone_node_tpus


def _resolve_tpu_topology(world_size: int) -> str:
    """`TORCH_TPU_TOPOLOGY` for a session of `world_size` TPU TensorCores.

    Derived from the session size alone: the TORCH_TPU_TOPOLOGY this process may have inherited describes the whole
    pod it runs in (a KubeRay TPU worker exports the host's 8-core topology), not the trainer or engine slot.
    An unusual shape can still be forced per run through --train-env-vars.
    """
    if world_size not in _TPU_V7X_TOPOLOGY_MAP:
        raise ValueError(
            f"no TPU v7x topology for {world_size} cores; known sizes are {sorted(_TPU_V7X_TOPOLOGY_MAP)}, "
            "or pass TORCH_TPU_TOPOLOGY through --train-env-vars"
        )
    return _TPU_V7X_TOPOLOGY_MAP[world_size]


def _compile_cache_env_vars(role: str, slot: str, *, node_local_default: bool, vllm: bool = False) -> dict[str, str]:
    """Compile-cache dirs of one `role` slot (e.g. `w4_off0`), so different degrees or slots never share a cache.

    The run's cache root is the tier-3 root its launcher exports (`--compile-cache-dir`); under it the caches survive
    a preempted node. Without one, a sub-host slot keeps node-local /tmp dirs and a full host inherits the defaults.
    """
    kinds = {"TORCHINDUCTOR_CACHE_DIR": "inductor_cache", "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "tpu_tier3"}
    if vllm:
        kinds["VLLM_CACHE_ROOT"] = "vllm_cache"
    if root := os.environ.get("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"):
        return {name: f"{root}/{role}_{slot}/{kind}" for name, kind in kinds.items()}
    if node_local_default:
        return {name: f"/tmp/{kind}_{slot}" for name, kind in kinds.items()}
    return {}


class TpuPlatform(Platform):
    device_type = "tpu"
    ray_accelerator_resource = "TPU"
    default_distributed_backend = "tpu_dist"
    default_attn_implementation = "sdpa"
    supports_pin_memory = False
    supports_colocate = False
    separate_role_placement_groups = True
    ray_assigns_device_ids = False
    # torch_tpu lowers compiled graphs to XLA
    compile_backend = "tpu"
    triton_kernels = False
    # torch_tpu runs FlexAttention as one dense L x L mask
    flex_attention = False
    # tpu_dist has no rooted reduce
    rooted_reduce = False
    # tpu_dist has no PREMUL_SUM
    premul_sum_reduce = False
    # device-to-device ICI (shared slice) or host-TCP tpu-sync (cross-slice / sub-host)
    weight_transfer_modes = frozenset({"ici", "tpu-sync"})
    # sglang-torchtpu has no --enable-prefill-weight-versions, so its server args lack the key
    prefill_weight_versions = False
    # the tpu compile backend takes static shapes only
    dynamic_shapes = False
    nvml_numa_affinity = False
    torch_memory_history = False
    # sglang-torchtpu turns "auto" into an fp8 KV cache without scales, which skews rollout logprobs against the
    # bf16 trainer; fp8 stays available through --sglang-kv-cache-dtype fp8_e4m3
    rollout_kv_cache_dtype = "bfloat16"
    noset_visible_devices_env_vars = {"RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1"}

    def bind_device_before_process_group(self, local_rank: int) -> None:
        # touching the device before init_process_group breaks the multi-process rendezvous
        pass

    def bind_device_after_process_group(self) -> None:
        # each process sees its one device, which the process group has brought up
        torch.accelerator.set_device_index(0)

    def current_device(self) -> torch.device:
        # On a CPU-only KubeRay head pod with MILES_ACCELERATOR=tpu, torch_tpu imports cleanly
        # with no local accelerator; avoid raising from current_device_index().
        if not torch.accelerator.is_available():
            return torch.device("cpu")
        return super().current_device()

    def empty_cache(self) -> None:
        # Explicit no-op on TPU: never evict compiled XLA programs.
        pass

    def device_name(self) -> str | None:
        return None

    def ray_actor_accelerator_options(self, num_devices: float, *, claim_whole_device: bool = False) -> dict[str, Any]:
        # Ray 2.56 rejects fractional TPU requests (valid chip options are 1, 2, 4, 8).
        # Per-device trainer actors claim whole TPUs; PG-backed rollout server launchers claim 0 TPUs
        # and spawn workers inside the STRICT_PACK placement group that holds the TPU bundles.
        if claim_whole_device and num_devices > 0:
            return {"resources": {self.ray_accelerator_resource: max(1, math.ceil(num_devices))}}
        return {}

    def create_placement_group_bundles(
        self,
        *,
        role: str,
        num_devices: int,
        devices_per_node: int,
        allocated_slices: dict[str, int],
        ray_nodes: list[dict[str, Any]],
    ) -> tuple[list[dict[str, float]], str, str | None, int]:
        slice_capacities, slice_per_node, standalone_node_tpus = _discover_tpu_slices(ray_nodes)
        if slice_capacities:
            sorted_labels = sorted(slice_capacities, key=_slice_sort_key)
            # First prefer completely unclaimed slices so trainer and rollout land on distinct slices.
            unclaimed = [
                label
                for label in sorted_labels
                if allocated_slices.get(label, 0) == 0 and slice_capacities[label] >= num_devices
            ]
            if unclaimed:
                slice_label = unclaimed[0]
                subhost_offset = 0
                allocated_slices[slice_label] = num_devices
            else:
                # Fall back to sharing a partially claimed slice (e.g. 4+4 sub-host on one 2x2x1 node).
                candidates = [
                    label
                    for label in sorted_labels
                    if slice_capacities[label] - allocated_slices.get(label, 0) >= num_devices
                ]
                if not candidates:
                    raise RuntimeError(
                        f"Insufficient TPU slice capacity for role={role!r} requesting {num_devices} TPUs; "
                        f"discovered slices={slice_capacities}, already allocated={allocated_slices}"
                    )
                slice_label = min(
                    candidates,
                    key=lambda label: (
                        allocated_slices.get(label, 0) / slice_capacities[label],
                        _slice_sort_key(label),
                    ),
                )
                subhost_offset = allocated_slices.get(slice_label, 0)
                allocated_slices[slice_label] = subhost_offset + num_devices

            per_node = slice_per_node.get(slice_label) or devices_per_node or num_devices
            strategy = "STRICT_PACK" if num_devices <= per_node else "PACK"
            bundles: list[dict[str, float]] = [
                {"CPU": 1, self.ray_accelerator_resource: 1, slice_label: 1e-4} for _ in range(num_devices)
            ]
            return bundles, strategy, slice_label, subhost_offset

        # Standalone TPU Ray cluster without KubeRay `tpu-group-*` labels.
        per_node = max(standalone_node_tpus) if standalone_node_tpus else (devices_per_node or num_devices)
        subhost_offset = 0
        if len(standalone_node_tpus) == 1:
            used = allocated_slices.get(_STANDALONE_SLICE_KEY, 0)
            if used + num_devices <= standalone_node_tpus[0]:
                subhost_offset = used
                allocated_slices[_STANDALONE_SLICE_KEY] = used + num_devices
        strategy = "STRICT_PACK" if num_devices <= per_node else "PACK"
        bundles = [{"CPU": 1, self.ray_accelerator_resource: 1} for _ in range(num_devices)]
        return bundles, strategy, None, subhost_offset

    def local_device_id(
        self, *, worker_in_pod_index: int | None, ray_device_ids: Callable[[], Sequence[int | str]]
    ) -> int:
        if worker_in_pod_index is not None:
            return worker_in_pod_index
        # worker_distributed_env_vars numbers each rank's device itself (RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS)
        if (local_rank := os.environ.get("LOCAL_RANK")) is not None:
            return int(local_rank)
        if not (ids := ray_device_ids()):
            raise RuntimeError("no LOCAL_RANK and Ray assigned this worker no TPU; cannot pick its local device")
        return int(ids[0])

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
        # Preserve the legacy shared-slice ICI path untouched when MILES_TPU_SHARED_SLICE is set.
        if os.environ.get("MILES_TPU_SHARED_SLICE"):
            return {}

        ips = list(bundle_node_ips) if bundle_node_ips else ["127.0.0.1"] * world_size
        topology = _resolve_tpu_topology(world_size)
        is_subhost = (devices_per_node > 0 and local_world_size < devices_per_node) or subhost_offset > 0

        if is_subhost:
            slot_rank = subhost_offset + local_rank
            port_base = subhost_port_base(subhost_offset)
            sb_addresses = ",".join(f"localhost:{port_base + 300 + r}" for r in range(local_world_size))
            return {
                "MILES_ACCELERATOR": self.device_type,
                "TORCH_TPU_TOPOLOGY": topology,
                "ALLOW_MULTIPLE_LIBTPU_LOAD": "1",
                "LOCAL_RANK": str(slot_rank),
                "LOCAL_WORLD_SIZE": str(subhost_offset + local_world_size),
                SLOT_DEVICE_ENV_VAR: str(slot_rank),
                "TORCH_TPU_SLICEBUILDER_ADDRESSES": sb_addresses,
                "MEGASCALE_COORDINATOR_ADDRESS": f"127.0.0.1:{port_base}",
                "MEGASCALE_PORT": str(port_base + 2 * local_rank),
                "MEGASCALE_DEBUG_PORT": str(port_base + 2 * local_rank + 1),
                "TPU_LOG_DIR": f"/tmp/tpu_logs/rank_{slot_rank}",
                **_compile_cache_env_vars("trainer", f"w{world_size}_off{subhost_offset}", node_local_default=True),
                "MILES_TPU_POP_WEBHOOK_ENV": "1",
            }

        unique_hostnames = list(dict.fromkeys(ips))
        sb_addresses = ",".join(
            f"{ip}:{_FULL_HOST_SLICEBUILDER_PORT_BASE + (idx % local_world_size)}" for idx, ip in enumerate(ips)
        )
        return {
            "MILES_ACCELERATOR": self.device_type,
            "LOCAL_RANK": str(local_rank),
            "LOCAL_WORLD_SIZE": str(local_world_size),
            "TPU_WORKER_HOSTNAMES": ",".join(unique_hostnames),
            "TORCH_TPU_TOPOLOGY": topology,
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": sb_addresses,
            **_compile_cache_env_vars("trainer", f"w{world_size}_off0", node_local_default=False),
        }

    def rollout_worker_env_vars(
        self,
        *,
        bundle_node_ips: list[str],
        tp_size: int,
        devices_per_node: int,
        subhost_offset: int = 0,
    ) -> dict[str, str]:
        ips = list(bundle_node_ips) if bundle_node_ips else ["127.0.0.1"] * tp_size
        topology = _resolve_tpu_topology(tp_size)
        is_subhost = (devices_per_node > 0 and tp_size < devices_per_node) or subhost_offset > 0

        if is_subhost:
            port_base = subhost_port_base(subhost_offset)
            sb_addresses = ",".join(f"localhost:{port_base + 300 + r}" for r in range(tp_size))
            return {
                "MILES_ACCELERATOR": self.device_type,
                "TPU_MULTIHOST_BACKEND": "mp",
                "VLLM_SERVER_DEV_MODE": "1",
                "TORCH_TPU_TOPOLOGY": topology,
                "TORCH_TPU_SLICEBUILDER_ADDRESSES": sb_addresses,
                "ALLOW_MULTIPLE_LIBTPU_LOAD": "1",
                "TPU_LOCAL_RANK_OFFSET": str(subhost_offset),
                "DEBUG_TPU_LOCAL_RANK_OFFSET": str(subhost_offset),
                "MEGASCALE_COORDINATOR_ADDRESS": f"127.0.0.1:{port_base}",
                "MILES_TPU_SUBHOST_PORT_BASE": str(port_base),
                "MILES_TPU_POP_WEBHOOK_ENV": "1",
                **_compile_cache_env_vars(
                    "rollout", f"tp{tp_size}_off{subhost_offset}", node_local_default=True, vllm=True
                ),
            }

        local_n = min(tp_size, devices_per_node) if devices_per_node > 0 else tp_size
        sb_addresses = ",".join(
            f"{ip}:{_FULL_HOST_SLICEBUILDER_PORT_BASE + (idx % local_n)}" for idx, ip in enumerate(ips)
        )
        return {
            "MILES_ACCELERATOR": self.device_type,
            "TPU_MULTIHOST_BACKEND": "mp",
            "VLLM_SERVER_DEV_MODE": "1",
            "TORCH_TPU_TOPOLOGY": topology,
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": sb_addresses,
            **_compile_cache_env_vars("rollout", f"tp{tp_size}_off0", node_local_default=False, vllm=True),
        }
