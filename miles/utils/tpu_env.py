"""TPU process environment layout shared by trainers and rollout launchers.

Kept free of torch / torch_tpu (and outside `miles.platforms`, whose import selects the platform and loads
torch_tpu) so a vLLM launcher can read it before any TPU runtime initializes.
"""

import os
from collections.abc import MutableMapping

# Pod-wide GKE TPU webhook env vars: they describe the whole host, so a sub-host libtpu session that inherits them
# joins the pod-wide MegaScale coordinator instead of its own.
GKE_WEBHOOK_ENV_VARS: tuple[str, ...] = (
    "MEGASCALE_NUM_SLICES",
    "MEGASCALE_SLICE_ID",
    "TPU_WORKER_HOSTNAMES",
    "TPU_WORKER_ID",
    "TPU_PROCESS_ADDRESSES",
    "TPU_PROCESS_PORT",
    "TPU_RUNTIME_METRICS_PORTS",
    "TORCH_TPU_DEVICES_PER_HOST",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_HOST_BOUNDS",
)

# The sub-host slot device a trainer rank binds. It cannot ride in the actor's runtime_env as TPU_VISIBLE_CHIPS: Ray
# reads the startup TPU_VISIBLE_CHIPS as the node's device list and indexes it with the TPU id it assigned, so any
# rank but 0 crashes in Ray core before user code. The rank exposes it to libtpu itself, before torch_tpu loads.
SLOT_DEVICE_ENV_VAR = "MILES_TPU_SLOT_DEVICE"

# each sub-host session owns a block of ports starting at base + stride * (its first local device index)
_SUBHOST_PORT_BASE = 21000
_SUBHOST_PORT_STRIDE = 20


def pop_gke_webhook_env_vars(env: MutableMapping[str, str] | None = None) -> None:
    """Remove the pod-wide GKE TPU webhook env vars from `env` (default: this process's environment)."""
    target = os.environ if env is None else env
    for name in GKE_WEBHOOK_ENV_VARS:
        target.pop(name, None)


def subhost_port_base(subhost_offset: int) -> int:
    """First port of the block owned by the sub-host session whose first local device is `subhost_offset`."""
    return _SUBHOST_PORT_BASE + _SUBHOST_PORT_STRIDE * subhost_offset


def expose_slot_device() -> None:
    """Make only this rank's sub-host slot device (`SLOT_DEVICE_ENV_VAR`) visible to libtpu; no-op without one."""
    if (device := os.environ.get(SLOT_DEVICE_ENV_VAR)) is not None:
        os.environ["TPU_VISIBLE_DEVICES"] = device
        os.environ["TPU_VISIBLE_CHIPS"] = device
