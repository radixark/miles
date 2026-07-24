import importlib
import logging
import os
import sys
from contextlib import nullcontext
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

_MUSA_PATCH_IMPORTED = False


def _append_musa_patch_path() -> None:
    patch_path = os.environ.get("MUSA_PATCH_PATH")
    if patch_path and patch_path not in sys.path:
        sys.path.append(patch_path)


def _try_import_musa_patch_once() -> bool:
    _append_musa_patch_path()
    try:
        importlib.import_module("musa_patch")
    except ImportError:
        return False
    except Exception as exc:
        logger.warning("Failed to import musa_patch: %s", exc)
        return False
    return True


import torch

_MUSA_PATCH_IMPORTED = _try_import_musa_patch_once()


def _musa() -> ModuleType | None:
    return getattr(torch, "musa", None)


def is_musa_available() -> bool:
    musa = _musa()
    return bool(musa is not None and getattr(musa, "is_available", lambda: False)())


def is_musa_environment() -> bool:
    return is_musa_available() or "MUSA_VISIBLE_DEVICES" in os.environ or bool(os.environ.get("MUSA_PATCH_PATH"))


def _musa_requested() -> bool:
    return "MUSA_VISIBLE_DEVICES" in os.environ or bool(os.environ.get("MUSA_PATCH_PATH"))


def _require_musa_available() -> None:
    if _musa_requested() and not is_musa_available():
        raise RuntimeError(
            "MUSA environment was requested via MUSA_VISIBLE_DEVICES or MUSA_PATCH_PATH, "
            "but torch.musa is unavailable. Check that musa_patch imports before torch and that MUSA runtime is installed."
        )


def is_cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def device_type() -> str:
    if is_musa_available():
        return "musa"
    if is_cuda_available():
        return "cuda"
    return "cpu"


def device_name(index: int | None = None) -> str:
    current_type = device_type()
    if current_type == "cpu":
        return "cpu"
    if index is None:
        index = current_device()
    return f"{current_type}:{index}"


def device(index: int | None = None) -> torch.device:
    return torch.device(device_name(index))


def accelerator_module() -> Any:
    if is_musa_available():
        return _musa()
    return torch.cuda


def set_device(index: int | str | torch.device) -> None:
    if device_type() == "cpu":
        return
    accelerator_module().set_device(index)


def current_device() -> int | str:
    if device_type() == "cpu":
        return "cpu"
    return accelerator_module().current_device()


def synchronize(device_arg: int | str | torch.device | None = None) -> None:
    if device_type() == "cpu":
        return
    module = accelerator_module()
    if hasattr(module, "synchronize"):
        if device_arg is None:
            module.synchronize()
        else:
            module.synchronize(device_arg)


def empty_cache() -> None:
    if device_type() == "cpu":
        return
    module = accelerator_module()
    if hasattr(module, "empty_cache"):
        module.empty_cache()


def ipc_collect() -> None:
    if device_type() == "cpu":
        return
    module = accelerator_module()
    if hasattr(module, "ipc_collect"):
        module.ipc_collect()


def mem_get_info(device_arg: int | str | torch.device | None = None) -> tuple[int, int]:
    if device_type() == "cpu":
        raise RuntimeError("accelerator memory info is unavailable on CPU")
    module = accelerator_module()
    if device_arg is None:
        device_arg = current_device()
    return module.mem_get_info(device_arg)


def memory_allocated(device_arg: int | str | torch.device | None = None) -> int:
    if device_type() == "cpu":
        return 0
    return accelerator_module().memory_allocated(device_arg)


def memory_reserved(device_arg: int | str | torch.device | None = None) -> int:
    if device_type() == "cpu":
        return 0
    return accelerator_module().memory_reserved(device_arg)


def max_memory_allocated(device_arg: int | str | torch.device | None = None) -> int:
    if device_type() == "cpu":
        return 0
    module = accelerator_module()
    if hasattr(module, "max_memory_allocated"):
        return module.max_memory_allocated(device_arg)
    return 0


def get_device_properties(device_arg: int | str | torch.device | None = None) -> Any:
    if device_type() == "cpu":
        return None
    module = accelerator_module()
    if device_arg is None:
        device_arg = current_device()
    return module.get_device_properties(device_arg)


def visible_devices_env_key() -> str:
    if "MUSA_VISIBLE_DEVICES" in os.environ:
        return "MUSA_VISIBLE_DEVICES"
    return "MUSA_VISIBLE_DEVICES" if is_musa_available() else "CUDA_VISIBLE_DEVICES"


def resolve_visible_device_id(physical_device_id: int | float | str) -> int:
    visible_devices = os.environ.get(visible_devices_env_key())
    physical_device_id = int(float(physical_device_id))
    if not visible_devices:
        return physical_device_id
    visible = [int(x) for x in visible_devices.split(",") if x.strip()]
    if physical_device_id in visible:
        return visible.index(physical_device_id)
    if 0 <= physical_device_id < len(visible):
        return physical_device_id
    raise RuntimeError(
        f"Device id {physical_device_id} is not valid under {visible_devices_env_key()}={visible_devices}. "
        f"Expected one of {visible} (physical) or 0..{len(visible) - 1} (local)."
    )


def process_group_backend(default: str = "nccl") -> str:
    if default == "nccl":
        if is_musa_available():
            return "mccl"
        _require_musa_available()
    return default


def weight_update_backend(default: str = "nccl") -> str:
    if default == "nccl":
        if is_musa_available():
            return "cpu:gloo,musa:mccl"
        _require_musa_available()
    return default


def stream_context(stream: Any):
    if stream is None or device_type() == "cpu":
        return nullcontext()
    module = accelerator_module()
    if hasattr(module, "stream"):
        return module.stream(stream)
    return nullcontext()


def Stream(*args, **kwargs):
    if device_type() == "cpu":
        return None
    module = accelerator_module()
    stream_cls = getattr(module, "Stream", None)
    if stream_cls is None:
        return None
    return stream_cls(*args, **kwargs)


def Event(*args, **kwargs):
    if device_type() == "cpu":
        return None
    module = accelerator_module()
    event_cls = getattr(module, "Event", None)
    if event_cls is None:
        return None
    return event_cls(*args, **kwargs)


def current_stream():
    if device_type() == "cpu":
        return None
    module = accelerator_module()
    if hasattr(module, "current_stream"):
        return module.current_stream()
    return None


def try_import_musa_patch() -> bool:
    global _MUSA_PATCH_IMPORTED
    if _MUSA_PATCH_IMPORTED:
        return True

    _append_musa_patch_path()

    try:
        importlib.import_module("musa_patch")
    except ImportError:
        if os.environ.get("MUSA_PATCH_PATH") or is_musa_available():
            logger.warning("musa_patch is not importable; continuing without it")
        return False
    except Exception as exc:
        logger.warning("Failed to import musa_patch: %s", exc)
        return False

    _MUSA_PATCH_IMPORTED = True
    logger.info("Imported musa_patch")
    return True
