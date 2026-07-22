"""In-memory stand-in for ``SGLangEngine`` (no CUDA, no sglang, no model)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import ray

logger = logging.getLogger(__name__)


# Methods that just ``_record + _maybe_fault + return X``. The value is the
# return value (no test asserts on its shape — sentinels keep the mock close
# to what the real method's HTTP response shape returns).
_RECORDING_METHODS: dict[str, Any] = {
    "health_generate": True,
    "release_memory_occupation": True,
    "resume_memory_occupation": True,
    "update_weights_from_disk": True,
    "update_weights_from_tensor": True,
    "flush_cache": True,
    "pause_generation": None,
    "continue_generation": None,
    "update_weight_version": None,
    "begin_weight_update": None,
    "end_weight_update": None,
    "init_weights_update_group": None,
    "destroy_weights_update_group": None,
    "update_weights_from_distributed": None,
    "load_lora_adapter_from_tensors": None,
    "unload_lora_adapter": None,
    "start_profile": None,
    "stop_profile": None,
    "check_weights": {"_mock": True},
    "get_server_info": {"_mock": True},
    "get_weight_version": "mock-v0",
    "get_parallelism_info": {"_mock": True},
    "get_remote_instance_transfer_engine_info": {"_mock": True},
}


def _make_recorder(name: str, return_value: Any) -> Callable:
    def method(self, *args, **kwargs):
        self._record(name, args, kwargs)
        self._maybe_fault(name)
        return return_value

    method.__name__ = name
    return method


class MockSGLangEngine:
    """Records every call into ``self.calls`` so tests can assert sequence and
    arguments. Fault injection is set via ``set_fault(method, exception)``."""

    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int = 0,
        sglang_overrides: dict | None = None,
        num_gpus_per_engine: int = 1,
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.sglang_overrides = sglang_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine

        self.initialized = False
        self.calls: list[tuple[str, tuple, dict]] = []
        self._faults: dict[str, BaseException] = {}
        self._port_seq = 20000
        self._lock = threading.Lock()

    def set_fault(self, method: str, exception: BaseException | None):
        if exception is None:
            self._faults.pop(method, None)
        else:
            self._faults[method] = exception

    def get_calls(self) -> list[tuple[str, tuple, dict]]:
        return list(self.calls)

    def get_init_kwargs(self) -> dict | None:
        for name, _args, kwargs in self.calls:
            if name == "init":
                return dict(kwargs)
        return None

    def init(self, **kwargs):
        self._record("init", (), kwargs)
        self._maybe_fault("init")
        self.initialized = True
        return None

    def shutdown(self):
        self._record("shutdown", (), {})
        self._maybe_fault("shutdown")
        self.initialized = False
        return True

    def simulate_crash(self):
        # Real SGLangEngine.simulate_crash calls self.shutdown() (only the http
        # server dies; the actor itself stays alive). Mirror that.
        self._record("simulate_crash", (), {})
        self.shutdown()

    def _get_current_node_ip_and_free_port(self, start_port: int = 15000, consecutive: int = 1):
        self._record("_get_current_node_ip_and_free_port", (), {"start_port": start_port, "consecutive": consecutive})
        with self._lock:
            port = max(self._port_seq, start_port)
            self._port_seq = port + consecutive
            return ("127.0.0.1", port)

    def _record(self, name: str, args: tuple, kwargs: dict) -> None:
        self.calls.append((name, args, kwargs))

    def _maybe_fault(self, method: str) -> None:
        exc = self._faults.pop(method, None)
        if exc is not None:
            raise exc


for _name, _retval in _RECORDING_METHODS.items():
    setattr(MockSGLangEngine, _name, _make_recorder(_name, _retval))


MockSGLangEngine = ray.remote(MockSGLangEngine)
