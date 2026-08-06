"""In-memory stand-in for ``SGLangEngine`` (no CUDA, no sglang, no model)."""

from __future__ import annotations

import logging
import ray

from miles.utils.misc import get_free_port
from miles.utils.test_utils.mock_sglang_http_server import MockSGLangHttpServer

logger = logging.getLogger(__name__)


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
        self._http_server: MockSGLangHttpServer | None = None

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

    def get_http_paths(self) -> list[str]:
        return self._http_server.paths if self._http_server is not None else []

    def get_http_payloads_of(self, path: str) -> list[dict | None]:
        return self._http_server.payloads_of(path) if self._http_server is not None else []

    def init(self, **kwargs):
        self._record("init", (), kwargs)
        self._maybe_fault("init")
        self._http_server = MockSGLangHttpServer(port=kwargs["port"])
        self.initialized = True
        return None

    def shutdown(self):
        self._record("shutdown", (), {})
        self._maybe_fault("shutdown")
        if self._http_server is not None:
            self._http_server.close()
        self.initialized = False
        return True

    def simulate_crash(self):
        # Real SGLangEngine.simulate_crash calls self.shutdown() (only the http
        # server dies; the actor itself stays alive). Mirror that.
        self._record("simulate_crash", (), {})
        self.shutdown()

    def _get_current_node_ip_and_free_port(self, start_port: int = 15000, consecutive: int = 1):
        self._record("_get_current_node_ip_and_free_port", (), {"start_port": start_port, "consecutive": consecutive})
        return ("127.0.0.1", get_free_port(start_port=start_port, consecutive=consecutive))

    def _get_node_ip(self):
        self._record("_get_node_ip", (), {})
        return "127.0.0.1"

    def _get_gpu_uuids(self, gpu_ids: list[int]):
        self._record("_get_gpu_uuids", (gpu_ids,), {})
        return [None] * len(gpu_ids)

    def _record(self, name: str, args: tuple, kwargs: dict) -> None:
        self.calls.append((name, args, kwargs))

    def _maybe_fault(self, method: str) -> None:
        exc = self._faults.pop(method, None)
        if exc is not None:
            raise exc


MockSGLangEngine = ray.remote(MockSGLangEngine)
