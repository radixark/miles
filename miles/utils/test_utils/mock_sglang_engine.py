"""In-memory stand-in for the engine ``CommandActor`` (no CUDA, no sglang, no model)."""

from __future__ import annotations

import logging
import shlex
from typing import Any

import ray

from miles.utils.misc import NodeProbeMixin, get_free_port
from miles.utils.test_utils.mock_sglang_http_server import MockSGLangHttpServer

logger = logging.getLogger(__name__)


def parse_cmd_flags(cmd: str) -> dict[str, Any]:
    """Naive ``--flag value`` scan of a launch command; enough for addressing asserts."""
    tokens = shlex.split(cmd)
    flags: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        name = token[2:].replace("-", "_")
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            value = tokens[index + 1]
            flags[name] = int(value) if value.isdigit() else value
            index += 2
        else:
            flags[name] = True
            index += 1
    return flags


class MockSGLangEngine:
    """Records every call into ``self.calls`` so tests can assert sequence and
    arguments. Fault injection is set via ``set_fault(method, exception)``."""

    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []
        self._faults: dict[str, BaseException] = {}
        self._http_server: MockSGLangHttpServer | None = None
        self._server_args: dict[str, Any] | None = None

    def set_fault(self, method: str, exception: BaseException | None):
        if exception is None:
            self._faults.pop(method, None)
        else:
            self._faults[method] = exception

    def inject_fault(self, mode: str) -> None:
        self._record("inject_fault", (), {"mode": mode})

    def get_calls(self) -> list[tuple[str, tuple, dict]]:
        return list(self.calls)

    def get_server_args(self) -> dict[str, Any] | None:
        return dict(self._server_args) if self._server_args is not None else None

    def get_http_paths(self) -> list[str]:
        return self._http_server.paths if self._http_server is not None else []

    def get_http_payloads_of(self, path: str) -> list[dict | None]:
        return self._http_server.payloads_of(path) if self._http_server is not None else []

    def run(self, cmd: str, envs: dict[str, str]):
        self._record("run", (), {"cmd": cmd, "envs": envs})
        self._maybe_fault("run")
        self._server_args = parse_cmd_flags(cmd)
        self._http_server = MockSGLangHttpServer(port=int(self._server_args["port"]))
        return None

    def shutdown(self):
        self._record("shutdown", (), {})
        self._maybe_fault("shutdown")
        if self._http_server is not None:
            self._http_server.close()
        return True

    def kill_subprocess(self):
        """Real CommandActor takes the actor down with the subprocess; the
        in-process mock cannot exit the test process, so only the server dies."""
        self._record("kill_subprocess", (), {})
        if self._http_server is not None:
            self._http_server.close()

    def _get_free_port_block(self, *, start_port: int, count: int) -> int:
        self._record("_get_free_port_block", (), {"start_port": start_port, "count": count})
        return get_free_port(start_port=start_port, consecutive=count)

    def _get_node_ip(self):
        self._record("_get_node_ip", (), {})
        return NodeProbeMixin._get_node_ip()

    def _to_local_gpu_ids(self, *, gpu_ids: list[int]) -> list[int]:
        self._record("_to_local_gpu_ids", (), {"gpu_ids": gpu_ids})
        return list(range(len(gpu_ids)))

    def _is_port_available(self, *, port: int) -> bool:
        self._record("_is_port_available", (), {"port": port})
        return True

    def _get_gpu_uuids(self, gpu_ids: list[int]):
        self._record("_get_gpu_uuids", (gpu_ids,), {})
        return [None] * len(gpu_ids)

    def _collect_env_report(self, *, role: str, rank: int, partial_env_report: str):
        self._record("_collect_env_report", (), {"role": role, "rank": rank, "partial_env_report": partial_env_report})

    def _record(self, name: str, args: tuple, kwargs: dict) -> None:
        self.calls.append((name, args, kwargs))

    def _maybe_fault(self, method: str) -> None:
        exc = self._faults.pop(method, None)
        if exc is not None:
            raise exc


MockSGLangEngine = ray.remote(MockSGLangEngine)
