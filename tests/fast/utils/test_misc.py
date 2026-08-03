import asyncio
import json
import logging
import socket
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import dataclass

import pytest

from miles.utils.env_report import ENV_REPORT_PREFIX
from miles.utils.http_utils import MILES_HOST_IP_ENV, get_host_info
from miles.utils.misc import (
    NodeProbeMixin,
    SimpleTicker,
    cancel_and_await_task,
    filter_keys,
    get_current_node_ip,
    get_free_port,
    get_gpu_uuids,
)


class TestFilterKeys:
    def test_projects_dict_by_keys(self):
        """filter_keys returns only the requested keys with their values."""
        d = {"a": 1, "b": 2, "c": 3}
        assert filter_keys(d, ["a", "c"]) == {"a": 1, "c": 3}

    def test_empty_interest_keys_returns_empty_dict(self):
        """An empty interest list yields an empty dict regardless of input."""
        assert filter_keys({"a": 1, "b": 2}, []) == {}

    def test_preserves_interest_keys_order(self):
        """Result key order follows interest_keys, not the source dict order."""
        d = {"a": 1, "b": 2, "c": 3}
        assert list(filter_keys(d, ["c", "a"]).keys()) == ["c", "a"]

    def test_full_subset_returns_all_entries(self):
        """Requesting every key returns the whole projection."""
        d = {"x": 10, "y": 20}
        assert filter_keys(d, ["x", "y"]) == {"x": 10, "y": 20}

    def test_duplicate_interest_key_collapses_to_single_entry(self):
        """A repeated interest key produces a single dict entry."""
        d = {"a": 1, "b": 2}
        assert filter_keys(d, ["a", "a"]) == {"a": 1}

    def test_missing_key_raises_key_error_and_logs(self, caplog):
        """A missing key raises KeyError and logs the error with context."""
        d = {"a": 1}
        with caplog.at_level(logging.ERROR, logger="miles.utils.misc"):
            with pytest.raises(KeyError):
                filter_keys(d, ["a", "missing"])
        assert any("filter_keys" in record.message for record in caplog.records)


@dataclass(frozen=True)
class _FakeGpuHandle:
    index: int


@dataclass(frozen=True)
class _FakeNvmlUuid:
    text: str

    def __str__(self) -> str:
        return self.text


class _FakeNvml:
    def __init__(
        self,
        *,
        uuid_by_index: dict[int, str],
        init_error: Exception | None = None,
        uuid_error_indices: frozenset[int] = frozenset(),
    ) -> None:
        self._uuid_by_index = uuid_by_index
        self._init_error = init_error
        self._uuid_error_indices = uuid_error_indices

    def nvmlInit(self) -> None:
        if self._init_error is not None:
            raise self._init_error

    def nvmlDeviceGetHandleByIndex(self, index: int) -> _FakeGpuHandle:
        return _FakeGpuHandle(index=index)

    def nvmlDeviceGetUUID(self, handle: _FakeGpuHandle) -> _FakeNvmlUuid:
        if handle.index in self._uuid_error_indices:
            raise RuntimeError(f"nvml uuid lookup failed for {handle.index}")
        return _FakeNvmlUuid(text=self._uuid_by_index[handle.index])


class TestGetGpuUuids:
    def test_get_gpu_uuids_returns_requested_nvml_uuids_in_order(self, monkeypatch) -> None:
        """Each requested gpu index is resolved through NVML, coerced to str, and answered in request order."""
        fake_nvml = _FakeNvml(uuid_by_index={0: "GPU-zero", 1: "GPU-one", 2: "GPU-two"})
        monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

        uuids = get_gpu_uuids([2, 0])

        assert uuids == ["GPU-two", "GPU-zero"]
        assert all(isinstance(uuid, str) for uuid in uuids)

    def test_get_gpu_uuids_returns_none_per_gpu_when_nvml_fails(self, monkeypatch) -> None:
        """A failing NVML init is swallowed and answered with exactly one None per requested gpu."""
        fake_nvml = _FakeNvml(uuid_by_index={}, init_error=RuntimeError("nvml unavailable"))
        monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

        assert get_gpu_uuids([0, 1, 3]) == [None, None, None]

    def test_get_gpu_uuids_returns_all_none_when_one_lookup_fails(self, monkeypatch) -> None:
        """A single failing uuid lookup yields all-None rather than a partial or short list."""
        fake_nvml = _FakeNvml(
            uuid_by_index={0: "GPU-zero", 1: "GPU-one"},
            uuid_error_indices=frozenset({1}),
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

        assert get_gpu_uuids([0, 1]) == [None, None]


class TestNodeProbeMixin:
    def test_get_node_ip_returns_nonempty_string(self):
        """The node ip probe answers with a usable address string."""
        node_ip = NodeProbeMixin._get_node_ip()
        assert isinstance(node_ip, str) and node_ip

    def test_get_free_port_block_returns_bindable_consecutive_ports(self) -> None:
        """A block request returns five ports that can be bound simultaneously."""
        candidate_start: int = get_free_port(start_port=15000, consecutive=10)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied_socket:
            occupied_socket.bind(("", candidate_start + 4))
            occupied_socket.listen()
            first_port: int = NodeProbeMixin._get_free_port_block(start_port=candidate_start, count=5)

            with ExitStack() as stack:
                for port in range(first_port, first_port + 5):
                    available_socket: socket.socket = stack.enter_context(
                        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    )
                    available_socket.bind(("", port))

    def test_get_gpu_uuids_returns_one_entry_per_gpu(self):
        """The uuid probe is best-effort: without NVML it still answers per gpu."""
        uuids = NodeProbeMixin._get_gpu_uuids([0, 1, 2])
        assert len(uuids) == 3
        assert all(uuid is None or isinstance(uuid, str) for uuid in uuids)

    def test_collect_env_report_forwards_probe_context(self, monkeypatch, capsys) -> None:
        """Role, rank and the launcher's partial report all reach the printed env report."""

        def _failing_pip_inspect(*args, **kwargs) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(args=["pip", "inspect"], returncode=1, stdout="", stderr="no pip")

        monkeypatch.setattr("miles.utils.env_report.subprocess.run", _failing_pip_inspect)

        NodeProbeMixin._collect_env_report(role="rollout", rank=7, partial_env_report='{"flavor": "probe"}')

        lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith(ENV_REPORT_PREFIX)]
        assert len(lines) == 1
        parsed = json.loads(lines[0].removeprefix(ENV_REPORT_PREFIX))
        assert parsed["role"] == "rollout"
        assert parsed["rank"] == 7
        assert parsed["launcher_env_report"] == {"flavor": "probe"}


async def _append(calls: list[int]) -> None:
    calls.append(1)


class TestSimpleTicker:
    async def test_it_keeps_calling_its_function(self):
        """The ticked work only makes progress while the loop keeps coming back."""
        calls: list[int] = []

        ticker = SimpleTicker(lambda: _append(calls), interval_seconds=0.0)
        await asyncio.sleep(0.02)
        await ticker.dispose()

        assert len(calls) > 1

    async def test_it_survives_a_failing_call(self):
        """A raising sweep must not silently kill the loop for every later round."""
        calls: list[int] = []

        async def _boom() -> None:
            calls.append(1)
            raise RuntimeError("tick exploded")

        ticker = SimpleTicker(_boom, interval_seconds=0.0)
        await asyncio.sleep(0.02)
        await ticker.dispose()

        assert len(calls) > 1

    async def test_dispose_stops_the_loop(self):
        """A surviving loop would keep working after its owner is gone."""
        calls: list[int] = []

        ticker = SimpleTicker(lambda: _append(calls), interval_seconds=0.0)
        await asyncio.sleep(0.02)
        await ticker.dispose()
        calls_after_dispose = len(calls)
        await asyncio.sleep(0.02)

        assert len(calls) == calls_after_dispose

    async def test_a_task_that_died_of_its_own_error_reports_it_on_dispose(self):
        """Cancelling a task that already failed must surface the failure, not hide it behind the cancellation."""

        async def _explode() -> None:
            raise RuntimeError("ticker died")

        task = asyncio.create_task(_explode())
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="ticker died"):
            await cancel_and_await_task(task)

    async def test_dispose_does_not_swallow_the_callers_cancellation(self):
        """A teardown that eats its caller's cancellation lets the shutdown path run on regardless."""
        ticker = SimpleTicker(lambda: _append([]), interval_seconds=1000.0)
        finished: list[str] = []

        async def _dispose() -> None:
            await ticker.dispose()
            finished.append("returned")

        task = asyncio.create_task(_dispose())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert task.cancelled() and finished == []

    async def test_disposing_twice_is_harmless(self):
        """Teardown paths overlap, so a second dispose must not raise."""
        ticker = SimpleTicker(lambda: _append([]), interval_seconds=0.0)

        await ticker.dispose()
        await ticker.dispose()


class TestCancelAndAwaitTask:
    async def test_it_swallows_the_cancellation_of_the_task_it_cancelled(self):
        """The cancellation the helper itself asked for is teardown noise, not something to raise at the caller."""
        task = asyncio.create_task(asyncio.Event().wait())
        await asyncio.sleep(0)

        await cancel_and_await_task(task)

        assert task.cancelled()

    async def test_it_returns_only_after_the_task_finished_unwinding(self):
        """Returning while the cancelled task is still unwinding lets it act once more after its owner is gone."""
        unwound: list[str] = []

        async def _slow_teardown() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                unwound.append("cleaned")

        task = asyncio.create_task(_slow_teardown())
        await asyncio.sleep(0)

        await cancel_and_await_task(task)

        assert task.done() and unwound == ["cleaned"]

    async def test_it_requests_teardown_before_propagating_the_callers_cancellation(self):
        """Being cancelled mid-teardown must still leave the task cancelled while the caller's cancellation travels on."""
        observed_cancel = asyncio.Event()
        release = asyncio.Event()
        returned: list[str] = []

        async def _absorbs_the_first_cancel() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                observed_cancel.set()
                await release.wait()

        task = asyncio.create_task(_absorbs_the_first_cancel())
        await asyncio.sleep(0)

        async def _caller() -> None:
            await cancel_and_await_task(task)
            returned.append("returned")

        caller = asyncio.create_task(_caller())
        await asyncio.wait_for(observed_cancel.wait(), timeout=2.0)
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        release.set()
        await task

        assert caller.cancelled() and returned == []

    async def test_a_shutdown_deadline_still_fires_when_the_task_is_slow_to_unwind(self):
        """Eating the deadline's cancellation would let a timed-out shutdown continue as if it had made its deadline."""
        release = asyncio.Event()
        returned: list[str] = []

        async def _absorbs_the_first_cancel() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()

        task = asyncio.create_task(_absorbs_the_first_cancel())
        await asyncio.sleep(0)

        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.01):
                await cancel_and_await_task(task)
                returned.append("returned")

        release.set()
        await task

        assert returned == []

    async def test_a_task_that_fails_while_unwinding_reports_its_error(self):
        """A teardown that raises its own error must not be mistaken for the cancellation the helper asked for."""

        async def _fails_on_teardown() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                raise RuntimeError("teardown exploded")

        task = asyncio.create_task(_fails_on_teardown())
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="teardown exploded"):
            await cancel_and_await_task(task)
