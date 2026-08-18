import asyncio
import logging
from types import SimpleNamespace

import pytest
import ray

from miles.utils.workers import worker_handle as worker_handle_module
from miles.utils.workers.ray_worker_handle import RayWorkerHandle
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError


class _ProbeableWorker:
    def demo(self) -> int:
        return 1


class TestBaseWorkerHandle:
    def test_rpc_handle_implements_the_contract(self):
        """The rpc client is a worker handle, so callers can hold the base type."""
        assert issubclass(RpcWorkerHandle, BaseWorkerHandle)

    def test_ray_handle_implements_the_contract(self):
        """The ray wrapper is a worker handle, so callers can hold the base type."""
        assert issubclass(RayWorkerHandle, BaseWorkerHandle)

    def test_a_handle_that_cannot_be_waited_out_says_so_rather_than_reporting_idle(self):
        """Callers read this answer off the base type, and a silent pass would report a busy worker idle."""

        class Minimal(BaseWorkerHandle):
            async def wait_ready(self, *, timeout: float) -> None: ...

            async def probe_is_dead(self) -> bool:
                return True

        with pytest.raises(NotImplementedError, match="running a call"):
            asyncio.run(Minimal().wait_idle(timeout=1.0))

    def test_incomplete_implementation_rejected(self):
        """A handle that implements neither wait_ready nor the death probe cannot be instantiated."""

        class Incomplete(BaseWorkerHandle):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_a_handle_implementing_only_wait_ready_is_rejected(self):
        """wait_dead is required on its own, so a readiness-only handle cannot be instantiated."""

        class ReadyOnly(BaseWorkerHandle):
            async def wait_ready(self, *, timeout: float) -> None: ...

        with pytest.raises(TypeError):
            ReadyOnly()

    def test_a_handle_implementing_only_wait_dead_is_rejected(self):
        """wait_ready is required on its own, so a death-only handle cannot be instantiated."""

        class DeadOnly(BaseWorkerHandle):
            async def wait_dead(self, *, timeout: float) -> None: ...

        with pytest.raises(TypeError):
            DeadOnly()


class TestWorkerUnreachableError:
    def test_is_plain_exception_without_submission_state(self) -> None:
        """The error carries only its message and standard exception state."""
        error = WorkerUnreachableError("boom")

        assert str(error) == "boom"
        assert not hasattr(error, "submitted")


class _FakeRemoteMethod:
    def __init__(self, coro_factories):
        self._coro_factories = list(coro_factories)
        self.call_count = 0
        self.args_seen: list[tuple] = []
        self.kwargs_seen: list[dict] = []

    def remote(self, *args, **kwargs):
        self.args_seen.append(args)
        self.kwargs_seen.append(kwargs)
        self.call_count += 1
        index = min(self.call_count - 1, len(self._coro_factories) - 1)
        return self._coro_factories[index]()


def _return_factory(value):
    async def _coro():
        return value

    return _coro


def _raise_factory(exc):
    async def _coro():
        raise exc

    return _coro


def _hang_factory():
    return asyncio.sleep(3600)


def _never_resolving_factory():
    return asyncio.get_running_loop().create_future()


def _ray_actor_error():
    return ray.exceptions.RayActorError()


def _ray_task_error():
    return ray.exceptions.RayTaskError.__new__(ray.exceptions.RayTaskError)


def _ray_actor_unavailable_error():
    return ray.exceptions.ActorUnavailableError("gcs connection reset", None)


def _make_handle(**methods) -> tuple[RayWorkerHandle, SimpleNamespace]:
    inner = SimpleNamespace(**methods)
    return RayWorkerHandle(inner), inner


def _make_monotonic(values):
    seq = list(values)
    state = {"i": 0}

    def _monotonic():
        i = state["i"]
        if i < len(seq):
            state["i"] += 1
            return seq[i]
        return seq[-1]

    return _monotonic


@pytest.mark.asyncio
class TestRayWorkerHandleDispatch:
    async def test_forwards_kwargs_and_returns_the_result(self):
        """The handle relays a keyword-only call to the underlying actor method."""
        handle, inner = _make_handle(echo=_FakeRemoteMethod([_return_factory(7)]))

        result = await handle.echo(value=7)

        assert result == 7
        assert inner.echo.kwargs_seen == [{"value": 7}]

    async def test_forwards_positional_args_unchanged(self):
        """Callers may pass positionally, exactly as they would on the in-process object."""
        handle, inner = _make_handle(echo=_FakeRemoteMethod([_return_factory(7)]))

        result = await handle.echo(7, flag=True)

        assert result == 7
        assert inner.echo.args_seen == [(7,)]
        assert inner.echo.kwargs_seen == [{"flag": True}]

    async def test_actor_death_is_reported_as_unreachable(self):
        """RayActorError means the worker process is gone, not that the call was bad."""
        handle, _inner = _make_handle(echo=_FakeRemoteMethod([_raise_factory(_ray_actor_error())]))

        with pytest.raises(WorkerUnreachableError):
            await handle.echo(value=1)

    async def test_application_errors_propagate_unchanged(self):
        """A failure inside the worker method must reach the caller as-is."""
        handle, _inner = _make_handle(train=_FakeRemoteMethod([_raise_factory(ValueError("boom"))]))

        with pytest.raises(ValueError, match="boom"):
            await handle.train(rollout_id=1)

    async def test_dunder_lookup_raises_attribute_error(self):
        """Serialization probes dunder attributes, which must not become remote calls."""
        handle, _inner = _make_handle()

        with pytest.raises(AttributeError):
            _ = handle.__custom_dunder__


@pytest.mark.asyncio
class TestRayWorkerHandleWaitReady:
    async def test_returns_when_the_probe_succeeds(self):
        """A constructed actor answers the readiness probe immediately."""
        handle, inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_return_factory(None)]))

        await handle.wait_ready(timeout=1.0)

        assert inner.__ray_ready__.call_count == 1

    async def test_actor_death_is_reported_as_unreachable(self):
        """A worker that died during startup can never become ready."""
        handle, _inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_raise_factory(_ray_actor_error())]))

        with pytest.raises(WorkerUnreachableError):
            await handle.wait_ready(timeout=1.0)

    async def test_a_hung_probe_times_out_as_unreachable(self):
        """A worker that never answers within the deadline is unreachable."""
        handle, _inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_hang_factory]))

        with pytest.raises(WorkerUnreachableError):
            await handle.wait_ready(timeout=0.01)

    async def test_allowing_the_uuid_to_change_is_accepted_and_changes_nothing(self):
        """Every run takes this path at startup, and refusing the flag aborted the ray backend there."""
        handle, inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_return_factory(None)]))

        await handle.wait_ready(timeout=1.0, allow_server_uuid_change=True)

        assert inner.__ray_ready__.call_count == 1


@pytest.mark.asyncio
class TestRayWorkerHandleWaitIdle:
    async def test_a_ray_actor_cannot_be_waited_out(self):
        """A ray actor tracks no calls, so this backend has to fail loudly rather than report a worker idle."""
        handle, _inner = _make_handle()

        with pytest.raises(NotImplementedError, match="rpc communication backend"):
            await handle.wait_idle(timeout=1.0)


@pytest.mark.asyncio
class TestRayWorkerHandleWaitDead:
    async def test_returns_immediately_when_actor_error_on_first_probe(self):
        """A dead actor whose first probe raises RayActorError is confirmed dead after one probe."""
        handle, inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_raise_factory(_ray_actor_error())]))

        await handle.wait_dead(timeout=120.0)

        assert inner.__ray_ready__.call_count == 1

    async def test_returns_immediately_when_task_error_on_first_probe(self):
        """RayTaskError on the first probe is also treated as confirmed actor death."""
        handle, inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_raise_factory(_ray_task_error())]))

        await handle.wait_dead(timeout=120.0)

        assert inner.__ray_ready__.call_count == 1

    async def test_a_temporarily_unavailable_actor_is_not_confirmed_dead(self, monkeypatch):
        """ActorUnavailableError means the actor may still be alive, so the probe is inconclusive and retried."""
        monkeypatch.setattr(worker_handle_module, "_WAIT_DEAD_PROBE_INTERVAL_SECONDS", 0.01)
        handle, inner = _make_handle(
            __ray_ready__=_FakeRemoteMethod(
                [_raise_factory(_ray_actor_unavailable_error()), _raise_factory(_ray_actor_error())]
            )
        )

        await asyncio.wait_for(handle.wait_dead(timeout=120.0), timeout=10.0)

        assert inner.__ray_ready__.call_count == 2

    async def test_a_persistently_unavailable_actor_only_gives_up_at_the_deadline(self, monkeypatch, caplog):
        """Unavailability alone never confirms death; the loop keeps probing until the deadline expires."""
        slept: list[float] = []

        async def _noop_sleep(seconds: float) -> None:
            slept.append(seconds)

        monkeypatch.setattr(worker_handle_module.asyncio, "sleep", _noop_sleep)
        monkeypatch.setattr(
            worker_handle_module, "time", SimpleNamespace(monotonic=_make_monotonic([0.0, 1.0, 2.0, 200.0]))
        )
        handle, inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_raise_factory(_ray_actor_unavailable_error())]))

        with caplog.at_level(logging.ERROR, logger=worker_handle_module.__name__):
            await handle.wait_dead(timeout=120.0)

        assert inner.__ray_ready__.call_count == 3
        assert "Timed out" in caplog.text

    async def test_a_hung_probe_is_timed_out_and_retried(self, monkeypatch):
        """A probe that never answers is abandoned on the probe budget, and the next probe confirms death."""
        monkeypatch.setattr(worker_handle_module, "_WAIT_DEAD_PROBE_INTERVAL_SECONDS", 0.01)
        handle, inner = _make_handle(
            __ray_ready__=_FakeRemoteMethod([_never_resolving_factory, _raise_factory(_ray_actor_error())])
        )

        await asyncio.wait_for(handle.wait_dead(timeout=120.0), timeout=10.0)

        assert inner.__ray_ready__.call_count == 2

    async def test_retries_after_timeout_then_confirms_death(self, monkeypatch):
        """A probe timeout is tolerated; the loop retries and confirms death on the next probe."""
        slept = []

        async def _noop_sleep(seconds):
            slept.append(seconds)

        monkeypatch.setattr(worker_handle_module.asyncio, "sleep", _noop_sleep)
        monkeypatch.setattr(worker_handle_module, "time", SimpleNamespace(monotonic=_make_monotonic([0.0, 1.0])))
        handle, inner = _make_handle(
            __ray_ready__=_FakeRemoteMethod(
                [
                    _raise_factory(asyncio.TimeoutError()),
                    _raise_factory(_ray_actor_error()),
                ]
            )
        )

        await handle.wait_dead(timeout=120.0)

        assert inner.__ray_ready__.call_count == 2
        assert slept == [1.0]

    async def test_a_live_worker_is_never_confirmed_dead_before_the_deadline(self, monkeypatch, caplog):
        """A worker whose readiness probe keeps succeeding is not reported dead until the deadline expires."""
        slept: list[float] = []

        async def _noop_sleep(seconds: float) -> None:
            slept.append(seconds)

        monkeypatch.setattr(worker_handle_module.asyncio, "sleep", _noop_sleep)
        monkeypatch.setattr(
            worker_handle_module, "time", SimpleNamespace(monotonic=_make_monotonic([0.0, 60.0, 200.0]))
        )
        handle, inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_return_factory(None)]))

        with caplog.at_level(logging.ERROR, logger="miles.utils.workers.ray_worker_handle"):
            await handle.wait_dead(timeout=120.0)

        assert inner.__ray_ready__.call_count == 2
        assert slept == [1.0]
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) == 1
        assert "Timed out after 120s waiting for" in error_records[0].getMessage()

    async def test_deadline_reached_returns_and_logs_error(self, monkeypatch, caplog):
        """When the timeout deadline is exceeded after a hung probe, it returns and logs an ERROR."""

        async def _noop_sleep(seconds):
            return None

        monkeypatch.setattr(worker_handle_module.asyncio, "sleep", _noop_sleep)
        monkeypatch.setattr(worker_handle_module, "time", SimpleNamespace(monotonic=_make_monotonic([0.0, 200.0])))
        handle, inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_raise_factory(asyncio.TimeoutError())]))

        with caplog.at_level(logging.ERROR, logger="miles.utils.workers.ray_worker_handle"):
            await handle.wait_dead(timeout=120.0)

        assert inner.__ray_ready__.call_count == 1
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) == 1
        assert "Timed out after 120s waiting for" in error_records[0].getMessage()


class TestRpcWorkerHandleWaitDead:
    @pytest.mark.asyncio
    async def test_a_server_that_cannot_be_reached_is_confirmed_dead(self):
        """A cell heals once its ranks are gone, and nothing answering at their address is that proof."""
        handle = RpcWorkerHandle(_ProbeableWorker, server_url="http://127.0.0.1:1")

        await handle.wait_dead(timeout=1.0)
