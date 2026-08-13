from __future__ import annotations

import threading
from typing import Any

import pytest

from miles.utils.workers.serving import serve_actor as serve_actor_module
from miles.utils.workers.serving.serve_actor import ServeActor, serve_until_stopped


class DemoWorker:
    def __init__(self, *, tag: str) -> None:
        self.tag = tag

    def echo(self, *, value: int) -> int:
        return value


def _build_worker(tag: str = "demo"):
    return lambda: DemoWorker(tag=tag)


@pytest.fixture
def started_threads(monkeypatch) -> list[dict[str, Any]]:
    started: list[dict[str, Any]] = []

    class _FakeThread:
        def __init__(self, *, target, kwargs, name, daemon) -> None:
            started.append(dict(target=target, kwargs=kwargs, name=name, daemon=daemon))

        def start(self) -> None:
            pass

    monkeypatch.setattr(serve_actor_module.threading, "Thread", _FakeThread)
    return started


class TestTheWorkerLivesInTheActor:
    def test_the_worker_is_built_when_the_actor_is_constructed(self):
        """RDT and the rest of the ray ecosystem need the worker inside the actor process, not beside it."""
        actor = ServeActor(build_worker=_build_worker(tag="built-here"))

        assert isinstance(actor._worker, DemoWorker) and actor._worker.tag == "built-here"

    def test_the_recipe_is_evaluated_exactly_once(self):
        """A second evaluation would build a second worker and serve one while the other holds the gpu."""
        calls: list[int] = []

        ServeActor(build_worker=lambda: calls.append(1) or DemoWorker(tag="x"))

        assert calls == [1]


class TestStartingTheServer:
    def test_the_server_runs_in_a_background_thread(self, started_threads):
        """The actor must stay answerable: the launcher still probes ports and kills it through ray."""
        actor = ServeActor(build_worker=_build_worker())

        actor.start_rpc_server(port=12345)

        assert len(started_threads) == 1
        assert started_threads[0]["daemon"] is True
        assert started_threads[0]["kwargs"]["port"] == 12345

    def test_the_app_serves_the_actor_s_own_worker(self, started_threads):
        """Serving a different object would answer calls that never reach the worker holding the gpu."""
        actor = ServeActor(build_worker=_build_worker())

        actor.start_rpc_server(port=12345)

        assert started_threads[0]["kwargs"]["app"].routes

    def test_the_server_can_only_be_started_once(self, started_threads):
        """Two servers on one worker would each bind a port and answer with a different boot uuid."""
        actor = ServeActor(build_worker=_build_worker())
        actor.start_rpc_server(port=12345)

        with pytest.raises(AssertionError):
            actor.start_rpc_server(port=12346)

    def test_the_server_thread_is_named_for_the_stacks_it_appears_in(self, started_threads):
        """A hung train step is read off thread names, and 'Thread-7' says nothing."""
        actor = ServeActor(build_worker=_build_worker())

        actor.start_rpc_server(port=12345)

        assert "rpc" in started_threads[0]["name"]


class TestTheActorDiesWithItsServer:
    def test_a_server_that_returns_takes_the_process_with_it(self, monkeypatch):
        """A live actor whose server stopped is a worker nobody can call and nothing reports as dead."""
        exits: list[int] = []
        monkeypatch.setattr(serve_actor_module.uvicorn, "run", lambda *args, **kwargs: None)
        monkeypatch.setattr(serve_actor_module.os, "_exit", lambda code: exits.append(code))

        serve_until_stopped(app=object(), port=12345)

        assert exits == [1]

    def test_a_server_that_crashes_takes_the_process_with_it(self, monkeypatch):
        """Binding an already taken port raises rather than returns, and must end the same way."""
        exits: list[int] = []

        def _explode(*args, **kwargs):
            raise OSError("address already in use")

        monkeypatch.setattr(serve_actor_module.uvicorn, "run", _explode)
        monkeypatch.setattr(serve_actor_module.os, "_exit", lambda code: exits.append(code))

        serve_until_stopped(app=object(), port=12345)

        assert exits == [1]

    def test_the_server_binds_every_interface(self, monkeypatch):
        """The driver reaches the worker by the node ip the launcher advertised, never by loopback."""
        seen: list[dict[str, Any]] = []
        monkeypatch.setattr(serve_actor_module.uvicorn, "run", lambda app, **kwargs: seen.append(kwargs) or None)
        monkeypatch.setattr(serve_actor_module.os, "_exit", lambda code: None)

        serve_until_stopped(app=object(), port=12345)

        assert seen[0] == dict(host="0.0.0.0", port=12345)


class TestFaultInjection:
    def test_the_actor_forwards_an_injected_fault_to_the_process(self, monkeypatch):
        """Fault injection targets the process the worker runs in, which is now the actor itself."""
        injected: list[str] = []
        monkeypatch.setattr(serve_actor_module, "_inject_fault", lambda *, mode: injected.append(mode))
        actor = ServeActor(build_worker=_build_worker())

        actor.inject_fault("hang")

        assert injected == ["hang"]


class TestTheRealThreadStarts:
    def test_start_rpc_server_really_spawns_a_thread(self, monkeypatch):
        """The fake thread of the other tests must not be the only proof that the server ever starts."""
        monkeypatch.setattr(serve_actor_module, "serve_until_stopped", lambda **kwargs: None)
        actor = ServeActor(build_worker=_build_worker())

        actor.start_rpc_server(port=12345)
        actor._server_thread.join(timeout=10.0)

        assert isinstance(actor._server_thread, threading.Thread) and not actor._server_thread.is_alive()
