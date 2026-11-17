"""A Miles actor whose message reaches an actor that has stopped keeps running; only the call fails."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import pkgutil
from types import SimpleNamespace

import pytest

pytest.importorskip("monarch")

from monarch._rust_bindings.monarch_hyperactor.supervision import SupervisionError
from monarch.actor import Actor, InvalidReferenceReason, MeshFailure, endpoint, this_host

from tests.ci.ci_register import register_cpu_ci
from tests.fast.utils.workers.monarch.monarch_context import in_monarch_context

import miles.utils.workers.monarch as monarch_package
from miles.utils.workers.monarch.base import MilesActor

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["rpc-comm"])

_CALL_TIMEOUT_SECONDS = 30.0


class _Target(Actor):
    @endpoint
    async def ping(self) -> bool:
        return True


class _CallsStoppedTargets:
    """The body both callers share: owns a mesh of targets, as the worker manager owns its cells' meshes."""

    def __init__(self, hosts) -> None:
        self._hosts = hosts

    async def __supervise__(self, failure: MeshFailure) -> bool:
        return True  # as the worker manager does: a lost process is left to the liveness scan

    @endpoint
    async def call_stopped_targets(self) -> str:
        procs = self._hosts.spawn_procs(per_host={"procs": 2}, name="targets")
        targets = procs.spawn("target", _Target)
        await targets.ping.call()
        await targets.stop()
        try:
            await asyncio.wait_for(targets.ping.call(), timeout=_CALL_TIMEOUT_SECONDS)
            outcome = "answered"
        except SupervisionError:
            outcome = "raised"
        except (TimeoutError, asyncio.TimeoutError):
            outcome = "timed out"
        await asyncio.sleep(1)  # the delivery failures reach this actor after its call has failed
        await procs.stop()
        return outcome

    @endpoint
    async def alive(self) -> bool:
        return True


class _MilesCaller(_CallsStoppedTargets, MilesActor):
    pass


class _PlainCaller(_CallsStoppedTargets, Actor):
    pass


class _Owner(Actor):
    """Spawns a caller and absorbs its failure, so a caller Monarch fails does not take the test process down."""

    def __init__(self, hosts) -> None:
        self._hosts = hosts

    async def __supervise__(self, failure: MeshFailure) -> bool:
        return True

    @endpoint
    async def run(self, caller_class: type) -> tuple[str, bool]:
        caller = self._hosts.spawn_procs(per_host={"procs": 1}, name="caller").spawn(
            "caller", caller_class, self._hosts
        )
        try:
            outcome = await caller.call_stopped_targets.call_one()
            return outcome, await caller.alive.call_one()
        except SupervisionError:
            return "caller failed", False


def _run_caller(caller_class: type) -> tuple[str, bool]:
    def run():
        hosts = this_host()
        owner = hosts.spawn_procs(per_host={"procs": 1}, name="owner").spawn("owner", _Owner, hosts)
        return owner.run.call_one(caller_class).get()

    return in_monarch_context(run)


class TestACallToAStoppedActor:
    def test_fails_the_call_but_not_a_miles_actor_that_made_it(self):
        assert _run_caller(_MilesCaller) == ("raised", True)

    def test_a_plain_actor_is_failed_by_monarch(self):
        """Pins the Monarch default MilesActor works around: if this starts failing, Monarch no longer fails the
        sender of a call to a stopped actor, and MilesActor._handle_invalid_reference can go."""
        assert _run_caller(_PlainCaller) == ("caller failed", False)


def _failure_with(reason: InvalidReferenceReason | None):
    return SimpleNamespace(root_delivery_failure=lambda: SimpleNamespace(invalid_reference_reason=reason))


class TestWhichDeliveryFailuresAreAbsorbed:
    @pytest.mark.parametrize("reason", [InvalidReferenceReason.ActorStopped, InvalidReferenceReason.ActorFailed])
    def test_a_destination_that_is_gone(self, reason: InvalidReferenceReason):
        assert MilesActor._handle_invalid_reference(None, _failure_with(reason)) is True

    @pytest.mark.parametrize(
        "reason",
        [
            InvalidReferenceReason.HandlerNotBound,
            InvalidReferenceReason.ProtocolMismatch,
            InvalidReferenceReason.ProcNotExist,
        ],
    )
    def test_anything_else_stays_fatal(self, reason: InvalidReferenceReason):
        assert MilesActor._handle_invalid_reference(None, _failure_with(reason)) is False

    def test_a_report_without_its_root_failure_stays_fatal(self):
        failure = SimpleNamespace(root_delivery_failure=lambda: None)
        assert MilesActor._handle_invalid_reference(None, failure) is False


def test_every_monarch_actor_miles_defines_is_a_miles_actor():
    actor_classes = {
        cls
        for module_info in pkgutil.iter_modules(monarch_package.__path__)
        for _, cls in inspect.getmembers(importlib.import_module(f"{monarch_package.__name__}.{module_info.name}"))
        if inspect.isclass(cls) and issubclass(cls, Actor) and cls.__module__.startswith(monarch_package.__name__)
    }

    assert actor_classes, "found no Monarch actor to check"
    assert {cls.__name__ for cls in actor_classes if not issubclass(cls, MilesActor)} == set()
