from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")


import pytest
from tests.fast.ray.rollout.conftest import make_args as _make_args

import miles.ray.rollout.eval_fleet as eval_fleet_mod
from miles.ray.rollout.eval_fleet import (
    EvalFleetInfo,
    EvalFleetPin,
    InferenceControllerEvalFleet,
    RolloutExecutorEvalFleet,
)
from miles.ray.rollout.rollout_server import RolloutServer
from miles.rollout.checkpoint_eval import EvalSkip
from miles.utils.context_lock import ContextLock
from miles.utils.workers.rpc.client.misc import RpcWorkerCallError, ServerRestartedError
from miles.utils.workers.worker_handle import WorkerUnreachableError
from miles.utils.workers.worker_spec import HostAndPort


def make_args(**overrides):
    defaults = dict(
        eval_num_gpus=1,
        eval_num_gpus_per_engine=1,
        sglang_model_routers={"default": ("10.0.0.1", 30000), "eval": ("10.0.0.2", 31000)},
    )
    defaults.update(overrides)
    return _make_args(**defaults)


class FakeEngine:
    """Stands in for the api client of one eval cell, with the two methods a pin calls."""

    def __init__(self, log):
        self.log = log
        self.weight_version = None

    async def update_weights_from_disk(self, model_path, load_format=None, weight_version=None):
        self.log.append(("update_weights_from_disk", (model_path,), dict(weight_version=weight_version)))
        self.weight_version = weight_version

    async def get_weight_version(self):
        self.log.append(("get_weight_version", (), {}))
        return self.weight_version


class FakeEvalServer:
    def __init__(self, engines):
        self._engines = engines
        self.context_lock = ContextLock("FakeEvalServer")
        self.router_ip = "10.0.0.2"
        self.router_port = 31000

    @property
    def api_clients(self):
        assert self.context_lock.held_in_current_context, "api_clients is read under the server's lock"
        return list(self._engines)


@pytest.fixture
def router_always_ready(monkeypatch):
    async def noop_router_ready(self, timeout=180.0):
        return None

    monkeypatch.setattr(eval_fleet_mod.InferenceControllerEvalFleet, "_wait_router_ready", noop_router_ready)


def make_fleet(args, engines):
    return InferenceControllerEvalFleet(args, srv=FakeEvalServer(engines))


class TestEvalFleetInfo:
    def test_describes_the_fleet_its_router_serves(self):
        """The description the executor retargets its eval args to comes from the server, not its own args."""
        fleet = make_fleet(make_args(eval_num_gpus=4, eval_num_gpus_per_engine=2), [])

        assert fleet.info == EvalFleetInfo(
            router=HostAndPort(host="10.0.0.2", port=31000), num_gpus=4, num_gpus_per_engine=2
        )


def _answers_version(version: str):
    async def get_weight_version():
        return version

    return get_weight_version


class TestTheFleetIsBuiltOnTheRealServer:
    def test_everything_the_fake_server_offers_a_real_one_offers_too(self):
        """A fake with an interface of its own is how the pin path came to call `.engines`, which no server has."""
        fake = FakeEvalServer([])
        offered = {name for name in (*vars(fake), *type(fake).__dict__) if not name.startswith("_")}

        assert offered <= set(dir(RolloutServer)) | set(RolloutServer.__annotations__)


class TestEvalFleetPinning:
    async def test_pins_every_engine_before_reporting_success(self, router_always_ready):
        """Every engine is reloaded from the snapshot before the pin reports no skip."""
        log = []
        fleet = make_fleet(make_args(), [FakeEngine(log), FakeEngine(log)])

        pin = await fleet.pin("/snap/step_5", "5")

        load_events = [e for e in log if e[0] == "update_weights_from_disk"]
        assert len(load_events) == 2
        assert all(e[2]["weight_version"] == "5" for e in load_events)
        assert pin == EvalFleetPin(skip_reason=None)

    async def test_requires_all_engines_to_match_and_retries(self, router_always_ready):
        """The router load-balances across engines, so one stale engine = mixed
        versions: the pin must fail even when the other engine matches, retry once,
        then degrade to an attributable skip."""
        log = []
        good, stale = FakeEngine(log), FakeEngine(log)
        stale.get_weight_version = _answers_version("999")
        fleet = make_fleet(make_args(), [good, stale])

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin.skip_reason == "pin_violation"
        assert len([e for e in log if e[0] == "update_weights_from_disk"]) == 4  # 2 engines x 2 attempts

    async def test_a_router_that_never_becomes_ready_skips_the_eval(self, monkeypatch):
        """Every other case here neutralises the router wait, so this branch is the only thing that reads it."""

        async def never_ready(self, timeout=180.0):
            raise TimeoutError("router never came up")

        monkeypatch.setattr(eval_fleet_mod.InferenceControllerEvalFleet, "_wait_router_ready", never_ready)
        fleet = make_fleet(make_args(), [FakeEngine([])])

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin.skip_reason == "unhealthy"

    async def test_a_cell_that_joined_between_attempts_is_pinned_too(self, router_always_ready):
        """Membership is read again per attempt, or a cell that joined mid-pin would serve the old weights."""
        log = []
        joined, stale = FakeEngine(log), FakeEngine(log)
        stale.get_weight_version = _answers_version("999")
        server = FakeEvalServer([stale])
        fleet = InferenceControllerEvalFleet(make_args(), srv=server)

        async def join_after_the_first_attempt(*_args, **_kwargs):
            server._engines = [joined]
            stale.get_weight_version = _answers_version("5")

        stale.update_weights_from_disk = join_after_the_first_attempt

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin == EvalFleetPin(skip_reason=None)
        assert joined.weight_version == "5"

    async def test_does_not_health_probe_the_server(self, router_always_ready):
        """The eval fleet has no fault tolerance: pin goes straight to the weight load."""
        server = FakeEvalServer([FakeEngine([])])
        assert not any(hasattr(server, name) for name in ("probe_and_mark_dead", "recover", "wait_all_engines_alive"))

        pin = await InferenceControllerEvalFleet(make_args(), srv=server).pin("/snap/step_5", "5")

        assert pin == EvalFleetPin(skip_reason=None)


class FakeInferenceController:
    def __init__(self, pins: list[EvalFleetPin]):
        self.calls: list[dict] = []
        self._pins = pins

    async def pin_eval_fleet(self, *, checkpoint_dir: str, weight_version: str) -> EvalFleetPin:
        self.calls.append(dict(checkpoint_dir=checkpoint_dir, weight_version=weight_version))
        pin = self._pins[len(self.calls) - 1]
        if isinstance(pin, Exception):
            raise pin
        return pin


class FakeControllerProvider:
    def __init__(self, controllers):
        self._controllers = controllers
        self.lookups = 0

    def get_handle(self, worker_name: str):
        self.lookups += 1
        if isinstance(handle := self._controllers[min(self.lookups, len(self._controllers)) - 1], Exception):
            raise handle
        return handle


@pytest.fixture
def fleet_states(monkeypatch):
    built = []
    monkeypatch.setattr(eval_fleet_mod, "GenerateState", lambda args: built.append(args) or f"fake-state-{len(built)}")
    return built


def make_session(controller, *, info=None):
    return make_session_over(FakeControllerProvider([controller]), info=info)


def make_session_over(provider, *, info=None):
    return RolloutExecutorEvalFleet(
        make_args(),
        info=info or EvalFleetInfo(router=HostAndPort(host="10.0.0.2", port=31000), num_gpus=2, num_gpus_per_engine=1),
        inference_controller_provider=provider,
    )


class TestRolloutExecutorEvalFleet:
    def test_builds_its_state_against_the_fleet_router(self, fleet_states):
        """The executor generates against the eval router and the fleet's gpu sizing, not the rollout ones."""
        make_session(FakeInferenceController([]))

        (state_args,) = fleet_states
        assert (state_args.sglang_router_ip, state_args.sglang_router_port) == ("10.0.0.2", 31000)
        assert (state_args.rollout_num_gpus, state_args.rollout_num_gpus_per_engine) == (2, 1)

    async def test_pins_over_rpc_and_returns_the_cached_state(self, fleet_states):
        """Pinning is the controller's call; the state is built once and handed back per point."""
        controller = FakeInferenceController([EvalFleetPin(skip_reason=None), EvalFleetPin(skip_reason=None)])
        session = make_session(controller)

        first = await session.pin("/snap/step_5", "5")
        second = await session.pin("/snap/step_6", "6")

        assert controller.calls == [
            dict(checkpoint_dir="/snap/step_5", weight_version="5"),
            dict(checkpoint_dir="/snap/step_6", weight_version="6"),
        ]
        assert first == second == "fake-state-1"
        assert len(fleet_states) == 1

    async def test_a_remote_skip_stays_an_attributable_skip(self, fleet_states):
        """The reason the controller skipped for must survive the wire as EvalSkip."""
        session = make_session(FakeInferenceController([EvalFleetPin(skip_reason="pin_violation")]))

        with pytest.raises(EvalSkip) as exc:
            await session.pin("/snap/step_5", "5")

        assert exc.value.reason == "pin_violation"

    async def test_the_controller_is_resolved_once_and_the_handle_reused(self, fleet_states):
        """Resolving a fresh handle per point would hand every point a fresh boot-uuid pin, so it is resolved once."""
        first, second = (
            FakeInferenceController([EvalFleetPin(skip_reason=None), EvalFleetPin(skip_reason=None)]),
            FakeInferenceController([EvalFleetPin(skip_reason=None)]),
        )
        provider = FakeControllerProvider([first, second])
        session = make_session_over(provider)

        await session.pin("/snap/step_5", "5")
        await session.pin("/snap/step_6", "6")

        assert (provider.lookups, len(first.calls), len(second.calls)) == (1, 2, 0)

    async def test_a_controller_that_cannot_be_reached_skips_the_point(self, fleet_states):
        """Losing the controller must skip one eval point, not raise into the driver's rollout loop."""
        session = make_session(FakeInferenceController([WorkerUnreachableError("controller is gone")]))

        with pytest.raises(EvalSkip) as exc:
            await session.pin("/snap/step_5", "5")

        assert exc.value.reason == "controller_unreachable"

    async def test_a_controller_that_answered_with_a_failure_is_not_a_skip(self, fleet_states):
        """A method that raised inside a reachable controller is our bug, and skipping every point would hide it."""
        session = make_session(FakeInferenceController([RpcWorkerCallError("assert self._eval_fleet is not None")]))

        with pytest.raises(RpcWorkerCallError):
            await session.pin("/snap/step_5", "5")

    async def test_a_timed_out_controller_skips_the_point(self, fleet_states):
        """A controller that never answers is unreachable in every way that matters to one eval point."""
        session = make_session(FakeInferenceController([TimeoutError("no answer")]))

        with pytest.raises(EvalSkip) as exc:
            await session.pin("/snap/step_5", "5")

        assert exc.value.reason == "controller_unreachable"

    async def test_the_pin_keeps_reporting_a_restarted_controller_across_points(self, fleet_states):
        """The kept handle keeps its boot-uuid pin, so a later point still sees the restart instead of a fresh pin."""
        controller = FakeInferenceController(
            [ServerRestartedError("boot uuid changed"), ServerRestartedError("boot uuid changed")]
        )
        provider = FakeControllerProvider([controller])
        session = make_session_over(provider)

        for checkpoint_dir, weight_version in (("/snap/step_5", "5"), ("/snap/step_6", "6")):
            with pytest.raises(EvalSkip):
                await session.pin(checkpoint_dir, weight_version)

        assert (provider.lookups, len(controller.calls)) == (1, 2)
