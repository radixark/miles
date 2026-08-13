from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")


import asyncio
import inspect
from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args as _make_args

import miles.ray.rollout.eval_fleet as eval_fleet_mod
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.ray.rollout.eval_fleet import (
    EvalFleetInfo,
    EvalFleetPin,
    InferenceControllerEvalFleet,
    RolloutExecutorEvalFleet,
)
from miles.ray.rollout.rollout_server import RolloutServer
from miles.rollout.checkpoint_eval import EvalSkip
from miles.utils.context_lock import ContextLock
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


class FlakyEngine(FakeEngine):
    def __init__(self, log, *, failures: int):
        super().__init__(log)
        self.remaining_failures = failures

    async def update_weights_from_disk(self, model_path, load_format=None, weight_version=None):
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise RuntimeError("transient http failure")
        await super().update_weights_from_disk(model_path, load_format=load_format, weight_version=weight_version)


class HangingEngine(FakeEngine):
    async def update_weights_from_disk(self, model_path, load_format=None, weight_version=None):
        self.log.append(("update_weights_from_disk", (model_path,), dict(weight_version=weight_version)))
        await asyncio.Event().wait()


@pytest.fixture
def router_probes(monkeypatch):
    probes: list[SimpleNamespace] = []

    async def record_probe(url, *, json_payload=None, timeout=180.0):
        probes.append(SimpleNamespace(url=url, json_payload=json_payload, timeout=timeout))

    monkeypatch.setattr(eval_fleet_mod, "wait_http_ok", record_probe)
    return probes


@pytest.fixture
def router_ready(monkeypatch):
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
    async def test_pins_every_engine_before_reporting_success(self, router_ready):
        """Every engine is reloaded from the snapshot before the pin reports no skip."""
        log = []
        fleet = make_fleet(make_args(), [FakeEngine(log), FakeEngine(log)])

        pin = await fleet.pin("/snap/step_5", "5")

        load_events = [e for e in log if e[0] == "update_weights_from_disk"]
        assert len(load_events) == 2
        assert all(e[2]["weight_version"] == "5" for e in load_events)
        assert pin == EvalFleetPin(skip_reason=None)

    async def test_requires_all_engines_to_match_and_retries(self, router_ready):
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

    async def test_a_cell_that_joined_between_attempts_is_pinned_too(self, router_ready):
        """Membership is read again per attempt, or a cell that joined mid-pin would serve the old weights."""
        log = []
        joined, stale = FakeEngine(log), FakeEngine(log)
        stale.get_weight_version = _answers_version("999")
        server = FakeEvalServer([stale])
        fleet = InferenceControllerEvalFleet(make_args(), srv=server)

        async def join_after_first_attempt(*_args, **_kwargs):
            server._engines = [joined]
            stale.get_weight_version = _answers_version("5")

        stale.update_weights_from_disk = join_after_first_attempt

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin == EvalFleetPin(skip_reason=None)
        assert joined.weight_version == "5"

    async def test_does_not_health_probe_the_server(self, router_ready):
        """The eval fleet has no fault tolerance: pin goes straight to the weight load."""
        server = FakeEvalServer([FakeEngine([])])
        assert not any(hasattr(server, name) for name in ("probe_and_mark_dead", "recover", "wait_all_engines_alive"))

        pin = await InferenceControllerEvalFleet(make_args(), srv=server).pin("/snap/step_5", "5")

        assert pin == EvalFleetPin(skip_reason=None)


class TestPinWeightTransport:
    async def test_pin_sends_the_checkpoint_dir_as_the_http_model_path(self, router_ready):
        """The snapshot directory reaches every engine as the model path of the weight load, not as any other field."""
        log = []
        fleet = make_fleet(make_args(), [FakeEngine(log), FakeEngine(log)])

        await fleet.pin("/snap/step_7", "7")

        assert [e for e in log if e[0] == "update_weights_from_disk"] == [
            ("update_weights_from_disk", ("/snap/step_7",), dict(weight_version="7")),
            ("update_weights_from_disk", ("/snap/step_7",), dict(weight_version="7")),
        ]

    async def test_the_calls_pin_makes_bind_against_the_real_api_client(self):
        """The fleet talks HTTP now, so the two calls it makes must stay bindable against the real client."""
        client = SGLangApiClient(server_url="http://10.0.0.2:31000")

        inspect.signature(SGLangApiClient.update_weights_from_disk).bind(
            client, model_path="/snap/step_7", weight_version="7"
        )
        inspect.signature(SGLangApiClient.get_weight_version).bind(client)

        assert inspect.iscoroutinefunction(SGLangApiClient.update_weights_from_disk)
        assert inspect.iscoroutinefunction(SGLangApiClient.get_weight_version)


class TestPinFailureModes:
    async def test_pin_retries_a_transient_load_failure_and_then_succeeds(self, router_ready):
        """One engine failing its first weight load is retried, and the eval point still runs."""
        log = []
        flaky = FlakyEngine(log, failures=1)
        fleet = make_fleet(make_args(), [FakeEngine(log), flaky])

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin == EvalFleetPin(skip_reason=None)
        assert flaky.remaining_failures == 0

    async def test_pin_skips_when_every_attempt_fails(self, router_ready):
        """A fleet that never loads its weights must degrade to an attributable skip, not a crash."""
        log = []
        fleet = make_fleet(make_args(), [FlakyEngine(log, failures=99)])

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin.skip_reason == "pin_violation"
        assert [e for e in log if e[0] == "get_weight_version"] == []

    async def test_pin_skips_when_a_weight_load_hangs_past_the_timeout(self, router_ready, monkeypatch):
        """A wedged engine must not park the controller event loop forever, so the load is deadlined."""
        monkeypatch.setattr(eval_fleet_mod, "EVAL_WEIGHT_LOAD_TIMEOUT_SECS", 0.01)
        log = []
        fleet = make_fleet(make_args(), [HangingEngine(log)])

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin.skip_reason == "pin_violation"
        assert len([e for e in log if e[0] == "update_weights_from_disk"]) == 2

    async def test_pin_skips_when_the_fleet_reads_no_client_at_all(self, router_ready):
        """An empty membership proves nothing about the served weights, so it must not pass the check vacuously."""
        fleet = make_fleet(make_args(), [])

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin.skip_reason == "pin_violation"


class TestRouterProbe:
    async def test_pin_probes_the_router_address_of_the_server_it_watches(self, router_probes):
        """The fleet owns no address of its own, so the probe must use the one its server reports."""
        server = FakeEvalServer([FakeEngine([])])
        server.router_ip, server.router_port = "10.9.9.9", 39999
        fleet = InferenceControllerEvalFleet(make_args(), srv=server)

        pin = await fleet.pin("/snap/step_5", "5")

        assert pin == EvalFleetPin(skip_reason=None)
        assert [probe.url for probe in router_probes] == ["http://10.9.9.9:39999/generate"]
        assert router_probes[0].json_payload["sampling_params"]["max_new_tokens"] == 1


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

    async def test_the_controller_is_resolved_again_for_every_point(self, fleet_states):
        """A controller that restarted answers on a new handle, and a session that kept the old one never heals."""
        first, second = (
            FakeInferenceController([EvalFleetPin(skip_reason=None)]),
            FakeInferenceController([EvalFleetPin(skip_reason=None)]),
        )
        provider = FakeControllerProvider([first, second])
        session = make_session_over(provider)

        await session.pin("/snap/step_5", "5")
        await session.pin("/snap/step_6", "6")

        assert (provider.lookups, len(first.calls), len(second.calls)) == (2, 1, 1)
