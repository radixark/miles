from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio
import inspect
import pickle
from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args as _make_args

import miles.ray.rollout.eval_fleet as eval_fleet_mod
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.ray.rollout.eval_fleet import EvalFleet
from miles.rollout.checkpoint_eval import EvalSkip
from miles.utils.context_lock import ContextLock


def make_args(**overrides):
    defaults = dict(
        eval_num_gpus=1,
        eval_num_gpus_per_engine=1,
        sglang_model_routers={"default": ("10.0.0.1", 30000), "eval": ("10.0.0.2", 31000)},
    )
    defaults.update(overrides)
    return _make_args(**defaults)


class FakeApiClient:
    def __init__(self, log: list) -> None:
        self.log = log
        self.weight_version: str | None = None
        self.weight_version_response: str | None = None

    async def update_weights_from_disk(
        self, model_path: str, load_format: str | None = None, weight_version: str | None = None
    ) -> None:
        self.log.append(("update_weights_from_disk", model_path, weight_version))
        self.weight_version = weight_version

    async def get_weight_version(self) -> str | None:
        self.log.append(("get_weight_version", None, None))
        if self.weight_version_response is not None:
            return self.weight_version_response
        return self.weight_version


@pytest.fixture
def fleet_env(monkeypatch):
    state_builds = []

    async def noop_router_ready(self, timeout=180.0):
        return None

    monkeypatch.setattr(eval_fleet_mod.EvalFleet, "_wait_router_ready", noop_router_ready)
    monkeypatch.setattr(
        eval_fleet_mod,
        "GenerateState",
        lambda args: state_builds.append(args) or "fake-fleet-state",
    )
    return SimpleNamespace(state_builds=state_builds)


def make_fleet(args, api_clients):
    return EvalFleet(args, api_clients=api_clients, router_host="10.0.0.2", router_port=31000)


def reachable_values(obj: object) -> list:
    values = list(vars(obj).values())
    return values + [nested for value in values if hasattr(value, "__dict__") for nested in vars(value).values()]


async def test_fleet_pins_every_engine_before_returning_the_state(fleet_env):
    log = []
    fleet = make_fleet(make_args(), [FakeApiClient(log), FakeApiClient(log)])

    state = await fleet.pin("/snap/step_5", "5")

    load_events = [e for e in log if e[0] == "update_weights_from_disk"]
    assert len(load_events) == 2
    assert all(e[2] == "5" for e in load_events)
    # The caller cannot generate before the pin: the state only exists as pin's return.
    assert state == "fake-fleet-state"
    # Built once at construction, not per eval.
    assert len(fleet_env.state_builds) == 1
    await fleet.pin("/snap/step_6", "6")
    assert len(fleet_env.state_builds) == 1


async def test_fleet_pin_requires_all_match_and_retries(fleet_env):
    """The router load-balances across engines, so one stale engine = mixed
    versions: the pin must fail even when the other engine matches, retry once,
    then degrade to an attributable skip."""
    log = []
    good, stale = FakeApiClient(log), FakeApiClient(log)
    stale.weight_version_response = "999"
    fleet = make_fleet(make_args(), [good, stale])

    with pytest.raises(EvalSkip) as exc:
        await fleet.pin("/snap/step_5", "5")

    assert exc.value.reason == "pin_violation"
    assert len([e for e in log if e[0] == "update_weights_from_disk"]) == 4  # 2 engines x 2 attempts


async def test_fleet_pin_does_not_health_probe_the_server(fleet_env):
    """The eval fleet has no fault tolerance: pin goes straight to the weight load."""
    fleet = make_fleet(make_args(), [FakeApiClient([])])
    assert not any(hasattr(fleet, name) for name in ("probe_and_mark_dead", "recover", "wait_all_engines_alive"))

    state = await fleet.pin("/snap/step_5", "5")

    assert state == "fake-fleet-state"


async def test_fleet_holds_a_snapshot_rather_than_the_live_rollout_server(fleet_env):
    """set_eval_fleet pickles the fleet into the executor actor, and a captured asyncio lock would never survive that."""
    fleet = make_fleet(make_args(), [FakeApiClient([])])

    pickle.loads(pickle.dumps(fleet))

    assert not any(isinstance(value, (ContextLock, asyncio.Lock)) for value in reachable_values(fleet))


class FlakyApiClient(FakeApiClient):
    def __init__(self, log: list, *, failures: int) -> None:
        super().__init__(log)
        self.remaining_failures = failures

    async def update_weights_from_disk(
        self, model_path: str, load_format: str | None = None, weight_version: str | None = None
    ) -> None:
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise RuntimeError("transient http failure")
        await super().update_weights_from_disk(model_path, load_format=load_format, weight_version=weight_version)


class HangingApiClient(FakeApiClient):
    async def update_weights_from_disk(
        self, model_path: str, load_format: str | None = None, weight_version: str | None = None
    ) -> None:
        self.log.append(("update_weights_from_disk", model_path, weight_version))
        await asyncio.Event().wait()


@pytest.fixture
def router_probe_env(monkeypatch):
    probes = []

    async def record_probe(url: str, *, json_payload=None, timeout: float = 180.0) -> None:
        probes.append(SimpleNamespace(url=url, json_payload=json_payload, timeout=timeout))

    monkeypatch.setattr(eval_fleet_mod, "wait_http_ok", record_probe)
    monkeypatch.setattr(eval_fleet_mod, "GenerateState", lambda args: "fake-fleet-state")
    return SimpleNamespace(probes=probes)


class TestPinWeightTransport:
    async def test_pin_sends_the_checkpoint_dir_as_the_http_model_path(self, fleet_env):
        """The snapshot directory reaches every engine as the model path of the weight load, not as any other field."""
        log = []
        fleet = make_fleet(make_args(), [FakeApiClient(log), FakeApiClient(log)])

        await fleet.pin("/snap/step_7", "7")

        assert [e for e in log if e[0] == "update_weights_from_disk"] == [
            ("update_weights_from_disk", "/snap/step_7", "7"),
            ("update_weights_from_disk", "/snap/step_7", "7"),
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
    async def test_pin_retries_a_transient_load_failure_and_then_succeeds(self, fleet_env):
        """One engine failing its first weight load is retried, and the eval point still runs."""
        log = []
        flaky = FlakyApiClient(log, failures=1)
        fleet = make_fleet(make_args(), [FakeApiClient(log), flaky])

        state = await fleet.pin("/snap/step_5", "5")

        assert state == "fake-fleet-state"
        assert flaky.remaining_failures == 0

    async def test_pin_skips_when_every_attempt_fails(self, fleet_env):
        """A fleet that never loads its weights must degrade to an attributable skip, not a crash."""
        log = []
        fleet = make_fleet(make_args(), [FlakyApiClient(log, failures=99)])

        with pytest.raises(EvalSkip) as exc:
            await fleet.pin("/snap/step_5", "5")

        assert exc.value.reason == "pin_violation"
        assert [e for e in log if e[0] == "get_weight_version"] == []

    async def test_pin_skips_when_a_weight_load_hangs_past_the_timeout(self, fleet_env, monkeypatch):
        """A wedged engine must not park the manager event loop forever, so the load is deadlined."""
        monkeypatch.setattr(eval_fleet_mod, "EVAL_WEIGHT_LOAD_TIMEOUT_SECS", 0.01)
        log = []
        fleet = make_fleet(make_args(), [HangingApiClient(log)])

        with pytest.raises(EvalSkip) as exc:
            await fleet.pin("/snap/step_5", "5")

        assert exc.value.reason == "pin_violation"
        assert len([e for e in log if e[0] == "update_weights_from_disk"]) == 2

    async def test_pin_skips_when_the_fleet_snapshot_holds_no_clients(self, fleet_env):
        """An empty snapshot proves nothing about the served weights, so it must not pass the version check vacuously."""
        fleet = make_fleet(make_args(), [])

        with pytest.raises(EvalSkip) as exc:
            await fleet.pin("/snap/step_5", "5")

        assert exc.value.reason == "pin_violation"


class TestRouterProbe:
    async def test_pin_probes_the_router_address_handed_to_the_constructor(self, router_probe_env):
        """The fleet no longer owns a server object, so the probe must use the router address it was constructed with."""
        args = make_args(sglang_model_routers={"default": ("10.0.0.1", 30000), "eval": ("10.0.0.2", 31000)})
        fleet = EvalFleet(args, api_clients=[FakeApiClient([])], router_host="10.9.9.9", router_port=39999)

        await fleet.pin("/snap/step_5", "5")

        assert [probe.url for probe in router_probe_env.probes] == ["http://10.9.9.9:39999/generate"]
        assert router_probe_env.probes[0].json_payload["sampling_params"]["max_new_tokens"] == 1

    async def test_pin_skips_as_unhealthy_when_the_router_never_answers(self, router_probe_env, monkeypatch):
        """A router that keeps 503ing after a revival must skip the point rather than dispatch into a dead route."""

        async def failing_probe(url: str, *, json_payload=None, timeout: float = 180.0) -> None:
            raise TimeoutError(url)

        monkeypatch.setattr(eval_fleet_mod, "wait_http_ok", failing_probe)
        fleet = make_fleet(make_args(), [FakeApiClient([])])

        with pytest.raises(EvalSkip) as exc:
            await fleet.pin("/snap/step_5", "5")

        assert exc.value.reason == "unhealthy"


class TestClientSnapshot:
    async def test_the_fleet_ignores_clients_appended_after_construction(self, fleet_env):
        """The fleet keeps its own snapshot, so a later mutation of the caller's list cannot change what it pins."""
        log = []
        api_clients = [FakeApiClient(log)]
        fleet = make_fleet(make_args(), api_clients)
        api_clients.append(FakeApiClient(log))

        await fleet.pin("/snap/step_5", "5")

        assert len([e for e in log if e[0] == "update_weights_from_disk"]) == 1
