from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio
from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args as _make_args

import miles.ray.rollout.eval_fleet as eval_fleet_mod
from miles.ray.rollout.eval_fleet import EvalFleet
from miles.rollout.checkpoint_eval import EvalSkip


def make_args(**overrides):
    defaults = dict(
        eval_num_gpus=1,
        eval_num_gpus_per_engine=1,
        sglang_model_routers={"default": ("10.0.0.1", 30000), "eval": ("10.0.0.2", 31000)},
    )
    defaults.update(overrides)
    return _make_args(**defaults)


class FakeRemoteMethod:
    def __init__(self, engine, name):
        self.engine = engine
        self.name = name

    def remote(self, *args, **kwargs):
        self.engine.log.append((self.name, args, kwargs))
        result = self.engine.responses[self.name](*args, **kwargs)
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(result)
        return fut


class FakeEngine:
    def __init__(self, log):
        self.log = log
        self.weight_version = None

        def load(model_path, weight_version=None):
            self.weight_version = weight_version
            return None

        self.responses = {
            "update_weights_from_disk": load,
            "get_weight_version": lambda: self.weight_version,
        }

    def __getattr__(self, name):
        if name in ("update_weights_from_disk", "get_weight_version"):
            return FakeRemoteMethod(self, name)
        raise AttributeError(name)


class FakeEvalServer:
    def __init__(self, engines):
        self._engines = engines

    @property
    def engines(self):
        return [SimpleNamespace(actor_handle=e) for e in self._engines]


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


def make_fleet(args, engines):
    return EvalFleet(args, srv=FakeEvalServer(engines))


async def test_fleet_pins_every_engine_before_returning_the_state(fleet_env):
    log = []
    fleet = make_fleet(make_args(), [FakeEngine(log), FakeEngine(log)])

    state = await fleet.pin("/snap/step_5", "5")

    load_events = [e for e in log if e[0] == "update_weights_from_disk"]
    assert len(load_events) == 2
    assert all(e[2]["weight_version"] == "5" for e in load_events)
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
    good, stale = FakeEngine(log), FakeEngine(log)
    stale.responses["get_weight_version"] = lambda: "999"
    fleet = make_fleet(make_args(), [good, stale])

    with pytest.raises(EvalSkip) as exc:
        await fleet.pin("/snap/step_5", "5")

    assert exc.value.reason == "pin_violation"
    assert len([e for e in log if e[0] == "update_weights_from_disk"]) == 4  # 2 engines x 2 attempts


async def test_fleet_pin_does_not_health_probe_the_server(fleet_env):
    """The eval fleet has no fault tolerance: pin goes straight to the weight load."""
    server = FakeEvalServer([FakeEngine([])])
    assert not any(hasattr(server, name) for name in ("probe_and_mark_dead", "recover", "wait_all_engines_alive"))

    state = await EvalFleet(make_args(), srv=server).pin("/snap/step_5", "5")

    assert state == "fake-fleet-state"
