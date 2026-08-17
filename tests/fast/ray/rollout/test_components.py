"""Factory contract for the role-separated rollout construction
(codex-rollout-fullparameter-design-0810 §4.3/§4.8/§8.2): the factory unpacks
(rollout_manager, num_rollout_per_epoch), returns two DISTINCT role objects
sharing one legacy handle (num_rollout_per_epoch is dropped: the tinker
driver has no epochs), the bundle disposes exactly once, and
future-shaped fakes can replace the factory without changing driver call
sites."""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio

import miles.ray.rollout.components as components_module
from miles.ray.rollout.components import InferenceEndpoint, RolloutComponents, create_rollout_components


class Remote:
    def __init__(self, log, name, value=None):
        self._log, self._name, self._value = log, name, value

    async def remote(self, *args):
        self._log.append((self._name, args))
        return self._value


def make_fake_manager(log):
    return SimpleNamespace(
        get_router_address=Remote(log, "get_router_address", ("10.0.0.7", 30001)),
        generate=Remote(log, "generate", {"batch": 1}),
        dispose=Remote(log, "dispose"),
    )


def build(monkeypatch, log):
    manager = make_fake_manager(log)
    monkeypatch.setattr(
        "miles.ray.placement_group.create_rollout_manager", lambda args, pg: (manager, 7), raising=True
    )
    components = create_rollout_components(SimpleNamespace(), pg=None)
    return components, manager


def test_factory_builds_two_role_views_over_one_legacy_handle(monkeypatch):
    log: list = []
    components, manager = build(monkeypatch, log)

    assert components.inference_controller is not components.rollout_executor
    # The raw combined actor is exposed ONLY as the factory's opaque
    # weight-update owner; the controller role never leaks it publicly.
    assert components.weight_update_owner is manager
    assert not hasattr(components.inference_controller, "manager")

    endpoint = asyncio.run(components.inference_controller.get_inference_endpoint())
    assert endpoint == InferenceEndpoint(host="10.0.0.7", port=30001)
    assert endpoint.base_url == "http://10.0.0.7:30001"

    # prepare_rollout is part of the controller port (PR #1842 boundary);
    # the legacy adapter accepts the call as a no-op.
    asyncio.run(components.inference_controller.prepare_rollout(3))
    assert asyncio.run(components.rollout_executor.generate(3)) == {"batch": 1}
    assert ("generate", (3,)) in log


def test_bundle_disposes_the_shared_actor_exactly_once(monkeypatch):
    log: list = []
    components, _ = build(monkeypatch, log)
    asyncio.run(components.dispose())
    asyncio.run(components.dispose())  # second call must be a no-op
    assert [name for name, _ in log].count("dispose") == 1


def test_future_shaped_fakes_satisfy_the_bundle_without_the_factory():
    """A split-world construction (separate controller/executor objects) fits
    the same bundle: driver call sites depend only on the role surface."""

    calls: list = []

    class FakeController:
        async def get_inference_endpoint(self):
            return InferenceEndpoint(host="h", port=1)

        async def prepare_rollout(self, rollout_id):
            calls.append(("prepare", rollout_id))

    class FakeExecutor:
        async def generate(self, rollout_id):
            calls.append(("generate", rollout_id))
            return rollout_id

    class FakeLifecycle:
        def __init__(self):
            self.disposed = 0

        async def dispose_once(self):
            self.disposed += 1

    lifecycle = FakeLifecycle()
    components = RolloutComponents(
        inference_controller=FakeController(),
        rollout_executor=FakeExecutor(),
        lifecycle=lifecycle,
        weight_update_owner=object(),
    )

    async def one_cycle():
        # The driver's per-rollout order: prepare on the controller role,
        # then generate on the executor role.
        await components.inference_controller.prepare_rollout(5)
        return await components.rollout_executor.generate(5)

    assert asyncio.run(one_cycle()) == 5
    assert calls == [("prepare", 5), ("generate", 5)]
    asyncio.run(components.dispose())
    assert lifecycle.disposed == 1


def test_module_never_imports_ray_directly():
    # The construction seam isolates Ray invocation shapes behind adapters.
    import inspect

    source = inspect.getsource(components_module)
    assert "import ray" not in source


def test_controller_port_covers_the_pr1842_prepare_boundary():
    """External review: the split controller's per-rollout responsibility is
    ``prepare_rollout()`` — the port must declare it so PR #1842's concrete
    drops in without a driver change."""
    from miles.ray.rollout.components import InferenceControllerPort

    assert hasattr(InferenceControllerPort, "prepare_rollout")


def test_tinker_driver_never_escapes_through_a_legacy_manager():
    """External review: the driver must reach the weight-update target only
    through the factory's opaque ``weight_update_owner`` — a future-shaped
    controller has no ``.manager`` to reach through."""
    from pathlib import Path

    import miles

    driver_source = (Path(miles.__file__).resolve().parent.parent / "train_tinker_backend.py").read_text()
    assert "inference_controller.manager" not in driver_source
    assert "weight_update_owner" in driver_source
    # The per-rollout prepare boundary is exercised before every generate.
    assert driver_source.index("prepare_rollout") < driver_source.index("rollout_executor.generate(")
