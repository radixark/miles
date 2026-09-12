import asyncio
from types import SimpleNamespace

from miles.ray.rollout.components import InferenceEndpoint, create_rollout_components


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
    assert components.weight_update_owner is manager
    assert not hasattr(components.inference_controller, "manager")

    endpoint = asyncio.run(components.inference_controller.get_inference_endpoint())
    assert endpoint == InferenceEndpoint(host="10.0.0.7", port=30001)
    assert endpoint.base_url == "http://10.0.0.7:30001"

    asyncio.run(components.inference_controller.prepare_rollout(3))
    assert asyncio.run(components.rollout_executor.generate(3)) == {"batch": 1}
    assert ("generate", (3,)) in log


def test_bundle_disposes_the_shared_actor_exactly_once(monkeypatch):
    log: list = []
    components, _ = build(monkeypatch, log)
    asyncio.run(components.dispose())
    asyncio.run(components.dispose())
    assert [name for name, _ in log].count("dispose") == 1
