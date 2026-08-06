from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch.nn as nn

from miles.backends.experimental.fsdp_utils.adaptations import routing_replay as rr
from miles.backends.experimental.fsdp_utils.adaptations.routing_replay import (
    RoutingReplayAdapter,
    discover_moe_modules,
    install_routing_replay,
    register_routing_replay_adapter,
)
from miles.utils.replay_base import routing_replay_manager


class _FakeRouter(nn.Module):
    def forward(self, x):
        return x


class _FakeDense(nn.Module):
    def forward(self, x):
        return x


def _model_with_layers(kinds):
    """kinds[i] == 'moe' builds a layer whose .mlp.gate is a _FakeRouter; 'dense' builds _FakeDense."""
    layers = nn.ModuleList()
    for kind in kinds:
        mlp = nn.Module()
        mlp.gate = _FakeRouter() if kind == "moe" else _FakeDense()
        layer = nn.Module()
        layer.mlp = mlp
        layers.append(layer)
    inner = nn.Module()
    inner.layers = layers
    model = nn.Module()
    model.model = inner
    return model


@pytest.fixture(autouse=True)
def _reset_manager():
    # The adapter registry is module-global and populated at spec-import time; snapshot it so
    # ad-hoc test adapters cannot leak into the next test's resolution.
    saved_adapters = list(rr._ADAPTERS)
    routing_replay_manager.enabled = False
    routing_replay_manager.replays = []
    routing_replay_manager.current = None
    yield
    rr._ADAPTERS[:] = saved_adapters
    routing_replay_manager.enabled = False
    routing_replay_manager.replays = []
    routing_replay_manager.current = None


def test_discover_returns_global_layer_index_skipping_dense():
    # GLM-4.7-Flash shape: layer 0 dense, the rest MoE.
    model = _model_with_layers(["dense"] + ["moe"] * 4)
    found = discover_moe_modules(model, "_FakeRouter")
    assert [idx for idx, _ in found] == [1, 2, 3, 4]


def test_discover_is_sorted_by_layer_index():
    # named_modules yields "layers.10" before "layers.2" is not guaranteed, so pin the order.
    model = _model_with_layers(["moe"] * 12)
    found = discover_moe_modules(model, "_FakeRouter")
    assert [idx for idx, _ in found] == list(range(12))


def test_discover_finds_layers_behind_an_extra_wrapper():
    # Qwen3.5's multimodal wrapper nests decoder layers under model.language_model.layers.
    model = _model_with_layers(["moe"] * 3)
    outer = nn.Module()
    outer.language_model = model
    assert [idx for idx, _ in discover_moe_modules(outer, "_FakeRouter")] == [0, 1, 2]


def test_install_assigns_stream_idx_equal_to_global_layer_index():
    routing_replay_manager.enabled = True
    register_routing_replay_adapter(
        RoutingReplayAdapter(
            name="fake", applies_to=lambda cfg: True, module_cls_name="_FakeRouter", install=lambda m: None
        )
    )
    model = _model_with_layers(["dense", "moe", "moe"])

    count = install_routing_replay(model, SimpleNamespace(model_type="fake"))

    assert count == 2
    assert [r.stream_idx for r in routing_replay_manager.replays] == [1, 2]


def test_install_calls_the_adapter_once_per_moe_layer():
    routing_replay_manager.enabled = True
    installed = []
    register_routing_replay_adapter(
        RoutingReplayAdapter(
            name="fake_counted",
            applies_to=lambda cfg: getattr(cfg, "model_type", None) == "fake_counted",
            module_cls_name="_FakeRouter",
            install=installed.append,
        )
    )
    model = _model_with_layers(["dense", "moe", "moe", "moe"])

    install_routing_replay(model, SimpleNamespace(model_type="fake_counted"))

    assert len(installed) == 3
    assert all(isinstance(m, _FakeRouter) for m in installed)


def test_install_is_a_noop_when_manager_disabled():
    routing_replay_manager.enabled = False
    register_routing_replay_adapter(
        RoutingReplayAdapter(
            name="fake_off", applies_to=lambda cfg: True, module_cls_name="_FakeRouter", install=lambda m: None
        )
    )
    model = _model_with_layers(["moe", "moe"])

    assert install_routing_replay(model, SimpleNamespace(model_type="fake_off")) == 0
    assert routing_replay_manager.replays == []


def test_install_raises_when_no_adapter_matches():
    routing_replay_manager.enabled = True
    register_routing_replay_adapter(
        RoutingReplayAdapter(
            name="picky",
            applies_to=lambda cfg: getattr(cfg, "model_type", None) == "something_else",
            module_cls_name="_FakeRouter",
            install=lambda m: None,
        )
    )
    with pytest.raises(ValueError, match="no routing-replay adapter"):
        install_routing_replay(_model_with_layers(["moe"]), SimpleNamespace(model_type="unknown_arch"))


def test_install_raises_when_adapter_matches_but_finds_no_layers():
    routing_replay_manager.enabled = True
    register_routing_replay_adapter(
        RoutingReplayAdapter(
            name="empty",
            applies_to=lambda cfg: getattr(cfg, "model_type", None) == "empty",
            module_cls_name="_NotPresent",
            install=lambda m: None,
        )
    )
    with pytest.raises(ValueError, match="found no MoE layers"):
        install_routing_replay(_model_with_layers(["moe"]), SimpleNamespace(model_type="empty"))
