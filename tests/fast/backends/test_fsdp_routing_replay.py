from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch.nn as nn

from miles.backends.fsdp_utils.adaptations import routing_replay
from miles.utils.arguments import resolve_fsdp_num_layers
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
    saved_adapters = list(routing_replay._ADAPTERS)
    routing_replay_manager.enabled = False
    routing_replay_manager.replays = []
    routing_replay_manager.current = None
    yield
    routing_replay._ADAPTERS[:] = saved_adapters
    routing_replay_manager.enabled = False
    routing_replay_manager.replays = []
    routing_replay_manager.current = None
    routing_replay_manager.stage = routing_replay.FALLTHROUGH


def test_discover_returns_global_layer_index_skipping_dense():
    model = _model_with_layers(["dense"] + ["moe"] * 4)
    found = routing_replay.discover_moe_modules(model, "_FakeRouter")
    assert [idx for idx, _ in found] == [1, 2, 3, 4]


def test_discover_is_sorted_by_layer_index():
    model = _model_with_layers(["moe"] * 12)
    found = routing_replay.discover_moe_modules(model, "_FakeRouter")
    assert [idx for idx, _ in found] == list(range(12))


def test_discover_finds_layers_behind_an_extra_wrapper():
    model = _model_with_layers(["moe"] * 3)
    outer = nn.Module()
    outer.language_model = model
    assert [idx for idx, _ in routing_replay.discover_moe_modules(outer, "_FakeRouter")] == [0, 1, 2]


def test_install_assigns_stream_idx_equal_to_global_layer_index():
    routing_replay_manager.enabled = True
    routing_replay.register_routing_replay_adapter(
        routing_replay.RoutingReplayAdapter(
            name="fake", applies_to=lambda cfg: True, module_cls_name="_FakeRouter", install=lambda m: None
        )
    )
    model = _model_with_layers(["dense", "moe", "moe"])

    count = routing_replay.install(model, SimpleNamespace(model_type="fake"))

    assert count == 2
    assert [r.stream_idx for r in routing_replay_manager.replays] == [1, 2]


def test_install_calls_the_adapter_once_per_moe_layer():
    routing_replay_manager.enabled = True
    installed = []
    routing_replay.register_routing_replay_adapter(
        routing_replay.RoutingReplayAdapter(
            name="fake_counted",
            applies_to=lambda cfg: getattr(cfg, "model_type", None) == "fake_counted",
            module_cls_name="_FakeRouter",
            install=installed.append,
        )
    )
    model = _model_with_layers(["dense", "moe", "moe", "moe"])

    routing_replay.install(model, SimpleNamespace(model_type="fake_counted"))

    assert len(installed) == 3
    assert all(isinstance(m, _FakeRouter) for m in installed)


def test_install_is_a_noop_when_manager_disabled():
    routing_replay_manager.enabled = False
    routing_replay.register_routing_replay_adapter(
        routing_replay.RoutingReplayAdapter(
            name="fake_off", applies_to=lambda cfg: True, module_cls_name="_FakeRouter", install=lambda m: None
        )
    )
    model = _model_with_layers(["moe", "moe"])

    assert routing_replay.install(model, SimpleNamespace(model_type="fake_off")) == 0
    assert routing_replay_manager.replays == []


def test_install_raises_when_no_adapter_matches():
    routing_replay_manager.enabled = True
    routing_replay.register_routing_replay_adapter(
        routing_replay.RoutingReplayAdapter(
            name="picky",
            applies_to=lambda cfg: getattr(cfg, "model_type", None) == "something_else",
            module_cls_name="_FakeRouter",
            install=lambda m: None,
        )
    )
    with pytest.raises(ValueError, match="no routing-replay adapter"):
        routing_replay.install(_model_with_layers(["moe"]), SimpleNamespace(model_type="unknown_arch"))


def test_install_raises_when_adapter_matches_but_finds_no_layers():
    routing_replay_manager.enabled = True
    routing_replay.register_routing_replay_adapter(
        routing_replay.RoutingReplayAdapter(
            name="empty",
            applies_to=lambda cfg: getattr(cfg, "model_type", None) == "empty",
            module_cls_name="_NotPresent",
            install=lambda m: None,
        )
    )
    with pytest.raises(ValueError, match="found no MoE layers"):
        routing_replay.install(_model_with_layers(["moe"]), SimpleNamespace(model_type="empty"))


def test_specs_register_adapters_for_every_supported_model_type():
    import miles.backends.fsdp_utils.adaptations.specs  # noqa: F401

    expected = {
        "qwen3_moe": "Qwen3MoeTopKRouter",
        "qwen3_5_moe_text": "Qwen3_5MoeTopKRouter",
        "glm4_moe_lite": "Glm4MoeLiteMoE",
    }
    for model_type, module_cls_name in expected.items():
        adapter = routing_replay.resolve_routing_replay_adapter(SimpleNamespace(model_type=model_type))
        assert adapter is not None, model_type
        assert adapter.module_cls_name == module_cls_name


def test_dense_archs_do_not_resolve_to_a_moe_adapter():
    import miles.backends.fsdp_utils.adaptations.specs  # noqa: F401

    for model_type in ("qwen3_5_text", "qwen3"):
        assert routing_replay.resolve_routing_replay_adapter(SimpleNamespace(model_type=model_type)) is None


def test_enable_follows_use_routing_replay():
    assert routing_replay.enable(Namespace(use_routing_replay=True, ci_test=False)) is True
    assert routing_replay.enable(Namespace(use_routing_replay=False, ci_test=False)) is False


def test_enable_defaults_false_when_arg_absent():
    assert routing_replay.enable(Namespace(ci_test=False)) is False


def test_enable_turns_on_the_replay_check_only_under_ci_test():
    routing_replay.enable(Namespace(use_routing_replay=True, ci_test=True))
    assert routing_replay_manager.enable_check_replay_result is True

    routing_replay.enable(Namespace(use_routing_replay=True, ci_test=False))
    assert routing_replay_manager.enable_check_replay_result is False


def test_pop_and_reduce_check_stats_resets_local_attempt(monkeypatch):
    group = object()
    reduce_stats = Mock(return_value=(5, 12))
    monkeypatch.setattr(routing_replay_manager, "enable_check_replay_result", True)
    monkeypatch.setattr(routing_replay, "reduce_check_stats", reduce_stats)
    routing_replay_manager.mismatched_tokens = 2
    routing_replay_manager.checked_tokens = 5

    assert routing_replay.pop_and_reduce_check_stats(group) == (5, 12)
    assert routing_replay_manager.pop_check_stats() == (0, 0)
    reduce_stats.assert_called_once_with((2, 5), group)


def test_log_prob_stage_is_fallthrough_when_disabled():
    routing_replay_manager.enabled = False
    assert routing_replay.log_prob_stage(Namespace(use_rollout_routing_replay=True)) == routing_replay.FALLTHROUGH


def test_log_prob_stage_replays_when_routing_came_from_the_rollout():
    routing_replay_manager.enabled = True
    assert routing_replay.log_prob_stage(Namespace(use_rollout_routing_replay=True)) == routing_replay.REPLAY_FORWARD


def test_log_prob_stage_records_for_the_non_rollout_variant():
    # --use-routing-replay alone has nothing to replay yet: the queues are only filled by fill(),
    # which is gated on use_rollout_routing_replay. Replaying here would pop an empty queue.
    routing_replay_manager.enabled = True
    assert routing_replay.log_prob_stage(Namespace(use_rollout_routing_replay=False)) == routing_replay.RECORD


def test_fill_is_skipped_for_the_non_rollout_variant():
    # Passing None for every downstream argument: reaching fill_replay_data would raise.
    routing_replay.fill(Namespace(use_rollout_routing_replay=False), None, None, None, None)


def test_stage_sets_and_restores():
    routing_replay_manager.stage = routing_replay.REPLAY_BACKWARD
    with routing_replay.stage(routing_replay.REPLAY_FORWARD):
        assert routing_replay_manager.stage == routing_replay.REPLAY_FORWARD
    assert routing_replay_manager.stage == routing_replay.REPLAY_BACKWARD


def test_stage_restores_on_exception():
    routing_replay_manager.stage = routing_replay.REPLAY_BACKWARD
    with pytest.raises(RuntimeError):
        with routing_replay.stage(routing_replay.REPLAY_FORWARD):
            raise RuntimeError("boom")
    assert routing_replay_manager.stage == routing_replay.REPLAY_BACKWARD


def test_stage_nests():
    routing_replay_manager.stage = routing_replay.FALLTHROUGH
    with routing_replay.stage(routing_replay.REPLAY_BACKWARD):
        with routing_replay.stage(routing_replay.REPLAY_FORWARD):
            assert routing_replay_manager.stage == routing_replay.REPLAY_FORWARD
        assert routing_replay_manager.stage == routing_replay.REPLAY_BACKWARD
    assert routing_replay_manager.stage == routing_replay.FALLTHROUGH


def test_ref_model_creation_does_not_install_routing_replay():
    import inspect

    from miles.backends.fsdp_utils.actor import FSDPTrainRayActor

    source = inspect.getsource(FSDPTrainRayActor._create_ref_model)
    assert "routing_replay.install" not in source


def test_resolve_num_layers_from_flat_config():
    cfg = SimpleNamespace(num_hidden_layers=48)
    assert resolve_fsdp_num_layers(cfg) == 48


def test_resolve_num_layers_unwraps_text_config():
    cfg = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=40))
    assert resolve_fsdp_num_layers(cfg) == 40


def test_resolve_num_layers_prefers_text_config_when_both_present():
    cfg = SimpleNamespace(num_hidden_layers=1, text_config=SimpleNamespace(num_hidden_layers=40))
    assert resolve_fsdp_num_layers(cfg) == 40


def test_resolve_num_layers_uses_get_text_config_when_available():
    text = SimpleNamespace(num_hidden_layers=47)
    cfg = SimpleNamespace(num_hidden_layers=1, get_text_config=lambda: text)
    assert resolve_fsdp_num_layers(cfg) == 47


def test_resolve_num_layers_falls_back_when_text_config_lacks_depth():
    cfg = SimpleNamespace(num_hidden_layers=32, text_config=SimpleNamespace())
    assert resolve_fsdp_num_layers(cfg) == 32


def test_resolve_num_layers_returns_none_when_absent():
    assert resolve_fsdp_num_layers(SimpleNamespace()) is None
