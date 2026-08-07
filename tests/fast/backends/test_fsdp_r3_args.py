from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

from miles.utils.arguments import resolve_fsdp_num_layers


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
