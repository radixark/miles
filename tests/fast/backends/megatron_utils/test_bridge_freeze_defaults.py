"""CPU regression tests for provider freeze defaults on bridge-built providers."""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

_FREEZE_FIELDS = ("freeze_language_model", "freeze_vision_model", "freeze_vision_projection")


def test_bridge_unfreezes_qwen3_vl_moe_defaults(apply_bridge_runtime_config, runtime_args):
    """Qwen3-VL MoE providers freeze the language model and ViT by default; miles must train them."""
    provider = SimpleNamespace(freeze_language_model=True, freeze_vision_model=True, freeze_vision_projection=False)

    apply_bridge_runtime_config(provider, runtime_args)

    assert all(getattr(provider, field) is False for field in _FREEZE_FIELDS)


def test_bridge_leaves_text_providers_without_freeze_fields(apply_bridge_runtime_config, runtime_args):
    provider = SimpleNamespace()

    apply_bridge_runtime_config(provider, runtime_args)

    assert not any(hasattr(provider, field) for field in _FREEZE_FIELDS)
