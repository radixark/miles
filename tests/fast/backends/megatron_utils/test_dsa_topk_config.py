import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def native_dsa_spec(monkeypatch):
    factory = Mock(return_value="native-spec")
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.models.gpt.experimental_attention_variant_module_specs",
        SimpleNamespace(get_transformer_block_with_experimental_attention_variant_spec=factory),
    )
    path = Path(__file__).resolve().parents[4] / "miles_plugins/models/glm5/megatron_spec.py"
    spec = importlib.util.spec_from_file_location("native_dsa_spec_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, factory


def _set_topk_environment(monkeypatch, *, tie_break, deterministic):
    envs = SimpleNamespace(
        SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK=SimpleNamespace(get=lambda: tie_break),
        SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC=SimpleNamespace(get=lambda: deterministic),
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", SimpleNamespace(envs=envs))


@pytest.mark.parametrize(
    ("tie_break", "deterministic", "expected"),
    [(None, False, 0), ("small", False, 1), ("large", True, 2), ("LARGE", False, 2)],
)
def test_native_spec_resolves_the_shared_flashinfer_policy_in_the_worker(
    monkeypatch, native_dsa_spec, tie_break, expected, deterministic
):
    module, _ = native_dsa_spec
    config = SimpleNamespace(dsa_indexer_topk_backend="flashinfer")
    # The worker environment is supplied after importing the spec factory.
    _set_topk_environment(monkeypatch, tie_break=tie_break, deterministic=deterministic)

    module.get_dsa_spec(None, config, vp_stage=2)
    assert config.dsa_indexer_topk_tie_break == expected
    assert config.dsa_indexer_topk_deterministic is deterministic


def test_invalid_flashinfer_policy_fails_before_constructing_native_layers(monkeypatch, native_dsa_spec):
    module, factory = native_dsa_spec
    _set_topk_environment(monkeypatch, tie_break="random", deterministic=True)
    with pytest.raises(RuntimeError, match="SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK"):
        module.get_dsa_spec(None, SimpleNamespace(dsa_indexer_topk_backend="flashinfer"), vp_stage=None)
    factory.assert_not_called()


def test_torch_backend_does_not_resolve_flashinfer_policy(monkeypatch, native_dsa_spec):
    module, _ = native_dsa_spec
    config = SimpleNamespace(dsa_indexer_topk_backend="torch")
    _set_topk_environment(monkeypatch, tie_break="invalid", deterministic=True)

    module.get_dsa_spec(None, config, vp_stage=None)
