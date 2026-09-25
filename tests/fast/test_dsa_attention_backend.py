"""The DSA attention backend switch: one value selects the kernel family for both DSA model plugins."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_plugin_backend_resolution():
    backend = _load("test_dsa_backend_module", "miles_plugins/models/dsa_backend.py")
    assert backend.resolve_dsa_attention_backend(None) == "tilelang"
    assert backend.resolve_dsa_attention_backend("tilelang") == "tilelang"
    assert backend.resolve_dsa_attention_backend("loom") == "loom"
    # ``megatron`` is a bridge-path choice; the plugin specs keep their fused kernels.
    assert backend.resolve_dsa_attention_backend("megatron") == "tilelang"
    assert backend.use_loom_dsa("loom") and not backend.use_loom_dsa("tilelang")
    with pytest.raises(ValueError, match="Unsupported DSA attention backend"):
        backend.resolve_dsa_attention_backend("nope")


def test_bridge_path_rejects_loom():
    bridge = _load("test_megatron_bridge_utils_module", "miles/utils/megatron_bridge_utils.py")
    provider = Namespace(dsa_attention_backend="megatron")
    bridge.apply_dsa_backend_args(provider, Namespace(dsa_attention_backend="tilelang"))
    assert provider.dsa_attention_backend == "tilelang"
    with pytest.raises(ValueError, match="only available with the miles_plugins model specs"):
        bridge.apply_dsa_backend_args(provider, Namespace(dsa_attention_backend="loom"))


def test_generated_package_registry_covers_both_architectures():
    """Every exported stage is exported for SM100a and SM103a from one source set that exists on disk."""
    package = REPO / "miles_plugins/models/dsa_train"
    registry = json.loads((package / "registry.json").read_text())
    module = registry["module"]
    assert set(module["arches"]) == {"sm_100a", "sm_103a"}
    assert module["sources"][0] == "cake_module.cu"
    for relative in module["sources"]:
        assert (package / "csrc" / relative).is_file(), relative
    stages = registry["stages"]
    expected = {
        *(f"dsa_attention_{kind}_h{h}_d512t{t}" for kind in ("fwd", "bwd") for h in (16, 32) for t in (64, 0)),
        *(f"dsa_indexer_logits_h{h}" for h in (8, 16, 32, 64)),
        *(f"dsa_indexer_bwd_h{h}" for h in (16, 32, 64)),
        "dsa_indexer_clean",
        "dsa_segmented_reduce",
    }
    assert set(stages) == expected
    for stage, record in stages.items():
        assert record["ffi_entry"] == stage
        assert record["kernel_symbol"] == f"kernel_{record['name']}", stage
        for relative in record["sources"]:
            assert relative in module["sources"], (stage, relative)
    # the launchers include the host surface shipped once, by the shared loader package
    assert (REPO / "miles_plugins/models/cake_native/csrc/cake_host_shim.h").is_file()
