"""The DSA attention backend switch: one value selects the kernel family for both DSA model plugins."""

import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TC_STAGES = {f"dsa_attention_fwd_tc_h{h}_d512t{t}" for h in (8, 16, 32, 64) for t in (64, 0)}
TC_BWD_STAGES = {f"dsa_attention_bwd_tc_h{h}_d512t{t}" for h in (8, 16, 32, 64) for t in (64, 0)}
INDEXER_TC_STAGES = {f"dsa_indexer_logits_tc_h{h}" for h in (8, 16, 32, 64)}
INDEXER_TC_BWD_STAGES = {f"dsa_indexer_bwd_tc_h{h}" for h in (32, 64)}


def _load(name: str, relative: str):
    """Execute one source file as module ``name`` without importing its package (keeps the plug-ins' GPU imports out).

    The module is registered in ``sys.modules`` before it runs: ``dataclasses`` (Python 3.12) and ``typing`` resolve
    a class's module through ``sys.modules[cls.__module__]`` while the body executes.
    """
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def test_plugin_backend_resolution():
    backend = _load("test_dsa_backend_module", "miles/kernels/attention/dsa/backend.py")
    assert backend.resolve_dsa_attention_backend(None) == "tilelang"
    assert backend.resolve_dsa_attention_backend("tilelang") == "tilelang"
    assert backend.resolve_dsa_attention_backend("cake") == "cake"
    # ``megatron`` is a bridge-path choice; the plugin specs keep their fused kernels.
    assert backend.resolve_dsa_attention_backend("megatron") == "tilelang"
    assert backend.use_cake_dsa("cake") and not backend.use_cake_dsa("tilelang")
    with pytest.raises(ValueError, match="Unsupported DSA attention backend"):
        backend.resolve_dsa_attention_backend("nope")


def test_bridge_path_rejects_cake(monkeypatch):
    # ``megatron_bridge_utils`` imports ``megatron.core.utils`` at module level.  Without a GPU driver (CPU-only
    # container, no libcuda) that import dies inside transformer_engine with OSError, which its ``except ImportError``
    # does not catch, so the three package entries are stubbed for this load; ``apply_dsa_backend_args`` is pure.
    for module_name in ("megatron", "megatron.core", "megatron.core.utils"):
        monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))
    sys.modules["megatron.core.utils"].unwrap_model = lambda model: [model]
    bridge = _load("test_megatron_bridge_utils_module", "miles/utils/megatron_bridge_utils.py")
    provider = Namespace(dsa_attention_backend="megatron")
    bridge.apply_dsa_backend_args(provider, Namespace(dsa_attention_backend="tilelang"))
    assert provider.dsa_attention_backend == "tilelang"
    with pytest.raises(ValueError, match="only available with the miles_plugins model specs"):
        bridge.apply_dsa_backend_args(provider, Namespace(dsa_attention_backend="cake"))


def test_generated_package_registry_covers_both_architectures():
    """Every exported stage is exported for SM100a and SM103a from one source set that exists on disk."""
    jit = _load("test_dsa_train_jit_module", "miles/kernels/attention/dsa/cake/_jit.py")
    assert set(jit.MODULES), "registry.json missing"
    expected = {
        *(f"dsa_attention_{kind}_h{h}_d512t{t}" for kind in ("fwd", "bwd") for h in (16, 32) for t in (64, 0)),
        *(f"dsa_attention_bwd_h32x2_d512t{t}" for t in (64, 0)),
        *(f"dsa_indexer_logits_h{h}" for h in (8, 16, 32, 64)),
        *(f"dsa_indexer_bwd_h{h}" for h in (16, 32, 64)),
        "dsa_indexer_clean",
        "dsa_segmented_reduce",
        "dsa_segmented_reduce_split4",
        "dsa_attention_delta",
        *TC_STAGES,
        *TC_BWD_STAGES,
        *INDEXER_TC_STAGES,
        *INDEXER_TC_BWD_STAGES,
    }
    assert set(jit.MODULES) == expected
    for stage, record in jit.MODULES.items():
        assert set(record["arches"]) == {"sm_100a", "sm_103a"}, stage
        assert record["kernel_symbol"] == f"kernel_{record['name']}", stage
        for relative in record["sources"]:
            assert (REPO / "miles/kernels/attention/dsa/cake/csrc" / relative).is_file(), (stage, relative)
    # the tcgen05 forward stages: the generator's dsa_h64_fwd ABI (four by-value tensor maps, no workspace), 384 threads,
    # built plain (precise math like the register-MMA stages: no compile flag)
    tc_args = (
        "q_latent q_rope kv_latent k_rope out o_lo lse indices topk_length sink num_queries num_kv topk idx_stride "
        "indices_offset k_rope_stride k_rope_offset has_topk_length derive_length scale_log2 heads"
    ).split()
    for stage in sorted(TC_STAGES):
        record = jit.MODULES[stage]
        assert [name for kind, name in record["arg_plan"] if kind != "grid"] == tc_args, stage
        assert "workspace" not in {kind for kind, _ in record["arg_plan"]}, stage
        assert record["block"] == [384, 1, 1] and record["tma_workspace_bytes"] == 0, stage
        assert record["compile_flags"] == [], stage
    assert len({jit.MODULES[stage]["name"] for stage in TC_STAGES}) == len(TC_STAGES)
    # the tcgen05 backward rows stages: the generator's dsa_h64_bwd ABI (seven by-value tensor maps, no workspace), 640
    # threads, built plain
    tc_bwd_args = (
        "q_latent q_rope dout dq_latent dq_rope kv_latent k_rope lse delta out o_lo indices topk_length dkv_f32 dkr_f32 "
        "dkv_stride dkr_stride dkr_col0 dkv_dst_map dkv_has_map num_queries num_kv topk idx_stride indices_offset "
        "has_topk_length token_base token_step scale_log2 sm_scale pass_lo pass_hi dq_mode dq_partial key_scratch pass_counts"
    ).split()
    for stage in sorted(TC_BWD_STAGES):
        record = jit.MODULES[stage]
        assert [name for kind, name in record["arg_plan"] if kind != "grid"] == tc_bwd_args, stage
        assert "workspace" not in {kind for kind, _ in record["arg_plan"]}, stage
        assert record["block"] == [640, 1, 1] and record["tma_workspace_bytes"] == 0, stage
        assert record["compile_flags"] == [], stage
    assert len({jit.MODULES[stage]["name"] for stage in TC_BWD_STAGES}) == len(TC_BWD_STAGES)
    # the tcgen05 indexer logits stages: the register-MMA logits ABI with ``index_q`` / ``index_k`` as by-value tensor maps
    # (no workspace), 320 threads, built plain
    indexer_tc_args = "index_q index_k weights cu_ks cu_ke logits seq_len seq_len_kv".split()
    for stage in sorted(INDEXER_TC_STAGES):
        record = jit.MODULES[stage]
        assert [name for kind, name in record["arg_plan"] if kind != "grid"] == indexer_tc_args, stage
        assert [kind for kind, _ in record["arg_plan"]][:2] == ["tma_buffer", "tma_buffer"], stage
        assert "workspace" not in {kind for kind, _ in record["arg_plan"]}, stage
        assert record["block"] == [320, 1, 1] and record["tma_workspace_bytes"] == 0, stage
        assert record["compile_flags"] == [], stage
    assert len({jit.MODULES[stage]["name"] for stage in INDEXER_TC_STAGES}) == len(INDEXER_TC_STAGES)
    # the register-MMA logits stages stay in the package as the fallback of every other head count / architecture
    for heads in (8, 16, 32, 64):
        assert jit.MODULES[f"dsa_indexer_logits_h{heads}"]["block"] == [256, 1, 1]
    # the tcgen05 indexer backward rows stages: the register-MMA ``dsa_indexer_bwd_h{H}`` ABI with ``index_k`` as a by-value
    # tensor map (no workspace), 256 threads, built plain
    indexer_tc_bwd_args = (
        "index_q index_k weights topk_indices grad_scores d_index_q d_weights partial heads topk".split()
    )
    for stage in sorted(INDEXER_TC_BWD_STAGES):
        record = jit.MODULES[stage]
        assert [name for kind, name in record["arg_plan"] if kind != "grid"] == indexer_tc_bwd_args, stage
        assert [
            name for kind, name in jit.MODULES["dsa_indexer_bwd_h32"]["arg_plan"] if kind != "grid"
        ] == indexer_tc_bwd_args
        assert "workspace" not in {kind for kind, _ in record["arg_plan"]}, stage
        assert record["block"] == [256, 1, 1] and record["tma_workspace_bytes"] == 0, stage
        assert record["compile_flags"] == [], stage
    assert len({jit.MODULES[stage]["name"] for stage in INDEXER_TC_BWD_STAGES}) == len(INDEXER_TC_BWD_STAGES)


def test_indexer_backward_dispatch_selects_the_tcgen05_stage_for_32_and_64_heads(monkeypatch):
    """Rows with exactly 32 or 64 indexer heads run the tcgen05 indexer backward rows kernel on both exported
    architectures (same partial contract as the register-MMA stage); other head counts, other architectures and the kill
    switch keep the register-MMA stages.
    """
    from miles.kernels.attention.dsa.cake import _kernels

    monkeypatch.delenv("MILES_DSA_INDEXER_BWD_TC", raising=False)
    assert _kernels.INDEXER_TC_BWD_HEADS == (32, 64)
    assert _kernels.INDEXER_TC_BWD_MIN_TOPK == {32: 1024, 64: 0}
    for arch in ("sm_100a", "sm_103a"):
        # GLM-5.2 (32 heads, topk 2048) and DeepSeek-V4 (64 heads, topk 512) rows run the tcgen05 stage
        assert _kernels.indexer_tc_bwd_stage(32, 2048, arch) == "dsa_indexer_bwd_tc_h32"
        assert _kernels.indexer_tc_bwd_stage(32, 1024, arch) == "dsa_indexer_bwd_tc_h32"
        assert _kernels.indexer_tc_bwd_stage(64, 512, arch) == "dsa_indexer_bwd_tc_h64"
        assert _kernels.indexer_tc_bwd_stage(64, 128, arch) == "dsa_indexer_bwd_tc_h64"
        for heads in (32, 64):
            assert _kernels.indexer_tc_bwd_stage(heads, 2048, arch) in INDEXER_TC_BWD_STAGES
        # 32-head rows below topk 1024 (the V4.1 indexer forms, topk 512) keep the register-MMA stage
        assert _kernels.indexer_tc_bwd_stage(32, 512, arch) is None
        assert _kernels.indexer_tc_bwd_stage(32, 992, arch) is None
        for heads in (1, 8, 16, 17, 33, 48, 65, 128):
            assert _kernels.indexer_tc_bwd_stage(heads, 2048, arch) is None
    assert _kernels.indexer_tc_bwd_stage(32, 2048, "sm_90a") is None
    assert _kernels.indexer_tc_bwd_stage(64, 512, "sm_120a") is None
    monkeypatch.setenv("MILES_DSA_INDEXER_BWD_TC", "0")
    assert _kernels.indexer_tc_bwd_stage(32, 2048, "sm_100a") is None
    assert _kernels.indexer_tc_bwd_stage(64, 512, "sm_103a") is None


def test_attention_backward_dispatch_selects_the_tcgen05_stages_for_8_16_32_and_64_heads(monkeypatch):
    """Rows with exactly 64 heads (``dsa_h64_bwd``) or exactly 8, 16 or 32 heads (the swap-AB schedule) run the tcgen05
    backward rows kernel on both exported architectures (one partial per slot); other head counts, other architectures and
    the kill switch keep the register-MMA / cluster-pair rows kernels.
    """
    from miles.kernels.attention.dsa.cake import _kernels

    monkeypatch.delenv("MILES_DSA_ATTENTION_BWD_TC", raising=False)
    for arch in ("sm_100a", "sm_103a"):
        for heads in (8, 16, 32, 64):
            for tail in (64, 0):
                assert (
                    _kernels.attention_tc_bwd_stage(heads, tail, arch) == f"dsa_attention_bwd_tc_h{heads}_d512t{tail}"
                )
                assert _kernels.attention_tc_bwd_stage(heads, tail, arch) in TC_BWD_STAGES
        for heads in (1, 4, 12, 24, 40, 48, 65, 128):
            assert _kernels.attention_tc_bwd_stage(heads, 0, arch) is None
    for heads in (8, 16, 32, 64):
        assert _kernels.attention_tc_bwd_stage(heads, 0, "sm_90a") is None
    monkeypatch.setenv("MILES_DSA_ATTENTION_BWD_TC", "0")
    for heads in (8, 16, 32, 64):
        assert _kernels.attention_tc_bwd_stage(heads, 0, "sm_100a") is None
    geo = _kernels.AttentionGeometry(64, 512, 0, 32, 2, 640, False, True)
    assert geo.kv_partials == 1 and geo.d_qk == 512
    geo8 = _kernels.AttentionGeometry(8, 512, 0, 16, 1, 640, False, True)
    assert geo8.kv_partials == 1 and geo8.block_h == 16
    geo16 = _kernels.AttentionGeometry(16, 512, 64, 16, 1, 2048, False, True)
    assert geo16.kv_partials == 1 and geo16.block_h == 16 and geo16.d_qk == 576
    geo32 = _kernels.AttentionGeometry(32, 512, 0, 32, 1, 640, False, True)
    assert geo32.kv_partials == 1 and geo32.block_h == 32 and geo32.head_blocks == 1
    assert _kernels.AttentionGeometry(64, 512, 0, 32, 2, 640, True).kv_partials == 1
    assert _kernels.AttentionGeometry(64, 512, 0, 32, 2, 640, False).kv_partials == 2


def test_attention_forward_dispatch_selects_the_tcgen05_stage_for_production_head_counts(monkeypatch):
    """8 / 16 / 32 / 64 heads run the tcgen05 forward on both exported architectures; other head counts and the kill switch
    keep the register-MMA stage chosen by ``attention_block_h``."""
    from miles.kernels.attention.dsa.cake import _kernels

    monkeypatch.delenv("MILES_DSA_ATTENTION_TC", raising=False)
    for arch in ("sm_100a", "sm_103a"):
        for heads in (8, 16, 32, 64):
            for tail in (64, 0):
                assert _kernels.attention_tc_stage(heads, tail, arch) == f"dsa_attention_fwd_tc_h{heads}_d512t{tail}"
                assert _kernels.attention_tc_stage(heads, tail, arch) in TC_STAGES
        for heads in (1, 4, 17, 48, 65, 128):
            assert _kernels.attention_tc_stage(heads, 0, arch) is None
    assert _kernels.attention_tc_stage(64, 0, "sm_90a") is None
    monkeypatch.setenv("MILES_DSA_ATTENTION_TC", "0")
    assert _kernels.attention_tc_stage(64, 0, "sm_100a") is None
    assert _kernels.attention_block_h(64) == 32 and _kernels.attention_head_blocks(64) == 2


def test_indexer_logits_dispatch_selects_the_tcgen05_stage_for_exported_head_counts(monkeypatch):
    """8 / 16 / 32 / 64 indexer heads run the tcgen05 logits stage on both exported architectures; other head counts, other
    architectures and the kill switch keep the register-MMA logits stage."""
    from miles.kernels.attention.dsa.cake import _kernels

    monkeypatch.delenv("MILES_DSA_INDEXER_TC", raising=False)
    for arch in ("sm_100a", "sm_103a"):
        for heads in (8, 16, 32, 64):
            assert _kernels.indexer_tc_stage(heads, arch) == f"dsa_indexer_logits_tc_h{heads}"
            assert _kernels.indexer_tc_stage(heads, arch) in INDEXER_TC_STAGES
            assert _kernels.INDEXER_TC_QUERY_COLUMNS % heads == 0
        for heads in (1, 4, 24, 48, 128):
            assert _kernels.indexer_tc_stage(heads, arch) is None
    assert _kernels.indexer_tc_stage(32, "sm_90a") is None
    monkeypatch.setenv("MILES_DSA_INDEXER_TC", "0")
    assert _kernels.indexer_tc_stage(32, "sm_100a") is None
    assert _kernels.INDEXER_TC_HEADS == (8, 16, 32, 64) and _kernels.INDEXER_HEADS == (8, 16, 32, 64)
