"""The opt-in cake DSA kernels: ``--miles-dsa-kernel-backend cake`` -> the cake rows of SparseAttentionConfig /
IndexerConfig -> the generated package in miles/kernels/attention/dsa/cake; plus that package's registry and
stage dispatch."""

import importlib.util
import sys
import types
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytest
import torch

import miles.kernels.attention.dsa as dsa_package
from miles.utils.args.configs.custom_megatron_plugins import DsaMegatronPluginsConfig
from miles_plugins.models.deepseek_v4.arguments import DSV4_SPEC_MODULE
from miles_plugins.models.glm5.arguments import MEGATRON_DSA_SPEC, MILES_DSA_SPEC

# the package re-exports the ``sparse_attention`` function under the submodule's name; take the modules themselves
dsa_sparse_attention = importlib.import_module("miles.kernels.attention.dsa.sparse_attention")
dsa_indexer = importlib.import_module("miles.kernels.attention.dsa.indexer")

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


@pytest.mark.parametrize("selection", [None, "auto", "cake"])
def test_kernel_backend_option_defaults_to_auto(selection):
    parser = ArgumentParser()
    DsaMegatronPluginsConfig.add_arguments(parser=parser)
    parsed = parser.parse_args([] if selection is None else ["--miles-dsa-kernel-backend", selection])
    config = DsaMegatronPluginsConfig.model_validate(vars(parsed))
    assert config.miles_dsa_kernel_backend == (selection or "auto")
    assert DsaMegatronPluginsConfig().miles_dsa_kernel_backend == "auto"


def test_kernel_backend_option_rejects_the_bridge_choices():
    parser = ArgumentParser()
    DsaMegatronPluginsConfig.add_arguments(parser=parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--miles-dsa-kernel-backend", "tilelang"])


def test_cake_configs_pair_forward_and_backward():
    config = dsa_sparse_attention.CAKE_SPARSE_ATTENTION_CONFIG
    assert (config.forward_backend, config.backward_backend) == ("cake", "cake")
    for forward, backward in (("cake", "tilelang"), ("cake", "cudnn"), ("flash_mla", "cake"), ("tilelang", "cake")):
        with pytest.raises(ValueError, match="runs forward and backward together"):
            dsa_sparse_attention.SparseAttentionConfig(forward_backend=forward, backward_backend=backward)
    # the measured default rows never pick the opt-in kernels
    for row in dsa_sparse_attention._TUNED.values():
        assert "cake" not in (row.forward_backend, row.backward_backend)
    indexer_config = dsa_indexer.CAKE_INDEXER_CONFIG
    assert (indexer_config.score_backend, indexer_config.backward_backend) == ("cake", "cake")
    assert dsa_indexer.IndexerConfig().backward_backend == "tilelang"
    for row in dsa_indexer._TUNED.values():
        assert "cake" not in (row.score_backend, row.backward_backend)


@pytest.fixture
def fake_cake(monkeypatch):
    """A stand-in for the generated package that records how the dispatch calls it (CPU, no kernel build)."""
    calls = []

    def sparse_attention(q, kv, indices, *, sm_scale, attn_sink, layout, d_v):
        calls.append(("sparse_attention", layout, tuple(q.shape), tuple(kv.shape), tuple(indices.shape), d_v))
        return torch.zeros(*q.shape[:-1], d_v)

    def indexer_logits(q, k, weights, ks, ke, *, layout):
        calls.append(("indexer_logits", layout, tuple(q.shape), tuple(k.shape)))
        if layout == "thd":
            return torch.full((q.shape[0], k.shape[0]), 1.0)
        return torch.full((q.shape[1], q.shape[0], k.shape[0]), 1.0)  # sbhd -> [B, S, S_kv]

    def indexer_topk_scores(q, k, weights, topk_indices, logits, *, layout):
        calls.append(("indexer_topk_scores", layout, weights.dtype))
        return torch.zeros(topk_indices.shape)

    stub = types.SimpleNamespace(
        sparse_attention=sparse_attention, indexer_logits=indexer_logits, indexer_topk_scores=indexer_topk_scores
    )
    monkeypatch.setattr(dsa_package, "cake", stub, raising=False)
    monkeypatch.setitem(sys.modules, "miles.kernels.attention.dsa.cake", stub)
    return calls


def test_sparse_attention_dispatches_cake_by_batch(fake_cake):
    config = dsa_sparse_attention.CAKE_SPARSE_ATTENTION_CONFIG
    # GLM-5 packs one sequence: the cake thd layout, RoPE tail kept in q / kv, d_v = 512
    q, kv, idx = torch.zeros(1, 6, 8, 576), torch.zeros(1, 6, 1, 576), torch.zeros(1, 6, 1, 64, dtype=torch.int32)
    out = dsa_sparse_attention.sparse_attention(q, kv, idx, 0.1, d_v=512, config=config)
    assert out.shape == (1, 6, 8, 512)
    assert fake_cake[-1] == ("sparse_attention", "thd", (6, 8, 576), (6, 1, 576), (6, 1, 64), 512)
    # DeepSeek-V4 batches: the cake bshd layout with the latent group squeezed
    q, kv, idx = torch.zeros(2, 6, 8, 512), torch.zeros(2, 7, 1, 512), torch.zeros(2, 6, 1, 64, dtype=torch.int32)
    out = dsa_sparse_attention.sparse_attention(q, kv, idx, 0.1, config=config)
    assert out.shape == (2, 6, 8, 512)
    assert fake_cake[-1] == ("sparse_attention", "bshd", (2, 6, 8, 512), (2, 7, 512), (2, 6, 64), 512)
    with pytest.raises(ValueError, match="single latent group"):
        dsa_sparse_attention.sparse_attention(
            q, torch.zeros(2, 7, 2, 512), torch.zeros(2, 6, 2, 64, dtype=torch.int32), 0.1, config=config
        )


@pytest.mark.parametrize("seq_len_kv", [8, 7], ids=["aligned", "padded"])
def test_indexer_logits_dispatch_cake_with_aligned_score_rows(fake_cake, seq_len_kv):
    config = dsa_indexer.CAKE_INDEXER_CONFIG
    ks, ke = torch.zeros(5, dtype=torch.int32), torch.full((5,), seq_len_kv, dtype=torch.int32)
    logits = dsa_indexer.indexer_logits(
        torch.zeros(5, 32, 128), torch.zeros(seq_len_kv, 128), torch.zeros(5, 32), ks, ke, config=config
    )
    assert fake_cake[-1] == ("indexer_logits", "thd", (5, 32, 128), (seq_len_kv, 128))
    assert logits.shape == (5, seq_len_kv) and logits.stride(0) % dsa_indexer.SCORE_ROW_ALIGN == 0
    assert torch.equal(logits, torch.ones(5, seq_len_kv))
    # sbhd: one launch for the whole batch
    logits = dsa_indexer.indexer_logits_sbhd(
        torch.zeros(5, 2, 64, 128), torch.zeros(seq_len_kv, 2, 128), torch.zeros(5, 2, 64), ks, ke, config=config
    )
    assert [call[0] for call in fake_cake].count("indexer_logits") == 2
    assert fake_cake[-1] == ("indexer_logits", "sbhd", (5, 2, 64, 128), (seq_len_kv, 2, 128))
    assert logits.shape == (2, 5, seq_len_kv) and logits.stride(1) % dsa_indexer.SCORE_ROW_ALIGN == 0
    flat = logits.reshape(10, seq_len_kv)
    assert flat.stride(0) % dsa_indexer.SCORE_ROW_ALIGN == 0


def test_lighting_indexer_routes_scores_and_backward_to_cake(fake_cake):
    config = dsa_indexer.CAKE_INDEXER_CONFIG
    ks, ke = torch.zeros(5, dtype=torch.int32), torch.full((5,), 8, dtype=torch.int32)
    seen = []

    def topk_fn(logits, topk, starts, ends):
        seen.append((tuple(logits.shape), topk))
        return torch.zeros(logits.shape[0], topk, dtype=torch.int32)

    weights = torch.zeros(5, 32, dtype=torch.bfloat16)
    scores, topk_indices = dsa_indexer.lighting_indexer(
        torch.zeros(5, 32, 128), torch.zeros(8, 128), weights, ks, ke, 4, topk_fn, config=config
    )
    assert seen == [((5, 8), 4)] and topk_indices.shape == (5, 4) and scores.shape == (5, 4)
    assert [call[0] for call in fake_cake] == ["indexer_logits", "indexer_topk_scores"]
    assert fake_cake[-1] == ("indexer_topk_scores", "thd", torch.float32)


def _megatron_arguments_module(monkeypatch):
    """miles/backends/megatron_utils/arguments.py without Megatron (CPU): stub the three modules it imports."""
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.tokenizers.utils.build_tokenizer",
        types.SimpleNamespace(vocab_size_with_padding=lambda size, args: size),
    )
    monkeypatch.setitem(
        sys.modules, "megatron.training.arguments", types.SimpleNamespace(parse_args=None, validate_args=None)
    )
    monkeypatch.setitem(sys.modules, "miles.utils.hf_utils.config", types.SimpleNamespace(load_hf_config=None))
    return _load("test_dsa_kernel_backend_megatron_arguments", "miles/backends/megatron_utils/arguments.py")


def _kernel_backend_args(**overrides):
    values = dict(
        miles_dsa_kernel_backend="cake",
        megatron_to_hf_mode="raw",
        multi_lora_n_adapters=0,
        spec=list(MILES_DSA_SPEC),
        dsa_impl="miles",
        dsv4_impl="megatron",
    )
    return Namespace(**(values | overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"spec": [DSV4_SPEC_MODULE, "get_dsv4_spec"], "dsv4_impl": "miles"},
        {"miles_dsa_kernel_backend": "auto", "megatron_to_hf_mode": "bridge", "spec": None},
    ],
    ids=["glm5-miles", "dsv4-miles", "auto-anywhere"],
)
def test_cake_is_accepted_on_the_plugin_specs(monkeypatch, overrides):
    module = _megatron_arguments_module(monkeypatch)
    module._validate_miles_dsa_kernel_backend(_kernel_backend_args(**overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"megatron_to_hf_mode": "bridge"}, "requires --megatron-to-hf-mode raw"),
        ({"multi_lora_n_adapters": 2}, "without multi-LoRA"),
        ({"dsa_impl": "megatron"}, "only wired into the miles DSA plugin specs"),
        ({"spec": list(MEGATRON_DSA_SPEC), "dsa_impl": "megatron"}, "only wired into the miles DSA plugin specs"),
        ({"spec": [DSV4_SPEC_MODULE, "get_dsv4_spec"], "dsv4_impl": "megatron"}, "only wired"),
        ({"spec": ["miles_plugins.models.glm5_next.glm5_next", "get_glm5_next_spec"]}, "only wired"),
        ({"spec": None}, "only wired"),
    ],
    ids=["bridge", "multi-lora", "native-dsa", "native-dsa-spec", "native-dsv4", "other-spec", "no-spec"],
)
def test_cake_is_rejected_where_it_would_be_ignored(monkeypatch, overrides, message):
    module = _megatron_arguments_module(monkeypatch)
    with pytest.raises(ValueError, match=message):
        module._validate_miles_dsa_kernel_backend(_kernel_backend_args(**overrides))


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
