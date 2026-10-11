"""The DSA attention backend switch: one value selects the kernel family for both DSA model plugins.

Also the ``cake`` top-k backend (``--miles-dsa-topk-backend cake``) on CPU: resolver, registry, input validation
ahead of any launch, argument parsing and the GLM-5 (top-k 2048 > 512) rejection.  Nothing here needs CUDA.
"""

import ast
import importlib.util
import sys
import types
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TOPK_STAGES = {f"dsa_topk_fp32_c{capacity}_k512" for capacity in (4096, 8192, 16384, 32768)}
TC_STAGES = {f"dsa_attention_fwd_tc_h{h}_d512t{t}" for h in (16, 32, 64) for t in (64, 0)}
TC_BWD_STAGES = {f"dsa_attention_bwd_tc_h64_d512t{t}" for t in (64, 0)}


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
        *TOPK_STAGES,
        *TC_STAGES,
        *TC_BWD_STAGES,
    }
    assert set(jit.MODULES) == expected
    for stage, record in jit.MODULES.items():
        assert set(record["arches"]) == {"sm_100a", "sm_103a"}, stage
        assert record["kernel_symbol"] == f"kernel_{record['name']}", stage
        for relative in record["sources"]:
            assert (REPO / "miles/kernels/attention/dsa/cake/csrc" / relative).is_file(), (stage, relative)
    for stage in TOPK_STAGES:
        record = jit.MODULES[stage]
        # The host wrapper binds these names and leaves the launch geometry to the record.
        assert [name for kind, name in record["arg_plan"] if kind != "grid"] == [
            "logits",
            "out",
            "row_end",
            "n_kv",
            "has_row_end",
        ], stage
        assert record["block"] == [256, 1, 1] and record["tma_workspace_bytes"] == 0, stage
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


def test_attention_backward_dispatch_selects_the_tcgen05_stage_for_64_heads(monkeypatch):
    """Rows with exactly 64 heads run the tcgen05 backward rows kernel on both exported architectures (one partial per
    slot); other head counts, other architectures and the kill switch keep the register-MMA / cluster-pair rows kernels.
    """
    from miles.kernels.attention.dsa.cake import _kernels

    monkeypatch.delenv("MILES_DSA_ATTENTION_BWD_TC", raising=False)
    for arch in ("sm_100a", "sm_103a"):
        for tail in (64, 0):
            assert _kernels.attention_tc_bwd_stage(64, tail, arch) == f"dsa_attention_bwd_tc_h64_d512t{tail}"
            assert _kernels.attention_tc_bwd_stage(64, tail, arch) in TC_BWD_STAGES
        for heads in (1, 16, 32, 40, 48, 65, 128):
            assert _kernels.attention_tc_bwd_stage(heads, 0, arch) is None
    assert _kernels.attention_tc_bwd_stage(64, 0, "sm_90a") is None
    monkeypatch.setenv("MILES_DSA_ATTENTION_BWD_TC", "0")
    assert _kernels.attention_tc_bwd_stage(64, 0, "sm_100a") is None
    geo = _kernels.AttentionGeometry(64, 512, 0, 32, 2, 640, False, True)
    assert geo.kv_partials == 1 and geo.d_qk == 512
    assert _kernels.AttentionGeometry(64, 512, 0, 32, 2, 640, True).kv_partials == 1
    assert _kernels.AttentionGeometry(64, 512, 0, 32, 2, 640, False).kv_partials == 2


def test_attention_forward_dispatch_selects_the_tcgen05_stage_for_production_head_counts(monkeypatch):
    """16 / 32 / 64 heads run the tcgen05 forward on both exported architectures; other head counts and the kill switch
    keep the register-MMA stage chosen by ``attention_block_h``."""
    from miles.kernels.attention.dsa.cake import _kernels

    monkeypatch.delenv("MILES_DSA_ATTENTION_TC", raising=False)
    for arch in ("sm_100a", "sm_103a"):
        for heads in (16, 32, 64):
            for tail in (64, 0):
                assert _kernels.attention_tc_stage(heads, tail, arch) == f"dsa_attention_fwd_tc_h{heads}_d512t{tail}"
                assert _kernels.attention_tc_stage(heads, tail, arch) in TC_STAGES
        for heads in (1, 8, 17, 48, 65, 128):
            assert _kernels.attention_tc_stage(heads, 0, arch) is None
    assert _kernels.attention_tc_stage(64, 0, "sm_90a") is None
    monkeypatch.setenv("MILES_DSA_ATTENTION_TC", "0")
    assert _kernels.attention_tc_stage(64, 0, "sm_100a") is None
    assert _kernels.attention_block_h(64) == 32 and _kernels.attention_head_blocks(64) == 2


# --------------------------------------------------------------------------- #
# cake top-k backend
# --------------------------------------------------------------------------- #


def test_core_topk_resolver_defers_the_plugin_import():
    """``get_dsa_topk_fn("cake")`` is a plain callable; the plugin (and its CUDA extensions) load on the first call."""
    topk = _load("test_dsa_topk_module", "miles/kernels/attention/dsa/topk.py")
    before = {name for name in sys.modules if name.startswith("miles_plugins")}
    fn = topk.get_dsa_topk_fn("cake")
    assert callable(fn) and fn is topk.cake_dsa_topk
    assert {name for name in sys.modules if name.startswith("miles_plugins")} == before
    assert topk.get_dsa_topk_fn("torch") is topk.torch_dsa_topk
    assert topk.get_dsa_topk_fn("flashinfer") is topk.flashinfer_dsa_topk
    with pytest.raises(ValueError, match="Unsupported miles DSA topk backend"):
        topk.get_dsa_topk_fn("nope")
    # The plugin validates before touching CUDA: GLM-5's 2048 exceeds the kernel's 512 slots.
    with pytest.raises(ValueError, match="512"):
        fn(torch.zeros(2, 8192), 2048)


def test_plugin_topk_backends_include_cake():
    from miles.kernels.attention.dsa.cake import ops

    assert set(ops._TOPK_BACKENDS) >= {"torch", "flashinfer", "cake"}
    assert ops.get_topk_fn("cake") is ops.cake_topk

    def replay_wrapper(logits, topk):  # a replay-manager style callable passes through untouched
        return logits

    assert ops.get_topk_fn(replay_wrapper) is replay_wrapper
    with pytest.raises(ValueError, match="unknown DSA top-k backend"):
        ops.get_topk_fn("nope")


def test_dsa_topk_validates_before_any_launch(monkeypatch):
    from miles.kernels.attention.dsa.cake import _kernels

    monkeypatch.setattr(_kernels, "_kernel", lambda name, device: pytest.fail(f"{name} was launched"))
    logits = torch.zeros(3, 1024)
    with pytest.raises(TypeError, match="float32"):
        _kernels.dsa_topk(logits.half(), 64)
    with pytest.raises(TypeError, match="2-D"):
        _kernels.dsa_topk(logits.unsqueeze(0), 64)
    with pytest.raises(ValueError, match="32768"):
        _kernels.dsa_topk(torch.zeros(1, 32769), 64)
    with pytest.raises(ValueError, match="512"):
        _kernels.dsa_topk(logits, 513)
    with pytest.raises(ValueError, match="1 <= topk"):
        _kernels.dsa_topk(logits, 0)
    # Valid requests on CPU tensors stop at the device check, still ahead of the kernel loader.
    with pytest.raises(ValueError, match="CUDA"):
        _kernels.dsa_topk(logits, 64)
    with pytest.raises(ValueError, match="CUDA"):
        _kernels.dsa_topk(torch.zeros(3, 100), 512)  # topk > num_keys: -inf columns are padded first
    assert [_kernels.topk_capacity(n) for n in (1, 4096, 4097, 8192, 8193, 16384, 16385, 32768)] == [
        4096,
        4096,
        8192,
        8192,
        16384,
        16384,
        32768,
        32768,
    ]
    for bad in (0, 32769):
        with pytest.raises(ValueError, match="1..32768"):
            _kernels.topk_capacity(bad)


def test_parser_accepts_the_cake_topk_backend():
    from miles_plugins.models.glm5.arguments import add_dsa_arguments, normalize_dsa_args

    parser = add_dsa_arguments(ArgumentParser())
    assert parser.parse_args([]).miles_dsa_topk_backend == "torch"
    assert parser.parse_args(["--miles-dsa-topk-backend", "cake"]).miles_dsa_topk_backend == "cake"
    with pytest.raises(SystemExit):
        parser.parse_args(["--miles-dsa-topk-backend", "nope"])
    # Native Megatron DSA cannot host the generated kernel.
    with pytest.raises(ValueError, match="only available with --dsa-impl miles"):
        normalize_dsa_args(Namespace(dsa_impl="megatron", miles_dsa_topk_backend="cake"), None)
    args = Namespace(dsa_impl="miles", miles_dsa_topk_backend="cake")
    normalize_dsa_args(args, None)
    assert args.miles_dsa_topk_backend == "cake"


def test_glm5_rejects_the_cake_topk_backend(monkeypatch):
    backend = _load("test_dsa_backend_topk_module", "miles/kernels/attention/dsa/backend.py")
    with pytest.raises(ValueError, match=r"index_topk <= 512.*GLM-5.*2048 keys per query"):
        backend.check_cake_topk_backend("cake", 2048, model="GLM-5")
    backend.check_cake_topk_backend("torch", 2048, model="GLM-5")
    backend.check_cake_topk_backend("flashinfer", 2048, model="GLM-5")
    # Within the limit: the device check only runs on CUDA hosts (the kernel loader enforces it again).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    backend.check_cake_topk_backend("cake", 512, model="DeepSeek-V4.1")
    backend.check_cake_topk_backend("cake", None, model="DeepSeek-V4.1")
    # The plugin specs call the guard (with the static key-row bound) before the model is built.
    guard, bound = "check_cake_topk_backend", "cake_topk_key_bound"
    for relative, scopes in (
        ("miles_plugins/models/glm5/glm5.py", {"get_glm5_spec": {guard, bound}, "__init__": {guard}}),
        ("miles_plugins/models/deepseek_v4/deepseek_v4.py", {"get_dsv4_spec": {guard, bound}}),
        ("miles_plugins/models/deepseek_v4_1/deepseek_v4_1.py", {"get_dsv41_spec": {guard, bound}}),
    ):
        called: dict[str, set] = {}
        for node in ast.walk(ast.parse((REPO / relative).read_text())):
            if isinstance(node, ast.FunctionDef):
                called.setdefault(node.name, set()).update(
                    getattr(call.func, "id", None) for call in ast.walk(node) if isinstance(call, ast.Call)
                )
        for scope, names in scopes.items():
            assert names <= called.get(scope, set()), (relative, scope, called.get(scope))


def test_cake_topk_backend_requires_blackwell(monkeypatch):
    backend = _load("test_dsa_backend_capability_module", "miles/kernels/attention/dsa/backend.py")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(ValueError, match=r"SM100a / SM103a.*10\.0 / 10\.3.*got 9\.0"):
        backend.check_cake_topk_backend("cake", 512, model="DeepSeek-V4.1")
    backend.check_cake_topk_backend("torch", 512, model="DeepSeek-V4.1")  # other backends never probe the device
    for capability in ((10, 0), (10, 3)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, cap=capability: cap)
        backend.check_cake_topk_backend("cake", 512, model="DeepSeek-V4.1")


def test_cake_topk_backend_rejects_long_key_rows(monkeypatch):
    backend = _load("test_dsa_backend_rows_module", "miles/kernels/attention/dsa/backend.py")
    assert backend.cake_topk_key_bound(None) is None
    assert backend.cake_topk_key_bound(4096) == 4096  # thd: the packed stream itself
    assert backend.cake_topk_key_bound(65536, (0, 2, 1)) == 65536  # V4.1: a ratio-1 layer keeps every key
    assert backend.cake_topk_key_bound(65536, (0, 2)) == 32768
    assert backend.cake_topk_key_bound(131073, (4,)) == 32769  # V4: ceil(seq_length / 4)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    backend.check_cake_topk_backend("cake", 512, model="DeepSeek-V4.1", max_keys=32768)
    backend.check_cake_topk_backend("cake", 512, model="DeepSeek-V4.1", max_keys=None)
    backend.check_cake_topk_backend("torch", 512, model="DeepSeek-V4.1", max_keys=10**6)
    with pytest.raises(ValueError, match=r"at most 32768 compressed keys.*DeepSeek-V4\.1.*32769 keys"):
        backend.check_cake_topk_backend("cake", 512, model="DeepSeek-V4.1", max_keys=32769)


def test_dsv41_topk_dump_is_env_gated(monkeypatch, tmp_path):
    """``MILES_DSV41_TOPK_DUMP_DIR`` writes sampled selections next to the torch reference; unset writes nothing."""
    dump = _load("test_dsv41_topk_dump_module", "miles_plugins/models/deepseek_v4_1/ops/topk_dump.py")
    topk_module = _load("test_dsa_topk_module_for_dump", "miles/kernels/attention/dsa/topk.py")
    torch.manual_seed(0)
    bsz, rows, n_kv, topk = 1, 3, 6, 4
    lens = torch.tensor([1, 4, 6])
    scores = torch.randn(bsz * rows, n_kv).masked_fill(torch.arange(n_kv)[None, :] >= lens[:, None], -torch.inf)
    # the seam's torch path on the same rows
    idx = topk_module.torch_dsa_topk(scores, topk).reshape(bsz, rows, topk).sort(dim=-1).values.to(torch.int64)
    idx = torch.where(idx < lens.unsqueeze(-1), idx, -1)

    monkeypatch.delenv("MILES_DSV41_TOPK_DUMP_DIR", raising=False)
    dump.topk_dump_config.cache_clear()
    assert dump.maybe_dump_topk("torch", 0, scores, idx, lens, topk) is None
    assert list(tmp_path.iterdir()) == []

    out_dir = tmp_path / "dumps"
    monkeypatch.setenv("MILES_DSV41_TOPK_DUMP_DIR", str(out_dir))
    monkeypatch.setenv("MILES_DSV41_TOPK_DUMP_EVERY", "2")
    monkeypatch.setenv("MILES_DSV41_TOPK_DUMP_MAX", "2")
    monkeypatch.setenv("MILES_DSV41_TOPK_DUMP_LAYERS", "0,3")
    dump.topk_dump_config.cache_clear()
    paths = [dump.maybe_dump_topk("torch", 0, scores, idx, lens, topk) for _ in range(6)]
    assert [path is not None for path in paths] == [True, False, True, False, False, False], "EVERY=2, MAX=2"
    assert paths[0] == str(out_dir / "rank0_layer0_call0.pt") and paths[2] == str(out_dir / "rank0_layer0_call2.pt")
    assert dump.maybe_dump_topk("torch", 1, scores, idx, lens, topk) is None, "layer 1 is not in LAYERS"
    assert dump.maybe_dump_topk("cake", 3, scores, idx, lens, topk) == str(out_dir / "rank0_layer3_call0.pt")
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "rank0_layer0_call0.pt",
        "rank0_layer0_call2.pt",
        "rank0_layer3_call0.pt",
    ]

    payload = torch.load(paths[0])
    assert set(payload) == {
        "schema",
        "backend",
        "layer",
        "rank",
        "call",
        "time",
        "topk",
        "n_kv",
        "deep_select",
        "compress_lens",
        "idx",
        "ref_idx",
        "topvals",
    }
    assert (payload["schema"], payload["backend"], payload["layer"], payload["rank"], payload["call"]) == (
        1,
        "torch",
        0,
        0,
        0,
    )
    assert payload["topk"] == topk and payload["n_kv"] == n_kv and payload["deep_select"] is False
    assert isinstance(payload["time"], float)
    assert torch.equal(payload["compress_lens"], lens)
    assert payload["idx"].dtype == torch.int32 and payload["ref_idx"].dtype == torch.int32
    assert torch.equal(payload["idx"], idx.to(torch.int32)) and torch.equal(payload["idx"], payload["ref_idx"])
    assert (payload["idx"][0, 0, :3] == -1).all() and payload["idx"][0, 0, 3] == 0  # lens 1: one key, -1 first
    assert payload["topvals"].shape == (bsz * rows, topk + 1) and payload["topvals"].dtype == torch.float32
    assert (payload["topvals"][:, 1:] <= payload["topvals"][:, :-1]).all(), "descending"
    assert torch.load(paths[2])["call"] == 2

    # The V4.1 seam calls the hook once per chunk on both selection paths.
    tree = ast.parse((REPO / "miles_plugins/models/deepseek_v4_1/ops/indexer.py").read_text())
    seam = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "indexer_select")
    calls = [
        node
        for node in ast.walk(seam)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "maybe_dump_topk"
    ]
    assert len(calls) == 2


def test_prebuild_is_scoped_to_the_stage_family(monkeypatch):
    """The attention / indexer operators never compile the top-k variants and vice versa; PREBUILD=0 stays lazy."""
    from miles.kernels.attention.dsa.cake import _jit, _kernels

    calls = []
    monkeypatch.setattr(_kernels, "device_arch", lambda device: "sm_100a")
    monkeypatch.setattr(_kernels, "prebuild", lambda arch, stages=None, max_workers=None: calls.append((arch, stages)))
    monkeypatch.setattr(_kernels, "kernel", lambda name, arch: (name, arch))
    monkeypatch.setattr(_kernels, "_PREBUILT", set())
    monkeypatch.delenv("MILES_DSA_TRAIN_PREBUILD", raising=False)
    assert _kernels._kernel("dsa_topk_fp32_c4096_k512", "cuda") == ("dsa_topk_fp32_c4096_k512", "sm_100a")
    assert len(calls) == 1 and calls[0][0] == "sm_100a" and set(calls[0][1]) == TOPK_STAGES
    _kernels._kernel("dsa_topk_fp32_c32768_k512", "cuda")
    assert len(calls) == 1, "a family is prebuilt once per architecture"
    _kernels._kernel("dsa_indexer_clean", "cuda")
    assert len(calls) == 2 and set(calls[1][1]) == set(_jit.MODULES) - TOPK_STAGES and len(calls[1][1]) == 17
    monkeypatch.setenv("MILES_DSA_TRAIN_PREBUILD", "0")
    monkeypatch.setattr(_kernels, "_PREBUILT", set())
    _kernels._kernel("dsa_topk_fp32_c4096_k512", "cuda")
    assert len(calls) == 2, "MILES_DSA_TRAIN_PREBUILD=0 builds lazily"
    # The real prebuild honours the stage filter (its loader is replaced, so nothing compiles here).
    loaded = []
    monkeypatch.setattr(_jit, "load", lambda stage, arch: loaded.append(stage))
    _jit.prebuild.__wrapped__("sm_100a", stages=("dsa_topk_fp32_c4096_k512", "dsa_indexer_clean"), max_workers=1)
    assert set(loaded) == {"dsa_topk_fp32_c4096_k512", "dsa_indexer_clean"}
