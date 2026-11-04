"""The opt-in Cake DSA kernels: ``--dsa-attention-backend cake`` -> the cake rows of SparseAttentionConfig /
IndexerConfig -> miles_ext_kernels.dsa through miles.kernels.attention.dsa.cake (CPU: the submodule's kernels are
stood in for)."""

import ast
import importlib
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from miles.kernels.attention.dsa import cake  # noqa: E402

# the package re-exports the ``sparse_attention`` function under the submodule's name; take the modules themselves
dsa_sparse_attention = importlib.import_module("miles.kernels.attention.dsa.sparse_attention")
dsa_indexer = importlib.import_module("miles.kernels.attention.dsa.indexer")


def test_the_cake_attention_config_pairs_forward_and_backward():
    config = dsa_sparse_attention.CAKE_SPARSE_ATTENTION_CONFIG
    assert (config.forward_backend, config.backward_backend) == ("cake", "cake")
    for forward, backward in (("cake", "tilelang"), ("cake", "cudnn"), ("flash_mla", "cake"), ("tilelang", "cake")):
        with pytest.raises(ValueError, match="runs forward and backward together"):
            dsa_sparse_attention.SparseAttentionConfig(forward_backend=forward, backward_backend=backward)


def test_the_measured_default_rows_never_pick_the_cake_kernels():
    for row in dsa_sparse_attention._TUNED.values():
        assert "cake" not in (row.forward_backend, row.backward_backend)
    assert dsa_indexer.CAKE_INDEXER_CONFIG.score_backend == "cake"
    assert dsa_indexer.IndexerConfig().score_backend != "cake"
    for row in dsa_indexer._TUNED.values():
        assert row.score_backend != "cake"


@pytest.fixture
def fake_ops(monkeypatch):
    """A stand-in for miles_ext_kernels.dsa.ops that records how the dispatch calls it (CPU, no kernel build)."""
    calls = []

    def sparse_attention(q, kv, indices, *, sm_scale, attn_sink, layout, d_v):
        calls.append(("sparse_attention", layout, tuple(q.shape), tuple(kv.shape), tuple(indices.shape), d_v))
        return torch.zeros(*q.shape[:-1], d_v)

    def indexer_logits(q, k, weights, ks, ke, *, layout):
        calls.append(("indexer_logits", layout, tuple(q.shape), tuple(k.shape)))
        if layout == "thd":
            return torch.full((q.shape[0], k.shape[0]), 1.0)
        return torch.full((q.shape[1], q.shape[0], k.shape[0]), 1.0)  # sbhd -> [B, S, S_kv]

    stub = types.SimpleNamespace(sparse_attention=sparse_attention, indexer_logits=indexer_logits)
    monkeypatch.setattr(cake, "ops", lambda: stub)
    return calls


def test_sparse_attention_routes_one_sequence_to_thd_and_a_batch_to_bshd(fake_ops):
    config = dsa_sparse_attention.CAKE_SPARSE_ATTENTION_CONFIG
    # GLM-5 packs one sequence: the thd layout, RoPE tail kept in q / kv, d_v = 512
    q, kv, idx = torch.zeros(1, 6, 8, 576), torch.zeros(1, 6, 1, 576), torch.zeros(1, 6, 1, 64, dtype=torch.int32)
    out = dsa_sparse_attention.sparse_attention(q, kv, idx, 0.1, d_v=512, config=config)
    assert out.shape == (1, 6, 8, 512)
    assert fake_ops[-1] == ("sparse_attention", "thd", (6, 8, 576), (6, 1, 576), (6, 1, 64), 512)
    # DeepSeek-V4 batches: the bshd layout with the latent group squeezed
    q, kv, idx = torch.zeros(2, 6, 8, 512), torch.zeros(2, 7, 1, 512), torch.zeros(2, 6, 1, 64, dtype=torch.int32)
    out = dsa_sparse_attention.sparse_attention(q, kv, idx, 0.1, config=config)
    assert out.shape == (2, 6, 8, 512)
    assert fake_ops[-1] == ("sparse_attention", "bshd", (2, 6, 8, 512), (2, 7, 512), (2, 6, 64), 512)
    with pytest.raises(ValueError, match="single latent group"):
        dsa_sparse_attention.sparse_attention(
            q, torch.zeros(2, 7, 2, 512), torch.zeros(2, 6, 2, 64, dtype=torch.int32), 0.1, config=config
        )
    assert len(fake_ops) == 2


@pytest.mark.parametrize("seq_len_kv", [8, 7], ids=["aligned", "padded"])
def test_indexer_logits_route_to_cake_and_keep_aligned_score_rows(fake_ops, seq_len_kv):
    config = dsa_indexer.CAKE_INDEXER_CONFIG
    ks, ke = torch.zeros(5, dtype=torch.int32), torch.full((5,), seq_len_kv, dtype=torch.int32)
    logits = dsa_indexer.indexer_logits(
        torch.zeros(5, 32, 128), torch.zeros(seq_len_kv, 128), torch.zeros(5, 32), ks, ke, config=config
    )
    assert fake_ops[-1] == ("indexer_logits", "thd", (5, 32, 128), (seq_len_kv, 128))
    assert logits.shape == (5, seq_len_kv) and logits.stride(0) % dsa_indexer.SCORE_ROW_ALIGN == 0
    assert torch.equal(logits, torch.ones(5, seq_len_kv))
    # sbhd: one launch for the whole batch
    logits = dsa_indexer.indexer_logits_sbhd(
        torch.zeros(5, 2, 64, 128), torch.zeros(seq_len_kv, 2, 128), torch.zeros(5, 2, 64), ks, ke, config=config
    )
    assert [call[0] for call in fake_ops] == ["indexer_logits", "indexer_logits"]
    assert fake_ops[-1] == ("indexer_logits", "sbhd", (5, 2, 64, 128), (seq_len_kv, 2, 128))
    assert logits.shape == (2, 5, seq_len_kv) and logits.stride(1) % dsa_indexer.SCORE_ROW_ALIGN == 0
    assert logits.reshape(10, seq_len_kv).stride(0) % dsa_indexer.SCORE_ROW_ALIGN == 0


@pytest.mark.parametrize(("heads", "head_dim"), [(8, 128), (16, 128), (32, 128), (64, 128)])
def test_the_cake_indexer_geometries_are_accepted(heads, head_dim):
    cake.check_indexer_geometry(heads, head_dim)


@pytest.mark.parametrize(("heads", "head_dim"), [(12, 128), (128, 128), (64, 64), (32, 256)])
def test_other_indexer_geometries_are_rejected(heads, head_dim):
    with pytest.raises(ValueError, match=f"this model has {heads} heads of dimension {head_dim}"):
        cake.check_indexer_geometry(heads, head_dim)


@pytest.fixture
def fake_ext_package(monkeypatch):
    """A stand-in for the miles_ext_kernels.dsa package: records configure() and the device check."""
    events = []
    package = types.ModuleType("miles_ext_kernels.dsa")
    package.configure = lambda args: events.append(("configure", vars(args)))
    ops = types.ModuleType("miles_ext_kernels.dsa.ops")
    ops.check_device = lambda device=None: events.append(("check_device",)) or "sm_100a"
    package.ops = ops
    monkeypatch.setitem(sys.modules, "miles_ext_kernels", types.ModuleType("miles_ext_kernels"))
    monkeypatch.setitem(sys.modules, "miles_ext_kernels.dsa", package)
    monkeypatch.setitem(sys.modules, "miles_ext_kernels.dsa.ops", ops)
    return events


def test_setup_hands_the_ext_arguments_to_the_kernels_and_checks_the_device(fake_ext_package):
    args = Namespace(miles_ext_kernels_args={"cake_dsa_variant": "fast", "cake_dsa_lazy_build": False})

    cake.setup(args, index_heads=32, index_head_dim=128)
    assert fake_ext_package == [
        ("configure", {"cake_dsa_variant": "fast", "cake_dsa_lazy_build": False}),
        ("check_device",),
    ]


def test_setup_checks_the_indexer_geometry_before_touching_the_kernels(fake_ext_package):
    with pytest.raises(ValueError, match="this model has 128 heads"):
        cake.setup(Namespace(miles_ext_kernels_args={}), index_heads=128, index_head_dim=128)
    assert fake_ext_package == []


def test_a_missing_submodule_names_the_flag_and_the_checkout(monkeypatch):
    for name in ("miles_ext_kernels", "miles_ext_kernels.dsa", "miles_ext_kernels.dsa.ops"):
        monkeypatch.setitem(sys.modules, name, None)  # an import of it raises ModuleNotFoundError
    with pytest.raises(ImportError, match="--use-miles-ext-kernels"):
        cake.ops()


_ROOT = Path(__file__).resolve().parents[2]
# The four miles DSA plugin specs, each with the modules that call the DSA kernels for its layers
_WIRED_SPECS = {
    "get_glm5_spec": ["miles_plugins/models/glm5/glm5.py"],
    "get_glm5_next_spec": [
        "miles_plugins/models/glm5_next/glm5_next.py",
        "miles_plugins/models/glm5_next/dsa.py",
        "miles_plugins/models/glm5_next/ops/kpool_indexer.py",
    ],
    "get_dsv4_spec": [
        "miles_plugins/models/deepseek_v4/deepseek_v4.py",
        "miles_plugins/models/deepseek_v4/ops/v4_indexer.py",
    ],
    "get_dsv41_spec": [
        "miles_plugins/models/deepseek_v4_1/deepseek_v4_1.py",
        "miles_plugins/models/deepseek_v4_1/ops/indexer.py",
    ],
}
_KERNEL_CALLS = {"sparse_attention", "indexer_logits", "indexer_logits_sbhd"}


def _tree(relative: str) -> ast.Module:
    return ast.parse((_ROOT / relative).read_text())


@pytest.mark.parametrize("spec", _WIRED_SPECS)
def test_every_dsa_kernel_call_of_a_wired_spec_takes_the_dispatcher_config(spec):
    """The model layers are GPU-only to build, so the wiring is read from the source: every sparse-attention and
    indexer-score call passes ``config=``, the row the layer picked (the cake row under --dsa-attention-backend cake,
    else None, the measured default)."""
    calls = [
        node
        for relative in _WIRED_SPECS[spec]
        for node in ast.walk(_tree(relative))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in _KERNEL_CALLS
    ]

    assert calls
    assert all("config" in {keyword.arg for keyword in call.keywords} for call in calls), [
        call.lineno for call in calls
    ]


@pytest.mark.parametrize("spec", _WIRED_SPECS)
def test_a_wired_spec_sets_the_cake_kernels_up_under_the_cake_backend(spec):
    tree = _tree(_WIRED_SPECS[spec][0])
    builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == spec)
    guarded_setups = [
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "args.dsa_attention_backend == 'cake'"
        and any(ast.unparse(statement).startswith("cake.setup(args, ") for statement in node.body)
    ]

    # the layers learn the choice from the spec: a ModuleSpec param (GLM) or the TransformerConfig (DeepSeek-V4)
    handed_to_layers = [
        node
        for node in ast.walk(builder)
        if (isinstance(node, ast.Dict) and "'dsa_attention_backend': args.dsa_attention_backend" in ast.unparse(node))
        or ast.unparse(node) == "config.miles_dsa_attention_backend = args.dsa_attention_backend"
    ]

    assert len(guarded_setups) == 1
    assert len(handed_to_layers) == 1
