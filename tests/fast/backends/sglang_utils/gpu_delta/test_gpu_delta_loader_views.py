"""CPU checks of the binding helpers model mappings share through ParameterBindings.

SGLang loader shards (merged column, dense TP slices) checked against the real loaders, bound views, the raw-mask
binding invariant and alias groups.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.layers import linear as sglang_linear
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MergedColumnParallelRepeatedLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

from miles.backends.sglang_utils.gpu_delta.bindings import (
    AliasGroups,
    ParameterBindings,
    TensorBinding,
    _direct_binding,
    dense_target,
    merged_column_target,
)
from miles.backends.sglang_utils.gpu_delta.byte_views import DestinationSegment, _DtypeMismatch


def _sglang_linear(cls, tp_size, tp_rank, *args, **kwargs):
    # A real SGLang linear on one TP rank without a distributed runtime: before distributed
    # initialization SGLang itself resolves a sharded partition group to a _LogicalGroup.
    group = None if tp_size == 1 else sglang_linear._LogicalGroup(tp_rank, tp_size)
    with patch.object(sglang_linear, "_resolve_linear_group", lambda parallel_group: group):
        return cls(*args, **kwargs)


def _bare_linear(cls, **attrs):
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    module.__dict__.update(attrs)
    return module


def _selection(slices):
    return tuple(slice(start, stop) for start, stop in slices)


def _loader_image(param, loader, value, shard):
    # SGLang's loader writes into a sentinel-filled parameter; unwritten rows keep the sentinel.
    param.data.fill_(-7)
    loader(param, value, shard)
    return param.data.clone()


@torch.no_grad()
def _helper_image(param, canonical, pieces):
    param.data.fill_(-7)
    for slices, view in pieces:
        view.copy_(canonical[_selection(slices)])
    return param.data.clone()


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 1), (4, 0), (4, 3)])
@pytest.mark.parametrize("conv", [False, True])
def test_merged_column_target_matches_the_sglang_loader_shard(tp_size, tp_rank, conv):
    # Four unequal parts; a 3-D conv weight is unsqueezed after construction, as Kimi-K3's KDA qkv_conv1d is.
    sizes = [8, 16, 4, 12]
    module = _sglang_linear(
        MergedColumnParallelLinear, tp_size, tp_rank, 6, sizes, bias=True, params_dtype=torch.bfloat16
    )
    if conv:
        module.weight.data = module.weight.data.unsqueeze(1)
    for param in (module.weight, module.bias):
        for shard, rows in enumerate(sizes):
            canonical = torch.randn(rows, *param.shape[1:]).bfloat16()
            expected = _loader_image(param, module.weight_loader, canonical, shard)
            view, slices = merged_column_target(param, module, shard, list(canonical.shape))
            assert view.data_ptr() >= param.data_ptr()
            assert slices[0] == [tp_rank * rows // tp_size, (tp_rank + 1) * rows // tp_size]
            actual = _helper_image(param, canonical, [(slices, view)])
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_loader_shard_helpers_reject_other_classes_shards_and_geometry():
    merged = _sglang_linear(MergedColumnParallelLinear, 2, 1, 6, [8, 8], bias=False)
    qkv = _sglang_linear(QKVParallelLinear, 2, 1, 6, 4, 4, 2, bias=False)
    subclass = type("FusedSubclass", (MergedColumnParallelLinear,), {})
    for module in (
        qkv,
        _bare_linear(subclass, output_sizes=[8, 8], use_presharded_weights=False, tp_group=None),
        _bare_linear(MergedColumnParallelRepeatedLinear),
    ):
        with pytest.raises(ValueError, match="unclassified merged column projection"):
            merged_column_target(merged.weight, module, 0, [8, 6])
    merged.use_presharded_weights = True
    with pytest.raises(ValueError, match="unclassified"):
        merged_column_target(merged.weight, merged, 0, [8, 6])
    merged.use_presharded_weights = False
    for shard in (2, -1, True, "0"):
        with pytest.raises(ValueError, match="unsupported merged column shard"):
            merged_column_target(merged.weight, merged, shard, [8, 6])
    for call in (
        lambda: merged_column_target(merged.weight, merged, 0, [16, 6]),
        lambda: merged_column_target(merged.weight, merged, 0, [8, 6, 1]),
    ):
        with pytest.raises(ValueError, match="geometry differs from its loader"):
            call()
    weight = merged.weight.detach()
    with pytest.raises(ValueError, match="no output_dim"):
        merged_column_target(weight, merged, 0, [8, 6])


@pytest.mark.parametrize("method", ["ModelOptFp4LinearMethod", "Fp8LinearMethod"])
def test_gate_up_shards_reject_a_quantized_merged_projection(method):
    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.gate_up_proj = _sglang_linear(MergedColumnParallelLinear, 1, 0, 2, [4, 4], params_dtype=torch.bfloat16)
    model.mlp.gate_up_proj.quant_method = type(method, (), {})()
    parameters = ParameterBindings(model, [("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1)])
    with pytest.raises(ValueError, match=f"^unsupported dense delta quantization {method}: mlp.gate_proj.weight$"):
        parameters.bind("mlp.gate_proj.weight", {"dtype": "BF16", "shape": [4, 2]}, "mlp.gate_proj.weight")


def test_dense_tp_slices_admit_only_exact_row_and_column_parallel_linears():
    group = SimpleNamespace(rank_in_group=1, world_size=2)
    for cls, local, dim, axis in (
        (ColumnParallelLinear, (4, 6), 0, "output_dim"),
        (RowParallelLinear, (8, 3), 1, "input_dim"),
    ):
        target = torch.zeros(local)
        setattr(target, axis, dim)
        _, slices = dense_target("w", {"shape": [8, 6]}, _bare_linear(cls, tp_group=group), target, [[0, 8], [0, 6]])
        assert slices[dim] == [local[dim], 2 * local[dim]]
    # Fused subclasses shard each part separately; one contiguous rank slice would be wrong.
    for cls in (MergedColumnParallelLinear, QKVParallelLinear):
        target = torch.zeros(4, 6)
        target.output_dim = 0
        with pytest.raises(ValueError, match="unclassified tensor-parallel mapping: w"):
            dense_target("w", {"shape": [8, 6]}, _bare_linear(cls, tp_group=group), target, [[0, 8], [0, 6]])
    quantized = _bare_linear(ColumnParallelLinear, tp_group=group, quant_method=type("Fp8LinearMethod", (), {})())
    with pytest.raises(ValueError, match="^unsupported dense delta quantization Fp8LinearMethod: w$"):
        dense_target("w", {"shape": [8, 6]}, quantized, torch.zeros(8, 6), [[0, 8], [0, 6]])


def test_bind_view_checks_dtype_quantization_selection_and_storage():
    parameters = ParameterBindings(torch.nn.Module())
    target = torch.zeros(4, 6, dtype=torch.bfloat16)
    meta = {"dtype": "BF16", "shape": [8, 6], "encoding": "xor_bytes"}
    binding = parameters.bind_view("w", meta, None, target, [[4, 8], [0, 6]])
    assert binding.slices == [[4, 8], [0, 6]] and binding.storage[0] is target and not binding.is_raw
    with pytest.raises(ValueError, match="unsupported checkpoint dtype: F8_E8M0"):
        parameters.bind_view("w", meta | {"dtype": "F8_E8M0"}, None, target, [[4, 8], [0, 6]])
    quantized = SimpleNamespace(quant_method=type("Fp8LinearMethod", (), {})())
    with pytest.raises(ValueError, match="unsupported dense delta quantization Fp8LinearMethod: w"):
        parameters.bind_view("w", meta, quantized, target, [[4, 8], [0, 6]])
    for slices in ([[6, 10], [0, 6]], [[4, 8]], [[4, 4], [0, 6]]):
        with pytest.raises(ValueError, match="canonical selection"):
            parameters.bind_view("w", meta, None, target, slices)
    with pytest.raises(ValueError, match="canonical/live dtype or shape mismatch"):
        parameters.bind_view("w", meta, None, target, [[0, 8], [0, 6]])


def test_raw_binding_targets_one_typed_tensor():
    # prepare_raw_xor XORs one typed target; a raw mask never splits into segments or across storages.
    target = torch.zeros(4, 6, dtype=torch.bfloat16)
    for storage, segments in (((target[0], target[1]), ()), ((target[0],), (DestinationSegment(((0, 2),), ()),))):
        with pytest.raises(ValueError, match="^a raw-mask binding targets one typed tensor: b$"):
            TensorBinding("b", "BF16", (8,), [[0, 8]], storage, encoding="xor_raw_bytes", segments=segments)


def test_sub_byte_f4_stays_an_unsupported_canonical_dtype():
    parameters = ParameterBindings(torch.nn.Module())
    meta = {"dtype": "F4", "shape": [4, 8], "encoding": "xor_bytes"}
    with pytest.raises(_DtypeMismatch, match="unsupported checkpoint dtype: F4"):
        parameters.bind_view("w", meta, None, torch.zeros(4, 4, dtype=torch.uint8), [[0, 4], [0, 8]])


def test_alias_groups_admit_only_disjoint_views_of_one_fused_buffer():
    fused = torch.zeros(10, 4, dtype=torch.bfloat16)

    def rows(name, start, stop, buffer=fused):
        return _direct_binding(
            name, {"dtype": "BF16", "shape": [stop - start, 4], "encoding": "xor_bytes"}, buffer[start:stop]
        )

    other = torch.zeros(2, 4, dtype=torch.bfloat16)
    groups = AliasGroups()
    groups.declare("q", "qkv", rows("q", 0, 4), fused)
    groups.declare("k", "qkv", rows("k", 4, 6), fused)
    groups.declare("o", "other", rows("o", 0, 2, other), other)
    assert groups("q", "k") and groups("k", "q")
    assert not groups("q", "o") and not groups("q", "undeclared") and not groups("undeclared", "other")
    with pytest.raises(ValueError, match="v is not a view of its fused buffer qkv"):
        groups.declare("v", "qkv", rows("v", 0, 4, torch.zeros(4, 4, dtype=torch.bfloat16)), fused)
    with pytest.raises(ValueError, match="v is not a view of its fused buffer qkv"):
        groups.declare("v", "qkv", rows("v", 6, 10), fused[:8])
    with pytest.raises(ValueError, match="fused views overlap in qkv: k and v"):
        groups.declare("v", "qkv", rows("v", 5, 8), fused)
    with pytest.raises(ValueError, match="declared twice: q"):
        groups.declare("q", "qkv", rows("q", 6, 8), fused)
    groups.declare("v", "qkv", rows("v", 6, 10), fused)
    assert groups("q", "v") and groups("v", "k")
