import sys
import types
from argparse import Namespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.utils.types import ParamInfo


def _install_import_stubs(monkeypatch):
    triton = types.ModuleType("triton")
    triton.jit = lambda fn: fn
    triton.cdiv = lambda x, y: (x + y - 1) // y
    tl = types.ModuleType("triton.language")
    tl.constexpr = int
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", tl)

    for name in [
        "sglang",
        "sglang.srt",
        "sglang.srt.utils",
        "sglang.srt.utils.patch_torch",
        "sglang.srt.weight_sync",
        "sglang.srt.weight_sync.tensor_bucket",
        "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.fp8_utils",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    sys.modules["sglang.srt.utils"].MultiprocessingSerializer = object
    sys.modules["sglang.srt.utils.patch_torch"].monkey_patch_torch_reductions = lambda: None
    sys.modules["sglang.srt.weight_sync.tensor_bucket"].FlattenedTensorBucket = object
    fp8_utils = sys.modules["sglang.srt.layers.quantization.fp8_utils"]
    fp8_utils.quant_weight_ue8m0 = lambda *args, **kwargs: None
    fp8_utils.transform_scale_ue8m0 = lambda x, **kwargs: x

    ray = types.ModuleType("ray")
    ray_actor = types.ModuleType("ray.actor")
    ray_util = types.ModuleType("ray.util")
    ray_scheduling = types.ModuleType("ray.util.scheduling_strategies")
    ray.remote = lambda *args, **kwargs: args[0] if args and callable(args[0]) and not kwargs else lambda obj: obj
    ray_actor.ActorHandle = object
    ray_scheduling.NodeAffinitySchedulingStrategy = object
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.actor", ray_actor)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", ray_scheduling)

    for name in [
        "megatron",
        "megatron.core",
        "megatron.core.utils",
        "megatron.core.transformer",
        "megatron.core.transformer.transformer_layer",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["megatron.core.transformer.transformer_layer"].get_transformer_layer_offset = lambda *args: 0
    sys.modules["megatron.core.utils"].unwrap_model = lambda model: model


@pytest.fixture
def direct_module(monkeypatch):
    module_names = [
        "miles.backends.megatron_utils.sglang",
        "miles.backends.megatron_utils.megatron_to_hf",
        "miles.backends.megatron_utils.megatron_to_hf.processors",
        "miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8",
        "miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_mxfp8",
        "miles.backends.megatron_utils.named_weights",
        "miles.backends.megatron_utils.update_weight.hf_weight_iterator",
        "miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct",
    ]
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)

    _install_import_stubs(monkeypatch)

    from miles.backends.megatron_utils.update_weight import hf_weight_iterator_direct

    yield hf_weight_iterator_direct

    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _param(name: str, size: int, *, src_rank: int = 0) -> ParamInfo:
    return ParamInfo(
        name=name,
        dtype=torch.float32,
        shape=torch.Size([size]),
        attrs={},
        size=size,
        src_rank=src_rank,
    )


def test_gather_batches_pack_by_size_only(direct_module, monkeypatch):
    # Atomicity is the base template's job; gather batches are pure size packing.
    params = [_param("layer.a", 4), _param("layer.b", 2), _param("layer.c", 4)]
    monkeypatch.setattr(direct_module, "_get_param_full_size", lambda info: info.size)

    batches = direct_module._pack_param_infos_by_size(Namespace(update_weight_buffer_size=6), params)
    assert [[param.name for param in batch] for batch in batches] == [["layer.a", "layer.b"], ["layer.c"]]

    batches = direct_module._pack_param_infos_by_size(
        Namespace(update_weight_buffer_size=6), params, size_multiplier=2
    )
    assert [[param.name for param in batch] for batch in batches] == [["layer.a"], ["layer.b"], ["layer.c"]]


@pytest.mark.parametrize("mode", [None, "broadcast_packed", "disk-delta", "gpu-delta"])
@pytest.mark.parametrize("expert", [False, True])
def test_batch_load_preserves_sync_before_gather_outside_gpu_delta(direct_module, monkeypatch, mode, expert):
    events = []
    value = torch.arange(4, dtype=torch.float32)
    info = _param("weight", 4)
    # Offline conversion namespaces do not carry a weight-transfer mode.
    args = Namespace(**({"update_weight_transfer_mode": mode} if mode is not None else {}))

    class Weight:
        def to(self, *, device, non_blocking):
            assert device == "cpu" and non_blocking
            events.append("load")
            return value

    def gather(args, params):
        events.append("gather")
        return [param for _, param in params]

    monkeypatch.setattr(direct_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(direct_module.torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(direct_module.torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(direct_module, "get_parallel_state", lambda: Namespace(ep=Namespace(size=1)))
    monkeypatch.setattr(direct_module, "all_gather_params_async", gather)
    load = direct_module._gather_megatron_expert_batch if expert else direct_module._materialize_non_expert_batch

    result = load(args, [info], {"weight": Weight()}, gather_pp=False)

    assert events == (["load", "gather"] if mode == "gpu-delta" else ["load", "sync", "gather"])
    assert len(result) == 1 and result[0][0] == "weight"
    assert torch.equal(result[0][1], value)


@pytest.mark.parametrize("materialize", [True, False])
@pytest.mark.parametrize("consume_locally", [True, False])
def test_owner_consumer_skips_gathers_and_normal_export_still_gathers(
    direct_module, monkeypatch, materialize, consume_locally
):
    events = []
    packed = ("expert.gate_proj.weight", torch.zeros(4, dtype=torch.uint8))
    scale = ("expert.gate_proj.weight_scale", torch.ones(1))
    local_name = "layer.experts.linear_fc1.weight0"
    remote = _param("layer.experts.linear_fc1.weight1", 4, src_rank=1)

    class Weight:
        def detach(self):
            return self

        def to(self, **kwargs):
            events.append("load")
            return packed[1]

    def convert(named_params):
        assert named_params[0][0] == local_name
        events.append("convert")
        yield [packed, scale]

    def consume(unit):
        assert unit == [packed, scale]
        events.append("consume")

    def gather(units, **kwargs):
        assert not consume_locally, "Owner-local consumption must skip every expert gather"
        events.append("gather")
        assert units == [[packed, scale]]
        return units

    iterator = direct_module.HfWeightIteratorDirect.__new__(direct_module.HfWeightIteratorDirect)
    iterator.args = Namespace()
    iterator._non_expert_batches = []
    iterator._expert_batches = [
        direct_module._ExpertBatch(param_infos=[_param(local_name, 4), remote], gathers=(gather, gather)),
        # An empty EDP-owner round must skip PP/EP gathers too; this is a
        # uniform consumer contract, not a decision based on local payload size.
        direct_module._ExpertBatch(param_infos=[remote], gathers=(gather, gather)) if consume_locally
        else direct_module._ExpertBatch(param_infos=[_param(local_name, 4)], gathers=(gather,)),
    ]
    iterator._convert_to_hf_param_units = convert
    iterator._convert_experts_before_gather = True
    iterator._expert_consumer = None
    if consume_locally:
        iterator.set_local_expert_consumer(consumer=consume)
    monkeypatch.setattr(direct_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(direct_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(direct_module, "_iter_mm_tower_units", lambda *args, **kwargs: iter(()))

    units = list(iterator._iter_hf_param_units({local_name: Weight()}, materialize=materialize))

    if consume_locally:
        assert units == []
        assert events == ["load", "convert", "consume"]
    else:
        assert units == ([[packed, scale], [packed, scale]] if materialize else [])
        assert events == ["load", "convert", "gather", "gather", "load", "convert", "gather"]


def test_gpu_delta_consumer_defers_failure_until_all_local_units_are_visited(direct_module, monkeypatch):
    from miles.backends.training_utils.weight_update.protocols.gpu_delta import UpdateWeightFromGpuDelta

    protocol = UpdateWeightFromGpuDelta(Namespace(custom_update_weight_post_write_path=None))
    failure, converted = ValueError("invalid canonical layout"), []

    def reject(name, tensor):
        raise failure

    protocol._match_layout = reject
    iterator = direct_module.HfWeightIteratorDirect.__new__(direct_module.HfWeightIteratorDirect)
    iterator._convert_experts_before_gather = True
    protocol.bind_iterator(iterator)

    class Weight:
        def detach(self):
            return self

        def to(self, **kwargs):
            return torch.zeros(1)

    def convert(named_params):
        converted.append(named_params[0][0])
        yield named_params

    iterator._convert_to_hf_param_units = convert
    monkeypatch.setattr(direct_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(direct_module.torch.cuda, "current_device", lambda: 0)
    batch = direct_module._ExpertBatch(
        param_infos=[_param("first", 1), _param("second", 1), _param("foreign", 1, src_rank=1)],
        gathers=(lambda *args, **kwargs: pytest.fail("Failed consumers must not enter expert gathers"),),
    )
    assert iterator._convert_and_gather_expert_batch(batch, {name: Weight() for name in ("first", "second")}) == []
    assert converted == ["first", "second"]
    assert protocol._error is failure  # after_base_weights performs the existing collective error check.


@pytest.mark.parametrize("materialize", [True, False])
def test_gpu_delta_etp2_gathers_complete_experts_before_sender_conversion(direct_module, monkeypatch, materialize):
    from miles.backends.training_utils.weight_update.protocols.gpu_delta import UpdateWeightFromGpuDelta

    name = "layer.experts.linear_fc1.weight0"
    # Unmarked TE grouped weights still need ETP gathering. Each shard contains
    # one gate row followed by one up row; conversion must see gate/gate/up/up.
    shards = [torch.tensor([[1.0, 2.0], [5.0, 6.0]]), torch.tensor([[3.0, 4.0], [7.0, 8.0]])]
    complete = torch.arange(1.0, 9.0).reshape(4, 2)
    info = ParamInfo(
        name=name, dtype=torch.float32, shape=shards[0].shape, size=shards[0].nbytes, src_rank=0,
        attrs={"tensor_model_parallel": False, "partition_dim": -1, "partition_stride": 1},
    )
    waited, converted = [], []
    etp_group = object()

    class Work:
        def wait(self):
            waited.append(True)

    def all_gather(buffers, tensor, *, group, async_op):
        assert group is etp_group and async_op
        assert torch.equal(tensor, shards[0])
        for buffer, shard in zip(buffers, shards, strict=True):
            buffer.copy_(shard)
        return Work()

    def convert(named_params):
        assert waited and len(named_params) == 1
        assert named_params[0][0] == name and torch.equal(named_params[0][1], complete)
        converted.append(name)
        yield [("expert.weight", named_params[0][1].to(torch.uint8)), ("expert.scale", torch.ones(1))]

    iterator = direct_module.HfWeightIteratorDirect.__new__(direct_module.HfWeightIteratorDirect)
    iterator.args = Namespace(swiglu=True, update_weight_transfer_mode="gpu-delta")
    iterator.placement = Namespace(gather_pp=False)
    iterator._non_expert_batches = []
    iterator._expert_batches = [direct_module._ExpertBatch(param_infos=[info], gathers=())]
    iterator._convert_experts_before_gather = False
    iterator._convert_to_hf_param_units = convert
    protocol = UpdateWeightFromGpuDelta(Namespace(custom_update_weight_post_write_path=None))
    protocol.send_bucket = lambda unit: pytest.fail("ETP-sharded experts must not use the owner-local consumer")
    protocol.bind_iterator(iterator)
    monkeypatch.setattr(direct_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(direct_module.dist, "all_gather", all_gather)
    monkeypatch.setattr(direct_module, "get_parallel_state", lambda: Namespace(
        etp=Namespace(size=2, group=etp_group), ep=Namespace(size=1)))
    monkeypatch.setattr(
        direct_module, "_load_or_allocate_params", lambda infos, weights, **kwargs: [weights[info.name].clone()]
    )
    monkeypatch.setattr(direct_module, "_iter_mm_tower_units", lambda *args, **kwargs: iter(()))

    units = list(iterator._iter_hf_param_units({name: shards[0]}, materialize=materialize))
    assert waited == [True]  # Non-senders must also complete the ordinary collective.
    assert converted == ([name] if materialize else [])
    if materialize:
        assert len(units) == 1 and [key for key, _ in units[0]] == ["expert.weight", "expert.scale"]
        assert torch.equal(units[0][0][1], complete.to(torch.uint8))
    else:
        assert units == []


def test_producer_discovery_installs_actual_owner_hook_and_preserves_plan(direct_module, monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path

    import safetensors.torch

    from miles.backends.training_utils import parallel

    path = Path(__file__).parents[3] / "manual" / "bench_gpu_delta_producer.py"
    spec = importlib.util.spec_from_file_location("gpu_delta_discovery_benchmark", path)
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    expert_name = "model.layers.3.mlp.experts.0.gate_proj.weight"
    dense_name = "model.norm.weight"
    weights = {expert_name: torch.zeros((2, 2), dtype=torch.uint8), dense_name: torch.ones(2)}
    safetensors.torch.save_file(weights, tmp_path / "model.safetensors")
    iterator = direct_module.HfWeightIteratorDirect.__new__(direct_module.HfWeightIteratorDirect)
    iterator._convert_experts_before_gather = True

    def buckets(values, *, materialize):
        assert materialize
        assert iterator._expert_consumer([(expert_name, values[expert_name])]) is None
        yield [(dense_name, values[dense_name])]

    iterator.iter_hf_weights = buckets
    monkeypatch.setattr(parallel, "get_parallel_state", lambda: Namespace(
        ep=Namespace(rank=0, size=1), edp=Namespace(rank=0)))
    monkeypatch.setattr(producer.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(producer, "_gather", lambda value: [value])

    def check(error, stage):
        if error is not None:
            raise error

    monkeypatch.setattr(producer, "_check", check)
    plan, ownership = producer._discover_plan(Namespace(hf_checkpoint=str(tmp_path), num_experts=2), iterator, weights)
    assert ownership["names"] == sorted(weights)
    assert ownership["routed_tensor_count"] == 1
    assert {entry["name"]: entry["encoding"] for entry in plan} == {
        expert_name: "xor_bytes", dense_name: "raw_bytes",
    }
