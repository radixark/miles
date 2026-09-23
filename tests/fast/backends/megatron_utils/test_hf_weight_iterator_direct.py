import sys
import types
from argparse import Namespace
from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import pytest
import torch
from tests.fast.utils.test_utils.fault_injector.fakes import _arm_marker_hook

from miles.utils.test_utils.fault_injector.models import FaultHookName
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
        "megatron.core.transformer",
        "megatron.core.transformer.transformer_layer",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["megatron.core.transformer.transformer_layer"].get_transformer_layer_offset = lambda *args: 0


@pytest.fixture
def direct_module(monkeypatch):
    module_names = [
        "miles.backends.megatron_utils.sglang",
        "miles.backends.megatron_utils.megatron_to_hf",
        "miles.backends.megatron_utils.megatron_to_hf.processors",
        "miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8",
        "miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_mxfp8",
        "miles.backends.megatron_utils.named_weights",
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


def _param(name: str, size: int) -> ParamInfo:
    return ParamInfo(
        name=name,
        dtype=torch.float32,
        shape=torch.Size([size]),
        attrs={},
        size=size,
        src_rank=0,
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


class _GatherLog:
    def __init__(self) -> None:
        self.log: list[object] = []

    def all_gather(self, buffers: list[torch.Tensor], tensor: torch.Tensor, *, group: str, async_op: bool) -> "_Done":
        self.log.append(("gather", group))
        for buffer in buffers:
            buffer.copy_(tensor)
        return _Done()

    def all_gather_object(self, output: list[object], obj: object, *, group: str) -> None:
        self.log.append(("gather_names", group))
        output[:] = [list(obj) for _ in output]


class _Done:
    def wait(self) -> None:
        pass


def _tp_param(size: int, *, tensor_model_parallel: bool = True) -> torch.Tensor:
    param = torch.arange(size, dtype=torch.float32)
    param.tensor_model_parallel = tensor_model_parallel
    param.partition_dim = 0
    param.partition_stride = 1
    return param


@pytest.fixture
def gather_log(direct_module, monkeypatch) -> _GatherLog:
    fake = _GatherLog()
    monkeypatch.setattr(direct_module, "dist", fake)
    return fake


class TestTensorParallelGatherFaultHook:
    def test_the_hook_fires_before_the_first_tensor_parallel_all_gather(
        self, direct_module, gather_log: _GatherLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A trainer fault armed at the all-gather must strike before any TP collective starts."""
        monkeypatch.setattr(
            direct_module,
            "get_parallel_state",
            lambda: SimpleNamespace(tp=SimpleNamespace(size=2, group="tp"), etp=SimpleNamespace(size=1, group="etp")),
        )
        _arm_marker_hook(
            monkeypatch, log=gather_log.log, hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER
        )
        names = ["decoder.layers.0.self_attention.linear_proj.weight", "decoder.layers.0.mlp.linear_fc2.bias"]

        gathered = direct_module.all_gather_params_async(
            Namespace(swiglu=False), [(_param(name, 2), _tp_param(2)) for name in names]
        )

        assert gather_log.log == [
            ("hook", FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER.value),
            ("gather", "tp"),
            ("gather", "tp"),
        ]
        assert [param.tolist() for param in gathered] == [[0.0, 1.0, 0.0, 1.0]] * 2

    @pytest.mark.parametrize("tp_size,tensor_model_parallel", [(1, True), (2, False)])
    def test_a_batch_without_any_collective_never_reaches_the_hook(
        self,
        direct_module,
        gather_log: _GatherLog,
        monkeypatch: pytest.MonkeyPatch,
        tp_size: int,
        tensor_model_parallel: bool,
    ) -> None:
        """Replicated params or a single TP rank must leave the all-gather fault armed for a real gather."""
        monkeypatch.setattr(
            direct_module,
            "get_parallel_state",
            lambda: SimpleNamespace(
                tp=SimpleNamespace(size=tp_size, group="tp"), etp=SimpleNamespace(size=1, group="etp")
            ),
        )
        _arm_marker_hook(
            monkeypatch, log=gather_log.log, hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER
        )

        direct_module.all_gather_params_async(
            Namespace(swiglu=False),
            [(_param("decoder.final_layernorm.weight", 2), _tp_param(2, tensor_model_parallel=tensor_model_parallel))],
        )

        assert gather_log.log == []

    def test_a_failing_hook_starts_no_collective(
        self, direct_module, gather_log: _GatherLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hook failure must abort the update before this rank enters the TP all-gather."""
        monkeypatch.setattr(
            direct_module,
            "get_parallel_state",
            lambda: SimpleNamespace(tp=SimpleNamespace(size=2, group="tp"), etp=SimpleNamespace(size=1, group="etp")),
        )
        _arm_marker_hook(
            monkeypatch, log=gather_log.log, hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER, fail=True
        )

        with pytest.raises(RuntimeError, match="failed"):
            direct_module.all_gather_params_async(
                Namespace(swiglu=False),
                [(_param("decoder.layers.0.self_attention.linear_proj.weight", 2), _tp_param(2))],
            )

        assert gather_log.log == [("hook", FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER.value)]
