import sys
import types
from argparse import Namespace
from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-b-cpu", labels=["weight-update"])

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
        "sglang.srt.layers.quantization.utils",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    sys.modules["sglang.srt.utils"].MultiprocessingSerializer = object
    sys.modules["sglang.srt.utils.patch_torch"].monkey_patch_torch_reductions = lambda: None
    sys.modules["sglang.srt.weight_sync.tensor_bucket"].FlattenedTensorBucket = object
    sys.modules["sglang.srt.layers.quantization.utils"].is_layer_skipped = lambda *args, **kwargs: False
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

    def get_rank(self) -> int:
        return 0

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


_EXPERT_NAMES = [
    "decoder.layers.0.mlp.experts.linear_fc1.weight0",
    "decoder.layers.0.mlp.experts.linear_fc2.weight0",
]


def _expert_info(name: str) -> ParamInfo:
    return ParamInfo(
        name=name,
        dtype=torch.float32,
        shape=torch.Size([2]),
        attrs={"tensor_model_parallel": False, "partition_dim": -1, "partition_stride": 1},
        size=2,
        src_rank=0,
    )


@pytest.fixture
def expert_parallel(direct_module, gather_log: _GatherLog, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        direct_module,
        "get_parallel_state",
        lambda: SimpleNamespace(
            tp=SimpleNamespace(size=1, group="tp"),
            etp=SimpleNamespace(size=1, group="etp"),
            ep=SimpleNamespace(size=2, group="ep"),
        ),
    )


def _materialize_experts(direct_module) -> list[tuple[str, torch.Tensor]]:
    return direct_module._gather_megatron_expert_batch(
        Namespace(swiglu=False),
        [_expert_info(name) for name in _EXPERT_NAMES],
        {name: torch.full((2,), float(index)) for index, name in enumerate(_EXPERT_NAMES)},
        gather_pp=False,
    )


class TestExpertParallelGatherFaultHook:
    def test_the_hook_fires_before_the_first_expert_parallel_all_gather(
        self, direct_module, gather_log: _GatherLog, expert_parallel: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On an EP run the all-gather fault must strike before the first expert collective, after the name exchange."""
        _arm_marker_hook(
            monkeypatch, log=gather_log.log, hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER
        )

        gathered = _materialize_experts(direct_module)

        assert gather_log.log == [
            ("gather_names", "ep"),
            ("hook", FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER.value),
            ("gather", "ep"),
            ("gather", "ep"),
        ]
        assert [name for name, _ in gathered] == _EXPERT_NAMES * 2

    def test_a_failing_hook_starts_no_expert_collective(
        self, direct_module, gather_log: _GatherLog, expert_parallel: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hook failure must abort before this rank enters the EP all-gather its peers wait on."""
        _arm_marker_hook(
            monkeypatch,
            log=gather_log.log,
            hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER,
            fail=True,
        )

        with pytest.raises(RuntimeError, match="failed"):
            _materialize_experts(direct_module)

        assert ("gather", "ep") not in gather_log.log

    def test_a_single_expert_rank_never_reaches_the_hook(
        self, direct_module, gather_log: _GatherLog, expert_parallel: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without EP peers there is no expert all-gather, so the one-shot fault must stay armed."""
        monkeypatch.setattr(
            direct_module,
            "get_parallel_state",
            lambda: SimpleNamespace(
                tp=SimpleNamespace(size=1, group="tp"),
                etp=SimpleNamespace(size=1, group="etp"),
                ep=SimpleNamespace(size=1, group="ep"),
            ),
        )
        _arm_marker_hook(
            monkeypatch, log=gather_log.log, hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER
        )

        assert [name for name, _ in _materialize_experts(direct_module)] == _EXPERT_NAMES
        assert gather_log.log == []


class TestExpertBatchConversion:
    def test_fp8_expert_batch_matches_scalar_and_honors_exclusions(
        self, direct_module, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import miles.backends.megatron_utils.megatron_to_hf as m2h
        from miles.backends.megatron_utils.megatron_to_hf.processors import quantizer_fp8

        monkeypatch.setattr(
            quantizer_fp8,
            "is_layer_skipped",
            lambda name, ignored, *_a, **_kw: any(p in name for p in (ignored or [])),
        )
        args = Namespace(
            backend=Namespace(vocab_size=128, kv_channels=64, num_attention_heads=4, num_query_groups=4, num_layers=2)
        )
        qcfg = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "ignored_layers": ["model.layers.1.mlp.experts"],
        }
        torch.manual_seed(0)
        dec, mtp = "module.module.decoder.layers", "module.module.mtp.layers.0.transformer_layer"
        named_params = [
            (f"{dec}.0.mlp.experts.linear_fc1.weight0", torch.randn(16, 8, dtype=torch.bfloat16)),
            (f"{dec}.0.mlp.experts.linear_fc1.weight1", torch.zeros(16, 8, dtype=torch.bfloat16)),
            (f"{dec}.0.mlp.experts.linear_fc2.weight0", torch.randn(8, 8, dtype=torch.bfloat16)),
            (f"{dec}.0.mlp.experts.linear_fc2.weight1", torch.randn(8, 8, dtype=torch.bfloat16)),
            (f"{dec}.1.mlp.experts.linear_fc2.weight0", torch.randn(8, 8, dtype=torch.bfloat16)),
            (f"{mtp}.mlp.experts.linear_fc2.weight0", torch.randn(8, 8, dtype=torch.bfloat16)),
        ]

        expected = [list(m2h.convert_to_hf(args, "deepseekv3", name, param, qcfg)) for name, param in named_params]
        actual = list(m2h.convert_experts_to_hf(args, "deepseekv3", iter(named_params), qcfg))

        for a_unit, e_unit in zip(actual, expected, strict=True):
            assert [n for n, _ in a_unit] == [n for n, _ in e_unit]
            for (_, a_t), (_, e_t) in zip(a_unit, e_unit, strict=True):
                assert a_t.stride() == e_t.stride()
                assert torch.equal(
                    a_t.contiguous().reshape(-1).view(torch.uint8), e_t.contiguous().reshape(-1).view(torch.uint8)
                )
        assert [n for n, _ in actual[4]] == ["model.layers.1.mlp.experts.0.down_proj.weight"]
        assert actual[4][0][1].data_ptr() == named_params[4][1].data_ptr()

    @pytest.mark.parametrize(
        ("qcfg", "routed_fmt"),
        [
            (None, None),
            ({"quant_method": "nvfp4"}, None),
            ({"quant_method": "compressed-tensors"}, None),
            ({"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic"}, "mxfp8"),
        ],
    )
    def test_non_fp8_or_routed_expert_formats_remain_lazy(
        self, direct_module, qcfg, routed_fmt, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import miles.backends.megatron_utils.megatron_to_hf as m2h

        consumed, packed_names = [], {"model.layers.0.mlp.experts.0.down_proj"}
        args = Namespace(**({"rollout_routed_expert_format": routed_fmt} if routed_fmt else {}))

        def _convert_one(a, model_name, name, param, cfg, packed):
            assert a is args and cfg is qcfg and packed is packed_names
            return [(name, param)]

        monkeypatch.setattr(m2h, "convert_to_hf", _convert_one)
        source = ((consumed.append(i) or f"expert{i}", torch.ones(2, 2)) for i in range(3))
        stream = m2h.convert_experts_to_hf(args, "qwen3moe", source, qcfg, packed_names)
        assert consumed == []
        assert next(stream)[0][0] == "expert0" and consumed == [0]
        assert [u[0][0] for u in stream] == ["expert1", "expert2"]

    @pytest.mark.parametrize("before_gather", [True, False])
    @pytest.mark.parametrize("materialize", [True, False])
    def test_direct_iterator_routes_expert_batches_and_respects_materialize(
        self,
        direct_module,
        expert_parallel: None,
        before_gather: bool,
        materialize: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        events: list[str] = []
        cpu_device = torch.device("cpu")

        def _gather_source(_a, infos, weights, **_kw):
            events.append("gather_source")
            return [(i.name, weights[i.name]) for i in infos]

        def _convert_experts(_a, _m, params, *_r):
            events.append("convert_experts")
            return [[(f"hf.{n}", t)] for n, t in params]

        monkeypatch.setattr(direct_module.torch, "device", lambda *_a, **_kw: cpu_device)
        monkeypatch.setattr(direct_module, "_gather_megatron_expert_batch", _gather_source)
        monkeypatch.setattr(direct_module, "convert_experts_to_hf", _convert_experts)

        it = object.__new__(direct_module.HfWeightIteratorDirect)
        it.args = Namespace(custom_model_provider_path=None)
        it.model_name, it.quantization_config, it.packed_weight_basenames = "qwen3moe", {"quant_method": "fp8"}, None
        it.placement = SimpleNamespace(gather_pp=False)
        it._convert_experts_before_gather = before_gather
        it._non_expert_batches = []
        it._expert_batches = [
            direct_module._ExpertBatch(
                param_infos=[_expert_info(n) for n in _EXPERT_NAMES],
                gathers=(lambda units, device: (events.append("gather_hf"), units)[1],),
            )
        ]

        emitted = list(it._iter_hf_param_units({n: torch.ones(2) for n in _EXPERT_NAMES}, materialize=materialize))
        expected = (
            ["convert_experts", "gather_hf"]
            if before_gather
            else ["gather_source", *(["convert_experts"] if materialize else [])]
        )
        assert events == expected
        assert [u[0][0] for u in emitted] == ([f"hf.{n}" for n in _EXPERT_NAMES] if materialize else [])
