import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.rematerialize_utils import build_main_cast_context


@pytest.mark.parametrize("precision_aware", [False, True])
def test_main_cast_context_selects_backend_optimizer_copy_and_preserves_extras(
    monkeypatch: pytest.MonkeyPatch, precision_aware: bool
) -> None:
    """The context selects the backend copy path and retains uncovered FP32 tensors."""
    module = torch.nn.Linear(2, 1, bias=False)
    model = SimpleNamespace(
        named_buffers=module.named_buffers,
        named_parameters=module.named_parameters,
        buffers=[],
        expert_parallel_buffers=[],
    )
    view = torch.zeros(2, dtype=torch.bfloat16)
    master = torch.tensor([2.0, 3.0])
    copied = []

    class HybridDeviceOptimizer:
        def __init__(self) -> None:
            self.gpu_params_map_cpu_copy = {view: master}
            self.param_to_fp32_param = {}

    sdk = ModuleType("megatron.core.optimizer.cpu_offloading.hybrid_optimizer")
    sdk.HybridDeviceOptimizer = HybridDeviceOptimizer
    monkeypatch.setitem(sys.modules, sdk.__name__, sdk)
    optimizer = SimpleNamespace(
        chained_optimizers=[
            SimpleNamespace(
                optimizer=HybridDeviceOptimizer(),
                _copy_main_params_to_model_params=lambda: copied.append("mcore"),
            )
        ]
    )
    args = SimpleNamespace(
        backend=SimpleNamespace(use_precision_aware_optimizer=precision_aware),
        check_rematerialize_param_from_master_weight=True,
    )

    context = build_main_cast_context(args, model=[model], optimizer=optimizer)
    context.cast_main_to_params()

    assert context.model_chunks == [model]
    assert context.check is True
    assert context.rematerializable_ids == {id(module.weight)}
    extras = list(context.extras_getter())
    assert len(extras) == 1
    assert extras[0][0] == "vp_stages.0.weight"
    assert extras[0][1] is module.weight
    assert copied == ([] if precision_aware else ["mcore"])
    assert torch.equal(view, master.to(torch.bfloat16) if precision_aware else torch.zeros_like(view))
