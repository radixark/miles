"""MXFP4 QAT in the NVMe optimizer store: routed experts reach the params projected onto the MXFP4 grid.

The trainer then computes with exactly the values an MXFP4 rollout engine decodes after a weight sync, while
dense params and the fp32 mains are untouched.
"""

from types import SimpleNamespace

import torch

from miles.utils.mxfp4 import project_mxfp4
from miles_plugins.optimizers.nvme_stream import NVMeOptimizerStateStore, _Entry

SHAPE = (8, 64)


def _store(qat: bool):
    torch.manual_seed(0)
    expert = torch.nn.Parameter(torch.zeros(SHAPE, dtype=torch.bfloat16))
    expert.allreduce = False  # expert-parallel: what Megatron sets on routed-expert weights
    dense = torch.nn.Parameter(torch.zeros(SHAPE, dtype=torch.bfloat16))
    expert_bias = torch.nn.Parameter(torch.zeros(SHAPE[0], dtype=torch.bfloat16))
    expert_bias.allreduce = False
    params = (expert, dense, expert_bias)

    param_data = torch.zeros(sum(p.numel() for p in params), dtype=torch.bfloat16)
    ranges, entries, offset = {}, [], 0
    for param in params:
        n = param.numel()
        ranges[param] = {"gbuf_world_in_bucket": SimpleNamespace(start=offset, end=offset + n, size=n)}
        entries.append(_Entry(param, (torch.randn(n) * 0.02).float(), 0))
        offset += n

    store = object.__new__(NVMeOptimizerStateStore)
    store.dist_opt = SimpleNamespace(
        _get_model_param_range_map=lambda p: ranges[p],
        model_param_gbuf_map={p: (0, torch.bfloat16, 0) for p in params},
        buffers=[SimpleNamespace(buckets=[SimpleNamespace(param_data=param_data)])],
    )
    store._mxfp4_qat_group_size = 32 if qat else None
    store.buckets = [SimpleNamespace(entries=entries)]
    store._mxfp4_projected = store._select_mxfp4_projected()
    return store, entries, ranges, param_data


def _written(param_data, ranges, entry):
    r = ranges[entry.model_param]["gbuf_world_in_bucket"]
    return param_data[r.start : r.end]


def test_only_routed_expert_weights_are_projected():
    store, (expert, dense, bias), ranges, param_data = _store(qat=True)
    mains = [e.main_param.clone() for e in (expert, dense, bias)]

    store._copy_main_to_model_params([expert, dense, bias])

    projected = project_mxfp4(expert.main_param.view(SHAPE).to(torch.bfloat16), 32).view(-1)
    assert torch.equal(_written(param_data, ranges, expert).view(torch.int16), projected.view(torch.int16))
    assert not torch.equal(_written(param_data, ranges, expert), expert.main_param.to(torch.bfloat16))
    assert torch.equal(_written(param_data, ranges, dense), dense.main_param.to(torch.bfloat16))
    assert torch.equal(_written(param_data, ranges, bias), bias.main_param.to(torch.bfloat16))
    for entry, main in zip((expert, dense, bias), mains, strict=True):
        assert torch.equal(entry.main_param, main), "the fp32 main must keep the full update"


def test_without_qat_every_param_gets_its_main():
    store, entries, ranges, param_data = _store(qat=False)

    store._copy_main_to_model_params(entries)

    for entry in entries:
        assert torch.equal(_written(param_data, ranges, entry), entry.main_param.to(torch.bfloat16))
