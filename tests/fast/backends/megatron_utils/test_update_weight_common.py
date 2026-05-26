from argparse import Namespace

import torch

from miles.backends.megatron_utils.update_weight import common
from miles.utils.types import ParamInfo


def test_tp_world_size_one_skips_weight_all_gather(monkeypatch):
    param = torch.nn.Parameter(torch.arange(4, dtype=torch.float32), requires_grad=False)
    param.tensor_model_parallel = True
    param.parallel_mode = None
    param.partition_dim = 0
    param.partition_stride = 1

    info = ParamInfo(
        name="module.module.decoder.layers.0.self_attention.linear_qkv.weight",
        dtype=param.dtype,
        shape=param.shape,
        attrs={
            "tensor_model_parallel": True,
            "parallel_mode": None,
            "partition_dim": 0,
            "partition_stride": 1,
        },
        size=param.numel() * param.element_size(),
        src_rank=0,
    )

    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_group", lambda: object())

    def _unexpected_all_gather(*args, **kwargs):
        raise AssertionError("world-size-1 TP params must not call dist.all_gather")

    monkeypatch.setattr(common.dist, "all_gather", _unexpected_all_gather)

    gathered = common.all_gather_params_async(Namespace(swiglu=False), [(info, param)])

    assert len(gathered) == 1
    assert gathered[0].data_ptr() == param.data.data_ptr()
    torch.testing.assert_close(gathered[0], param.data)


def test_grouped_moe_weight1_expands_to_global_expert_slices():
    args = Namespace(num_experts=8, hidden_size=2, swiglu=True)
    param = torch.arange(24, dtype=torch.float32).view(2, 12)
    param.tensor_model_parallel = True
    param.parallel_mode = None

    expanded = list(
        common._iter_grouped_moe_expert_params(
            args,
            layer_idx=3,
            rest="mlp.experts.weight1",
            param=param,
            expert_offset=4,
            ep_size=4,
        )
    )

    assert [name for name, _ in expanded] == [
        "module.module.decoder.layers.3.mlp.experts.linear_fc1.weight4",
        "module.module.decoder.layers.3.mlp.experts.linear_fc1.weight5",
    ]
    torch.testing.assert_close(expanded[0][1], param.view(2, 2, 6)[0].transpose(0, 1))
    torch.testing.assert_close(expanded[1][1], param.view(2, 2, 6)[1].transpose(0, 1))
    assert expanded[0][1].tensor_model_parallel is True
    assert expanded[0][1].partition_dim == 0
    assert expanded[0][1].partition_stride == 2


def test_grouped_moe_weight2_expands_to_global_expert_slices():
    args = Namespace(num_experts=8, hidden_size=2, swiglu=True)
    param = torch.arange(12, dtype=torch.float32).view(6, 2)
    param.tensor_model_parallel = True
    param.parallel_mode = None

    expanded = list(
        common._iter_grouped_moe_expert_params(
            args,
            layer_idx=3,
            rest="mlp.experts.weight2",
            param=param,
            expert_offset=4,
            ep_size=4,
        )
    )

    assert [name for name, _ in expanded] == [
        "module.module.decoder.layers.3.mlp.experts.linear_fc2.weight4",
        "module.module.decoder.layers.3.mlp.experts.linear_fc2.weight5",
    ]
    torch.testing.assert_close(expanded[0][1], param.view(2, 3, 2)[0].transpose(0, 1))
    torch.testing.assert_close(expanded[1][1], param.view(2, 3, 2)[1].transpose(0, 1))
    assert expanded[0][1].tensor_model_parallel is True
    assert expanded[0][1].partition_dim == 1
    assert expanded[0][1].partition_stride == 1
