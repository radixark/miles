from types import SimpleNamespace

import torch

from miles_plugins.models.glm5_next import lora


def test_export_fused_fc1_splits_each_tp_shard_before_gather(monkeypatch) -> None:
    local_a = torch.tensor([[1.0], [2.0]])
    local_b = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]])
    remote_a = torch.tensor([[3.0], [4.0]])
    remote_gate_b = torch.tensor([[50.0, 51.0], [60.0, 61.0]])
    remote_up_b = torch.tensor([[70.0, 71.0], [80.0, 81.0]])
    gather_results = iter(
        (
            torch.cat((local_a, remote_a), dim=0),
            torch.cat((local_b[:2], remote_gate_b), dim=0),
            torch.cat((local_b[2:], remote_up_b), dim=0),
        )
    )
    monkeypatch.setattr(lora, "_gather_tp", lambda tensor, dim: next(gather_results))
    wrapper = SimpleNamespace(
        adapters=[
            SimpleNamespace(
                input_is_parallel=False,
                linear_in=SimpleNamespace(weight=local_a),
                linear_out=SimpleNamespace(weight=local_b),
            )
        ]
    )

    exported = dict(lora._export_linear(wrapper, 0, 4, "model.layers.0.mlp.", "gate_up"))

    assert torch.equal(
        exported["model.layers.0.mlp.gate_proj.lora_B.weight"],
        torch.cat((local_b[:2], remote_gate_b), dim=0),
    )
    assert torch.equal(
        exported["model.layers.0.mlp.up_proj.lora_B.weight"],
        torch.cat((local_b[2:], remote_up_b), dim=0),
    )
