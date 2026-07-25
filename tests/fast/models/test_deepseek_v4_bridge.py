import pytest
import torch


@pytest.fixture(scope="module")
def bridge_stub():
    pytest.importorskip("mbridge")
    from miles_plugins.mbridge.deepseekv4 import DeepseekV4Bridge

    bridge = DeepseekV4Bridge.__new__(DeepseekV4Bridge)
    bridge.make_vocab_size_divisible_by = None
    return bridge


def test_tid2eid_preserves_integer_expert_indices(bridge_stub):
    bridge_stub.dtype = torch.bfloat16
    tid2eid = torch.tensor([[0, 257, 381, 382, 383]], dtype=torch.int64)

    converted = bridge_stub._weight_to_mcore_format(
        "decoder.layers.0.mlp.router.tid2eid", [tid2eid]
    )

    assert converted.dtype == torch.int64
    assert torch.equal(converted, tid2eid)
    assert converted.is_contiguous()
