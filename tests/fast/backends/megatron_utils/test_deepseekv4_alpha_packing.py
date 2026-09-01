import torch

from miles.backends.megatron_utils.megatron_to_hf.deepseekv4 import convert_deepseekv4_to_hf


def _alpha_name(layer: int, site: str, seg: str) -> str:
    return f"module.module.decoder.layers.{layer}.{site}_hyper_connection.alpha_{seg}"


def test_alphas_pack_once_complete_regardless_of_arrival_order():
    assert convert_deepseekv4_to_hf(None, _alpha_name(3, "self_attention", "res"), torch.tensor(3.0)) == []
    assert convert_deepseekv4_to_hf(None, _alpha_name(3, "self_attention", "pre"), torch.tensor(1.0)) == []
    out = convert_deepseekv4_to_hf(None, _alpha_name(3, "self_attention", "post"), torch.tensor(2.0))
    assert [name for name, _tensor in out] == ["model.layers.3.hc_attn_scale"]
    assert torch.equal(out[0][1], torch.tensor([1.0, 2.0, 3.0]))


def test_alpha_sites_accumulate_independently():
    assert convert_deepseekv4_to_hf(None, _alpha_name(0, "self_attention", "pre"), torch.tensor(1.0)) == []
    assert convert_deepseekv4_to_hf(None, _alpha_name(0, "mlp", "pre"), torch.tensor(10.0)) == []
    assert convert_deepseekv4_to_hf(None, _alpha_name(1, "mlp", "pre"), torch.tensor(100.0)) == []
    assert convert_deepseekv4_to_hf(None, _alpha_name(0, "mlp", "post"), torch.tensor(20.0)) == []
    assert convert_deepseekv4_to_hf(None, _alpha_name(0, "self_attention", "post"), torch.tensor(2.0)) == []

    out = convert_deepseekv4_to_hf(None, _alpha_name(0, "mlp", "res"), torch.tensor(30.0))
    assert [name for name, _tensor in out] == ["model.layers.0.hc_ffn_scale"]
    assert torch.equal(out[0][1], torch.tensor([10.0, 20.0, 30.0]))

    out = convert_deepseekv4_to_hf(None, _alpha_name(0, "self_attention", "res"), torch.tensor(3.0))
    assert torch.equal(out[0][1], torch.tensor([1.0, 2.0, 3.0]))

    assert convert_deepseekv4_to_hf(None, _alpha_name(1, "mlp", "post"), torch.tensor(200.0)) == []
    out = convert_deepseekv4_to_hf(None, _alpha_name(1, "mlp", "res"), torch.tensor(300.0))
    assert torch.equal(out[0][1], torch.tensor([100.0, 200.0, 300.0]))


def test_alpha_packing_resets_for_the_next_sync():
    for round_offset in (0.0, 5.0):
        for seg, value in (("pre", 1.0), ("post", 2.0)):
            assert convert_deepseekv4_to_hf(None, _alpha_name(7, "mlp", seg), torch.tensor(value + round_offset)) == []
        out = convert_deepseekv4_to_hf(None, _alpha_name(7, "mlp", "res"), torch.tensor(3.0 + round_offset))
        assert torch.equal(out[0][1], torch.tensor([1.0 + round_offset, 2.0 + round_offset, 3.0 + round_offset]))
