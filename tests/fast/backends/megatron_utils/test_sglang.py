from miles.backends.megatron_utils.sglang import _accept_grouped_moe


def _requant_before_grouped_moe(weight_block_size, output_dtype=None, weight_shape=None):
    return weight_block_size is not None


def test_grouped_moe_is_dropped_for_sglang_without_it() -> None:
    """The ROCm image's SGLang (sglang-miles-v0.5.22-final) has no `grouped_moe`;
    FP8 weight updates there must still get the requant decision instead of a TypeError."""
    should_requant = _accept_grouped_moe(_requant_before_grouped_moe)

    assert should_requant(weight_block_size=[128, 128], grouped_moe=True) is True
    assert should_requant(weight_block_size=None, grouped_moe=False) is False


def test_sglang_with_grouped_moe_is_called_unchanged() -> None:
    def requant(weight_block_size, output_dtype=None, weight_shape=None, grouped_moe=False):
        return grouped_moe

    assert _accept_grouped_moe(requant) is requant
