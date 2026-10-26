import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig as HFVisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from miles_plugins.models.qwen3_5.mm_towers import _mrope_position_ids, _scatter_rows
from miles_plugins.models.qwen3_5.vision import Qwen3_5VisionConfig, Qwen3_5VisionTower, _PatchEmbed, _rot_pos_ids

_VISION_CONFIG = dict(
    depth=2,
    hidden_size=64,
    num_heads=4,
    intermediate_size=128,
    hidden_act="gelu_pytorch_tanh",
    in_channels=3,
    patch_size=4,
    temporal_patch_size=2,
    spatial_merge_size=2,
    out_hidden_size=32,
    num_position_embeddings=36,
)


def test_tower_loads_the_hf_checkpoint_layout():
    hf = Qwen3_5VisionModel(HFVisionConfig(**_VISION_CONFIG))
    tower = Qwen3_5VisionTower(Qwen3_5VisionConfig(**_VISION_CONFIG))

    assert {k: v.shape for k, v in tower.state_dict().items()} == {k: v.shape for k, v in hf.state_dict().items()}


def test_config_rejects_deepstack_mergers():
    with pytest.raises(ValueError, match="deepstack"):
        Qwen3_5VisionConfig.from_hf({**_VISION_CONFIG, "deepstack_visual_indexes": [3]})


def test_patch_embed_gemm_matches_conv3d():
    torch.manual_seed(0)
    embed = _PatchEmbed(Qwen3_5VisionConfig(**_VISION_CONFIG)).double()
    pixel_values = torch.randn(10, 3 * 2 * 4 * 4, dtype=torch.float64)

    expected = embed.proj(pixel_values.view(-1, 3, 2, 4, 4)).view(-1, 64)

    torch.testing.assert_close(embed(pixel_values), expected)


def test_rotary_positions_follow_the_spatial_merge_block_order():
    # A 2x4 grid arrives as two 2x2 merge blocks, each read row by row.
    assert _rot_pos_ids(2, 4, 2).tolist() == [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]]


def test_rows_land_only_in_this_sequence_parallel_window():
    decoder_input = torch.zeros(4, 1, 2)
    rows = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    # Rank 1 holds packed tokens 4..7.
    merged = _scatter_rows(decoder_input, rows, torch.tensor([3, 4, 6]), sp_rank=1)

    assert merged[:, 0, 0].tolist() == [2.0, 0.0, 3.0, 0.0]


def test_mrope_positions_continue_over_the_padding():
    positions = torch.tensor([[0, 0, 0], [1, 1, 1], [1, 2, 3], [5, 5, 5]])

    position_ids = _mrope_position_ids(positions, torch.zeros(1, 6, dtype=torch.long))

    assert position_ids.shape == (3, 1, 6)
    assert position_ids[:, 0].tolist() == [[0, 1, 1, 5, 6, 7], [0, 1, 2, 5, 6, 7], [0, 1, 3, 5, 6, 7]]
