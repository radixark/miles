"""The frozen NaViT tower loads from released-checkpoint tensor names (q/k rows re-interleaved like SGLang's loader)
and samples the position table in the checkpoint's xy layout."""

import pytest
import torch
from safetensors.torch import save_file

from miles_plugins.models.muse_spark.vision.mm_towers import load_tower_state
from miles_plugins.models.muse_spark.vision.navit import (
    MuseSparkVisionConfig,
    MuseSparkVisionTower,
    patch_grid_positions,
)
from miles_plugins.models.muse_spark.weight_mapping import NON_TEXT_PREFIXES

_CONFIG = MuseSparkVisionConfig(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=48,
    vision_output_dim=128,
    image_size=56,
    projector_input_dim=24,
    llm_hidden_size=40,
)


def _sglang_rotary_rows(weight: torch.Tensor, heads: int) -> torch.Tensor:
    # sglang/srt/models/muse_spark_omni.py load_weights, q/k projections of the vision tower.
    shape = weight.shape
    return weight.reshape(heads, 2, shape[0] // heads // 2, *shape[1:]).transpose(1, 2).reshape(shape)


def test_every_parameter_has_a_distinct_released_name():
    tower = MuseSparkVisionTower(_CONFIG)
    names = {key: tower.checkpoint_name(key) for key in tower.state_dict()}

    assert len(set(names.values())) == len(names)
    assert all(name.startswith(NON_TEXT_PREFIXES[:3]) for name in names.values())
    assert (
        names["vision_model.encoder.layers.1.attn.out_proj.weight"] == "model.vision_tower.layers.1.attn.proj.weight"
    )
    assert names["vision_model.encoder.layers.0.ln_1.bias"] == "model.vision_tower.layers.0.norm1.bias"
    assert names["vision_model.encoder.layers.0.mlp.c_proj.weight"] == "model.vision_tower.layers.0.mlp.fc2.weight"
    assert names["vision_model.adapter.c_proj.weight"] == "model.vision_adapter.fc2.weight"
    assert names["vision_projection.weight"] == "model.vision_projection.weight"


def test_load_from_released_names(tmp_path):
    tower = MuseSparkVisionTower(_CONFIG)
    stored = {tower.checkpoint_name(key): torch.randn(value.shape) for key, value in tower.state_dict().items()}
    stored["model.language_model.embed_tokens.weight"] = torch.zeros(3)
    stored["model.audio_tower.proj.weight"] = torch.zeros(3)
    save_file(stored, str(tmp_path / "model.safetensors"))

    state = load_tower_state(tmp_path, tower)

    assert state.keys() == tower.state_dict().keys()
    heads = _CONFIG.num_attention_heads
    for key, value in state.items():
        source = stored[tower.checkpoint_name(key)]
        rotary = ".attn.q_proj." in key or ".attn.k_proj." in key
        assert torch.equal(value, _sglang_rotary_rows(source, heads) if rotary else source), key
    tower.load_state_dict(state, strict=True)


_POSITION_TABLE_SIDE = 32


def _reference_position_cells(grid_h: int, grid_w: int) -> torch.Tensor:
    # sglang test/registered/unit/multimodal/test_muse_spark_omni.py (#223): the cells the checkpoint's xy-layout
    # position table is read at, per patch in patch order.
    patch = torch.arange(grid_h * grid_w, dtype=torch.float64)
    row = torch.div(patch, grid_h, rounding_mode="floor") + 0.5
    col = patch % grid_h + 0.5
    return torch.stack((row * _POSITION_TABLE_SIDE / grid_w - 0.5, col * _POSITION_TABLE_SIDE / grid_h - 0.5), dim=-1)


@pytest.mark.parametrize("grid", [(24, 40), (40, 24), (46, 46), (58, 90)])
def test_position_grid_follows_the_checkpoint_xy_layout(grid):
    positions = patch_grid_positions(*grid, device=torch.device("cpu"))

    assert positions.dtype == torch.float32
    cells = ((positions.double() + 1) * _POSITION_TABLE_SIDE - 1) / 2
    torch.testing.assert_close(cells.flip(-1), _reference_position_cells(*grid), atol=1e-5, rtol=0)
