"""Step5 vision tower: token counts, feature order, the 2D RoPE and the strict checkpoint loader."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from miles_plugins.models.step5.vision import (
    Step5VisionConfig,
    Step5VisionEncoder,
    feature_order,
    load_step5_vision_encoder,
    rope_angles,
    rotate_half,
)

# 16-pixel overviews and 8-pixel crops: 8x8 and 4x4 patch grids, so 4 and 1 tokens after the two
# stride-2 convolutions.
_TINY = Step5VisionConfig(width=32, layers=2, heads=4, patch_size=2, image_size=16, mlp_width=48, text_hidden_size=24)
_STEP5 = Step5VisionConfig(
    width=1536, layers=47, heads=16, patch_size=14, image_size=728, mlp_width=8960, text_hidden_size=4096
)
_CPU = torch.device("cpu")


def _tiny_encoder() -> Step5VisionEncoder:
    torch.manual_seed(0)
    encoder = Step5VisionEncoder(_TINY)
    for param in encoder.parameters():
        torch.nn.init.normal_(param, std=0.1)
    return encoder.requires_grad_(False).eval()


def test_step5_overviews_give_169_tokens_and_crops_81():
    with torch.device("meta"):
        encoder = Step5VisionEncoder(_STEP5)
        features = encoder.features(
            torch.empty(2, 3, 728, 728), torch.empty(3, 3, 504, 504), torch.tensor([1, 2], device=_CPU)
        )
    assert features.shape == (2 * 169 + 3 * 81, 4096)


def test_features_put_each_images_crops_before_its_overview():
    encoder = _tiny_encoder()
    overviews, crops = torch.randn(3, 3, 16, 16), torch.randn(3, 3, 8, 8)

    features = encoder.features(overviews, crops, torch.tensor([2, 0, 1]))

    with torch.no_grad():
        overview, crop = encoder._encode(overviews), encoder._encode(crops)
    assert overview.shape == (3, 4, 24) and crop.shape == (3, 1, 24)
    # Image 0 has crops 0 and 1, image 1 none, image 2 crop 2.
    expected = torch.cat([crop[0], crop[1], overview[0], overview[1], crop[2], overview[2]])
    torch.testing.assert_close(features, expected, rtol=0, atol=0)


def test_images_without_crops_need_no_crop_pixels():
    encoder = _tiny_encoder()
    overviews = torch.randn(2, 3, 16, 16)

    features = encoder.features(overviews, torch.empty(0, 3, 8, 8), torch.tensor([0, 0]))

    with torch.no_grad():
        torch.testing.assert_close(features, encoder._encode(overviews).flatten(0, 1), rtol=0, atol=0)


def test_crop_counts_must_match_the_pixels():
    with pytest.raises(AssertionError, match="num_patches"):
        _tiny_encoder().features(torch.randn(1, 3, 16, 16), torch.randn(2, 3, 8, 8), torch.tensor([1]))


def test_feature_order_takes_crop_rows_from_after_all_overviews():
    # Two images of 3 overview tokens; image 0 has one crop of 2 tokens, image 1 two crops.
    assert feature_order([1, 2], overview_tokens=3, crop_tokens=2) == [6, 7, 0, 1, 2, 8, 9, 10, 11, 3, 4, 5]
    assert feature_order([], overview_tokens=3, crop_tokens=2) == []


def test_rope_turns_the_first_half_by_column_and_the_second_by_row():
    angles = rope_angles(2, 3, 8, device=_CPU)

    inv_freq = torch.tensor([1.0, 0.01])  # a 4-wide 1D RoPE with theta 1e4
    for row in range(2):
        for col in range(3):
            expected = torch.cat([col * inv_freq, row * inv_freq]).repeat_interleave(2)
            torch.testing.assert_close(angles[row * 3 + col], expected)


def test_a_smaller_grid_has_the_native_tables_angles():
    """SGLang looks a smaller grid up in its native table; building it directly gives the same numbers."""
    native, small = rope_angles(8, 8, 8, device=_CPU), rope_angles(3, 5, 8, device=_CPU)
    rows, cols = torch.meshgrid(torch.arange(3), torch.arange(5), indexing="ij")
    torch.testing.assert_close(small, native[(rows * 8 + cols).flatten()], rtol=0, atol=0)


def test_rotate_half_turns_adjacent_pairs():
    assert rotate_half(torch.tensor([1.0, 2.0, 3.0, 4.0])).tolist() == [-2.0, 1.0, -4.0, 3.0]


def test_position_embedding_is_itself_at_the_native_grid():
    encoder = _tiny_encoder()
    native = encoder.vision_model._position_embedding(8, 8)
    torch.testing.assert_close(native[0], encoder.vision_model.positional_embedding, rtol=0, atol=0)


def _write_checkpoint(path, tensors: dict[str, torch.Tensor]) -> None:
    save_file(tensors, path / "model-00001.safetensors")
    weight_map = {name: "model-00001.safetensors" for name in tensors}
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def _tower_tensors() -> dict[str, torch.Tensor]:
    return {name: tensor.bfloat16().contiguous() for name, tensor in _tiny_encoder().state_dict().items()}


def test_loader_takes_every_tower_tensor_and_freezes_the_tower(tmp_path):
    tower = _tower_tensors()
    _write_checkpoint(tmp_path, tower | {"model.embed_tokens.weight": torch.zeros(4, 24, dtype=torch.bfloat16)})

    encoder = load_step5_vision_encoder(str(tmp_path), _TINY, device=_CPU)

    assert encoder.state_dict().keys() == tower.keys()
    for name, param in encoder.named_parameters():
        torch.testing.assert_close(param, tower[name], rtol=0, atol=0)
        assert not param.requires_grad
    assert not encoder.training


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda tower: tower.pop("vision_model.ln_pre.bias"), "Missing key"),
        (lambda tower: tower.update({"vision_model.ln_post.weight": torch.ones(32).bfloat16()}), "Unexpected key"),
    ],
)
def test_loader_matches_tower_names_strictly(tmp_path, edit, match):
    tower = _tower_tensors()
    edit(tower)
    _write_checkpoint(tmp_path, tower)
    with pytest.raises(RuntimeError, match=match):
        load_step5_vision_encoder(str(tmp_path), _TINY, device=_CPU)


def test_loader_rejects_weights_that_are_not_bf16(tmp_path):
    tower = _tower_tensors()
    tower["vit_large_projector.weight"] = tower["vit_large_projector.weight"].float()
    _write_checkpoint(tmp_path, tower)
    with pytest.raises(ValueError, match="vit_large_projector.weight"):
        load_step5_vision_encoder(str(tmp_path), _TINY, device=_CPU)


def _hf_config(**vision_overrides) -> SimpleNamespace:
    vision = dict(
        width=1536,
        layers=47,
        heads=16,
        patch_size=14,
        image_size=728,
        mlp_ratio=8960 / 1536,
        hidden_act="quick_gelu",
        use_rope2d=True,
    )
    return SimpleNamespace(
        vision_config=SimpleNamespace(**(vision | vision_overrides)), text_config=SimpleNamespace(hidden_size=4096)
    )


def test_config_reads_the_checkpoints_dimensions():
    assert Step5VisionConfig.from_hf(_hf_config()) == _STEP5


def test_config_rejects_another_activation():
    with pytest.raises(ValueError, match="hidden_act='gelu'"):
        Step5VisionConfig.from_hf(_hf_config(hidden_act="gelu"))
