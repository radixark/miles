"""The Inkling towers read SGLang's token-space media fields under their own names."""

import torch

from miles_plugins.models.inkling.mm_towers import _tower_media_kwargs


def test_token_space_fields_map_to_the_tower_names_and_leave_the_rest():
    patches, positions, dmel = torch.zeros(2, 1), torch.tensor([3, 4]), torch.zeros(5, 80)
    kwargs = {
        "vision_patches_bthwc": patches,
        "mm_image_positions": positions,
        "dmel_bins": dmel,
        "mm_audio_positions": torch.tensor([7, 8]),
        "input_ids": torch.tensor([[1]]),
    }

    media = _tower_media_kwargs(kwargs)

    assert media == {"mm_vision_patches": patches, "mm_vision_positions": positions, "mm_audio_dmel": dmel}
    # mm_audio_positions already is the towers' name; the model kwargs stay for the text forward.
    assert set(kwargs) == {"mm_audio_positions", "input_ids"}


def test_legacy_fields_are_not_media_kwargs():
    kwargs = {"mm_vision_patches": torch.zeros(1), "mm_audio_dmel": torch.zeros(1)}
    assert _tower_media_kwargs(kwargs) == {}
    assert set(kwargs) == {"mm_vision_patches", "mm_audio_dmel"}
