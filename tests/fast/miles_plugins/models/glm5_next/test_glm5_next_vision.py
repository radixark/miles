import pytest
import torch
from transformers.models.glm_ocr.configuration_glm_ocr import GlmOcrVisionConfig

from miles_plugins.models.glm5_next.vision import Glm5NextVisionModel


def _tiny_tower() -> Glm5NextVisionModel:
    config = GlmOcrVisionConfig(
        depth=2,
        hidden_size=32,
        out_hidden_size=24,
        intermediate_size=48,
        num_heads=4,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        in_channels=3,
    )
    config.swiglu_limit = 7.0
    config.projection_intermediate_size = 40
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Glm5NextVisionModel(config).eval()


def test_encoding_a_subset_of_images_gives_their_rows_of_encoding_all():
    """Each CP rank encodes only its own screenshots; that is exact only if images do not mix in the tower."""
    tower = _tiny_tower()
    grid = torch.tensor([[1, 4, 6], [1, 2, 4], [2, 4, 4]])
    patches = grid.prod(dim=-1)
    pixel_values = torch.randn(int(patches.sum()), 3 * 2 * 14 * 14)
    with torch.no_grad():
        everything = tower(pixel_values, grid_thw=grid).pooler_output
        subset = tower(
            torch.cat([pixel_values[: patches[0]], pixel_values[patches[0] + patches[1] :]]), grid_thw=grid[[0, 2]]
        ).pooler_output
    rows = patches // 4
    assert everything.shape == (int(rows.sum()), 24)
    expected = torch.cat([everything[: rows[0]], everything[rows[0] + rows[1] :]])
    torch.testing.assert_close(subset, expected)


def test_png_rebuild_falls_back_to_the_engine_processor(monkeypatch):
    """AutoProcessor gives a bare tokenizer for GLM-5.3 on transformers 5.12.1;
    the encoder then uses SGLang's Glm5NextProcessor instead of failing."""
    import sys
    import types

    import importlib.util
    from pathlib import Path

    # The module file alone: the package __init__ imports the fla kernels.
    path = Path(__file__).resolve().parents[5] / "miles/backends/torchtitan_utils/models/glm5_next/vision.py"
    spec = importlib.util.spec_from_file_location("titan_glm5_next_vision", path)
    titan_vision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(titan_vision)

    class FakeProcessor:
        image_processor = object()

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs.get("trust_remote_code") is True
            return cls()

    module = types.ModuleType("sglang.srt.configs.glm5_next_processing")
    module.Glm5NextProcessor = FakeProcessor
    monkeypatch.setitem(sys.modules, "sglang.srt.configs.glm5_next_processing", module)
    assert callable(getattr(titan_vision.Glm5NextImageEncoder, "encode", None))
    monkeypatch.setattr(titan_vision, "load_processor", lambda *args, **kwargs: None)
    encoder = titan_vision.Glm5NextImageEncoder.__new__(titan_vision.Glm5NextImageEncoder)
    encoder._hf_checkpoint = "/ckpt"
    assert encoder.image_processor is FakeProcessor.image_processor

    monkeypatch.setitem(sys.modules, "sglang.srt.configs.glm5_next_processing", None)
    encoder = titan_vision.Glm5NextImageEncoder.__new__(titan_vision.Glm5NextImageEncoder)
    encoder._hf_checkpoint = "/ckpt"
    with pytest.raises(RuntimeError, match="has no image processor"):
        _ = encoder.image_processor
