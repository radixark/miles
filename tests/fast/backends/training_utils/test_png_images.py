"""PNG images rebuilt into pixel_values per micro-batch (data/png_images.py)."""

from __future__ import annotations

import io

import pytest
import torch
from PIL import Image

from miles.backends.training_utils.data import png_images


class FakeImageProcessor:
    """Grids like a 16 px patch processor; pixel rows carry the image's mean color."""

    def __init__(self):
        self.calls = 0

    def __call__(self, images, return_tensors="pt"):
        self.calls += 1
        (image,) = images
        grid = torch.tensor([[1, image.height // 16, image.width // 16]], dtype=torch.int64)
        patches = int(grid[0].prod())
        mean = torch.tensor(image.resize((1, 1)).getpixel((0, 0)), dtype=torch.float32) / 255
        return {"pixel_values": mean.repeat(patches, 1), "image_grid_thw": grid}


def _png(width, height, color):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _packed(pngs):
    data = b"".join(pngs)
    return (
        torch.frombuffer(bytearray(data), dtype=torch.uint8),
        torch.tensor([len(p) for p in pngs], dtype=torch.int64),
    )


@pytest.fixture
def processor(monkeypatch):
    fake = FakeImageProcessor()
    monkeypatch.setattr(png_images, "_hf_checkpoint", "fake/checkpoint")
    monkeypatch.setattr(png_images, "_image_processor", fake)
    return fake


def test_two_pngs_become_their_pixel_rows_in_order_and_the_png_keys_go(processor):
    pngs = [_png(64, 64, (255, 0, 0)), _png(128, 64, (0, 0, 255))]
    image_png, lengths = _packed(pngs)
    mm = {"image_png": image_png, "image_png_lengths": lengths, "image_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 8]])}
    out = png_images.rebuild_png_images(mm)
    assert set(out) == {"pixel_values", "image_grid_thw"}
    assert out["pixel_values"].shape == (16 + 32, 3)
    assert torch.allclose(out["pixel_values"][:16], torch.tensor([1.0, 0.0, 0.0]).repeat(16, 1))
    assert torch.allclose(out["pixel_values"][16:], torch.tensor([0.0, 0.0, 1.0]).repeat(32, 1))
    assert processor.calls == 2, "one processor call per image"
    assert mm["image_grid_thw"] is out["image_grid_thw"]


def test_a_dict_without_pngs_is_untouched_and_both_forms_at_once_are_refused(processor):
    plain = {"pixel_values": torch.zeros(2, 3), "image_grid_thw": torch.tensor([[1, 1, 2]])}
    assert png_images.rebuild_png_images(plain) is plain
    image_png, lengths = _packed([_png(16, 16, (0, 0, 0))])
    with pytest.raises(ValueError, match="both pixel_values and image_png"):
        png_images.rebuild_png_images({**plain, "image_png": image_png, "image_png_lengths": lengths})
    with pytest.raises(ValueError, match="needs image_png_lengths"):
        png_images.rebuild_png_images({"image_png": image_png, "image_grid_thw": torch.tensor([[1, 1, 1]])})


def test_a_grid_that_differs_from_the_shipped_one_is_refused(processor):
    image_png, lengths = _packed([_png(64, 64, (0, 0, 0))])
    with pytest.raises(ValueError, match="differs from the shipped"):
        png_images.rebuild_png_images(
            {"image_png": image_png, "image_png_lengths": lengths, "image_grid_thw": torch.tensor([[1, 2, 2]])}
        )


def test_lengths_that_do_not_cover_the_bytes_are_refused(processor):
    image_png, lengths = _packed([_png(16, 16, (0, 0, 0))])
    with pytest.raises(ValueError, match="lengths sum"):
        png_images.split_pngs(image_png, lengths - 1)


def test_without_a_configured_checkpoint_the_rebuild_says_so(monkeypatch):
    monkeypatch.setattr(png_images, "_hf_checkpoint", None)
    monkeypatch.setattr(png_images, "_image_processor", None)
    image_png, lengths = _packed([_png(16, 16, (0, 0, 0))])
    with pytest.raises(RuntimeError, match="no checkpoint is configured"):
        png_images.rebuild_png_images(
            {"image_png": image_png, "image_png_lengths": lengths, "image_grid_thw": torch.tensor([[1, 1, 1]])}
        )


def test_configure_resets_the_processor_when_the_checkpoint_changes(monkeypatch):
    monkeypatch.setattr(png_images, "_hf_checkpoint", "a")
    monkeypatch.setattr(png_images, "_image_processor", object())
    png_images.configure("a")
    assert png_images._image_processor is not None
    png_images.configure("b")
    assert png_images._image_processor is None and png_images._hf_checkpoint == "b"


def test_the_processor_comes_from_load_processor(monkeypatch):
    """load_processor knows the engine's processor for a model transformers cannot resolve."""
    from miles.utils import processing_utils

    class Proc:
        image_processor = object()

    monkeypatch.setattr(processing_utils, "load_processor", lambda path, **kw: Proc())
    assert png_images.load_image_processor("/any") is Proc.image_processor
    monkeypatch.setattr(processing_utils, "load_processor", lambda path, **kw: None)
    with pytest.raises(RuntimeError, match="no image_processor"):
        png_images.load_image_processor("/any")
