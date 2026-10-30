"""Pixel tensors from PNG bytes, per micro-batch, on the training side.

The session server ships a sample's images as PNG bytes
(``multimodal_train_inputs = {"image_png": uint8 [sum of lengths],
"image_png_lengths": int64 [n], "image_grid_thw": int64 [n, 3]}``,
miles.rollout.session.media) instead of the processor's ``pixel_values``: a
1024x576 screenshot is about 170 KB as PNG and 5.4 MB as the fp32 patch
tensor, and a screenshot episode replays every earlier screenshot in every
turn, so the pixel form does not fit the rollout batch in host memory.

``get_batch`` (data/rollout.py) concatenates the dict per micro-batch and
then calls ``rebuild_png_images``: each PNG is decoded and run through the
checkpoint's own image processor, the grid is checked against the shipped
one, and the concatenated ``pixel_values`` replaces the PNG keys, so the
model forward receives what the processor path produced. The processor is
the one ``AutoProcessor`` gives for ``args.hf_checkpoint``, the class the
rollout side counted and gridded with; ``configure`` records the checkpoint
once per process, and the processor loads on first use.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

PNG_KEYS = ("image_png", "image_png_lengths")

_image_processor: Any = None
_hf_checkpoint: str | None = None


def configure(hf_checkpoint: str | None) -> None:
    """Record the checkpoint whose image processor rebuilds PNG images."""
    global _hf_checkpoint, _image_processor
    if hf_checkpoint != _hf_checkpoint:
        _hf_checkpoint, _image_processor = hf_checkpoint, None


def image_processor() -> Any:
    """The configured checkpoint's image processor, loaded on first use."""
    global _image_processor
    if _image_processor is None:
        if _hf_checkpoint is None:
            raise RuntimeError(
                "a batch carries image_png but no checkpoint is configured for the rebuild "
                "(png_images.configure runs when the rollout batch is fetched)"
            )
        _image_processor = load_image_processor(_hf_checkpoint)
    return _image_processor


def load_image_processor(hf_checkpoint: str) -> Any:
    """The checkpoint's image processor through ``load_processor``, which
    also knows the engine's processor for a model transformers cannot
    resolve (GLM-5.3 on a transformers without ``glm5_next``)."""
    from miles.utils.processing_utils import load_processor

    processor = load_processor(hf_checkpoint, trust_remote_code=True)
    found = getattr(processor, "image_processor", None)  # config-access-exempt: a text processor has none
    if found is None:
        raise RuntimeError(f"{hf_checkpoint}: AutoProcessor has no image_processor to rebuild images with")
    logger.info("png_images: rebuilding pixel_values with %s from %s", type(found).__name__, hf_checkpoint)
    return found


def split_pngs(image_png: torch.Tensor, image_png_lengths: torch.Tensor) -> list[bytes]:
    data = image_png.detach().to("cpu").contiguous().numpy().tobytes()
    out, offset = [], 0
    for length in image_png_lengths.detach().to("cpu").tolist():
        out.append(data[offset : offset + length])
        offset += length
    if offset != len(data):
        raise ValueError(f"image_png holds {len(data)} bytes but the lengths sum to {offset}")
    return out


def pixel_values_from_pngs(processor: Any, image_png, image_png_lengths, image_grid_thw) -> torch.Tensor:
    """Decode each PNG, run the image processor on it alone, check its grid
    against the shipped one, and concatenate the ``[patches, features]``
    tensors in image order."""
    from PIL import Image

    pngs = split_pngs(image_png, image_png_lengths)
    grids = image_grid_thw.detach().to("cpu")
    if grids.shape[0] != len(pngs):
        raise ValueError(f"{len(pngs)} PNG(s) but {int(grids.shape[0])} image grid row(s)")
    chunks = []
    for index, png in enumerate(pngs):
        with Image.open(io.BytesIO(png)) as decoded:
            image = decoded.convert("RGB")
        output = processor(images=[image], return_tensors="pt")
        grid = output["image_grid_thw"][0].to(torch.int64)
        if not torch.equal(grid, grids[index].to(torch.int64)):
            raise ValueError(
                f"image {index}: the rebuilt grid {grid.tolist()} differs from the shipped {grids[index].tolist()}; "
                "the rollout side and the trainer run different image processors"
            )
        chunks.append(output["pixel_values"])
    return torch.cat(chunks, dim=0)


def rebuild_png_images(multimodal_train_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The micro-batch's multimodal dict with ``pixel_values`` rebuilt from
    its PNG keys (removed), on the device of ``image_grid_thw``; a dict
    without PNG keys is returned as is."""
    if "image_png" not in multimodal_train_inputs:
        return multimodal_train_inputs
    if "pixel_values" in multimodal_train_inputs:
        raise ValueError("a batch carries both pixel_values and image_png")
    if "image_png_lengths" not in multimodal_train_inputs or "image_grid_thw" not in multimodal_train_inputs:
        raise ValueError("image_png needs image_png_lengths and image_grid_thw")
    grid = multimodal_train_inputs["image_grid_thw"]
    pixel_values = pixel_values_from_pngs(
        image_processor(),
        multimodal_train_inputs["image_png"],
        multimodal_train_inputs["image_png_lengths"],
        grid,
    )
    out = {k: v for k, v in multimodal_train_inputs.items() if k not in PNG_KEYS}
    out["pixel_values"] = pixel_values.to(device=grid.device)
    return out
