"""Vision features for the image-token rows of one packed sequence, under context parallelism.

A microbatch reaches every CP rank with the images of the whole sequence: either ``pixel_values``
(``[patches, features]``, the images back to back) or the PNG bytes of each image (``image_png``, a
uint8 tensor of the PNGs back to back, and ``image_png_lengths``), plus ``image_grid_thw``
(``[images, 3]``). Image ``i`` fills ``t * h * w / merge**2`` image-token rows, in sequence order.
A rank holds a contiguous ``1 / cp`` slice of the sequence, so it finds the global feature row of each
of its image tokens from the global image-token mask and encodes only the images those rows fall in.

No torchtitan imports, so the CPU tests run without it.
"""

import io
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from PIL import Image


@dataclass(frozen=True)
class ImageInputs:
    """One microbatch's images, as the batch carries them. Exactly one of ``pixel_values`` and
    ``image_png`` is set."""

    grid_thw: torch.Tensor
    pixel_values: torch.Tensor | None = None
    image_png: torch.Tensor | None = None
    image_png_lengths: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (self.pixel_values is None) == (self.image_png is None):
            raise ValueError("images need exactly one of pixel_values and image_png")
        if self.image_png is not None and self.image_png_lengths is None:
            raise ValueError("image_png needs image_png_lengths")

    @property
    def num_images(self) -> int:
        return self.grid_thw.shape[0]


def image_token_counts(grid_thw: torch.Tensor, merge_size: int) -> torch.Tensor:
    """Image-token rows each image fills: its patches over the spatial merge."""
    return grid_thw.prod(dim=-1) // (merge_size * merge_size)


def local_image_rows(global_is_image: torch.Tensor, local_indices: torch.Tensor | None):
    """``(positions, rows)``: the local positions of this rank's image tokens, and the global feature
    row each one takes. ``local_indices`` maps local positions to sequence positions (None without CP)."""
    global_rows = global_is_image.cumsum(0) - 1
    if local_indices is None:
        positions = global_is_image.nonzero(as_tuple=True)[0]
        return positions, global_rows[positions]
    local_is_image = global_is_image.index_select(0, local_indices)
    positions = local_is_image.nonzero(as_tuple=True)[0]
    return positions, global_rows.index_select(0, local_indices.index_select(0, positions))


def image_features_for_rows(
    rows: torch.Tensor,
    tokens_per_image: torch.Tensor,
    encode: Callable[[list[int]], torch.Tensor],
) -> torch.Tensor:
    """The feature row for each global row in ``rows``, encoding only the images the rows fall in.
    ``encode(images)`` returns the features of those images, back to back, in the order given."""
    row_ends = tokens_per_image.cumsum(0)
    row_starts = row_ends - tokens_per_image
    image_of_row = torch.bucketize(rows, row_ends, right=True)
    images = image_of_row.unique(sorted=True)
    selected_counts = tokens_per_image.index_select(0, images)
    selected_starts = selected_counts.cumsum(0) - selected_counts

    features = encode(images.tolist())
    if features.shape[0] != int(selected_counts.sum()):
        raise ValueError(
            f"the vision tower returned {features.shape[0]} feature row(s) for images {images.tolist()}, "
            f"which take {int(selected_counts.sum())} image-token row(s)"
        )
    slot = torch.searchsorted(images, image_of_row)
    index = selected_starts.index_select(0, slot) + rows - row_starts.index_select(0, image_of_row)
    return features.index_select(0, index.to(features.device))


def image_features_for_local_tokens(
    global_is_image: torch.Tensor,
    local_indices: torch.Tensor | None,
    *,
    tokens_per_image: torch.Tensor,
    encode: Callable[[list[int]], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(positions, features)`` for this rank's image tokens; features is None when it holds none."""
    num_tokens, num_rows = int(global_is_image.sum()), int(tokens_per_image.sum())
    if num_tokens != num_rows:
        raise ValueError(
            f"image token/feature mismatch: the sequence has {num_tokens} image token(s) but its "
            f"{tokens_per_image.numel()} image(s) fill {num_rows} feature row(s)"
        )
    positions, rows = local_image_rows(global_is_image, local_indices)
    if positions.numel() == 0:
        return positions, None
    tokens_per_image = tokens_per_image.to(rows.device)
    return positions, image_features_for_rows(rows, tokens_per_image, encode)


def write_image_features(
    h_TD: torch.Tensor,
    tokens: torch.Tensor,
    *,
    image_token_id: int,
    tokens_per_image: torch.Tensor,
    encode: Callable[[list[int]], torch.Tensor],
    cp_group: dist.ProcessGroup | None = None,
    local_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """The embedding output ``[T, D]`` with this rank's image-token rows replaced by vision features.
    Under CP, ``local_indices`` are the sequence positions of this rank's tokens; the ranks hold
    contiguous slices in rank order, so the gathered image-token mask is already in sequence order."""
    with torch.no_grad():
        is_image = tokens.flatten() == image_token_id
        global_is_image = is_image
        if cp_group is not None:
            local = is_image.to(torch.uint8)
            gathered = local.new_empty(dist.get_world_size(cp_group) * local.shape[0])
            dist.all_gather_into_tensor(gathered, local, group=cp_group)
            global_is_image = gathered.bool()
        positions, features = image_features_for_local_tokens(
            global_is_image, local_indices, tokens_per_image=tokens_per_image, encode=encode
        )
    if features is None:
        return h_TD
    return h_TD.index_put((positions,), features.to(h_TD.dtype))


def split_pngs(image_png: torch.Tensor, image_png_lengths: torch.Tensor) -> list[bytes]:
    data = image_png.detach().cpu().contiguous().numpy().tobytes()
    lengths = image_png_lengths.detach().cpu().tolist()
    if sum(lengths) != len(data):
        raise ValueError(f"image_png holds {len(data)} bytes but image_png_lengths sum to {sum(lengths)}")
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + int(length))
    return [data[start:end] for start, end in zip(offsets[:-1], offsets[1:], strict=True)]


def pixel_values_from_png(image_processor, png: bytes, grid_thw: torch.Tensor) -> torch.Tensor:
    """One image's ``[patches, features]`` pixels, rebuilt by the processor the rollout ran. A grid that
    differs from the recorded one means the trainer and the rollout run different processors."""
    with Image.open(io.BytesIO(png)) as decoded:
        image = decoded.convert("RGB")
    output = image_processor(images=[image], return_tensors="pt")
    grid = output["image_grid_thw"][0].to(torch.int64)
    if not torch.equal(grid, grid_thw.cpu().to(torch.int64)):
        raise ValueError(f"the rebuilt image grid {grid.tolist()} differs from the recorded {grid_thw.tolist()}")
    return output["pixel_values"]


def select_pixel_values(
    inputs: ImageInputs, images: list[int], image_processor_fn: Callable[[], object]
) -> torch.Tensor:
    """The pixels of ``images``, back to back. PNGs are decoded only for the selected images."""
    if inputs.pixel_values is not None:
        patch_ends = inputs.grid_thw.prod(dim=-1).cumsum(0).tolist()
        patch_starts = [0, *patch_ends[:-1]]
        return torch.cat([inputs.pixel_values[patch_starts[i] : patch_ends[i]] for i in images], dim=0)
    pngs = split_pngs(inputs.image_png, inputs.image_png_lengths)
    if len(pngs) != inputs.num_images:
        raise ValueError(f"{len(pngs)} PNG(s) but {inputs.num_images} image grid row(s)")
    image_processor = image_processor_fn()
    return torch.cat([pixel_values_from_png(image_processor, pngs[i], inputs.grid_thw[i]) for i in images], dim=0)
