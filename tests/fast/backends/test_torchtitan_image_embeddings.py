import io

import pytest
import torch
import torch.distributed as dist
from PIL import Image
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.torchtitan_utils.image_embeddings import (
    ImageInputs,
    image_features_for_local_tokens,
    image_token_counts,
    select_pixel_values,
    write_image_features,
)

_IMAGE_ID = 9
_MERGE = 2
_SEQ_LEN = 64
# grids (t, h, w) fill 4, 6, 2 and 8 image-token rows after the 2x2 merge
_GRIDS = torch.tensor([[1, 4, 4], [1, 4, 6], [1, 2, 4], [2, 4, 4]])
# image token spans in the sequence; with CP 4 the 8 headtail chunks are 8 tokens wide, so spans
# 6..9, 14..19 and 40..47 cross chunk boundaries
_SPANS = [(6, 10), (14, 20), (30, 32), (40, 48)]


def _sequence() -> torch.Tensor:
    tokens = torch.arange(_SEQ_LEN) % 7 + 20
    for start, end in _SPANS:
        tokens[start:end] = _IMAGE_ID
    return tokens


def _headtail_indices(seq_len: int, cp: int, rank: int) -> torch.Tensor:
    """torchtitan's headtail split: 2 * cp chunks, rank r holds chunk r and chunk 2 * cp - 1 - r."""
    chunks = torch.arange(seq_len).chunk(2 * cp)
    return torch.cat([chunks[rank], chunks[2 * cp - 1 - rank]])


def _row_features(tokens_per_image: torch.Tensor, calls: list):
    """An encoder whose feature for global row g is the vector [g, g]: the mapping shows in the values."""
    row_starts = (tokens_per_image.cumsum(0) - tokens_per_image).tolist()

    def encode(images: list[int]) -> torch.Tensor:
        calls.append(images)
        rows = [torch.arange(row_starts[i], row_starts[i] + int(tokens_per_image[i])) for i in images]
        return torch.cat(rows).float().unsqueeze(-1).expand(-1, 2)

    return encode


def _images_overlapping(indices: torch.Tensor, tokens: torch.Tensor) -> list[int]:
    held = set(indices.tolist())
    return [i for i, (start, end) in enumerate(_SPANS) if held & set(range(start, end))]


@pytest.mark.parametrize("cp", [1, 2, 4])
def test_each_rank_gets_the_global_feature_row_of_each_of_its_image_tokens(cp):
    tokens = _sequence()
    is_image = tokens == _IMAGE_ID
    global_rows = is_image.cumsum(0) - 1
    tokens_per_image = image_token_counts(_GRIDS, _MERGE)
    assert tokens_per_image.tolist() == [4, 6, 2, 8]

    for rank in range(cp):
        local_indices = _headtail_indices(_SEQ_LEN, cp, rank)
        calls = []
        positions, features = image_features_for_local_tokens(
            is_image,
            local_indices,
            tokens_per_image=tokens_per_image,
            encode=_row_features(tokens_per_image, calls),
        )
        sequence_positions = local_indices[positions]
        assert torch.equal(sequence_positions, local_indices[is_image[local_indices]])
        assert torch.equal(features[:, 0].long(), global_rows[sequence_positions])
        # only the screenshots whose token span overlaps this rank's chunks are encoded, once each
        assert calls == [_images_overlapping(local_indices, tokens)]


def test_without_cp_every_image_token_is_written_in_order():
    tokens = _sequence()
    tokens_per_image = image_token_counts(_GRIDS, _MERGE)
    calls = []
    positions, features = image_features_for_local_tokens(
        tokens == _IMAGE_ID, None, tokens_per_image=tokens_per_image, encode=_row_features(tokens_per_image, calls)
    )
    assert torch.equal(features[:, 0].long(), torch.arange(20))
    assert calls == [[0, 1, 2, 3]]
    assert positions.numel() == 20


def test_a_rank_without_image_tokens_runs_no_tower():
    tokens = torch.full((_SEQ_LEN,), 3)
    tokens[60:64] = _IMAGE_ID
    calls = []
    tokens_per_image = image_token_counts(_GRIDS[:1], _MERGE)
    positions, features = image_features_for_local_tokens(
        tokens == _IMAGE_ID,
        _headtail_indices(_SEQ_LEN, 4, 3),
        tokens_per_image=tokens_per_image,
        encode=_row_features(tokens_per_image, calls),
    )
    assert positions.numel() == 0 and features is None and calls == []


def test_a_token_count_that_disagrees_with_the_grids_is_refused():
    tokens = _sequence()
    tokens[6] = 0
    with pytest.raises(ValueError, match="19 image token.*fill 20 feature row"):
        image_features_for_local_tokens(
            tokens == _IMAGE_ID,
            None,
            tokens_per_image=image_token_counts(_GRIDS, _MERGE),
            encode=lambda images: torch.zeros(0, 2),
        )


def test_a_tower_that_returns_the_wrong_row_count_is_refused():
    with pytest.raises(ValueError, match="returned 3 feature row"):
        image_features_for_local_tokens(
            _sequence() == _IMAGE_ID,
            None,
            tokens_per_image=image_token_counts(_GRIDS, _MERGE),
            encode=lambda images: torch.zeros(3, 2),
        )


def test_pixel_tensors_are_sliced_per_selected_image():
    patches = _GRIDS.prod(dim=-1)
    pixel_values = torch.cat([torch.full((int(n), 3), float(i)) for i, n in enumerate(patches)])
    inputs = ImageInputs(grid_thw=_GRIDS, pixel_values=pixel_values)
    selected = select_pixel_values(inputs, [1, 3], image_processor_fn=lambda: pytest.fail("no PNGs here"))
    assert selected.shape[0] == patches[1] + patches[3]
    assert selected[: patches[1]].eq(1).all() and selected[patches[1] :].eq(3).all()


def _png(color: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 4), (color, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeImageProcessor:
    """Returns one patch row per image whose value is the image's red channel."""

    def __init__(self):
        self.decoded = []

    def __call__(self, images, return_tensors):
        red = images[0].getpixel((0, 0))[0]
        self.decoded.append(red)
        return {"pixel_values": torch.full((2, 3), float(red)), "image_grid_thw": torch.tensor([[1, 1, 2]])}


def _png_inputs(colors: list[int], grid=(1, 1, 2)) -> ImageInputs:
    pngs = [_png(c) for c in colors]
    return ImageInputs(
        grid_thw=torch.tensor([grid] * len(colors)),
        image_png=torch.frombuffer(bytearray(b"".join(pngs)), dtype=torch.uint8),
        image_png_lengths=torch.tensor([len(p) for p in pngs]),
    )


def test_pngs_are_decoded_only_for_the_selected_images():
    processor = _FakeImageProcessor()
    selected = select_pixel_values(_png_inputs([10, 20, 30]), [0, 2], image_processor_fn=lambda: processor)
    assert processor.decoded == [10, 30]
    assert selected[:, 0].tolist() == [10.0, 10.0, 30.0, 30.0]


def test_a_rebuilt_grid_that_differs_from_the_recorded_one_is_refused():
    with pytest.raises(ValueError, match="rebuilt image grid"):
        select_pixel_values(_png_inputs([10], grid=(1, 2, 2)), [0], image_processor_fn=_FakeImageProcessor)


def test_images_need_exactly_one_pixel_source():
    with pytest.raises(ValueError, match="exactly one"):
        ImageInputs(grid_thw=_GRIDS)


def _gloo_cp_worker(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        tokens = _sequence()
        local_indices = _headtail_indices(_SEQ_LEN, world_size, rank)
        restore_indices = torch.cat([_headtail_indices(_SEQ_LEN, world_size, r) for r in range(world_size)]).argsort()
        tokens_per_image = image_token_counts(_GRIDS, _MERGE)
        h_BLD = torch.full((1, local_indices.numel(), 2), -1.0, requires_grad=True)
        calls = []
        out = write_image_features(
            h_BLD,
            tokens[local_indices].unsqueeze(0),
            image_token_id=_IMAGE_ID,
            tokens_per_image=tokens_per_image,
            encode=_row_features(tokens_per_image, calls),
            cp_group=dist.group.WORLD,
            local_indices=local_indices,
            restore_indices=restore_indices,
        )
        is_image = (tokens == _IMAGE_ID)[local_indices]
        global_rows = ((tokens == _IMAGE_ID).cumsum(0) - 1)[local_indices]
        assert torch.equal(out[0, is_image, 0].long(), global_rows[is_image])
        assert out[0, ~is_image].eq(-1).all()
        assert calls == [_images_overlapping(local_indices, tokens)]
        # image rows take no gradient into the embedding; text rows keep theirs
        out.sum().backward()
        assert torch.equal(h_BLD.grad[0, :, 0], (~is_image).float())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_the_cp_gather_maps_rows_across_real_ranks(world_size):
    run_multiprocess(_gloo_cp_worker, world_size=world_size)
