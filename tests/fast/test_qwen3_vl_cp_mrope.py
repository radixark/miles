"""CPU unit tests for the Qwen3-VL THD packed mRoPE fix.

CP reconstruction (issue #1296): under context parallelism each rank's THD row holds only its
zigzag chunks of every packed segment. `_reassemble_full_row` de-interleaves the all-gathered
per-rank rows back to the full natural-order row so per-segment MRoPE positions can be rebuilt
and re-sliced. These tests check that reconstruction is the exact inverse of `slice_with_cp`
(the function miles uses to shard the tokens), and that re-slicing with
`_natural_to_zigzag_slice` round-trips.

Video grid rows: a video expands to grid_t frames, each <vision_start><video_pad>...<vision_end>,
but has one `video_grid_thw` row, so each packed segment must get the rows of its own videos:

    segment 0: [S V V][S V V][S V]  video A (grid_t=2) + video C (grid_t=1)  ->  rows [A, C]
    segment 1: [S V]                video B (grid_t=1)                       ->  rows [B]
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from miles_plugins.models.qwen3_vl import (
    _natural_to_zigzag_slice,
    _num_videos_in_segment,
    _reassemble_full_row,
    _segment_positions,
)


def _slice_with_cp(tokens, cp_size, cp_rank, pad_value=0):
    """Reference copy of cp_utils.slice_with_cp's THD zigzag slicing (per sample)."""
    token_len = len(tokens)
    chunk = (token_len + 2 * cp_size - 1) // (2 * cp_size)
    pad = 2 * cp_size * chunk - token_len
    if pad:
        tokens = F.pad(tokens, (0, pad), value=pad_value)
    s1, e1 = chunk * cp_rank, chunk * (cp_rank + 1)
    s2, e2 = chunk * (2 * cp_size - cp_rank - 1), chunk * (2 * cp_size - cp_rank)
    return torch.cat([tokens[s1:e1], tokens[s2:e2]])


def _build_like_get_batch(sample_lens, cp_size, pad_size=8):
    """Mimic miles get_batch THD+CP packing: per-sample zigzag slice, concat, pad, cu*cp."""
    samples = []
    base = 1
    for L in sample_lens:
        samples.append(torch.arange(base, base + L))  # unique nonzero ids
        base += L
    per_rank = []
    for r in range(cp_size):
        row = torch.cat([_slice_with_cp(t, cp_size, r) for t in samples])
        per_rank.append(row)
    cu = [0]
    for t in samples:
        cu.append(cu[-1] + _slice_with_cp(t, cp_size, 0).size(0))
    final_pad = (pad_size - per_rank[0].size(0) % pad_size) % pad_size
    if final_pad:
        per_rank = [F.pad(row, (0, final_pad), value=0) for row in per_rank]
        cu.append(cu[-1] + final_pad)
    cu = [x * cp_size for x in cu]
    return samples, per_rank, cu


@pytest.mark.parametrize(
    "cp_size,sample_lens",
    [(2, [10, 7, 13]), (2, [16, 16]), (4, [20, 9, 30, 5]), (2, [3]), (4, [40, 17])],
)
def test_reassemble_is_inverse_of_slice_with_cp(cp_size, sample_lens):
    samples, per_rank, cu = _build_like_get_batch(sample_lens, cp_size)
    local_len = per_rank[0].size(0)
    assert cu[-1] == cp_size * local_len

    full = _reassemble_full_row(per_rank, cu, cp_size)
    assert full is not None and full.numel() == cu[-1]

    # Each real sample's tokens reappear (in order) at the start of its segment.
    for i, t in enumerate(samples):
        seg = full[cu[i] : cu[i + 1]]
        assert torch.equal(seg[: t.numel()], t)

    # Re-slicing the full row per segment recovers exactly each rank's local chunks.
    for r in range(cp_size):
        recon = []
        for i in range(len(cu) - 1):
            recon.append(_natural_to_zigzag_slice(full[cu[i] : cu[i + 1]], cp_size, r, dim=0))
        assert torch.equal(torch.cat(recon), per_rank[r])


def test_reassemble_bails_on_indivisible_segment():
    # A segment length not divisible by 2*cp -> None (caller falls back to dense path).
    cu = [0, 6]  # 6 not divisible by 2*cp=4
    gathered = [torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
    assert _reassemble_full_row(gathered, cu, cp_size=2) is None


VISION_START, IMAGE_PAD, VIDEO_PAD, TEXT = 7, 8, 9, 0


def test_packed_videos_get_their_own_grid_rows():
    """Counting frames as videos would hand video B's row to segment 0 and none to segment 1."""
    video_a = [VISION_START, VIDEO_PAD, VIDEO_PAD, VISION_START, VIDEO_PAD, VIDEO_PAD]
    video_c = [VISION_START, VIDEO_PAD]
    segment_a = [TEXT, *video_a, TEXT, *video_c, TEXT]
    segment_b = [TEXT, VISION_START, VIDEO_PAD, TEXT]
    flat = torch.tensor(segment_a + segment_b)
    cu = [0, len(segment_a), len(segment_a) + len(segment_b)]
    video_grid_thw = torch.tensor([[2, 2, 4], [1, 2, 2], [1, 2, 2]])
    model = SimpleNamespace(
        config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=IMAGE_PAD,
        video_token_id=VIDEO_PAD,
        vision_start_token_id=VISION_START,
    )
    grids_per_segment = []

    def recording_get_rope_index(merge, image_id, video_id, start_id, input_ids, video_grid_thw=None, **_):
        grids_per_segment.append(video_grid_thw.tolist())
        return torch.zeros(3, 1, input_ids.shape[1], dtype=torch.long), None

    _segment_positions(model, flat, cu, torch.tensor(cu), {"video_grid_thw": video_grid_thw}, recording_get_rope_index)

    assert grids_per_segment == [[[2, 2, 4], [1, 2, 2]], [[1, 2, 2]]]


@pytest.mark.parametrize("first_video, num_frames", [(0, 4), (1, 7)])
def test_segment_off_a_video_boundary_is_rejected(first_video, num_frames):
    """A segment must hold whole videos; a partial one would read another video's grid."""
    with pytest.raises(ValueError):
        _num_videos_in_segment([3, 2, 4], first_video=first_video, num_frames=num_frames)
