import pytest
import torch

from miles.utils.sampling_mask import RolloutSamplingMask


def test_rollout_sampling_mask_builds_private_int32_storage():
    mask = RolloutSamplingMask.from_mask_list([[1, 3], [4, 2], [5, 6, 7]])

    ids, lengths = mask._select_masks(range(3))

    assert len(mask) == 3
    assert ids.dtype == torch.int32
    assert lengths.dtype == torch.long
    assert ids.tolist() == [1, 3, 4, 2, 5, 6, 7]
    assert lengths.tolist() == [2, 2, 3]
    assert not hasattr(mask, "ids")
    assert not hasattr(mask, "offsets")


def test_rollout_sampling_mask_requires_non_empty_mask():
    with pytest.raises(ValueError, match="every response token needs a non-empty sampling mask"):
        RolloutSamplingMask.from_mask_list([[], [1]])


def test_rollout_sampling_mask_owns_input_storage():
    ids = torch.tensor([5, 7], dtype=torch.int16)
    offsets = torch.tensor([0, 1, 2], dtype=torch.int32)
    mask = RolloutSamplingMask(ids=ids, offsets=offsets)

    ids[0] = 99
    offsets[1] = 99
    selected_ids, lengths = mask._select_masks(range(2))

    assert selected_ids.tolist() == [5, 7]
    assert lengths.tolist() == [1, 1]


def test_rollout_sampling_mask_validates_csr_offsets():
    with pytest.raises(ValueError, match="offsets must start at zero and end at the flattened id count"):
        RolloutSamplingMask(ids=torch.tensor([0, 1]), offsets=torch.tensor([0, 1]))


@pytest.mark.parametrize("ids", [torch.tensor([1.5]), torch.tensor([True]), torch.tensor([[1]])])
def test_rollout_sampling_mask_rejects_non_integer_or_non_1d_tensor_ids(ids):
    with pytest.raises(ValueError, match="must be one-dimensional integers"):
        RolloutSamplingMask(ids=ids, offsets=[0, 1])


def test_select_masks_slices_a_contiguous_range_and_reports_lengths():
    mask = RolloutSamplingMask.from_mask_list([[1, 3], [4, 2], [5, 6, 7], [8]])

    ids, lengths = mask._select_masks(range(1, 3))

    assert ids.tolist() == [4, 2, 5, 6, 7]
    assert lengths.tolist() == [2, 3]


def test_select_masks_concatenates_disjoint_runs_in_order():
    mask = RolloutSamplingMask.from_mask_list([[1, 3], [4, 2], [5, 6, 7], [8]])

    ids, lengths = mask._select_masks(torch.tensor([0, 1, 3]))

    assert ids.tolist() == [1, 3, 4, 2, 8]
    assert lengths.tolist() == [2, 2, 1]


@pytest.mark.parametrize("token_indices", [range(0), torch.empty(0, dtype=torch.long)])
def test_select_masks_handles_no_tokens(token_indices):
    mask = RolloutSamplingMask.from_mask_list([[1]])

    ids, lengths = mask._select_masks(token_indices)

    assert ids.numel() == 0
    assert lengths.numel() == 0


@pytest.mark.parametrize("token_indices", [range(1, 2), [1]])
def test_select_masks_rejects_out_of_range_position(token_indices):
    mask = RolloutSamplingMask.from_mask_list([[0]])

    with pytest.raises(ValueError, match=r"response indices must be in \[0, 1\)"):
        mask._select_masks(token_indices)
