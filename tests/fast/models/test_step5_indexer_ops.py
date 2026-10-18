import torch

from miles_plugins.models.step5.indexer_ops import top_regions
from miles_plugins.models.step5.region_selection import TOP_REGIONS


def test_top_regions_matches_deepselect_permuted_ties_after_initial_window():
    """DeepSelect visits the tail before its permuted prefix once there are >8192 candidates."""
    scores = torch.zeros(2, 1, 8193)
    candidates = torch.tensor([8192, 8193])

    got = top_regions(scores, candidates)

    assert got[0, 0].tolist() == list(range(TOP_REGIONS))

    perm_len = 16  # ceil(8193 / 512), with the 8-segment tail removed.
    first_segment = 0x22262226 % perm_len
    expected = list(range(first_segment * 512, first_segment * 512 + TOP_REGIONS - 1)) + [8192]
    assert got[1, 0].tolist() == sorted(expected)
