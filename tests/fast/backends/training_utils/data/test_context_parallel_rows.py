import torch

from miles.backends.training_utils.data.context_parallel import natural_to_local_cp_rows, scatter_cp_rows


def test_rows_on_another_context_parallel_rank_are_skipped():
    """natural_to_local_cp_rows marks a row held elsewhere with -1; indexing with it would overwrite the last token."""
    decoder_input = torch.zeros(4, 1, 2)
    rows = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    merged = scatter_cp_rows(decoder_input, rows, torch.tensor([-1, 2, -1]), sp_rank=0)

    assert merged[:, 0, 0].tolist() == [0.0, 0.0, 2.0, 0.0]
    assert decoder_input.abs().sum() == 0


def test_only_the_rows_in_this_sequence_parallel_window_are_written():
    """Rank 1 of a sequence-parallel pair holds packed tokens 4..7; the other rows belong to rank 0."""
    decoder_input = torch.zeros(4, 1, 2)
    rows = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    merged = scatter_cp_rows(decoder_input, rows, torch.tensor([3, 4, 6]), sp_rank=1)

    assert merged[:, 0, 0].tolist() == [2.0, 0.0, 3.0, 0.0]


def test_the_producer_and_the_consumer_agree_on_the_sentinel():
    """A token on the other CP rank comes out of natural_to_local_cp_rows as -1 and never reaches decoder_input."""
    positions = natural_to_local_cp_rows(torch.arange(8), sample_start=0, sample_end=8, cp_size=2, cp_rank=0)
    rows = torch.arange(8, dtype=torch.float).view(8, 1) + 1

    merged = scatter_cp_rows(torch.zeros(4, 1, 1), rows, positions, sp_rank=0)

    # Rank 0 holds chunks 0 and 3 of the zigzag layout: tokens 0, 1, 6, 7.
    assert positions.tolist() == [0, 1, -1, -1, -1, -1, 2, 3]
    assert merged[:, 0, 0].tolist() == [1.0, 2.0, 7.0, 8.0]
