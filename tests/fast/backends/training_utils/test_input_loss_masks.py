"""Mask alignment must be established before packing and context-parallel slicing."""

import pytest
import torch

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data import rollout as data_utils
from miles.backends.training_utils.parallel import GroupInfo, ParallelState


def _parallel_state(cp_size: int, cp_rank: int) -> ParallelState:
    group = GroupInfo(rank=0, size=1, group=None)
    cp_group = GroupInfo(rank=cp_rank, size=cp_size, group=None)
    return ParallelState(
        intra_dp=group,
        intra_dp_cp=cp_group,
        cp=cp_group,
        tp=GroupInfo(rank=0, size=2, group=None),
        pp=group,
        ep=group,
        etp=group,
        indep_dp=group,
    )


@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
@pytest.mark.parametrize("allgather_cp", [False, True])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_mask_alignment_survives_packing_and_cp(
    monkeypatch: pytest.MonkeyPatch,
    qkv_format: str,
    allgather_cp: bool,
    cp_size: int,
) -> None:
    # Different lengths, multiple assistant spans, an empty response, a masked
    # response, and a prompt-free sample. Token IDs uniquely identify positions.
    # Masks use the production dtype from get_rollout_data (torch.int).
    tokens = [torch.arange(100 * i + 1, 100 * i + n + 1) for i, n in enumerate([13, 11, 5, 9, 7, 1])]
    masks = [
        torch.tensor(values, dtype=torch.int)
        for values in ([1, 1, 0, 0, 1, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1, 1], [], [0] * 6, [1] * 7, [1])
    ]
    original_masks = [mask.clone() for mask in masks]
    rollout = {
        "tokens": tokens,
        "loss_masks": masks,
        "total_lengths": [len(t) for t in tokens],
        "response_lengths": [len(mask) for mask in masks],
        "max_seq_lens": [24] * len(tokens),
    }
    # Independent oracle keyed by the actual input token at each position:
    # mask[i] == 1 iff tokens[i] is a supervised response token. Neither the
    # collator's padding helper nor its CP slicing builds it.
    input_weights = {0: 0}
    for token_ids, mask in zip(tokens, masks, strict=True):
        weights = [0] * (len(token_ids) - len(mask)) + mask.tolist()
        for position, token_id in enumerate(token_ids.tolist()):
            input_weights[token_id] = weights[position]

    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    total = 0
    seen_tokens = []
    for cp_rank in range(cp_size):
        state = _parallel_state(cp_size, cp_rank)
        monkeypatch.setattr(data_utils, "get_parallel_state", lambda state=state: state)
        monkeypatch.setattr(context_parallel, "get_parallel_state", lambda state=state: state)
        batch = data_utils.get_batch(
            data_utils.DataIterator(rollout, micro_batch_size=len(tokens)),
            list(rollout),
            pad_multiplier=7,
            qkv_format=qkv_format,
            allgather_cp=allgather_cp,
        )
        assert "full_loss_masks" not in batch
        flat_tokens = batch["tokens"].flatten().tolist()
        seen_tokens.extend(token_id for token_id in flat_tokens if token_id != 0)
        expected = torch.tensor([input_weights[token_id] for token_id in flat_tokens], dtype=torch.int)
        torch.testing.assert_close(batch["input_loss_masks"].flatten(), expected)
        assert batch["input_loss_masks"].shape == batch["tokens"].shape
        total += batch["input_loss_masks"].sum().item()

    assert sorted(seen_tokens) == sorted(torch.cat(tokens).tolist())
    assert total == sum(input_weights.values())
    for mask, original in zip(masks, original_masks, strict=True):
        torch.testing.assert_close(mask, original)
