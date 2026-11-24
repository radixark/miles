"""Trie-packed micro-batches (--trie-packing): the step schedule, and each scheduled micro-batch as one trie row.

``get_data_iterator`` hands its ``DataIterator`` a ``TrieMicroBatches``; ``get_batch`` then reads each micro-batch's
trie row from it, the same row on every pass over the rollout.
"""

import logging

import torch
import torch.distributed as dist

from miles.backends.training_utils.data.trie_packing import (
    WINDOW_START_MULTIPLE,
    TrieLayout,
    TrieModelSpec,
    micro_batch_trie_layout,
    pack_trie_bins,
    response_logit_rows,
    trie_input_loss_mask,
)
from miles.platforms import current_platform
from miles.utils.ft_utils.process_group_utils import GeneralPGUtil
from miles.utils.seqlen_balancing import expand_bins_by_splitting
from miles.utils.types import RolloutBatch

logger = logging.getLogger(__name__)


class TrieMicroBatches:
    """A rollout's trie micro-batches: the samples each holds, and its trie row, built on first use."""

    def __init__(
        self,
        rollout_data: RolloutBatch,
        model_spec: TrieModelSpec,
        micro_batch_indices: list[list[int]],
    ) -> None:
        self.rollout_data = rollout_data
        self.model_spec = model_spec
        self.micro_batch_indices = micro_batch_indices
        self._layouts: dict[int, TrieLayout] = {}

    def layout(self, index: int, *, pad_multiple: int) -> TrieLayout:
        """The trie row of micro-batch ``index``."""
        if index not in self._layouts:
            rows = self.micro_batch_indices[index]
            self._layouts[index] = micro_batch_trie_layout(
                [self.rollout_data["tokens"][i].tolist() for i in rows],
                [self.rollout_data["total_lengths"][i] for i in rows],
                [self.rollout_data["response_lengths"][i] for i in rows],
                self.model_spec,
                window_start_multiple=WINDOW_START_MULTIPLE,
                pad_multiple=pad_multiple,
            )
        return self._layouts[index]

    def fill_batch(self, index: int, batch: dict, *, pad_multiple: int) -> dict:
        """Fill micro-batch ``index`` as its trie row: shared prefixes stored once, at their logical positions.

        The model scores only the rows that predict a response (``trie_logit_rows``), so the logits
        hold the responses back to back and ``response_row_starts`` tells the loss where each begins.
        """
        for key in ("adapter_slots", "multimodal_train_inputs", "witness_ids", "mrope_position_ids"):
            assert batch.get(key) is None, f"trie packing does not support {key}"
        layout = self.layout(index, pad_multiple=pad_multiple)
        device = current_platform.current_device()
        total_lengths, response_lengths = batch["total_lengths"], batch["response_lengths"]
        num_tokens = layout.num_tokens
        batch["unconcat_tokens"] = batch["tokens"]
        batch["tokens"] = layout.tokens.to(device).unsqueeze(0)
        batch["position_ids"] = layout.positions.to(device).unsqueeze(0)
        batch["cu_seqlens"] = torch.tensor([0, num_tokens], dtype=torch.int, device=device)
        batch["cu_seqlens_host"] = (0, num_tokens)
        batch["max_seqlen"] = num_tokens
        batch["trie_layout"] = layout
        batch["trie_logit_rows"] = response_logit_rows(layout, total_lengths, response_lengths).to(device)
        response_row_starts = [0]
        for response_length in response_lengths[:-1]:
            response_row_starts.append(response_row_starts[-1] + response_length)
        batch["response_row_starts"] = response_row_starts
        batch["input_loss_masks"] = trie_input_loss_mask(
            layout, batch["loss_masks"], total_lengths, response_lengths
        ).to(device)
        return batch


def schedule_trie_micro_batches(
    rollout_data: RolloutBatch,
    model_spec: TrieModelSpec,
    *,
    num_steps: int,
    num_local_gbs: int,
    max_tokens_per_gpu: int,
    dp_group,
) -> tuple[TrieMicroBatches, list[int]]:
    """Per step, this rank's samples as prefix tries of at most ``max_tokens_per_gpu`` distinct tokens, split until
    every DP rank runs the same number of micro-batches; and the number of micro-batches per step."""
    tokens, total_lengths = rollout_data["tokens"], rollout_data["total_lengths"]
    steps, logical, physical = [], 0, 0
    for step in range(num_steps):
        start, end = step * num_local_gbs, (step + 1) * num_local_gbs
        tries = pack_trie_bins(
            [tokens[i].tolist() for i in range(start, end)],
            max_tokens_per_gpu,
        )
        steps.append([list(trie.tags) for trie in tries])
        logical += sum(total_lengths[start:end])
        physical += sum(len(trie.token_ids) for trie in tries)
    logger.info(
        f"trie packing: {logical} tokens -> {physical} on this rank "
        f"({logical / max(physical, 1):.2f}x), tries per step {[len(bins) for bins in steps]}"
    )
    counts = torch.tensor([len(bins) for bins in steps], dtype=torch.int, device=current_platform.current_device())
    GeneralPGUtil.create(dp_group).all_reduce(counts, dp_group, op=dist.ReduceOp.MAX)
    num_microbatches = counts.tolist()
    micro_batch_indices = []
    for step, (bins, target) in enumerate(zip(steps, num_microbatches, strict=True)):
        start = step * num_local_gbs
        # splitting a trie gives up only the sharing between its two halves
        expand_bins_by_splitting(bins, target, total_lengths[start : start + num_local_gbs])
        assert len(bins) == target, f"step {step}: {num_local_gbs} samples cannot fill {target} trie micro-batches"
        micro_batch_indices.extend([[start + i for i in sorted(trie_bin)] for trie_bin in bins])
    trie_micro_batches = TrieMicroBatches(rollout_data, model_spec, micro_batch_indices)
    return trie_micro_batches, num_microbatches
