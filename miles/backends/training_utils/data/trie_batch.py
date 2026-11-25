"""Trie-packed micro-batches (--trie-packing): the step schedule, and each scheduled micro-batch as one trie row.

``get_data_iterator`` hands its ``DataIterator`` a ``TrieMicroBatches``; ``get_batch`` and the routing replay
fill then read each micro-batch's trie row from it, the same row on every pass over the rollout.
"""

import logging
from collections.abc import Sequence

import torch
import torch.distributed as dist

from miles.backends.training_utils.data.token_space_media_process import resolved_multimodal_train_inputs
from miles.backends.training_utils.data.trie_packing import (
    WINDOW_START_MULTIPLE,
    MediaItem,
    TrieLayout,
    TrieModelSpec,
    media_items,
    micro_batch_trie_layout,
    pack_trie_bins,
    response_logit_rows,
    routing_aware_packing_keys,
    trie_input_loss_mask,
    trie_media_inputs,
)
from miles.platforms import current_platform
from miles.utils.ft_utils.process_group_utils import GeneralPGUtil
from miles.utils.output_store.plan import is_segments_column
from miles.utils.seqlen_balancing import expand_bins_by_splitting
from miles.utils.types import RolloutBatch

logger = logging.getLogger(__name__)


class TrieMicroBatches:
    """A rollout's trie micro-batches: the samples each holds, and its trie row, built on first use.

    ``media`` and ``packing_keys`` are each sample's media items and packing keys (``trie_media_items``,
    ``trie_packing_keys``), as the schedule packed them.
    """

    def __init__(
        self,
        rollout_data: RolloutBatch,
        model_spec: TrieModelSpec,
        micro_batch_indices: list[list[int]],
        *,
        media: list[list[MediaItem]] | None,
        packing_keys: list[list[int]] | None,
    ) -> None:
        self.rollout_data = rollout_data
        self.model_spec = model_spec
        self.micro_batch_indices = micro_batch_indices
        self._media = media
        self._packing_keys = packing_keys
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
                media=_take(self._media, rows),
                packing_keys=_take(self._packing_keys, rows),
            )
        return self._layouts[index]

    def fill_batch(self, index: int, batch: dict, *, pad_multiple: int) -> dict:
        """Fill micro-batch ``index`` as its trie row: shared prefixes stored once, at their logical positions.

        The model scores only the rows that predict a response (``trie_logit_rows``), so the logits
        hold the responses back to back and ``response_row_starts`` tells the loss where each begins.
        The row's media are each item once, in token order (``trie_media_inputs``).
        """
        for key in ("adapter_slots", "witness_ids", "mrope_position_ids"):
            assert batch.get(key) is None, f"trie packing does not support {key}"
        layout = self.layout(index, pad_multiple=pad_multiple)
        device = current_platform.current_device()
        total_lengths, response_lengths = batch["total_lengths"], batch["response_lengths"]
        if batch.get("multimodal_train_inputs") is not None:
            media = _take(self._media, self.micro_batch_indices[index])
            assert media is not None, "trie packing lays out no media for this model"
            batch["multimodal_train_inputs"] = trie_media_inputs(layout, media, self.model_spec.media)
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


def _take(per_sample: list | None, rows: Sequence[int]) -> list | None:
    return None if per_sample is None else [per_sample[i] for i in rows]


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
    # the schedule packs media and routing as the trie rows will, so their budget holds
    media = trie_media_items(rollout_data, model_spec)
    packing_keys = trie_packing_keys(rollout_data, media)
    tokens, total_lengths = rollout_data["tokens"], rollout_data["total_lengths"]
    steps, logical, physical = [], 0, 0
    for step in range(num_steps):
        start, end = step * num_local_gbs, (step + 1) * num_local_gbs
        tries = pack_trie_bins(
            [tokens[i].tolist() for i in range(start, end)],
            max_tokens_per_gpu,
            None if media is None else media[start:end],
            None if packing_keys is None else packing_keys[start:end],
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
    trie_micro_batches = TrieMicroBatches(
        rollout_data, model_spec, micro_batch_indices, media=media, packing_keys=packing_keys
    )
    return trie_micro_batches, num_microbatches


def trie_media_items(rollout_data: RolloutBatch, model_spec: TrieModelSpec) -> list[list[MediaItem]] | None:
    """Each sample's media items, for a model whose trie rows lay media out; None without media."""
    multimodal = rollout_data.get("multimodal_train_inputs")
    if multimodal is None or not model_spec.media:
        return None
    multimodal = resolved_multimodal_train_inputs(multimodal)
    return [
        media_items(tokens, sample, model_spec.media)
        for tokens, sample in zip(rollout_data["tokens"], multimodal, strict=True)
    ]


def trie_packing_keys(rollout_data: RolloutBatch, media: list[list[MediaItem]] | None) -> list[list[int]] | None:
    """Each sample's packing keys under rollout routing replay (``routing_aware_packing_keys``), None without it.

    Rows held in the output store are read per micro-batch, after the schedule, so their samples pack on tokens.
    """
    routed_experts = rollout_data.get("rollout_routed_experts")
    if routed_experts is None or is_segments_column(routed_experts):
        return None
    return routing_aware_packing_keys(rollout_data["tokens"], routed_experts, media)
