"""Multi-LoRA slots' compute members shared by every backend's adapter slots."""

from collections.abc import Sequence

import torch


class SlotComputeMembers:
    """Multi-LoRA slots' compute members: claim, free and routing, checked here, done by the backend's hooks.

    A slot is claimed from ``init`` to ``clear``.
    """

    def _count_slots(self) -> None:
        # multi-LoRA runs PP=1, so every rank holds slot modules
        self.num_slots = self._local_num_slots()
        assert self.num_slots > 0, "a multi-LoRA model holds no slot modules"
        self._claimed_slots: set[int] = set()

    def init(self, slot: int, lora_rank: int, lora_alpha: float, seed: int | None) -> None:
        """Claim ``slot``, masking its weights past ``lora_rank``.

        With ``seed`` the weights are first re-drawn through the RNG tracker from it; without, they stay as the
        model built them or the last ``clear`` re-drew them.
        """
        self._assert_slot(slot)
        # a re-draw would wipe the tenant's adapter
        assert slot not in self._claimed_slots, f"slot {slot} is already claimed"
        self._init_slot(slot, lora_rank, lora_alpha, seed)
        self._claimed_slots.add(slot)

    def clear(self, slot: int) -> None:
        """Free ``slot``; its weights are re-drawn through the RNG tracker for the next claim."""
        self._assert_slot(slot)
        self._clear_slot(slot)
        self._claimed_slots.discard(slot)

    def route_tokens(self, token_counts_by_slot: Sequence[int] | torch.Tensor) -> None:
        """Route the next forward's tokens on this rank (after any CP split), sorted by slot, until the next call.

        Every TP rank must pass the same counts: a row-parallel adapter reduces the slots' partial sums together.
        Recomputation reruns the forward inside the backward, after the forward step has returned.
        """
        assert self._multi_lora, "single LoRA routes no tokens"
        counts_tensor = torch.as_tensor(token_counts_by_slot)
        assert counts_tensor.dim() == 1 and not (
            counts_tensor.is_floating_point() or counts_tensor.is_complex() or counts_tensor.dtype == torch.bool
        ), f"token counts must be a 1-D integer sequence, got {counts_tensor.dtype} of shape {tuple(counts_tensor.shape)}"
        # one host sync per forward
        counts = tuple(counts_tensor.tolist())
        assert len(counts) == self.num_slots, f"{len(counts)} token counts for {self.num_slots} slots"
        assert min(counts) >= 0, f"negative token counts {counts}"
        unclaimed = [slot for slot, count in enumerate(counts) if count and slot not in self._claimed_slots]
        assert not unclaimed, f"tokens routed to unclaimed slots {unclaimed}"
        self._install_routing(counts)

    def _assert_slot(self, slot: int) -> None:
        assert self._multi_lora, "single LoRA's adapter is built with the model"
        assert 0 <= slot < self.num_slots, f"slot {slot} is not in [0, {self.num_slots})"
