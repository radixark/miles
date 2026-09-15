from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miles.backends.megatron_utils.lora.slot_capacity import RankProbe


def expert_groups_per_slot(model) -> int:
    from megatron.bridge.peft.multi_lora_layers import MultiLoRAGroupedExpertLinear

    return max(
        (
            module.num_local_experts
            for chunk in model
            for module in chunk.modules()
            if isinstance(module, MultiLoRAGroupedExpertLinear)
        ),
        default=0,
    )


def grouped_mm_max_groups() -> int | None:
    import torch

    if not hasattr(torch, "_grouped_mm") or not torch.cuda.is_available():
        return None

    def fits(groups: int) -> bool:
        rows = 16 * groups
        try:
            torch._grouped_mm(
                torch.zeros(rows, 16, dtype=torch.bfloat16, device="cuda"),
                torch.zeros(groups, 16, 16, dtype=torch.bfloat16, device="cuda"),
                offs=torch.arange(16, rows + 1, 16, dtype=torch.int32, device="cuda"),
            )
            torch.cuda.synchronize()
            return True
        except RuntimeError:
            return False

    if not fits(1):
        return None
    low, high = 1, 2
    while high <= 1 << 16 and fits(high):
        low, high = high, high * 2
    if high > 1 << 16:
        return None
    while high - low > 1:
        mid = (low + high) // 2
        low, high = (mid, high) if fits(mid) else (low, mid)
    return low


def max_capacity_for_once_fb(probes: list[RankProbe]) -> tuple[int, str] | None:
    groups = max(probe.expert_groups_per_slot for probe in probes)
    max_groups = min((probe.grouped_mm_max_groups for probe in probes if probe.grouped_mm_max_groups), default=None)
    if not groups or not max_groups:
        return None
    return max_groups // groups, f"torch._grouped_mm's {max_groups}-group limit ({groups} local experts per slot)"
