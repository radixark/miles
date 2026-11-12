from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils import parallel
from miles.backends.training_utils.loss.hub.math_utils import compute_ess_ratio_contribution, compute_opsm_mask
from miles.backends.training_utils.parallel import GroupInfo, ParallelState


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    group = GroupInfo(rank=0, size=1, group=None)
    state = ParallelState(
        intra_dp=group, intra_dp_cp=group, cp=group, tp=group, pp=group, ep=group, etp=group, indep_dp=group
    )
    monkeypatch.setattr(parallel, "_parallel_state", state)


def _rollout_mask_sums(rollout_ids: list[int], loss_masks: list[torch.Tensor]) -> torch.Tensor:
    totals: dict[int, float] = {}
    for rollout_id, mask in zip(rollout_ids, loss_masks, strict=True):
        totals[rollout_id] = totals.get(rollout_id, 0.0) + float(mask.sum())
    return torch.tensor([totals[rollout_id] for rollout_id in rollout_ids])


def _ess(ppo_kl: list[torch.Tensor], loss_masks: list[torch.Tensor], **kwargs) -> float:
    lengths = [mask.numel() for mask in loss_masks]
    kwargs.setdefault("calculate_per_token_loss", False)
    return compute_ess_ratio_contribution(
        ppo_kl=torch.cat(ppo_kl),
        loss_masks=loss_masks,
        total_lengths=lengths,
        response_lengths=lengths,
        qkv_format="thd",
        max_seq_lens=None,
        **kwargs,
    ).item()


def test_on_policy_ess_adds_one_per_rollout_however_it_is_split() -> None:
    rollout_ids = [0, 0, 0, 1, 1, 2]
    loss_masks = [torch.ones(n) for n in (4, 2, 3, 5, 1, 6)]
    ppo_kl = [torch.zeros(mask.numel()) for mask in loss_masks]

    per_rollout = _ess(ppo_kl, loss_masks, denominators=_rollout_mask_sums(rollout_ids, loss_masks))

    assert per_rollout == pytest.approx(3.0)
    assert _ess(ppo_kl, loss_masks) == pytest.approx(6.0)


def test_ess_of_a_split_rollout_is_its_token_weighted_mean() -> None:
    loss_masks = [torch.ones(2), torch.tensor([1.0, 0.0, 1.0])]
    ppo_kl = [torch.tensor([0.0, -0.5]), torch.tensor([0.3, 9.0, 0.0])]
    weights = [torch.tensor([0.0, 0.5]).exp(), torch.tensor([-0.3, 0.0]).exp()]
    ess = [w.sum() ** 2 / (w.numel() * (w * w).sum()) for w in weights]

    per_rollout = _ess(ppo_kl, loss_masks, denominators=_rollout_mask_sums([0, 0], loss_masks))

    assert per_rollout == pytest.approx(float(ess[0] * 2 / 4 + ess[1] * 2 / 4), rel=1e-5)


def test_single_sample_rollouts_and_per_token_ess_are_unchanged() -> None:
    loss_masks = [torch.ones(3), torch.tensor([1.0, 1.0, 0.0, 1.0])]
    ppo_kl = [torch.tensor([0.1, -0.2, 0.3]), torch.tensor([0.0, 0.4, 5.0, -0.1])]
    single = _rollout_mask_sums([0, 1], loss_masks)

    assert _ess(ppo_kl, loss_masks, denominators=single) == pytest.approx(_ess(ppo_kl, loss_masks))
    assert _ess(ppo_kl, loss_masks, denominators=single, calculate_per_token_loss=True) == pytest.approx(
        _ess(ppo_kl, loss_masks, calculate_per_token_loss=True)
    )


def _opsm_clipfrac(loss_masks: list[torch.Tensor], advantage: float, **kwargs) -> float:
    log_probs = [torch.zeros(mask.numel()) for mask in loss_masks]
    old_log_probs = [torch.ones(mask.numel()) for mask in loss_masks]
    _, clipfrac = compute_opsm_mask(
        args=Namespace(opsm_delta=0.5),
        full_log_probs=log_probs,
        full_old_log_probs=old_log_probs,
        advantages=[torch.full((mask.numel(),), advantage) for mask in loss_masks],
        loss_masks=loss_masks,
        **kwargs,
    )
    return clipfrac.item()


def test_opsm_clipfrac_counts_a_masked_rollout_once_however_it_is_split() -> None:
    rollout_ids = [0, 0, 0, 1]
    loss_masks = [torch.ones(3), torch.ones(2), torch.ones(1), torch.ones(4)]
    denominators = _rollout_mask_sums(rollout_ids, loss_masks)

    assert _opsm_clipfrac(loss_masks, -1.0, denominators=denominators) == pytest.approx(2.0)
    assert _opsm_clipfrac(loss_masks, 1.0, denominators=denominators) == 0.0
    assert _opsm_clipfrac(loss_masks, -1.0) == pytest.approx(4.0)


def test_opsm_clipfrac_ignores_the_masked_prefix_of_a_later_piece() -> None:
    loss_masks = [torch.ones(3), torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])]

    assert _opsm_clipfrac(loss_masks, -1.0, denominators=_rollout_mask_sums([0, 0], loss_masks)) == pytest.approx(1.0)
    assert _opsm_clipfrac(loss_masks, -1.0) == pytest.approx(2.0)


def test_opsm_clipfrac_sums_to_one_rollout_across_context_parallel_ranks() -> None:
    loss_masks = [torch.ones(3), torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])]
    denominators = _rollout_mask_sums([0, 0], loss_masks)
    halves = [(slice(0, mask.numel() // 2), slice(mask.numel() // 2, None)) for mask in loss_masks]

    total = 0.0
    for rank in range(2):
        local_masks = [mask[halves[i][rank]] for i, mask in enumerate(loss_masks)]
        _, clipfrac = compute_opsm_mask(
            args=Namespace(opsm_delta=0.5),
            full_log_probs=[torch.zeros(mask.numel()) for mask in loss_masks],
            full_old_log_probs=[torch.ones(mask.numel()) for mask in loss_masks],
            advantages=[torch.full((local.numel(),), -1.0) for local in local_masks],
            loss_masks=loss_masks,
            denominators=denominators,
            local_loss_masks=local_masks,
        )
        total += clipfrac.item()

    assert total == pytest.approx(1.0)
