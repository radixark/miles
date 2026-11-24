"""Verify the BPO custom loss plugin math, validation, and `policy_loss_function` hook integration."""

import math
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from examples.experimental.bpo.bpo_loss import BpoConfig, bpo_loss_function, compute_bpo_policy_loss
from pydantic import ValidationError

BPO_LOSS_PATH = "examples.experimental.bpo.bpo_loss.bpo_loss_function"


def _pg_args(**overrides: object) -> SimpleNamespace:
    return SimpleNamespace(
        **(
            dict(
                advantage_estimator="grpo",
                eps_clip=0.2,
                eps_clip_high=0.28,
                eps_clip_c=None,
                bpo_smoothing=0.1,
                bpo_weight_cap=3.0,
            )
            | overrides
        )
    )


def _bpo_ref(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    smoothing: float = 0.1,
    cap: float = 3.0,
    eps_low: float = 0.2,
    eps_high: float = 0.28,
) -> tuple[torch.Tensor, torch.Tensor]:
    w = (1.0 + smoothing - old_log_probs.detach().exp()) / (1.0 + smoothing - log_probs.detach().exp())
    clipped = ((advantages.detach() > 0) & (w > 1.0 + eps_high)) | ((advantages.detach() < 0) & (w < 1.0 - eps_low))
    loss = -(advantages.detach() * w.clamp(max=cap) * (~clipped)) * log_probs
    return loss, clipped.to(log_probs.dtype)


class TestBpoConfigAndMath:
    def test_bpo_validation(self) -> None:
        assert (BpoConfig().bpo_smoothing, BpoConfig().bpo_weight_cap) == (0.1, 3.0)
        for bad, match in [
            (dict(bpo_smoothing=0.0), "bpo-smoothing"),
            (dict(bpo_smoothing=1e-50), "bpo-smoothing"),
            (dict(bpo_weight_cap=-1.0), "bpo-weight-cap"),
            (dict(bpo_weight_cap=1e100), "bpo-weight-cap"),
        ]:
            with pytest.raises(ValidationError, match=match):
                BpoConfig(**bad)

        for bad, match in [
            (dict(advantage_estimator="gspo"), "advantage-estimator grpo"),
            (dict(eps_clip_c=3.0), "eps-clip-c"),
            (dict(eps_clip=1.0), "eps-clip"),
            (dict(eps_clip_high=-0.1), "eps-clip-high"),
        ]:
            with pytest.raises(ValueError, match=match):
                bpo_loss_function(_pg_args(**bad), {}, torch.zeros(1, 2, 2), torch.sum)

    def test_bpo_formula_cap_nonfinite_and_precision(self) -> None:
        # Eqs. (14)-(15), stop-gradient through weight/old_log_probs/advantages, and uncapped-weight masking
        probs = [0.1, 0.5, 0.9, 0.999999]
        old_probs = [0.9, 0.4, 0.1, 0.01]
        lp = torch.tensor(probs * 3, dtype=torch.float64).log().requires_grad_()
        old_lp = torch.tensor(old_probs * 3, dtype=torch.float64).log().requires_grad_()
        adv = torch.tensor([1.0] * 4 + [-2.0] * 4 + [0.0] * 4, dtype=torch.float64, requires_grad=True)

        pg_loss, pg_clip = compute_bpo_policy_loss(
            args=_pg_args(bpo_weight_cap=1.1), log_probs=lp, old_log_probs=old_lp, ppo_kl=old_lp - lp, advantages=adv
        )
        ref_loss, ref_clip = _bpo_ref(lp, old_lp, adv, cap=1.1)
        torch.testing.assert_close(pg_loss, ref_loss)
        torch.testing.assert_close(pg_clip, ref_clip)
        grad, old_grad, adv_grad = torch.autograd.grad(pg_loss.sum(), (lp, old_lp, adv), allow_unused=True)
        torch.testing.assert_close(grad, torch.autograd.grad(ref_loss.sum(), lp)[0])
        assert old_grad is None and adv_grad is None

        # Non-finite gating (log_p=-inf/NaN -> 0, valid old_log_p=-inf q=0 kept) + bf16->fp32 promotion + empty slice
        lp_edge = torch.tensor([-torch.inf, torch.nan, math.log(0.5)], requires_grad=True)
        old_edge = torch.tensor([math.log(0.2), math.log(0.5), -torch.inf])
        adv_edge = torch.tensor([-1.0, 1.0, -1.0])
        edge_loss, edge_clip = compute_bpo_policy_loss(
            args=_pg_args(eps_clip=0.0, eps_clip_high=0.0),
            log_probs=lp_edge,
            old_log_probs=old_edge,
            ppo_kl=old_edge - lp_edge,
            advantages=adv_edge,
        )
        assert torch.isfinite(edge_loss).all() and edge_clip.sum().item() == 0.0
        torch.testing.assert_close(
            torch.autograd.grad(edge_loss.sum(), lp_edge)[0], torch.tensor([0.0, 0.0, 1.1 / 0.6])
        )

        lp_bf16 = torch.tensor([-0.69140625], dtype=torch.bfloat16, requires_grad=True)
        old_bf16 = torch.tensor([-1.1015625], dtype=torch.bfloat16)
        loss_bf16, clip_bf16 = compute_bpo_policy_loss(
            args=_pg_args(),
            log_probs=lp_bf16,
            old_log_probs=old_bf16,
            ppo_kl=old_bf16 - lp_bf16,
            advantages=torch.ones(1, dtype=torch.bfloat16),
        )
        assert loss_bf16.dtype == torch.float32 and clip_bf16.dtype == torch.float32 and clip_bf16.item() == 1.0

        empty = torch.empty(0, requires_grad=True)
        e_loss, e_clip = compute_bpo_policy_loss(
            args=_pg_args(), log_probs=empty, old_log_probs=empty, ppo_kl=empty, advantages=empty
        )
        assert e_loss.shape == (0,) and e_clip.shape == (0,)


class TestBpoPipelineIntegration:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def _process_group(cls, tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
        from miles.backends.training_utils import parallel

        saved = parallel._parallel_state
        created = not dist.is_initialized()
        if created:
            dist.init_process_group(
                "gloo", init_method=f"file://{tmp_path_factory.mktemp('bpo') / 'rv'}", rank=0, world_size=1
            )
        try:
            yield
        finally:
            parallel._parallel_state = saved
            if created and dist.is_initialized():
                dist.destroy_process_group()

    @staticmethod
    def _custom_args(**overrides: object):
        from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

        from miles.utils.args.custom_function import CustomFunctionConfig
        from miles.utils.args.custom_view import ImmutableNamespace

        make_parallel_state()
        hook = CustomFunctionConfig(
            path=BPO_LOSS_PATH,
            config=BpoConfig(
                bpo_smoothing=overrides.pop("bpo_smoothing", 0.1),
                bpo_weight_cap=overrides.pop("bpo_weight_cap", 3.0),
            ),
        )
        return ImmutableNamespace(
            **dict(
                make_args(
                    loss_type="custom_loss",
                    custom_loss_function_path=hook,
                    eps_clip=overrides.pop("eps_clip", 0.2),
                    eps_clip_high=overrides.pop("eps_clip_high", 0.28),
                    true_on_policy_mode=False,
                    **overrides,
                )
            )
        )

    @pytest.mark.parametrize("use_rollout_logprobs", [False, True])
    def test_bpo_dispatch_end_to_end(self, use_rollout_logprobs: bool) -> None:
        from miles.backends.training_utils.loss import objective

        args = self._custom_args(
            use_rollout_logprobs=use_rollout_logprobs,
            observe_training_entropy=True,
            entropy_coef=0.03,
            use_kl_loss=True,
            kl_loss_coef=0.2,
            kl_loss_type="k2",
            bpo_smoothing=0.25,
            bpo_weight_cap=1.5,
        )
        logits = torch.randn((1, 6, 5), generator=torch.Generator().manual_seed(7))
        logits[0, 1] = torch.tensor([-2.0, -2.0, 3.0, -2.0, -2.0])  # token 0 -> high prob -> BPO-clipped
        logits.requires_grad_(True)
        old_lp = torch.tensor([math.log(0.05), torch.nan, math.log(0.4), math.log(0.2)], requires_grad=True)
        adv = torch.tensor([1.0, torch.nan, -2.0, 0.5], requires_grad=True)
        batch = dict(
            unconcat_tokens=[torch.tensor([0, 1, 2, 3, 4, 0])],
            total_lengths=[6],
            response_lengths=[4],
            loss_masks=[torch.tensor([1, 0, 1, 1])],
            advantages=[adv],
            rollout_log_probs=[old_lp if use_rollout_logprobs else torch.full((4,), -0.5)],
            log_probs=[torch.full((4,), -0.5) if use_rollout_logprobs else old_lp],
            ref_log_probs=[torch.full((4,), -1.5)],
        )

        dense_logits = logits.detach().clone().requires_grad_()
        loss, _, log = objective.loss_function(args, batch, 1, logits, num_rollouts=1)
        metrics = dict(zip(log["keys"], log["values"][1:], strict=True))

        dist_lp = dense_logits[0, 1:5].log_softmax(-1)
        selected = dist_lp[torch.arange(4), torch.tensor([2, 3, 4, 0])]
        active = torch.tensor([True, False, True, True])
        pg_tokens, clipped = _bpo_ref(selected[active], old_lp[active], adv[active], smoothing=0.25, cap=1.5)
        assert clipped.sum().item() >= 1.0
        expected_pg = pg_tokens.mean()
        entropy = -(dist_lp.exp() * dist_lp).sum(-1)[active]
        kl_pen = (selected[active] + 1.5).square() / 2
        expected = expected_pg + (-0.03 * entropy + 0.2 * kl_pen).mean()

        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(metrics["pg_loss"], expected_pg.detach())
        torch.testing.assert_close(metrics["pg_clipfrac"], clipped.mean())
        grad, old_grad, adv_grad = torch.autograd.grad(loss, (logits, old_lp, adv), allow_unused=True)
        torch.testing.assert_close(grad, torch.autograd.grad(expected, dense_logits)[0])
        assert old_grad is None and adv_grad is None

    def test_policy_loss_hook_active_nan_and_opsm(self) -> None:
        from miles.backends.training_utils.loss import objective
        from miles.backends.training_utils.loss.hub.losses import policy_loss_function

        # Active-token NaN in old_log_probs falls back to w=1
        args = self._custom_args(use_rollout_logprobs=True, entropy_coef=0.0, observe_training_entropy=False)
        logits = torch.randn((1, 4, 4), generator=torch.Generator().manual_seed(31), requires_grad=True)
        dense_logits = logits.detach().clone().requires_grad_()
        old_lp = torch.tensor([torch.nan, math.log(0.4), math.log(0.2)])
        adv = torch.tensor([1.0, -1.0, 0.5])
        batch = dict(
            unconcat_tokens=[torch.tensor([0, 1, 2, 3])],
            total_lengths=[4],
            response_lengths=[3],
            loss_masks=[torch.ones(3)],
            advantages=[adv],
            rollout_log_probs=[old_lp],
        )
        loss, _, _ = objective.loss_function(args, batch, 1, logits, num_rollouts=1)
        selected = dense_logits[0, 0:3].log_softmax(-1)[torch.arange(3), torch.tensor([1, 2, 3])]
        clean_old = torch.where(old_lp.isnan(), selected.detach(), old_lp)
        expected = _bpo_ref(selected, clean_old, adv)[0].mean()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(
            torch.autograd.grad(loss, logits)[0], torch.autograd.grad(expected, dense_logits)[0]
        )

        # OPSM: rejected -inf hook -> 0 loss/grad; kept fp16 hook promotes to fp32 (no int32-mask overflow)
        opsm_batch = batch | dict(advantages=[torch.full((3,), -1.0)], rollout_log_probs=[torch.zeros(3)])
        loss_inf, _ = policy_loss_function(
            self._custom_args(
                use_rollout_logprobs=True,
                use_opsm=True,
                opsm_delta=0.05,
                entropy_coef=0.0,
                observe_training_entropy=False,
            ),
            opsm_batch,
            logits,
            torch.mean,
            pg_loss_fn=lambda *, log_probs, **_: (
                torch.full_like(log_probs, -torch.inf) + 0.0 * log_probs,
                torch.zeros_like(log_probs),
            ),
        )
        assert loss_inf.item() == 0.0 and torch.autograd.grad(loss_inf, logits)[0].abs().max().item() == 0.0

        n = 8192
        mask_i32 = torch.ones(n, dtype=torch.int32)
        fp16_batch = dict(
            unconcat_tokens=[torch.zeros(n + 1, dtype=torch.long)],
            total_lengths=[n + 1],
            response_lengths=[n],
            loss_masks=[mask_i32],
            advantages=[torch.ones(n)],
            rollout_log_probs=[torch.full((n,), -math.log(2.0))],
        )
        loss_fp16, _ = policy_loss_function(
            self._custom_args(
                use_rollout_logprobs=True,
                use_opsm=True,
                opsm_delta=1.0,
                entropy_coef=0.0,
                observe_training_entropy=False,
            ),
            fp16_batch,
            torch.zeros((1, n + 1, 2)),
            lambda x: (x * mask_i32).sum() / mask_i32.sum(),
            pg_loss_fn=lambda *, log_probs, **_: (
                torch.full_like(log_probs, 10.0, dtype=torch.float16),
                torch.zeros_like(log_probs),
            ),
        )
        torch.testing.assert_close(loss_fp16, torch.tensor(10.0))
