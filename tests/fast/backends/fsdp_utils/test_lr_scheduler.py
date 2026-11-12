import pytest
import torch

from miles.backends.fsdp_utils.config import FsdpArgsNamespace
from miles.backends.fsdp_utils.lr_scheduler import get_lr_scheduler
from miles.utils.args.runtime import TrainerConfig


class TestSchedulerConfigurationScope:
    @pytest.mark.parametrize("decay_iters,expected", [(None, 12), (7, 7)])
    def test_iteration_derivation_uses_backend_fields_and_restores_immutability(
        self, decay_iters: int | None, expected: int
    ) -> None:
        """Scheduler setup derives missing iteration counts while retaining explicit decay and immutable ownership."""
        backend = FsdpArgsNamespace(
            global_batch_size=4,
            lr_decay_iters=decay_iters,
            lr_wsd_decay_iters=None,
            lr_warmup_fraction=None,
            lr_warmup_iters=0,
            lr_warmup_init=0.0,
            lr=0.1,
            min_lr=0.0,
            lr_decay_style="linear",
            use_checkpoint_lr_scheduler=True,
            override_lr_scheduler=False,
            lr_wsd_decay_style=None,
        )
        args = TrainerConfig.model_construct(
            backend=backend, num_rollout=6, rollout_batch_size=4, n_samples_per_prompt=2
        )
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)

        scheduler = get_lr_scheduler(args=args, optimizer=optimizer)

        assert backend.train_iters == 12
        assert backend.lr_decay_iters == expected
        assert scheduler.lr_decay_steps == expected
        assert scheduler.get_last_lr() == [0.1]
        assert "train_iters" not in vars(args)
        with pytest.raises(AttributeError, match="immutable"):
            backend.train_iters = 99


def _scheduler(*, num_rollout: int, rollouts_per_round: int, global_batch_size: int, decay_style: str = "cosine"):
    backend = FsdpArgsNamespace(
        global_batch_size=global_batch_size,
        lr_decay_iters=None,
        lr_wsd_decay_iters=None,
        lr_warmup_fraction=None,
        lr_warmup_iters=0,
        lr_warmup_init=0.0,
        lr=1.0,
        min_lr=0.0,
        lr_decay_style=decay_style,
        use_checkpoint_lr_scheduler=False,
        override_lr_scheduler=False,
        lr_wsd_decay_style=None,
    )
    args = TrainerConfig.model_construct(
        backend=backend, num_rollout=num_rollout, rollout_batch_size=rollouts_per_round, n_samples_per_prompt=1
    )
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    return backend, optimizer, get_lr_scheduler(args, optimizer)


def test_rounds_of_multi_sample_rollouts_decay_over_one_step_per_round():
    backend, optimizer, scheduler = _scheduler(num_rollout=4, rollouts_per_round=4, global_batch_size=8)
    rates = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert backend.train_iters == 4
    assert rates[:3] == sorted(rates[:3], reverse=True) and min(rates[:3]) > 0
    assert rates[3] == pytest.approx(0.0, abs=1e-12)


def test_one_round_of_multi_sample_rollouts_builds_a_scheduler():
    backend, _, _ = _scheduler(num_rollout=1, rollouts_per_round=4, global_batch_size=8, decay_style="constant")

    assert backend.train_iters == 1


def test_a_global_batch_dividing_the_round_keeps_its_step_count():
    backend, _, _ = _scheduler(num_rollout=3, rollouts_per_round=8, global_batch_size=4)

    assert backend.train_iters == 6
