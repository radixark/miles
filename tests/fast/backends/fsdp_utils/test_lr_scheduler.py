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
