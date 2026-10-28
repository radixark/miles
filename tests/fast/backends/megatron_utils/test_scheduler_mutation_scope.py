import pytest
import torch
from tests.fast.backends.megatron_utils.backend_scope_fixtures import _parsed_trainer

from miles.backends.megatron_utils.model import get_optimizer_param_scheduler


class TestMegatronSchedulerMutationScope:
    @pytest.mark.parametrize("decay_iters,expected", [(None, 12), (7, 7)])
    def test_real_scheduler_derives_iterations_without_leaving_backend_mutable(
        self, decay_iters: int | None, expected: int
    ) -> None:
        """Megatron scheduling derives private iteration counts and revokes its mutation lease."""
        flags = [
            "--num-rollout",
            "6",
            "--rollout-batch-size",
            "4",
            "--n-samples-per-prompt",
            "2",
            "--global-batch-size",
            "4",
            "--lr",
            "0.1",
            "--min-lr",
            "0",
            "--lr-decay-style",
            "linear",
        ]
        if decay_iters is not None:
            flags.extend(["--lr-decay-iters", str(decay_iters)])
        args = _parsed_trainer(*flags)
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)

        scheduler = get_optimizer_param_scheduler(args=args, optimizer=optimizer)

        assert args.backend.train_iters == 12
        assert args.backend.lr_decay_iters == expected
        assert scheduler.lr_decay_steps == expected * 4
        assert optimizer.param_groups[0]["lr"] == 0.1
        assert "train_iters" not in vars(args)
        with pytest.raises(AttributeError, match="immutable"):
            args.backend.train_iters = 99
