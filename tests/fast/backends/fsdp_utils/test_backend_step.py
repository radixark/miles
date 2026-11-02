from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from miles.backends.fsdp_utils.actor import FSDPTrainRayActor


class TestBackendStepOptions:
    def test_step_uses_backend_clip_limit_and_reports_updated_optimizer_lr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An FSDP step clips at its backend limit and reports the learning rate after scheduling."""
        actor = object.__new__(FSDPTrainRayActor)
        actor.args = SimpleNamespace(backend=SimpleNamespace(clip_grad=0.75))
        actor.model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            actor.model.weight.fill_(1)
        actor.model.weight.grad = torch.full_like(actor.model.weight, 2)
        actor.optimizer = torch.optim.SGD(actor.model.parameters(), lr=0.1)
        actor.lr_scheduler = torch.optim.lr_scheduler.StepLR(actor.optimizer, step_size=1, gamma=0.5)
        clip = Mock(return_value=SimpleNamespace(full_tensor=lambda: torch.tensor(2.5)))
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)

        result = actor._apply_step()

        assert clip.call_count == 1
        assert clip.call_args.args[1] == 0.75
        assert result.grad_norm == 2.5
        assert result.extra_metrics == {"lr-pg_0": 0.05}
        torch.testing.assert_close(actor.model.weight, torch.tensor([[0.8]]))
