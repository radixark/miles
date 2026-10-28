from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from miles.backends.fsdp_utils import checkpoint
from miles.backends.fsdp_utils.config import FsdpArgsNamespace


@pytest.fixture
def cpu_checkpoint_actor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(checkpoint.dist, "get_rank", lambda *args, **kwargs: 0)
    monkeypatch.setattr(checkpoint.dist, "get_world_size", lambda *args, **kwargs: 1)
    monkeypatch.setattr(checkpoint.dist, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scheduler.step()
    return SimpleNamespace(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        args=SimpleNamespace(
            backend=FsdpArgsNamespace(
                load=str(tmp_path),
                save=str(tmp_path),
                no_save_optim=False,
                no_load_optim=False,
                no_load_rng=False,
                ckpt_step=None,
            ),
            start_rollout_id=None,
        ),
        global_step=11,
        micro_step=13,
    )
