"""The torchtitan reference model moves between host and GPU around its pass, except under FSDP CPU offload."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])


class _Part(torch.nn.Linear):
    def __init__(self, moves: list[str]):
        super().__init__(2, 2)
        self._moves = moves

    def to(self, *args, **kwargs):
        self._moves.append("device")
        return self

    def cpu(self):
        self._moves.append("cpu")
        return self


def _build_ref(cpu_offload: bool, moves: list[str]):
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils import actor as actor_mod

    part = _Part(moves)
    trainer = SimpleNamespace(
        model_parts=[part],
        configure_loss_reduction=lambda: None,
        engine=SimpleNamespace(load_checkpoint=lambda: None),
        step_runner=lambda: "ref-runner",
    )
    config = SimpleNamespace(training=SimpleNamespace(enable_cpu_offload=cpu_offload))
    actor = object.__new__(actor_mod.TorchtitanTrainRayActor)
    actor.trainer = SimpleNamespace(parallelism=None)
    with (
        patch.object(actor_mod, "build_trainer_config", return_value=config),
        patch.object(actor_mod, "TitanTrainer", return_value=trainer),
        patch.object(torch.accelerator, "empty_cache"),
        patch.object(actor_mod.current_platform, "current_device", return_value="device"),
    ):
        assert actor._build_ref_runner(Namespace(ref_load="/ref")) == "ref-runner"
        with actor._ref_context():
            moves.append("forward")
    return part


def test_ref_parts_stay_put_under_fsdp_cpu_offload():
    moves = []
    part = _build_ref(cpu_offload=True, moves=moves)
    # FSDP refuses a forward when CPU-offloaded parameters sit on the GPU
    assert moves == ["forward"]
    assert not part.training
    assert not any(p.requires_grad for p in part.parameters())


def test_ref_parts_move_to_the_gpu_for_their_pass_without_offload():
    moves = []
    _build_ref(cpu_offload=False, moves=moves)
    assert moves == ["cpu", "device", "forward", "cpu"]
