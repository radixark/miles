"""A part's forward stages its own routers' ids; a second part on the rank (virtual PP) must not advance them."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import pytest
import torch
import torch.nn as nn

from torchtitan.models.common.moe import TokenChoiceTopKRouter

from miles.backends.torchtitan_utils import routing_replay
from miles.backends.training_utils.replay import routing_replay as shared_replay
from miles.utils import replay_base
from miles.utils.replay_base import routing_replay_manager


class _Gate(nn.Module):
    def forward(self, x):
        return x


class _Router(TokenChoiceTopKRouter):
    """A real router with the attributes its forward reads; install() finds routers by type."""

    def __init__(self):
        nn.Module.__init__(self)
        self.gate = _Gate()
        self.score_func = nn.Softmax(dim=-1)
        self.num_experts = 8
        self.top_k = 2
        self.route_norm = False
        self.route_norm_epsilon = 1e-20
        self.route_scale = 1.0
        self.aux_loss = None
        self.register_buffer("tokens_per_expert_E", torch.zeros(8), persistent=False)


class _Part(nn.Module):
    """A model part with one router at decoder layer ``layer``."""

    def __init__(self, layer: int):
        super().__init__()
        self.layers = nn.ModuleDict({str(layer): _Router()})

    def forward(self, scores):
        (router,) = self.layers.values()
        _, expert_ids, _ = router(scores)
        return expert_ids


@pytest.fixture
def cpu_replay(monkeypatch):
    """The replay hands its entries out on the current accelerator; this suite runs on CPU."""
    monkeypatch.setattr(replay_base.current_platform, "current_device", lambda: torch.device("cpu"))
    routing_replay_manager.enabled = True
    routing_replay_manager.enable_check_replay_result = False
    routing_replay_manager.stage = shared_replay.REPLAY_FORWARD
    routing_replay_manager.replays = []
    yield
    routing_replay_manager.enabled = False
    routing_replay_manager.replays = []
    routing_replay_manager.stage = shared_replay.FALLTHROUGH


def _queue_microbatches(count: int) -> None:
    """Queue ``count`` entries per replay; replay k's entry for microbatch m names expert 4k + m."""
    for k, replay in enumerate(routing_replay_manager.replays):
        # record() pins a host buffer, which needs an accelerator
        replay.top_indices_list = [
            torch.full((4, 2), 4 * k + microbatch, dtype=torch.long) for microbatch in range(count)
        ]


def test_each_part_stages_only_its_own_routers(cpu_replay):
    parts = [_Part(0), _Part(1)]
    routing_replay.install(parts)
    _queue_microbatches(2)
    scores = torch.rand(4, 8)

    with routing_replay.consumption_guard(parts, 2):
        for microbatch in range(2):
            for k, part in enumerate(parts):
                assert part(scores).unique().tolist() == [4 * k + microbatch]
    assert all(replay.staged_forward is None for replay in routing_replay_manager.replays)


def test_the_train_pass_stages_the_forward_and_leaves_the_backward_cursor(cpu_replay):
    part = _Part(0)
    routing_replay.install([part])
    _queue_microbatches(2)
    (replay,) = routing_replay_manager.replays
    scores = torch.rand(4, 8)

    routing_replay_manager.stage = shared_replay.REPLAY_BACKWARD
    with routing_replay.consumption_guard([part], 2):
        for microbatch in range(2):
            assert part(scores).unique().tolist() == [microbatch]
            assert replay.staged_forward is None
    assert replay.backward_index == 0
