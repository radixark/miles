"""The routing queues must line up with the microbatches, one entry each.

Nothing in the replay mechanism enforces that alignment: a queue serves whoever
asks next, so one unexpected forward shifts every later lookup by a microbatch.
That is not a crash and not obviously wrong in the metrics either -- adjacent
microbatches of a GRPO group share a prompt, so a shifted replay reproduces the
prompt's routing exactly and only diverges over the response.

The unexpected forward is real: a pipeline schedule infers the shapes its stages
exchange by running one forward per stage over microbatch 0, and repeats that
whenever the pass changes direction (in RL, every log-prob-then-train pair).
"""

from tests.ci.ci_register import register_cuda_ci

# Needs a GPU only because the shared replay queue hands its entries out on the
# current CUDA device; the alignment being tested is device-independent.
register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["torchtitan", "replay"])

import pytest
import torch
import torch.nn as nn

from torchtitan.models.common.moe import TokenChoiceTopKRouter

from miles.backends.torchtitan_utils import routing_replay
from miles.utils.replay_base import routing_replay_manager


class _Gate(nn.Module):
    def forward(self, x):
        return x


class _Router(TokenChoiceTopKRouter):
    """A real router with the attributes its forward reads, and nothing else.

    Subclassing the real class matters: install() finds routers by type and
    rebinds torchtitan's own forward, so the queue is exercised through the code
    that actually runs in training.
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.gate = _Gate()
        self.score_func = "softmax"
        self.num_expert_groups = None
        self.top_k = 2
        self.route_norm = False
        self.route_scale = 1.0


class _Part(nn.Module):
    """A model part with one router, like a pipeline stage's submodule."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({"0": _Router()})

    def forward(self, scores):
        _, expert_ids, _ = self.layers["0"](scores)
        return expert_ids


@pytest.fixture
def part():
    routing_replay_manager.enabled = True
    routing_replay_manager.enable_check_replay_result = False
    routing_replay_manager.stage = routing_replay.REPLAY_FORWARD
    routing_replay_manager.replays = []
    part = _Part()
    routing_replay.install([part])
    yield part
    routing_replay_manager.enabled = False
    routing_replay_manager.replays = []
    routing_replay_manager.stage = routing_replay.FALLTHROUGH


def _queue_microbatches(count):
    """Queue one entry per microbatch, entry k naming expert k."""
    replay = routing_replay_manager.replays[0]
    for k in range(count):
        replay.record(torch.full((4, 2), k, dtype=torch.long))
    return replay


def test_each_microbatch_replays_its_own_entry(part):
    _queue_microbatches(3)
    scores = torch.rand(1, 4, 8, device="cuda")

    with routing_replay.consumption_guard([part], 3):
        for expected in range(3):
            assert part(scores).unique().tolist() == [expected]


def test_the_schedules_metadata_forward_does_not_consume(part):
    _queue_microbatches(2)
    scores = torch.rand(1, 4, 8, device="cuda")

    with routing_replay.consumption_guard([part], 2):
        routing_replay.bypass_schedule_initialization([part])
        inferred = part(scores)
        # Bypassed: the router chose for itself rather than replaying entry 0.
        assert inferred.unique().tolist() != [0]

        for expected in range(2):
            assert part(scores).unique().tolist() == [expected]


def test_consecutive_passes_share_one_queue(part):
    """The queues are filled per rollout and read across its optimizer steps, so
    a pass is held to how far it advanced, not to where it ended up."""
    _queue_microbatches(4)
    scores = torch.rand(1, 4, 8, device="cuda")

    for step in range(2):
        with routing_replay.consumption_guard([part], 2):
            for offset in range(2):
                assert part(scores).unique().tolist() == [step * 2 + offset]


def test_the_bypass_ends_at_the_first_real_microbatch(part):
    """The probing forward and the backward that follows it both fall through;
    the next forward is microbatch 0 and replays again."""
    _queue_microbatches(2)
    scores = torch.rand(1, 4, 8, device="cuda")

    with routing_replay.consumption_guard([part], 2):
        routing_replay.bypass_schedule_initialization([part])
        part(scores)
        assert routing_replay_manager.stage == routing_replay.FALLTHROUGH
        for expected in range(2):
            assert part(scores).unique().tolist() == [expected]
        assert routing_replay_manager.stage == routing_replay.REPLAY_FORWARD


def test_a_recompute_pass_that_lost_its_place_is_reported(part):
    """Activation checkpointing reads a second cursor, one entry per microbatch.
    Anything else means the recompute is replaying other microbatches' routing --
    which is what the schedule's probing backward caused."""
    replay = _queue_microbatches(3)

    with pytest.raises(RuntimeError, match="recompute pass is replaying"):
        with routing_replay.consumption_guard([part], 2):
            replay.pop_forward()
            replay.pop_forward()
            replay.pop_backward()


def test_a_stray_forward_is_reported(part):
    _queue_microbatches(3)
    scores = torch.rand(1, 4, 8, device="cuda")

    with pytest.raises(RuntimeError, match="advanced 3 times over a pass of 2"):
        with routing_replay.consumption_guard([part], 2):
            for _ in range(3):
                part(scores)


def test_the_reference_model_is_left_alone(part):
    """Only the actor's routers replay; a second trainer's parts must not be
    held to the actor's queues."""
    _queue_microbatches(1)
    other = _Part()

    with routing_replay.consumption_guard([other], 7):
        pass
