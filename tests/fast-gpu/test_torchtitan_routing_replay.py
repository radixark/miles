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
register_cuda_ci(
    est_time=60, suite="stage-b-2-gpu-h200", labels=["torchtitan", "replay"], hardware=["hopper"], num_gpus=2
)

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from torchtitan.models.common.moe import TokenChoiceTopKRouter

from miles.backends.torchtitan_utils import routing_replay
from miles.backends.training_utils.replay import routing_replay as shared_replay
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
        self._debug_force_load_balance = False


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
    routing_replay_manager.stage = shared_replay.REPLAY_FORWARD
    routing_replay_manager.replays = []
    part = _Part()
    routing_replay.install([part])
    yield part
    routing_replay_manager.enabled = False
    routing_replay_manager.replays = []
    routing_replay_manager.stage = shared_replay.FALLTHROUGH


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
        assert routing_replay_manager.stage == shared_replay.FALLTHROUGH
        for expected in range(2):
            assert part(scores).unique().tolist() == [expected]
        assert routing_replay_manager.stage == shared_replay.REPLAY_FORWARD


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


def test_a_record_pass_queues_one_entry_per_microbatch(part):
    """Plain --use-routing-replay runs the log-prob pass in the record stage:
    nothing is popped, one entry lands per microbatch, and the train pass
    replays them. The guard held every pass to a pop per microbatch, so that
    first pass raised before any entry existed."""
    scores = torch.rand(1, 4, 8, device="cuda")
    replay = routing_replay_manager.replays[0]

    with shared_replay.stage(shared_replay.RECORD), routing_replay.consumption_guard([part], 2):
        for _ in range(2):
            part(scores)

    assert len(replay.top_indices_list) == 2
    assert replay.forward_index == 0
    with routing_replay.consumption_guard([part], 2):
        for _ in range(2):
            part(scores)


def test_a_record_pass_short_of_an_entry_is_reported(part):
    scores = torch.rand(1, 4, 8, device="cuda")

    with pytest.raises(RuntimeError, match="recorded 1 entries and popped 0 over a record pass of 2"):
        with shared_replay.stage(shared_replay.RECORD), routing_replay.consumption_guard([part], 2):
            part(scores)


def test_a_fallthrough_pass_leaves_the_queues_alone(part):
    """The reference model's pass and a probe forward fall through: the router
    chooses for itself and the queues stay as they were."""
    _queue_microbatches(2)
    scores = torch.rand(1, 4, 8, device="cuda")
    replay = routing_replay_manager.replays[0]

    with shared_replay.stage(shared_replay.FALLTHROUGH), routing_replay.consumption_guard([part], 3):
        for _ in range(3):
            part(scores)

    assert (replay.forward_index, replay.backward_index, len(replay.top_indices_list)) == (0, 0, 2)
    with pytest.raises(RuntimeError, match="touched its queue"):
        with shared_replay.stage(shared_replay.FALLTHROUGH), routing_replay.consumption_guard([part], 1):
            replay.pop_forward()


def test_an_uninstalled_second_trainer_is_left_alone(part):
    """A second trainer's parts without replay hooks are not held to the
    actor's queues."""
    _queue_microbatches(1)
    other = _Part()

    with routing_replay.consumption_guard([other], 7):
        pass


def test_the_reference_model_replays_the_same_routes_as_the_actor(part):
    """The reference trainer's routers are installed too (torchtitan actor
    _build_ref_runner) and register under the same layer index, so fill()
    records the rollout's stream into both queues and the reference pass
    replays the routes the actor replays: identical weights then give
    identical log-probs and a step-0 KL of 0 to the reference. Before, the
    reference fell through and routed on its own (kl_loss 0.0096 at step 0 on
    GLM-5.3-Flash with identical weights). The guard around each pass holds
    only that trainer's queues: the reference pass leaves the actor's queue
    where it was and must not be reported for it."""
    reference = _Part()
    routing_replay.install([reference])
    actor_replay, reference_replay = routing_replay_manager.replays
    assert actor_replay is not reference_replay
    assert actor_replay.stream_idx == reference_replay.stream_idx == 0
    for replay in (actor_replay, reference_replay):
        for k in range(2):
            replay.record(torch.full((4, 2), k, dtype=torch.long))
    scores = torch.rand(1, 4, 8, device="cuda")

    args = SimpleNamespace(use_rollout_routing_replay=True)
    with shared_replay.stage(shared_replay.ref_log_prob_stage(args)):
        with routing_replay.consumption_guard([reference], 2):
            reference_routes = [reference(scores).unique().tolist() for _ in range(2)]
    with shared_replay.stage(shared_replay.log_prob_stage(args)):
        with routing_replay.consumption_guard([part], 2):
            actor_routes = [part(scores).unique().tolist() for _ in range(2)]
    assert reference_routes == actor_routes == [[0], [1]]
    # the reference pass consumed its own queue, not the actor's
    assert (reference_replay.forward_index, actor_replay.forward_index) == (2, 2)


def test_the_reference_pass_falls_through_without_rollout_replay():
    """Recording-only routing replay (no rollout routes) leaves the reference
    pass as it was: the stage is FALLTHROUGH, and so is the disabled case."""
    args = SimpleNamespace(use_rollout_routing_replay=False)
    routing_replay_manager.enabled = True
    try:
        assert shared_replay.ref_log_prob_stage(args) == shared_replay.FALLTHROUGH
        assert shared_replay.log_prob_stage(args) == shared_replay.RECORD
        on = SimpleNamespace(use_rollout_routing_replay=True)
        assert shared_replay.ref_log_prob_stage(on) == shared_replay.REPLAY_FORWARD
    finally:
        routing_replay_manager.enabled = False
    assert shared_replay.ref_log_prob_stage(on) == shared_replay.FALLTHROUGH


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
