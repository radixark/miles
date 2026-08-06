from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.backends.experimental.fsdp_utils.actor import (
    resolve_routing_replay_enabled,
    routing_replay_stage,
)
from miles.utils.replay_base import routing_replay_manager


@pytest.fixture(autouse=True)
def _reset_manager():
    routing_replay_manager.enabled = False
    routing_replay_manager.stage = "fallthrough"
    yield
    routing_replay_manager.enabled = False
    routing_replay_manager.stage = "fallthrough"


def test_enabled_follows_use_routing_replay():
    # --use-rollout-routing-replay sets use_routing_replay during arg validation.
    assert resolve_routing_replay_enabled(Namespace(use_routing_replay=True)) is True
    assert resolve_routing_replay_enabled(Namespace(use_routing_replay=False)) is False


def test_enabled_defaults_false_when_arg_absent():
    assert resolve_routing_replay_enabled(Namespace()) is False


def test_stage_context_sets_and_restores():
    routing_replay_manager.stage = "replay_backward"
    with routing_replay_stage("replay_forward"):
        assert routing_replay_manager.stage == "replay_forward"
    assert routing_replay_manager.stage == "replay_backward"


def test_stage_context_restores_on_exception():
    routing_replay_manager.stage = "replay_backward"
    with pytest.raises(RuntimeError):
        with routing_replay_stage("replay_forward"):
            raise RuntimeError("boom")
    assert routing_replay_manager.stage == "replay_backward"


def test_stage_context_nests():
    # The training step sets replay_forward inside a replay_backward region; leaving the
    # forward must put the region back so backward recompute reads the backward cursor.
    routing_replay_manager.stage = "fallthrough"
    with routing_replay_stage("replay_backward"):
        assert routing_replay_manager.stage == "replay_backward"
        with routing_replay_stage("replay_forward"):
            assert routing_replay_manager.stage == "replay_forward"
        assert routing_replay_manager.stage == "replay_backward"
    assert routing_replay_manager.stage == "fallthrough"


def test_ref_model_creation_does_not_install_routing_replay():
    # The ref pass runs as fallthrough. If _create_ref_model ever installed R3, the ref
    # model's Replay objects would land in the same manager.replays list, doubling its
    # length and invalidating every stream_idx during fill_replay_data.
    import inspect

    from miles.backends.experimental.fsdp_utils.actor import FSDPTrainRayActor

    source = inspect.getsource(FSDPTrainRayActor._create_ref_model)
    assert "install_routing_replay" not in source
