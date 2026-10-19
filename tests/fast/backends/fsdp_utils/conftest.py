from argparse import Namespace

import pytest

from miles.backends.fsdp_utils import actor as actor_module
from miles.backends.training_utils import parallel as parallel_module
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles.ray.train_actor import TrainRayActor
from miles.utils.ft_utils.heartbeat_utils import SimpleHeartbeat
from miles.utils.init_once import InitOnce
from miles.utils.timer import Timer


@pytest.fixture
def fsdp_three_rank_state(monkeypatch: pytest.MonkeyPatch) -> ParallelState:
    trivial = GroupInfo(rank=0, size=1, group=None)
    dp_group = GroupInfo(rank=1, size=3, group=None)
    state = ParallelState(
        intra_dp=dp_group,
        intra_dp_cp=dp_group,
        cp=trivial,
        tp=trivial,
        pp=trivial,
        ep=trivial,
        etp=trivial,
        indep_dp=trivial,
    )

    def init_common(
        self: TrainRayActor,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> None:
        self.args = args
        self.role = role

    monkeypatch.setattr(parallel_module, "_parallel_state", None)
    monkeypatch.setattr(Timer(), "start_time", {})
    monkeypatch.setattr(Timer(), "timers", {})
    monkeypatch.setattr(TrainRayActor, "_init_common", init_common)
    monkeypatch.setattr(actor_module, "create_fsdp_parallel_state", lambda _args: state)
    return state


@pytest.fixture
def fsdp_debug_actor() -> actor_module.FSDPTrainRayActor:
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor._heartbeat = SimpleHeartbeat()
    actor._init_once = InitOnce(type(actor).__name__)
    return actor
