from pathlib import Path

from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.views import read_training_events
from tests.utils.soak.ft.checkers.fault_hook_dispatch import assert_hook_dispatches, assert_p2p_receiver_failures
from tests.utils.soak.ft.checkers.trainer_peer_progress import assert_trainer_peers_progress
from tests.utils.soak.ft.types import FaultTrigger

from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend

DEFAULT_FAULT_TRIGGERS: frozenset[FaultTrigger] = frozenset({FaultTrigger.TIMER})
HOOK_TRAIN_ARGS: str = "--update-weights-timeout 600 "


def resolve(requested: list[FaultTrigger] | None) -> frozenset[FaultTrigger]:
    return frozenset(requested) if requested else DEFAULT_FAULT_TRIGGERS


def compute_test_name_suffix(triggers: frozenset[FaultTrigger]) -> str:
    return "" if triggers == DEFAULT_FAULT_TRIGGERS else "_" + "_".join(sorted(triggers))


def compute_hook_train_args(triggers: frozenset[FaultTrigger]) -> str:
    return HOOK_TRAIN_ARGS if FaultTrigger.HOOK in triggers else ""


def assert_hook_evidence(
    triggers: frozenset[FaultTrigger],
    *,
    ft_components: tuple[str, ...],
    config: command_utils.ExecuteTrainConfig,
    events: list[SoakEvent],
    dump_dir: str | Path,
) -> None:
    if FaultTrigger.HOOK not in triggers:
        return
    training_events = read_training_events(events, dump_dir=dump_dir)
    assert_hook_dispatches(events, training_events=training_events)
    if "rollout" in ft_components and config.cluster_backend is ClusterBackend.RAY:
        assert_p2p_receiver_failures(events, training_events=training_events)
    if "train" in ft_components:
        assert_trainer_peers_progress(events, training_events=training_events)
