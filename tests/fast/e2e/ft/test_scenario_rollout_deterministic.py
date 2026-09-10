import json
import random
import shlex
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.e2e.ft.conftest_ft import scenario_rollout_deterministic
from tests.e2e.ft.conftest_ft.modes import MODES
from tests.e2e.ft.conftest_ft.scenario_rollout_deterministic import (
    DETERMINISTIC_INFERENCE_ENV_VARS,
    NUM_ROLLOUTS,
    TERMINAL_FAULT_FREE_ROLLOUTS,
    _build_args,
    _compute_fault_progress_windows,
)
from tests.fast.utils.soak.utils import typed_cell
from tests.utils.soak.config import SoakPolicy, create_tail_policy
from tests.utils.soak.core import SoakActionScheduler
from tests.utils.soak.fault_forms import InjectFaultForm
from tests.utils.soak.state import SoakAdmissionClosedEvent, SoakObservation, SoakScheduleEvent

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.ray.specs.train import compute_trainer_num_cells
from miles.utils.audit_utils.event_logger.models import TrainGroupStepEndEvent
from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity
from miles.utils.external_utils import command_utils
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import FaultTarget
from miles.utils.workers.types import ClusterBackend

_BASE = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)


def test_environment_evidence_uses_the_actual_non_independent_trainer_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rollout-only FT keeps all trainer ranks in one cell when checking environment evidence."""
    mode = MODES["kill_rollout__dp4"]
    args = SimpleNamespace(
        indep_dp=False, actor_num_nodes=mode.train_num_nodes, actor_num_gpus_per_node=mode.train_gpus_per_node
    )
    num_cells = compute_trainer_num_cells(args, role="actor")
    expected = {
        (cell, rank)
        for cell in range(num_cells)
        for rank in range(mode.train_num_nodes * mode.train_gpus_per_node // num_cells)
    }
    observed: list[set[tuple[int, int]]] = []
    monkeypatch.setattr(scenario_rollout_deterministic, "read_events", lambda path: [])
    monkeypatch.setattr(scenario_rollout_deterministic, "assert_identified_engine_checksums", lambda **kwargs: None)
    monkeypatch.setattr(scenario_rollout_deterministic, "assert_published_weight_checksums", lambda events: None)
    monkeypatch.setattr(scenario_rollout_deterministic, "compare_deterministic_sides", lambda **kwargs: None)
    monkeypatch.setattr(
        scenario_rollout_deterministic,
        "assert_deterministic_environment",
        lambda events, **kwargs: observed.append(kwargs["trainer_ranks"]),
    )

    scenario_rollout_deterministic._compare(str(tmp_path), mode)

    assert num_cells == 1
    assert observed == [expected, expected]


def test_the_actual_backend_owns_the_single_api_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Kubernetes override must not inherit API setup from the default Ray configuration."""
    monkeypatch.setattr(
        command_utils, "default_config", lambda: command_utils.ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY)
    )
    config = command_utils.ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, namespace="explicit")
    argv = shlex.split(_build_args(MODES["kill_rollout__dp4"], str(tmp_path), config=config))
    assert argv.count("--api-server-port") == 1
    assert argv.count("--api-server-host") == 1
    assert "--fault-witness-enable" in argv


def test_rollout_deterministic_uses_the_shared_deterministic_recipe_without_true_on_policy(tmp_path: Path) -> None:
    """The rollout-FT comparison must use pure deterministic inference rather than true-on-policy."""
    args = _build_args(MODES["kill_rollout__dp4"], dump_dir=str(tmp_path))

    assert "--sglang-enable-deterministic-inference " in args
    assert "--sglang-attention-backend flashinfer " in args
    assert '"SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "false"' in args
    assert "--rollout-health-check-interval 1.0 " in args
    assert "--deterministic-mode " in args
    assert "--context-parallel-size " not in args
    assert "--true-on-policy-mode" not in args
    assert "--sglang-true-on-policy-contract" not in args
    assert "--true-on-policy-contract" not in args
    assert "--sglang-attention-backend fa3" not in args
    assert "--recompute-logprobs-via-prefill" not in args
    argv = shlex.split(args)
    assert argv.count("--inference-env-vars") == 1
    assert json.loads(argv[argv.index("--inference-env-vars") + 1]) == DETERMINISTIC_INFERENCE_ENV_VARS
    assert argv[argv.index("--update-weight-transfer-mode") + 1] == "p2p"
    assert argv[argv.index("--sglang-router-policy") + 1] == "round_robin"
    assert argv[argv.index("--ft-components") + 1] == "rollout"
    assert argv[argv.index("--ft-components") + 2].startswith("--")


def test_deterministic_rollout_rejects_colocated_topology(tmp_path: Path) -> None:
    """The comparison must exercise the disaggregated transfer path on both sides."""
    with pytest.raises(AssertionError, match="disaggregated"):
        _build_args(MODES["kill_rollout__dp4__colocate"], dump_dir=str(tmp_path))


class TestComputeFaultProgressWindows:
    def test_two_crashes_before_any_rollout_finished_share_one_window(self) -> None:
        """Effects before the first completion share a progress window without proving an active generation hit."""
        crashed = _compute_fault_progress_windows(
            injected_at=[_BASE, _BASE + timedelta(seconds=10)], rollout_completions=[(0, _BASE + timedelta(hours=1))]
        )

        assert crashed == {0}

    def test_crashes_on_either_side_of_a_finished_rollout_are_two_windows(self) -> None:
        """A completed rollout separates two fault progress windows."""
        crashed = _compute_fault_progress_windows(
            injected_at=[_BASE, _BASE + timedelta(seconds=20)],
            rollout_completions=[(0, _BASE + timedelta(seconds=10))],
        )

        assert crashed == {0, 1}

    def test_a_run_with_no_crashes_has_no_windows(self) -> None:
        """An empty result must not read as coverage; the caller's floor is what rejects it."""
        assert _compute_fault_progress_windows(injected_at=[], rollout_completions=[(0, _BASE)]) == set()

    def test_repeated_metrics_from_one_rollout_count_as_one_completed_rollout(self) -> None:
        """Repeated metric events must not advance an injection by multiple rollout windows."""
        crashed = _compute_fault_progress_windows(
            injected_at=[_BASE + timedelta(seconds=20)],
            rollout_completions=[
                (0, _BASE + timedelta(seconds=10)),
                (0, _BASE + timedelta(seconds=15)),
            ],
        )

        assert crashed == {1}


@pytest.mark.parametrize("case", ["normal", "discarded", "other_trainer", "closed", "absent"])
def test_rollout_fault_window_requires_normal_progress_and_stays_closed(case: str) -> None:
    """Only the actor's normal training opens admission, and closure permanently overrides progress."""
    step = TrainGroupStepEndEvent(
        timestamp=_BASE,
        source=TrainerControllerProcessIdentity(trainer_id="critic" if case == "other_trainer" else "actor"),
        rollout_id=0,
        cell_outcomes={
            0: [TrainStepOutcome.DISCARDED_SHOULD_RETRY if case == "discarded" else TrainStepOutcome.NORMAL]
        },
    )
    cells = [typed_cell(f"rollout-{index}", "rollout") for index in range(2)]
    observation = SoakObservation(
        cells=cells,
        fault_targets={
            cell["metadata"]["name"]: FaultTarget(
                cell_id=cell["metadata"]["name"], sub_index=0, workers_hash="generation-0"
            )
            for cell in cells
        },
        training_events=[] if case == "absent" else [step, step],
    )
    form = InjectFaultForm(base_url="http://control", failure_mode=FailureMode.SIGKILL)
    scheduler = SoakActionScheduler(
        rng=random.Random(0),
        mean_intervals={"rollout": 1},
        forms={"rollout": [form]},
        policy=SoakPolicy(start_after_rollout_id=0),
    )
    events = [SoakScheduleEvent(due_of_type={"rollout": 0}), observation]
    if case == "closed":
        events.append(SoakAdmissionClosedEvent())
    assert (scheduler.choose(events=events, now=1) is not None) == (case == "normal")
    tail = create_tail_policy(num_rollout=NUM_ROLLOUTS, min_tail_rollouts=TERMINAL_FAULT_FREE_ROLLOUTS)
    assert NUM_ROLLOUTS - tail.close_after_rollout_id - 1 >= 3
