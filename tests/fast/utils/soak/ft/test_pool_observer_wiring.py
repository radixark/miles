from pathlib import Path

import pytest
from tests.fast.utils.soak.ft.actions.test_resize import _form
from tests.utils.soak.ft.actions.inject_fault import InjectFaultForm
from tests.utils.soak.ft.observers import create_cell_observer

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.test_utils.fault_injector.actions.process import KillProcessAction
from miles.utils.workers.types import ClusterBackend


class TestPoolObserverWiring:
    def test_resize_forms_observe_the_real_cell_type_and_keep_fault_target_types(self, tmp_path: Path) -> None:
        """Pool forms select their workload while cell faults retain their hook observations."""
        observer = create_cell_observer(
            base_url="http://api",
            cell_types={"pool", "rollout"},
            forms={
                "pool": [_form(tmp_path)],
                "rollout": [
                    InjectFaultForm(base_url="http://api", action=KillProcessAction(), through_trainer_hook=True)
                ],
            },
            config=ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, namespace="isolated", run_id="same"),
        )

        assert observer.pool_workloads_of_cell_type == {"actor": "train"}
        assert observer.cell_types == {"actor", "rollout"}
        assert observer.fault_target_cell_types == frozenset({"actor", "rollout"})
        assert observer.namespace == "isolated"
        assert observer.release == "miles-run-same-all"

    @pytest.mark.parametrize("kind", ["pool", "actor"])
    def test_forms_in_the_wrong_kind_are_rejected(self, tmp_path: Path, kind: str) -> None:
        """A misplaced form cannot silently omit a pool or bypass fault observation wiring."""
        wrong_form = (
            InjectFaultForm(base_url="http://api", action=KillProcessAction()) if kind == "pool" else _form(tmp_path)
        )
        with pytest.raises(AssertionError, match="Not every form"):
            create_cell_observer(
                base_url="http://api",
                cell_types={kind},
                forms={kind: [wrong_form]},
                config=ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, namespace="isolated"),
            )


class TestResizeRequestTiming:
    @pytest.mark.parametrize("rollout_id,expected", [(0, "pending"), (1, "due"), (2, "missed")])
    def test_scheduler_admits_only_the_due_step(self, tmp_path: Path, rollout_id: int, expected: str) -> None:
        """The public request selector cannot issue a pending resize or silently skip a missed window."""
        import random
        from dataclasses import replace

        from tests.fast.utils.soak.soak_fakes import _at
        from tests.utils.soak.core.events import SoakObservationEvent
        from tests.utils.soak.ft.types import PoolTarget, ResizeStep

        from miles.utils.audit_utils.event_logger.models import TrainGroupStepEndEvent
        from miles.utils.audit_utils.process_identity import TrainerControllerProcessIdentity

        form = replace(_form(tmp_path), schedule=(ResizeStep(at_rollout=2, replicas=4),))
        observation = SoakObservationEvent(
            targets=[],
            new_sut_events=[
                TrainGroupStepEndEvent(
                    timestamp=_at(1),
                    source=TrainerControllerProcessIdentity(trainer_id="actor"),
                    rollout_id=rollout_id,
                    attempt=0,
                    role="actor",
                    cell_outcomes={},
                )
            ],
        )
        kwargs = dict(
            target=PoolTarget(identity="train", alive=True, ready=True, replicas=2),
            observation=observation,
            events=[observation],
            rng=random.Random(0),
        )
        if expected == "missed":
            with pytest.raises(AssertionError, match="scheduler missed"):
                form.maybe_create_request(**kwargs)
        else:
            request = form.maybe_create_request(**kwargs)
            if expected == "pending":
                assert request is None
            else:
                assert request is not None
                assert request.form_name == "resize:actor"
                assert request.details.step == ResizeStep(at_rollout=2, replicas=4)
