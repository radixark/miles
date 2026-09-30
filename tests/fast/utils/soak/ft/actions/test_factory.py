import pytest
from tests.utils.soak.ft.actions import factory as factory_module
from tests.utils.soak.ft.actions.base import CellFaultForms
from tests.utils.soak.ft.actions.factory import (
    HOOK_FAULT_LIFETIME_SECONDS,
    HOOK_FAULT_MAX_DELAY_MS,
    compute_mean_interval_seconds_of_kind,
    create_cell_fault_forms,
)
from tests.utils.soak.ft.actions.inject_fault import InjectFaultForm
from tests.utils.soak.ft.actions.pod import DeletePodFaultForm, ExecSigkillFaultForm, ExecSigstopFaultForm
from tests.utils.soak.ft.types import ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE, FaultTrigger

from miles.utils.external_utils import command_utils
from miles.utils.test_utils.fault_injector.actions.process import (
    DeadlockThreadAction,
    ExitProcessAction,
    KillProcessAction,
    SegfaultProcessAction,
    StopProcessAction,
)
from miles.utils.test_utils.fault_injector.models import FaultHookName
from miles.utils.workers.types import ClusterBackend

_BEFORE_ALL_GATHER = FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER
_BEFORE_SEND = FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND


@pytest.fixture(autouse=True)
def _fixed_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory_module, "compute_base_url", lambda config: f"http://{config.cluster_backend}:1")


def _forms(backend: ClusterBackend, *triggers: FaultTrigger) -> CellFaultForms:
    config = command_utils.ExecuteTrainConfig(cluster_backend=backend, namespace="ns", run_id="run-1")
    return create_cell_fault_forms(config, triggers=frozenset(triggers))


def _hook_forms(forms: CellFaultForms, kind: str) -> list[InjectFaultForm]:
    return [form for form in forms[kind] if isinstance(form, InjectFaultForm) and form.hook_name is not None]


class TestTimerForms:
    def test_ray_timer_forms_inject_process_faults_at_random_instants(self) -> None:
        """Timer faults on Ray are direct process faults without any hook, delay or lifetime."""
        forms = _forms(ClusterBackend.RAY, FaultTrigger.TIMER)

        assert [(type(f), f.action, f.hook_name) for f in forms[ACTOR_CELL_TYPE]] == [
            (InjectFaultForm, KillProcessAction(), None),
            (InjectFaultForm, ExitProcessAction(), None),
            (InjectFaultForm, SegfaultProcessAction(), None),
        ]
        assert [(f.action, f.hook_name, f.through_trainer_hook) for f in forms[ROLLOUT_CELL_TYPE]] == [
            (KillProcessAction(), None, False)
        ]
        assert all(f.base_url == f"http://{ClusterBackend.RAY}:1" for kind_forms in forms.values() for f in kind_forms)

    def test_kubernetes_timer_forms_add_pod_faults_and_never_inject_into_engines(self) -> None:
        """Engines on Kubernetes are hit through their pods, and trainers also lose their pod."""
        forms = _forms(ClusterBackend.KUBERNETES, FaultTrigger.TIMER)

        assert [type(f) for f in forms[ACTOR_CELL_TYPE]] == [InjectFaultForm] * 3 + [DeletePodFaultForm]
        assert [type(f) for f in forms[ROLLOUT_CELL_TYPE]] == [
            ExecSigkillFaultForm,
            ExecSigstopFaultForm,
            DeletePodFaultForm,
        ]


class TestHookForms:
    @pytest.mark.parametrize("backend", [ClusterBackend.RAY, ClusterBackend.KUBERNETES])
    def test_every_trainer_hook_is_paired_with_every_trainer_hook_action(self, backend: ClusterBackend) -> None:
        """Each hook can kill, stop or deadlock the trainer that reaches it."""
        forms = _forms(backend, FaultTrigger.HOOK)

        assert [(f.hook_name, f.action) for f in forms[ACTOR_CELL_TYPE]] == [
            (hook, action)
            for hook in (_BEFORE_ALL_GATHER, _BEFORE_SEND)
            for action in (KillProcessAction(), StopProcessAction(), DeadlockThreadAction())
        ]
        assert all(f.lifetime_seconds == HOOK_FAULT_LIFETIME_SECONDS == 300.0 for f in forms[ACTOR_CELL_TYPE])
        assert not any(f.through_trainer_hook for f in forms[ACTOR_CELL_TYPE])

    def test_a_deadlock_is_never_delayed_and_other_hook_faults_may_be(self) -> None:
        """A delayed deadlock would fire off the training thread, which the fault request refuses."""
        forms = _forms(ClusterBackend.RAY, FaultTrigger.HOOK)

        delays = {(f.hook_name, f.action.kind): f.max_delay_ms for f in forms[ACTOR_CELL_TYPE]}
        assert {key: delay for key, delay in delays.items() if key[1] == DeadlockThreadAction().kind} == {
            (_BEFORE_ALL_GATHER, DeadlockThreadAction().kind): 0,
            (_BEFORE_SEND, DeadlockThreadAction().kind): 0,
        }
        assert {delay for key, delay in delays.items() if key[1] != DeadlockThreadAction().kind} == {
            HOOK_FAULT_MAX_DELAY_MS
        }
        assert HOOK_FAULT_MAX_DELAY_MS == 1000.0

    def test_ray_engines_are_killed_from_another_trainers_send_hook(self) -> None:
        """Only the send hook sits inside a transfer to the engine, and the engine itself reaches no hook."""
        forms = _forms(ClusterBackend.RAY, FaultTrigger.HOOK)

        [form] = forms[ROLLOUT_CELL_TYPE]
        assert (form.action, form.hook_name, form.through_trainer_hook) == (KillProcessAction(), _BEFORE_SEND, True)
        assert (form.lifetime_seconds, form.max_delay_ms) == (300.0, 1000.0)

    def test_kubernetes_engines_get_no_hook_forms(self) -> None:
        """Kubernetes engines cannot be reached through a trainer's api server action."""
        assert _forms(ClusterBackend.KUBERNETES, FaultTrigger.HOOK)[ROLLOUT_CELL_TYPE] == []


class TestTriggerCombination:
    @pytest.mark.parametrize("backend", [ClusterBackend.RAY, ClusterBackend.KUBERNETES])
    def test_both_triggers_draw_from_the_union_of_their_forms(self, backend: ClusterBackend) -> None:
        """Combining triggers must neither drop nor duplicate either trigger's forms."""
        both = _forms(backend, FaultTrigger.TIMER, FaultTrigger.HOOK)
        timer = _forms(backend, FaultTrigger.TIMER)
        hook = _forms(backend, FaultTrigger.HOOK)

        for kind in (ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE):
            assert [f.name for f in both[kind]] == [f.name for f in hook[kind]] + [f.name for f in timer[kind]]

    @pytest.mark.parametrize("backend", [ClusterBackend.RAY, ClusterBackend.KUBERNETES])
    def test_form_names_stay_unique_within_a_kind(self, backend: ClusterBackend) -> None:
        """The runner finds a request's form by name, so two forms sharing one would be confused."""
        forms = _forms(backend, FaultTrigger.TIMER, FaultTrigger.HOOK)

        for kind_forms in forms.values():
            names = [f.name for f in kind_forms]
            assert len(names) == len(set(names))

    def test_both_kinds_are_always_present(self) -> None:
        """A kind without forms is an empty list, so the runner can still filter by the soaked kinds."""
        assert set(_forms(ClusterBackend.KUBERNETES, FaultTrigger.HOOK)) == {ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE}

    def test_hook_forms_are_the_only_ones_carrying_a_hook(self) -> None:
        """Timer forms that armed hooks would wait for an update the timer never promised."""
        assert _hook_forms(_forms(ClusterBackend.RAY, FaultTrigger.TIMER), ACTOR_CELL_TYPE) == []
        assert len(_hook_forms(_forms(ClusterBackend.RAY, FaultTrigger.HOOK), ACTOR_CELL_TYPE)) == 6


class TestComputeMeanIntervalSecondsOfKind:
    def test_each_component_maps_to_its_cell_type_and_own_interval(self) -> None:
        """Swapping the intervals would soak trainers at the engines' rate and the reverse."""
        assert compute_mean_interval_seconds_of_kind(
            ("train", "rollout"), trainer_crash_interval_seconds=11.0, rollout_crash_interval_seconds=13.0
        ) == {ACTOR_CELL_TYPE: 11.0, ROLLOUT_CELL_TYPE: 13.0}
        assert compute_mean_interval_seconds_of_kind(
            ("rollout",), trainer_crash_interval_seconds=11.0, rollout_crash_interval_seconds=13.0
        ) == {ROLLOUT_CELL_TYPE: 13.0}
