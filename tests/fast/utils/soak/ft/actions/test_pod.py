import random

import pytest
from tests.fast.utils.soak.k8s_utils.pod_fakes import _api_error, _FakePodApi, _live_pod, _patch_pod_api
from tests.fast.utils.soak.soak_fakes import _at, _cell_target, _fault_target, _observation
from tests.utils.soak.core.types import SoakActionEvidence, SoakActionRequest
from tests.utils.soak.ft.actions.factory import create_cell_fault_forms
from tests.utils.soak.ft.actions.pod import BasePodFaultForm, DeletePodFaultForm
from tests.utils.soak.ft.types import CellTarget, InjectFaultDetails, PodDetails
from tests.utils.soak.k8s_utils.pod_manipulation import PodDeletedEvidence, SoakPodTarget
from tests.utils.soak.k8s_utils.pod_processes import ProcessIdentity, ProcessTarget

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.workers.types import ClusterBackend, DeployComponent

_RUN_ID = "run-a"
_RELEASE = ReleaseName(run_id=_RUN_ID, deploy_component=DeployComponent.ALL, deploy_instance_id=None).serialize()


def _engine_target(pod_uid: str, *, pattern: str = "sglang::", pid: int = 42) -> ProcessTarget:
    return ProcessTarget(
        pod_uid=pod_uid,
        boot_id="boot-a",
        pid_namespace="pidns",
        init_start_ticks=11,
        pattern=pattern,
        processes=[ProcessIdentity(pid=pid, start_ticks=7)],
    )


def _pod(name: str, *, engine: ProcessTarget | None = None, **overrides: object) -> SoakPodTarget:
    return SoakPodTarget(
        **{
            "namespace": "ns",
            "release": _RELEASE,
            "name": name,
            "uid": f"uid-{name}",
            "process_targets": {} if engine is None else {"engine": engine},
            **overrides,
        }
    )


def _rollout_with(*pods: SoakPodTarget) -> CellTarget:
    return _cell_target(kind="rollout").model_copy(update={"pods": list(pods)})


def _create(form: BasePodFaultForm, target: CellTarget, *, seed: int = 0) -> SoakActionRequest | None:
    return form.maybe_create_request(
        target=target, observation=_observation([target], at=_at(0)), events=[], rng=random.Random(seed)
    )


def _pod_request(form: BasePodFaultForm, pod: SoakPodTarget) -> SoakActionRequest:
    return SoakActionRequest(
        request_id="req-1", target=_rollout_with(pod), form_name=form.name, details=PodDetails(pod=pod)
    )


async def _execute(form: BasePodFaultForm, request: SoakActionRequest) -> list[SoakActionEvidence]:
    reported: list[SoakActionEvidence] = []
    await form.execute(request, report_applied=reported.append)
    return reported


class TestBasePodFaultForm:
    @pytest.mark.parametrize("kwargs", [{"namespace": "", "run_id": _RUN_ID}, {"namespace": "ns", "run_id": ""}])
    def test_a_form_without_namespace_or_run_id_is_refused(self, kwargs: dict[str, str]) -> None:
        """Without both the form cannot tell which release's pods it may harm."""
        with pytest.raises(AssertionError):
            DeletePodFaultForm(**kwargs)

    def test_the_release_is_the_all_component_release_of_the_run(self) -> None:
        """Pod forms target the single release that installs every component of the run."""
        assert DeletePodFaultForm(namespace="ns", run_id=_RUN_ID).release == _RELEASE
        assert DeletePodFaultForm(namespace="ns", run_id="run-b").release != _RELEASE

    def test_a_target_without_pods_is_declined(self) -> None:
        """No observed pod means there is nothing to delete."""
        assert _create(DeletePodFaultForm(namespace="ns", run_id=_RUN_ID), _rollout_with()) is None

    def test_the_pod_choice_is_seeded_among_all_candidates(self) -> None:
        """The same seed picks the same pod and different seeds can reach every pod."""
        form = DeletePodFaultForm(namespace="ns", run_id=_RUN_ID)
        target = _rollout_with(_pod("pod-a"), _pod("pod-b"))

        chosen = {_create(form, target, seed=seed).details.pod.name for seed in range(20)}

        assert chosen == {"pod-a", "pod-b"}
        assert _create(form, target, seed=3).details == _create(form, target, seed=3).details

    def test_the_request_carries_the_form_name_target_and_pod(self) -> None:
        """The request binds the chosen pod to the observed cell incarnation and this form."""
        form = DeletePodFaultForm(namespace="ns", run_id=_RUN_ID)
        target = _rollout_with(_pod("pod-a"))

        request = _create(form, target)

        assert (request.form_name, request.target, request.details) == (
            "delete_pod",
            target,
            PodDetails(pod=_pod("pod-a")),
        )

    @pytest.mark.parametrize(
        "pod",
        [
            pytest.param(_pod("pod-a", namespace="other"), id="namespace"),
            pytest.param(_pod("pod-a", release="miles-run-b"), id="release"),
        ],
    )
    async def test_a_request_for_another_release_is_refused_before_any_api_call(
        self, monkeypatch: pytest.MonkeyPatch, pod: SoakPodTarget
    ) -> None:
        """A pod outside this run's namespace and release must never be harmed."""
        api = _FakePodApi(reads=[_live_pod(pod.uid), _api_error(404)])
        _patch_pod_api(monkeypatch, api)
        form = DeletePodFaultForm(namespace="ns", run_id=_RUN_ID)

        with pytest.raises(AssertionError, match="different release"):
            await _execute(form, _pod_request(form, pod))

        assert api.calls == []

    async def test_a_request_without_pod_details_is_refused(self) -> None:
        """Details of another form cannot be misread as a pod."""
        form = DeletePodFaultForm(namespace="ns", run_id=_RUN_ID)
        details = InjectFaultDetails(fault_target=_fault_target("c"))
        request = SoakActionRequest(target=_rollout_with(), form_name=form.name, details=details)

        with pytest.raises(AssertionError, match="names no pod"):
            await _execute(form, request)


class TestDeletePodFaultForm:
    async def test_a_confirmed_deletion_is_reported_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The form reports exactly the deletion evidence of the requested pod."""
        _patch_pod_api(monkeypatch, _FakePodApi(reads=[_live_pod("uid-pod-a"), _api_error(404)]))
        form = DeletePodFaultForm(namespace="ns", run_id=_RUN_ID)

        reported = await _execute(form, _pod_request(form, _pod("pod-a")))

        assert reported == [PodDeletedEvidence(namespace="ns", pod_name="pod-a", pod_uid="uid-pod-a")]

    async def test_a_refused_deletion_reports_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A pod replaced before injection must not be counted as harmed."""
        _patch_pod_api(monkeypatch, _FakePodApi(reads=[_live_pod("uid-successor")]))
        form = DeletePodFaultForm(namespace="ns", run_id=_RUN_ID)

        with pytest.raises(AssertionError):
            await _execute(form, _pod_request(form, _pod("pod-a")))


class TestCreateCellFaultFormsPodForms:
    def test_kubernetes_timer_forms_add_pod_faults_with_unique_names(self) -> None:
        """Actors and rollouts may lose their pod, each form named once."""
        config = command_utils.ExecuteTrainConfig(
            cluster_backend=ClusterBackend.KUBERNETES, namespace="ns", run_id=_RUN_ID
        )

        forms = create_cell_fault_forms(base_url="http://api:18080", config=config)

        rollout_names = [form.name for form in forms["rollout"]]
        assert rollout_names == ["delete_pod"]
        assert "delete_pod" in [form.name for form in forms["actor"]]
        for kind_forms in forms.values():
            assert len({form.name for form in kind_forms}) == len(kind_forms)
        assert all(form.namespace == "ns" and form.release == _RELEASE for form in forms["rollout"])

    def test_ray_timer_forms_have_no_pod_faults(self) -> None:
        """Ray runs have no pods to harm, so no pod form is scheduled."""
        config = command_utils.ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY, run_id=_RUN_ID)

        forms = create_cell_fault_forms(base_url="http://api:18080", config=config)

        assert not any(isinstance(form, BasePodFaultForm) for kind_forms in forms.values() for form in kind_forms)
