import json
import random
import subprocess
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from tests.fast.utils.soak.utils import NAMESPACE, RUN_ID, api_server_fault_forms, config_of, typed_cell
from tests.utils.soak import fault_forms
from tests.utils.soak.process_target import ProcessIdentity, ProcessTarget
from tests.utils.soak.state import SoakActionRequest, SoakPodTarget

from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import FaultTarget
from miles.utils.workers.types import ClusterBackend, DeployComponent


@pytest.mark.parametrize("status_code", [200, 412, 503, None])
async def test_async_http_injection_uses_the_recorded_target_and_propagates_failure(
    monkeypatch: pytest.MonkeyPatch, status_code: int | None
) -> None:
    """A receipt resolves an ambiguous submission without submitting the fault twice."""
    sent: list[httpx.Request] = []

    def respond(http_request: httpx.Request) -> httpx.Response:
        sent.append(http_request)
        if http_request.method == "GET":
            return httpx.Response(status_code=200, json=receipt)
        if status_code is None:
            raise httpx.ReadTimeout("Reply lost", request=http_request)
        return httpx.Response(status_code=status_code)

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        fault_forms.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs),
    )
    form = fault_forms.InjectFaultForm(base_url="http://control", failure_mode=FailureMode.SIGKILL)
    identity = FaultTarget(cell_id="actor-7", sub_index=0, workers_hash="generation-0")
    request = SoakActionRequest(
        form_name=form.name, target=typed_cell("actor-7", "actor"), harms_cell=True, fault_target=identity
    )
    receipt = {
        "request_id": request.request_id,
        "target": identity.model_dump(mode="json"),
        "mode": "sigkill",
        "exited_pids": [42],
    }
    if status_code == 412:
        with pytest.raises(httpx.HTTPStatusError):
            await form.execute(request)
        assert len(sent) == 1
    else:
        assert await form.execute(request) == receipt
        assert len(sent) == 2
        assert sent[1].url.path == f"/api/v1/fault-receipts/{request.request_id}"
    assert sent[0].url.path == "/api/v1/cells/actor-7/inject-fault"
    assert json.loads(sent[0].content) == {
        "mode": "sigkill",
        "sub_index": 0,
        "expected_target": identity.model_dump(mode="json"),
        "request_id": request.request_id,
    }


@pytest.mark.parametrize("returncode", [0, 1])
async def test_async_pod_exec_uses_the_selected_pod_and_rejects_a_missing_process(
    monkeypatch: pytest.MonkeyPatch, returncode: int
) -> None:
    """No matching process is a failed injection, even when kubectl itself launched successfully."""
    form = fault_forms.ExecSigkillFaultForm(
        namespace=NAMESPACE, run_id=RUN_ID, container="engine", process_pattern="sglang::"
    )
    release = ReleaseName(run_id=RUN_ID, deploy_component=DeployComponent.ALL, deploy_instance_id=None).serialize()
    request = SoakActionRequest(
        target=typed_cell("rollout-engine-7", "rollout"),
        form_name=form.name,
        harms_cell=True,
        pod=SoakPodTarget(
            namespace=NAMESPACE,
            release=release,
            name="selected-pod",
            uid="selected-uid",
            process_targets={
                "engine": ProcessTarget(
                    pod_uid="selected-uid",
                    boot_id="boot",
                    pid_namespace="pid:[1]",
                    init_start_ticks=1,
                    pattern="sglang::",
                    processes=[ProcessIdentity(pid=42, start_ticks=2)],
                )
            },
        ),
    )
    receipt = {
        "request_id": request.request_id,
        "target": request.pod.process_targets["engine"].model_dump(mode="json"),
        "exited_pids": [42],
    }
    command = AsyncMock(
        return_value=subprocess.CompletedProcess(args=[], returncode=returncode, stdout=json.dumps(receipt), stderr="")
    )
    monkeypatch.setattr(fault_forms, "run_command", command)
    if returncode == 0:
        assert await form.execute(request) == receipt
    else:
        with pytest.raises(AssertionError, match="No process matching"):
            await form.execute(request)
    assert command.await_count == 1
    assert command.call_args.args[0] == [
        "kubectl",
        "exec",
        "--stdin",
        "--namespace",
        NAMESPACE,
        "selected-pod",
        "--container",
        "engine",
        "--",
        "python3",
        "-m",
        "tests.utils.soak.process_target",
        "kill",
        request.request_id,
    ]
    assert (
        ProcessTarget.model_validate_json(command.call_args.kwargs["stdin_data"])
        == request.pod.process_targets["engine"]
    )


def test_ray_draws_the_in_process_kills_for_a_trainer_cell() -> None:
    """Ray owns no pods, so the only fault it can be asked for is a kill inside the worker."""
    forms = api_server_fault_forms()["actor"]

    assert [form.name for form in forms] == [f"inject_fault:{one.value}" for one in fault_forms.FAILURE_MODES]


def test_ray_draws_a_sigkill_only_for_a_rollout_cell() -> None:
    """The engine is a subprocess, and exit, segfault and deadlock are faults only its own code can commit."""
    forms = api_server_fault_forms()["rollout"]

    assert [form.name for form in forms] == [f"inject_fault:{FailureMode.SIGKILL.value}"]


def test_kubernetes_draws_the_kills_plus_pod_deletion_for_a_trainer_cell() -> None:
    """Trainer workers are served over rpc on k8s, so pod deletion joins the kills instead of replacing them."""
    forms_of = fault_forms.create_cell_fault_forms(
        base_url="http://control", config=config_of(ClusterBackend.KUBERNETES)
    )

    forms = forms_of["actor"]

    assert [form.name for form in forms] == [
        *(f"inject_fault:{one.value}" for one in fault_forms.FAILURE_MODES),
        fault_forms.DELETE_POD_FORM_NAME,
    ]


def test_every_kill_is_its_own_form_so_the_draw_stays_uniform() -> None:
    """Folding the kills into one form would make pod deletion half of every trainer injection."""
    forms_of = fault_forms.create_cell_fault_forms(
        base_url="http://control", config=config_of(ClusterBackend.KUBERNETES)
    )

    assert len(forms_of["actor"]) == len(fault_forms.FAILURE_MODES) + 1


def test_a_kubernetes_run_without_a_namespace_fails_before_the_soak_starts() -> None:
    """kubectl would otherwise delete pods in whatever namespace the kubeconfig happens to point at."""
    with pytest.raises(AssertionError, match="needs the namespace"):
        fault_forms.create_cell_fault_forms(
            base_url="http://control", config=config_of(ClusterBackend.KUBERNETES, namespace="")
        )


def test_an_inject_fault_form_posts_the_failure_mode_it_was_built_for(monkeypatch) -> None:
    """The form's name must describe what it actually does, or a soak log explains nothing."""
    posted: list[tuple[str, dict]] = []
    requests = MagicMock()
    requests.post.side_effect = lambda url, json, timeout: posted.append((url, json)) or MagicMock()
    monkeypatch.setattr(fault_forms, "requests", requests)

    forms = api_server_fault_forms()["actor"]
    form = next(one for one in forms if one.name == f"inject_fault:{FailureMode.SEGFAULT.value}")
    form.inject(typed_cell("actor-0", "actor"), random.Random(0))

    assert posted == [("http://control/api/v1/cells/actor-0/inject-fault", {"mode": "segfault", "sub_index": 0})]


def test_the_delete_pod_form_never_reaches_the_api_server(monkeypatch) -> None:
    """Routing it through inject-fault would test the production path, not an outsider."""
    seen: list[dict] = []
    monkeypatch.setattr(fault_forms, "delete_one_pod_of_cell", lambda **kwargs: seen.append(kwargs) or "pod")
    requests = MagicMock()
    monkeypatch.setattr(fault_forms, "requests", requests)

    forms_of = fault_forms.create_cell_fault_forms(
        base_url="http://control", config=config_of(ClusterBackend.KUBERNETES)
    )
    cell = typed_cell("actor-0", "actor")
    next(
        one for one in forms_of[fault_forms.ROLLOUT_CELL_TYPE] if one.name == fault_forms.DELETE_POD_FORM_NAME
    ).inject(cell, random.Random(0))

    assert [one["cell_id"] for one in seen] == ["actor-0"]
    assert [one["release"] for one in seen] == [
        ReleaseName(run_id=RUN_ID, deploy_component=DeployComponent.ALL, deploy_instance_id=None).serialize()
    ]
    assert [one["namespace"] for one in seen] == [NAMESPACE]
    requests.post.assert_not_called()


def test_a_kubernetes_engine_can_be_crashed_in_place_as_well_as_deleted() -> None:
    """An engine pod has no rpc server to take a kill, so it is reached with kubectl exec or by deletion."""
    forms = fault_forms.create_cell_fault_forms(base_url="http://control", config=config_of(ClusterBackend.KUBERNETES))

    assert [form.name for form in forms[fault_forms.ROLLOUT_CELL_TYPE]] == [
        fault_forms.EXEC_SIGKILL_FORM_NAME,
        fault_forms.DELETE_POD_FORM_NAME,
    ]


def test_ray_gains_no_exec_form() -> None:
    """There is no pod to reach into, and its engines already take an in-process kill."""
    forms = fault_forms.create_cell_fault_forms(base_url="http://control", config=config_of(ClusterBackend.RAY))

    assert fault_forms.EXEC_SIGKILL_FORM_NAME not in [form.name for form in forms[fault_forms.ROLLOUT_CELL_TYPE]]
