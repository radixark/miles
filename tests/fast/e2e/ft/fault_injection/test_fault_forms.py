import random
from unittest.mock import MagicMock

import pytest
from tests.e2e.ft.conftest_ft.fault_injection import fault_forms
from tests.fast.e2e.ft.fault_injection.utils import NAMESPACE, RUN_ID, api_server_fault_forms, config_of, typed_cell

from miles.utils.external_utils.command_utils.helm_backend.naming import RunNames
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.types import ClusterBackend


def test_ray_draws_the_in_process_kills_for_a_trainer_cell() -> None:
    """Ray owns no pods, so the only fault it can be asked for is a kill inside the worker."""
    forms = api_server_fault_forms()["actor"]

    assert [form.name for form in forms] == [f"inject_fault:{one.value}" for one in fault_forms.FAILURE_MODES]


def test_ray_draws_the_same_kills_for_a_rollout_cell() -> None:
    """On ray an engine is supervised by an actor that can be asked to die, exactly like a trainer cell."""
    forms = api_server_fault_forms()["rollout"]

    assert [form.name for form in forms] == [f"inject_fault:{one.value}" for one in fault_forms.FAILURE_MODES]


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


def test_kubernetes_draws_pod_deletion_only_for_a_rollout_cell() -> None:
    """A k8s engine pod runs sglang as its entrypoint with no rpc server, so a kill would blow up at runtime."""
    forms_of = fault_forms.create_cell_fault_forms(
        base_url="http://control", config=config_of(ClusterBackend.KUBERNETES)
    )

    forms = forms_of["rollout"]

    assert [form.name for form in forms] == [fault_forms.DELETE_POD_FORM_NAME]


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
    assert [one["release"] for one in seen] == [RunNames.release(run_id=RUN_ID)]
    assert [one["namespace"] for one in seen] == [NAMESPACE]
    requests.post.assert_not_called()
