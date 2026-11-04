import pytest
from kubernetes_asyncio import client
from tests.fast.utils.soak.k8s_utils.pod_fakes import _api_error, _FakePodApi, _live_pod, _patch_pod_api, _pod_target
from tests.utils.soak.k8s_utils import pod_manipulation
from tests.utils.soak.k8s_utils.pod_manipulation import PodDeletedEvidence, delete_observed_pod


class TestDeleteObservedPod:
    async def test_the_delete_carries_the_observed_uid_and_resource_version_as_preconditions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The API server must refuse the delete if the pod was replaced or changed since it was read."""
        api = _FakePodApi(reads=[_live_pod("uid-a", resource_version="rv-9"), _api_error(404)])
        _patch_pod_api(monkeypatch, api)

        await delete_observed_pod(_pod_target())

        [(_, delete)] = [call for call in api.calls if call[0] == "delete"]
        assert delete["name"] == "pod-a" and delete["namespace"] == "ns"
        assert delete["body"].preconditions.uid == "uid-a"
        assert delete["body"].preconditions.resource_version == "rv-9"

    async def test_a_404_after_the_delete_is_evidence_for_the_observed_pod(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once the pod name is gone the evidence names the exact pod incarnation deleted."""
        _patch_pod_api(monkeypatch, _FakePodApi(reads=[_live_pod("uid-a"), _api_error(404)]))

        evidence = await delete_observed_pod(_pod_target())

        assert evidence == PodDeletedEvidence(namespace="ns", pod_name="pod-a", pod_uid="uid-a")

    async def test_it_waits_while_the_original_uid_is_still_terminating(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A still-present original pod is not yet deleted, so confirmation keeps polling."""
        api = _FakePodApi(
            reads=[
                _live_pod("uid-a"),
                _live_pod("uid-a", deleting=True),
                _live_pod("uid-a", deleting=True),
                _live_pod("uid-b"),
            ]
        )
        _patch_pod_api(monkeypatch, api)

        evidence = await delete_observed_pod(_pod_target())

        assert [kind for kind, _ in api.calls] == ["read", "delete", "read", "read", "read"]
        assert evidence.pod_uid == "uid-a"

    async def test_a_replacement_pod_under_the_same_name_confirms_the_deletion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A StatefulSet successor with a new UID proves the observed incarnation is gone."""
        _patch_pod_api(monkeypatch, _FakePodApi(reads=[_live_pod("uid-a"), _live_pod("uid-b")]))

        assert (await delete_observed_pod(_pod_target())).pod_uid == "uid-a"

    @pytest.mark.parametrize(
        "before",
        [
            pytest.param(_live_pod("uid-other"), id="replaced_before_delete"),
            pytest.param(_live_pod("uid-a", deleting=True), id="already_deleting"),
            pytest.param(_live_pod("uid-a", resource_version=None), id="no_resource_version"),
        ],
    )
    async def test_a_pod_that_fails_a_precondition_is_never_deleted(
        self, monkeypatch: pytest.MonkeyPatch, before: object
    ) -> None:
        """Replaced, already-deleting or unversioned pods are refused before any delete is sent."""
        api = _FakePodApi(reads=[before])
        _patch_pod_api(monkeypatch, api)

        with pytest.raises(AssertionError):
            await delete_observed_pod(_pod_target())

        assert [kind for kind, _ in api.calls] == ["read"]

    async def test_a_target_without_an_observed_uid_is_refused_before_any_api_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a UID the delete could hit whichever pod holds the name now."""
        api = _FakePodApi(reads=[_live_pod("")])
        _patch_pod_api(monkeypatch, api)

        with pytest.raises(AssertionError, match="observed UID"):
            await delete_observed_pod(_pod_target(uid=""))

        assert api.calls == []

    async def test_a_404_on_the_first_read_is_raised_not_counted_as_deleted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pod that vanished before injection was not deleted by the soak."""
        api = _FakePodApi(reads=[_api_error(404)])
        _patch_pod_api(monkeypatch, api)

        with pytest.raises(client.ApiException):
            await delete_observed_pod(_pod_target())

        assert [kind for kind, _ in api.calls] == ["read"]

    async def test_a_rejected_delete_is_raised_without_evidence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 409 precondition failure means nothing was deleted by this request."""
        api = _FakePodApi(reads=[_live_pod("uid-a"), _api_error(404)], delete_error=_api_error(409))
        _patch_pod_api(monkeypatch, api)

        with pytest.raises(client.ApiException) as info:
            await delete_observed_pod(_pod_target())

        assert info.value.status == 409
        assert [kind for kind, _ in api.calls] == ["read", "delete"]

    async def test_a_non_404_error_while_confirming_is_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only a 404 proves absence; a 500 during confirmation is not a deletion."""
        _patch_pod_api(monkeypatch, _FakePodApi(reads=[_live_pod("uid-a"), _api_error(500)]))

        with pytest.raises(client.ApiException) as info:
            await delete_observed_pod(_pod_target())

        assert info.value.status == 500

    async def test_a_pod_that_never_goes_away_times_out_without_evidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A delete the kubelet never completes must end as a bounded failure, not a success."""
        monkeypatch.setattr(pod_manipulation, "KUBECTL_TIMEOUT_SECONDS", 0.5)
        _patch_pod_api(monkeypatch, _FakePodApi(reads=[_live_pod("uid-a")]))

        with pytest.raises(TimeoutError):
            await delete_observed_pod(_pod_target())
