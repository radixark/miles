import pytest
from tests.fast.utils.workers.rpc.server.test_app import _client, _Worker

from miles.utils.workers.env_vars import POD_UID_ENV_VAR
from miles.utils.workers.rpc.common.protocol import BOOT_UUID_HEADER, EXPECTED_BOOT_UUID_HEADER, POD_UID_HEADER


class TestPodIdentityHeaders:
    @pytest.mark.parametrize("path,status", [("/v1/health", 200), ("/missing", 404)])
    async def test_success_and_error_responses_identify_the_serving_pod(
        self, monkeypatch: pytest.MonkeyPatch, path: str, status: int
    ) -> None:
        """A caller can identify the responding pod even when its requested route fails."""
        monkeypatch.setenv(POD_UID_ENV_VAR, "pod-a")

        async with _client(_Worker()) as client:
            response = await client.get(path)

        assert response.status_code == status
        assert response.headers[POD_UID_HEADER] == "pod-a"
        assert response.headers[BOOT_UUID_HEADER]

    async def test_a_rejected_stale_boot_still_reports_the_current_pod(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stale fenced request must reveal the replacement identity without executing the call."""
        monkeypatch.setenv(POD_UID_ENV_VAR, "replacement-pod")

        async with _client(_Worker()) as client:
            response = await client.get("/v1/health", headers={EXPECTED_BOOT_UUID_HEADER: "old-boot"})

        assert response.status_code == 412
        assert response.headers[POD_UID_HEADER] == "replacement-pod"
        assert response.headers[BOOT_UUID_HEADER] != "old-boot"

    async def test_a_process_outside_kubernetes_omits_the_pod_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A worker without a pod UID must not invent a Kubernetes identity."""
        monkeypatch.delenv(POD_UID_ENV_VAR, raising=False)

        async with _client(_Worker()) as client:
            response = await client.get("/v1/health")

        assert POD_UID_HEADER not in response.headers
        assert response.headers[BOOT_UUID_HEADER]

    async def test_pod_identity_is_fixed_when_the_server_is_created(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Later environment mutation cannot relabel an already serving worker process."""
        monkeypatch.setenv(POD_UID_ENV_VAR, "original-pod")

        async with _client(_Worker()) as client:
            monkeypatch.setenv(POD_UID_ENV_VAR, "different-pod")
            response = await client.get("/v1/health")

        assert response.headers[POD_UID_HEADER] == "original-pod"
