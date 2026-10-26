import pytest
from tests.fast.utils.workers.k8s_client_fakes import _ConfigException, _KubernetesClientProbe

from miles.utils.workers.k8s_client import core_v1_api


class TestCoreV1Api:
    @pytest.mark.parametrize("outside_cluster", [False, True])
    async def test_loaded_client_is_closed_after_the_consumer_finishes(
        self, kubernetes_client_probe: _KubernetesClientProbe, outside_cluster: bool
    ) -> None:
        """Both configuration routes yield a usable API and close its owned client."""
        probe = kubernetes_client_probe
        if outside_cluster:
            probe.incluster_error = _ConfigException("not in a cluster")

        async with core_v1_api() as api:
            assert api is probe.core_api
            assert probe.open_clients == 1
            assert probe.closed_clients == 0

        assert probe.incluster_loads == 1
        assert probe.kubeconfig_loads == int(outside_cluster)
        assert probe.closed_clients == 1

    async def test_consumer_failure_closes_the_client_and_preserves_the_error(
        self, kubernetes_client_probe: _KubernetesClientProbe
    ) -> None:
        """An API consumer failure must not leak its HTTP client or disappear."""
        error = RuntimeError("consumer failed")

        with pytest.raises(RuntimeError) as raised:
            async with core_v1_api():
                raise error

        assert raised.value is error
        assert kubernetes_client_probe.closed_clients == 1

    async def test_unexpected_incluster_failure_does_not_fall_back_to_another_cluster(
        self, kubernetes_client_probe: _KubernetesClientProbe
    ) -> None:
        """Only missing in-cluster configuration permits loading a local kubeconfig."""
        probe = kubernetes_client_probe
        probe.incluster_error = OSError("service account unreadable")

        with pytest.raises(OSError, match="service account unreadable"):
            async with core_v1_api():
                pytest.fail("configuration failed before opening a client")

        assert probe.kubeconfig_loads == 0
        assert probe.open_clients == 0

    async def test_failed_kubeconfig_load_propagates_without_opening_a_client(
        self, kubernetes_client_probe: _KubernetesClientProbe
    ) -> None:
        """A failed fallback cannot produce an API using unrelated default credentials."""
        probe = kubernetes_client_probe
        probe.incluster_error = _ConfigException("not in a cluster")
        probe.kubeconfig_error = _ConfigException("no kubeconfig")

        with pytest.raises(_ConfigException, match="no kubeconfig"):
            async with core_v1_api():
                pytest.fail("configuration failed before opening a client")

        assert probe.open_clients == 0
