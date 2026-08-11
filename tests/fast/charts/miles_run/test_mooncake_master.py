from typing import Any

from tests.fast.charts.utils import RUN_RELEASE_NAME, objects_of_kind, render_run, requires_helm

MOONCAKE_MASTER_NAME = f"{RUN_RELEASE_NAME}-miles-run-mooncake-master"


def _mooncake_object(objects: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    matches = [obj for obj in objects_of_kind(objects, kind) if obj["metadata"]["name"] == MOONCAKE_MASTER_NAME]
    assert len(matches) == 1
    return matches[0]


@requires_helm
class TestMooncakeMaster:
    def test_a_run_that_does_not_need_mooncake_renders_no_master(self) -> None:
        """The default run must not launch or expose a Mooncake master that none of its workers use."""
        objects = render_run()

        assert not any(
            obj["kind"] in {"Service", "StatefulSet"} and obj["metadata"]["name"] == MOONCAKE_MASTER_NAME
            for obj in objects
        )

    def test_the_service_and_master_process_share_names_selectors_and_ports(self) -> None:
        """The singleton master and its Service must agree on identity, selection, and both configured ports."""
        rpc_port = 51051
        metrics_port = 51052
        objects = render_run(
            "--set",
            "run.mooncake.enabled=true",
            "--set",
            f"run.mooncake.rpcPort={rpc_port}",
            "--set",
            f"run.mooncake.metricsPort={metrics_port}",
        )
        service = _mooncake_object(objects, "Service")
        stateful_set = _mooncake_object(objects, "StatefulSet")
        container = stateful_set["spec"]["template"]["spec"]["containers"][0]
        service_ports = {port["name"]: port["port"] for port in service["spec"]["ports"]}
        container_ports = {port["name"]: port["containerPort"] for port in container["ports"]}

        assert service["metadata"]["name"] == stateful_set["metadata"]["name"]
        assert service["spec"]["selector"] == stateful_set["spec"]["selector"]["matchLabels"]
        assert service["spec"]["selector"].items() <= stateful_set["spec"]["template"]["metadata"]["labels"].items()
        assert stateful_set["spec"]["serviceName"] == service["metadata"]["name"]
        assert stateful_set["spec"]["replicas"] == 1
        assert service_ports == container_ports == {"rpc": rpc_port, "metrics": metrics_port}
        assert container["command"] == [
            "mooncake_master",
            "--rpc_port",
            str(rpc_port),
            "--metrics_port",
            str(metrics_port),
        ]
