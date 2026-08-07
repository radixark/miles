import json

from tests.fast.charts.utils import NAMESPACE, RUN_RELEASE_NAME, objects_of_kind, render_run, requires_helm

from miles.utils.external_utils.command_utils.helm_backend import mooncake, naming

INIT_KWARGS = {"master_server_address": "127.0.0.1:50051", "local_hostname": "localhost"}


def _argv(**overrides: object) -> list[str]:
    kwargs = {**INIT_KWARGS, **overrides}
    return [
        "python",
        "train.py",
        "--object-store-backend",
        "mooncake",
        "--mooncake-store-init-kwargs",
        json.dumps(kwargs),
        "--lr",
        "1e-6",
    ]


def _rewritten_kwargs(argv: list[str]) -> dict[str, object]:
    return json.loads(argv[argv.index("--mooncake-store-init-kwargs") + 1])


class TestUsesMooncake:
    def test_recognises_the_backend_flag_and_its_value(self):
        """Everything else mooncake needs is only rendered when the run is detected as using it."""
        assert mooncake.uses_mooncake(_argv())

    def test_ignores_a_run_that_names_another_object_store(self):
        """A run on the default backend must not gain a master StatefulSet it never talks to."""
        argv = ["python", "train.py", "--object-store-backend", "none"]

        assert not mooncake.uses_mooncake(argv)

    def test_ignores_the_word_appearing_as_some_other_value(self):
        """The flag's value is what selects the backend, not the word turning up anywhere in the argv."""
        assert not mooncake.uses_mooncake(["python", "train.py", "--run-name", "mooncake"])

    def test_ignores_a_trailing_flag_with_no_value(self):
        """A truncated argv must read as "not mooncake" rather than index past the end of the list."""
        assert not mooncake.uses_mooncake(["python", "train.py", "--object-store-backend"])


class TestMasterPortOf:
    def test_reads_the_port_the_run_configured(self):
        """The chart publishes this port, so reading it wrong points every client at a closed socket."""
        assert mooncake.master_port_of(_argv(), default_port=0) == 50051

    def test_falls_back_when_the_address_carries_no_port(self):
        """A bare host is a legal address, and the caller's default is the only other answer available."""
        argv = _argv(master_server_address="127.0.0.1")

        assert mooncake.master_port_of(argv, default_port=7) == 7

    def test_falls_back_when_the_run_configured_nothing(self):
        """A run may leave the kwargs out entirely, which is not a reason to fail before helm is even called."""
        assert mooncake.master_port_of(["python", "train.py"], default_port=7) == 7


class TestWithClusterMaster:
    def test_points_the_master_address_at_the_in_cluster_service(self):
        """The launcher's own loopback address means nothing inside a pod, which would hang every client."""
        rewritten = mooncake.with_cluster_master(_argv(), "mooncake.myns.svc.cluster.local")

        assert _rewritten_kwargs(rewritten)["master_server_address"] == "mooncake.myns.svc.cluster.local:50051"

    def test_keeps_the_port_the_run_configured(self):
        """The Service publishes the port the values carry, so rewriting the host must not move the port."""
        rewritten = mooncake.with_cluster_master(_argv(master_server_address="1.2.3.4:60000"), "host")

        assert _rewritten_kwargs(rewritten)["master_server_address"] == "host:60000"

    def test_keeps_every_other_init_kwarg(self):
        """The kwargs are rewritten as a whole, and a dropped one changes how the store is built."""
        rewritten = mooncake.with_cluster_master(_argv(), "host")

        assert _rewritten_kwargs(rewritten)["local_hostname"] == "localhost"

    def test_leaves_the_rest_of_the_argv_untouched(self):
        """Only the address is cluster-specific; every other argument is the experiment itself."""
        rewritten = mooncake.with_cluster_master(_argv(), "host")

        assert rewritten[:5] == _argv()[:5]
        assert rewritten[-2:] == ["--lr", "1e-6"]

    def test_passes_a_non_mooncake_run_through_unchanged(self):
        """A run that never asked for mooncake has no address to rewrite, and no kwargs to invent."""
        argv = ["python", "train.py", "--lr", "1e-6"]

        assert mooncake.with_cluster_master(argv, "host") == argv


@requires_helm
class TestServiceNameCoupling:
    def test_the_host_it_builds_is_the_service_the_chart_renders(self):
        """The pods dial this name; a chart rename would leave the launcher pointing at nothing."""
        objects = render_run("--set", "run.mooncake.enabled=true")
        services = [
            obj for obj in objects_of_kind(objects, "Service") if obj["metadata"]["name"].endswith("mooncake-master")
        ]

        assert len(services) == 1
        assert mooncake.master_service_host(RUN_RELEASE_NAME, NAMESPACE) == (
            f"{services[0]['metadata']['name']}.{NAMESPACE}.svc.cluster.local"
        )

    def test_the_component_name_is_the_one_the_values_carry(self):
        """The launcher names this Service and dials it, so the two must come out of the same call."""
        objects = render_run("--set", "run.mooncake.enabled=true")
        services = [
            obj for obj in objects_of_kind(objects, "Service") if obj["metadata"]["name"].endswith("mooncake-master")
        ]

        assert services[0]["metadata"]["name"] == naming.component_name(RUN_RELEASE_NAME, mooncake.COMPONENT)
