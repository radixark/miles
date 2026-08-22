from miles.utils.external_utils.command_utils.helm_backend.launcher import observability
from miles.utils.workers.worker_provider.kubernetes.helm.env import INSTANCE_LABEL


class TestFarewell:
    def test_tells_the_user_how_to_look_again_and_how_to_stop(self):
        """The release outlives the launcher, so both commands are the only way back to the run."""
        message = observability.farewell(namespace="rl", release="miles-run-x", workload="r-miles-run-orchestrator")

        assert "kubectl logs" in message
        assert "tear down earlier: helm uninstall -n rl miles-run-x" in message

    def test_the_farewell_offers_the_whole_release_as_well_as_the_orchestrator(self):
        """Sending a user to one statefulset makes every other pod of their run invisible."""
        message = observability.farewell(namespace="rl", release="miles-run-x", workload="r-miles-run-orchestrator")

        assert "statefulset/r-miles-run-orchestrator" in message
        assert f"--selector {INSTANCE_LABEL}=miles-run-x" in message

    def test_the_farewell_always_says_where_this_summary_stops(self):
        """A user who mistakes a vanilla launcher for monitoring will not notice what it never reported."""
        message = observability.farewell(namespace="rl", release="miles-run-x", workload="r-miles-run-orchestrator")

        assert observability._OBSERVABILITY_BOUNDARY in message
        assert "observability stack" in message

    def test_says_the_release_removes_itself_when_it_will(self):
        """A user told to uninstall by hand would keep doing it, and wonder why nothing was there."""
        message = observability.farewell(namespace="rl", release="miles-run-x", workload="r-miles-run-orchestrator")

        assert "uninstalls itself about two minutes" in message
        assert "tear down earlier: helm uninstall -n rl miles-run-x" in message
