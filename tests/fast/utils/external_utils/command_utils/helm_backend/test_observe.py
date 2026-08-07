from miles.utils.external_utils.command_utils.helm_backend import observe


def _pod(name="p", phase="Running", ready=True, restarts=0, scheduling_gated=False):
    return observe.PodStatus(name=name, phase=phase, ready=ready, restarts=restarts, scheduling_gated=scheduling_gated)


def _event(pod_name="p", reason="FailedScheduling", message="no node", count=1, event_type="Warning"):
    return observe.PodEvent(pod_name=pod_name, reason=reason, message=message, count=count, type=event_type)


class TestStartupSummary:
    def test_says_nothing_is_up_before_the_first_pod(self):
        """helm returns before the scheduler has acted, and an empty summary would read as a hang."""
        assert observe.startup_summary([]) == "no pods yet"

    def test_separates_a_serving_pod_from_one_still_starting(self):
        """A Running pod that is not ready is loading a model, which is the phase users wait through."""
        summary = observe.startup_summary([_pod(), _pod(ready=False)])

        assert summary == "2 pods: 1 running, 1 starting"

    def test_counts_a_gated_pod_apart_from_a_pending_one(self):
        """A colocate pod waits for its trainer's node on purpose; calling that pending looks like a stall."""
        summary = observe.startup_summary([_pod(phase="Pending"), _pod(phase="Pending", scheduling_gated=True)])

        assert summary == "2 pods: 1 pending, 1 gated"

    def test_surfaces_failures_and_restarts(self):
        """A crash looping pod is the one thing a user must not have to go looking for."""
        summary = observe.startup_summary([_pod(phase="Failed", ready=False), _pod(restarts=3)])

        assert "1 failed" in summary and "1 restarted" in summary

    def test_omits_the_categories_that_are_empty(self):
        """A healthy run should read as one short line, not a table of zeroes."""
        assert observe.startup_summary([_pod()]) == "1 pods: 1 running"


class TestStartupEvents:
    def test_names_the_reason_a_pod_cannot_be_scheduled(self):
        """An unschedulable pod stays Pending forever, and its phase alone never says why."""
        summary = observe.startup_summary(
            [_pod(phase="Pending", ready=False)],
            [_event(reason="FailedScheduling", message="0/8 nodes are available: insufficient nvidia.com/gpu")],
        )

        assert "FailedScheduling" in summary
        assert "insufficient nvidia.com/gpu" in summary

    def test_names_the_image_a_pod_could_not_pull(self):
        """An image pull failure is the other startup hang users hit, and it is invisible in the pod phase."""
        summary = observe.startup_summary(
            [_pod(phase="Pending", ready=False)],
            [_event(reason="Failed", message="pull access denied for miles:dev", count=4)],
        )

        assert "Failed x4" in summary
        assert "pull access denied" in summary

    def test_keeps_quiet_about_the_ordinary_events_of_a_healthy_run(self):
        """Pulled, Created and Started arrive for every pod and would bury the one line that matters."""
        summary = observe.startup_summary(
            [_pod()], [_event(reason="Pulled", message="image ready", event_type="Normal")]
        )

        assert summary == "1 pods: 1 running"

    def test_reports_the_busiest_events_first_and_says_how_many_it_left_out(self):
        """A large run repeats one failure per pod, and printing all of them hides the summary line."""
        events = [_event(pod_name=f"p{index}", count=index) for index in range(1, 9)]

        summary = observe.startup_summary([_pod()], events)

        assert summary.splitlines()[1].startswith("  p8: FailedScheduling x8")
        assert summary.splitlines()[-1] == "  ... and 3 more warning events"


class TestIsSettled:
    def test_is_not_settled_while_a_pod_is_still_coming_up(self):
        """Printing the summary once more is how the user sees progress."""
        assert not observe.is_settled([_pod(), _pod(phase="Pending", ready=False)])

    def test_is_settled_once_every_pod_serves_or_failed(self):
        """A failed pod will not improve on its own, so waiting for it is waiting forever."""
        assert observe.is_settled([_pod(), _pod(phase="Failed", ready=False)])

    def test_is_not_settled_with_no_pods_at_all(self):
        """An empty list means the scheduler has not started, not that everything is fine."""
        assert not observe.is_settled([])


class TestScaleHint:
    def test_stays_quiet_for_a_run_a_user_can_read(self):
        """Advice nobody needs trains people to skip the output that matters."""
        assert observe.scale_hint([_pod() for _ in range(10)]) is None

    def test_points_a_large_run_at_the_cluster_dashboards(self):
        """A summary of hundreds of pods is worse than the tool built for it."""
        hint = observe.scale_hint([_pod() for _ in range(200)])

        assert "200 pods" in hint


class TestCommands:
    def test_follows_the_orchestrator_container_by_name(self):
        """A pod may gain a sidecar, and kubectl then refuses to guess which container to read."""
        command = observe.follow_log_command(namespace="rl", workload="r-miles-run-orchestrator")

        assert command[-2:] == ["-c", "orchestrator"]
        assert "statefulset/r-miles-run-orchestrator" in command

    def test_tells_the_user_how_to_look_again_and_how_to_stop(self):
        """The release outlives the launcher, so both commands are the only way back to the run."""
        message = observe.farewell(namespace="rl", release="miles-run-x", workload="r-miles-run-orchestrator")

        assert "kubectl logs" in message
        assert "helm uninstall -n rl miles-run-x" in message

    def test_reaches_every_pod_of_the_release_and_not_just_the_orchestrator(self):
        """A worker that crashed is the reason the run stopped, and its log is in another workload entirely."""
        command = observe.release_log_command(namespace="rl", release="miles-run-x")

        assert command[command.index("--selector") + 1] == f"{observe.RELEASE_LABEL}=miles-run-x"
        assert "--all-containers" in command
        assert "-c" not in command

    def test_the_farewell_offers_the_whole_release_as_well_as_the_orchestrator(self):
        """Sending a user to one statefulset makes every other pod of their run invisible."""
        message = observe.farewell(namespace="rl", release="miles-run-x", workload="r-miles-run-orchestrator")

        assert "statefulset/r-miles-run-orchestrator" in message
        assert f"--selector {observe.RELEASE_LABEL}=miles-run-x" in message

    def test_the_farewell_always_says_where_this_summary_stops(self):
        """A user who mistakes a vanilla launcher for monitoring will not notice what it never reported."""
        message = observe.farewell(namespace="rl", release="miles-run-x", workload="r-miles-run-orchestrator")

        assert observe.observability_boundary() in message
        assert "observability stack" in message

    def test_the_boundary_does_not_wait_for_a_large_run_to_be_mentioned(self):
        """The >50 pod hint used to be the only place it appeared, so small runs were never told."""
        small_run = [_pod() for _ in range(3)]

        assert observe.scale_hint(small_run) is None
        assert "observability stack" in observe.farewell(namespace="rl", release="x", workload="w")
