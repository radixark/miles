from __future__ import annotations

import pytest

from miles.utils.workers.types import (
    ClusterBackend,
    HotRestartComponent,
    WorkerCommBackend,
    parse_hot_restart,
    resolve_worker_comm_backend,
)


class TestParseHotRestart:
    def test_an_empty_value_asks_for_no_hot_restart(self):
        """Every ordinary launch passes this, and it must not plan a restart of anything."""
        assert parse_hot_restart("") == []

    def test_the_two_components_are_parsed_together(self):
        """This is the only accepted value, and it names the pair the feature replaces."""
        assert parse_hot_restart("orchestration,rollout_executor") == [
            HotRestartComponent.ORCHESTRATION,
            HotRestartComponent.ROLLOUT_EXECUTOR,
        ]

    def test_whitespace_around_the_names_is_ignored(self):
        """The value travels through an env var, where a stray space is a typo rather than an intent."""
        assert parse_hot_restart(" orchestration , rollout_executor ") == [
            HotRestartComponent.ORCHESTRATION,
            HotRestartComponent.ROLLOUT_EXECUTOR,
        ]

    def test_a_component_that_cannot_be_hot_restarted_is_refused(self):
        """Everything else is taken over by the new script rather than replaced with it."""
        with pytest.raises(ValueError):
            parse_hot_restart("orchestration,trainer")

    @pytest.mark.parametrize("value", ["orchestration", "rollout_executor"])
    def test_either_component_alone_is_parsed_and_left_to_the_planner_to_refuse(self, value: str):
        """Parsing answers what the flag names; which combinations a hot restart supports is the planner's call."""
        assert parse_hot_restart(value) == [HotRestartComponent(value)]


class TestAHotRestartRequestThatNamesNoComponent:
    @pytest.mark.parametrize("value", [",", " ", " , ", ",,"])
    def test_a_non_empty_value_naming_nothing_is_refused(self, value: str):
        """It parsed to an empty list, which every caller reads as an ordinary launch of the whole fleet."""
        with pytest.raises(AssertionError, match="names no component"):
            parse_hot_restart(value)

    def test_the_refusal_names_the_components_that_could_have_been_asked_for(self):
        """The value travels through an env var, where the only fix is knowing what may be written there."""
        with pytest.raises(AssertionError, match="orchestration"):
            parse_hot_restart(",")

    def test_an_empty_value_is_still_the_ordinary_launch(self):
        """Every run that is not being hot restarted passes this, and it must not be refused."""
        assert parse_hot_restart("") == []

    def test_a_value_that_does_name_a_component_is_still_parsed(self):
        """The refusal is for a value that names nothing, not for the requests the feature exists for."""
        assert parse_hot_restart("orchestration,") == [HotRestartComponent.ORCHESTRATION]


class TestClusterBackendPredicates:
    @pytest.mark.parametrize(
        ("backend", "ray_cluster", "driver_launched", "auto_resumed"),
        [
            (ClusterBackend.RAY, True, True, False),
            (ClusterBackend.KUBERNETES, False, False, True),
            (ClusterBackend.MONARCH, False, True, False),
        ],
    )
    def test_each_backend_answers_by_the_reason_a_caller_asks(
        self, backend: ClusterBackend, ray_cluster: bool, driver_launched: bool, auto_resumed: bool
    ):
        """Callers ask why they care, not which backend it is, so a new backend answers in one place."""
        assert (
            backend.provides_ray_cluster(),
            backend.launches_workers_from_driver(),
            backend.auto_resumes_cells(),
        ) == (ray_cluster, driver_launched, auto_resumed)

    def test_monarch_speaks_rpc_by_default(self):
        """Phase 1 of the monarch backend serves every worker over the rpc wire kubernetes already speaks."""
        assert (
            resolve_worker_comm_backend(cluster_backend=ClusterBackend.MONARCH, requested=None)
            is WorkerCommBackend.RPC
        )

    def test_monarch_refuses_ray_actor_calls(self):
        with pytest.raises(AssertionError, match="not available under --cluster-backend monarch"):
            resolve_worker_comm_backend(cluster_backend=ClusterBackend.MONARCH, requested=WorkerCommBackend.RAY.value)
