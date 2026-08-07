from miles.utils.external_utils.command_utils.helm_backend import naming

LONGEST_RUN_ID = "a" * 40


class TestReleaseName:
    def test_a_release_is_the_chart_name_and_the_run_id(self):
        """The launcher finds a run's release again from the run id alone, so the rule is fixed."""
        assert naming.release_name("260101-000000-000") == "miles-run-260101-000000-000"

    def test_the_same_run_id_always_names_the_same_release(self):
        """Relaunching a run upgrades its release; a fresh name would deploy a second copy instead."""
        assert naming.release_name(LONGEST_RUN_ID) == naming.release_name(LONGEST_RUN_ID)


class TestComponentName:
    def test_an_object_is_the_release_the_chart_name_and_the_component(self):
        """Every object of a run is traceable to the release that made it."""
        assert naming.component_name("myrun", "orchestrator") == "myrun-miles-run-orchestrator"

    def test_a_release_that_already_carries_the_chart_name_is_not_told_it_twice(self):
        """The launcher's own releases start with the chart name, and doubling it wastes the budget."""
        assert naming.component_name("miles-run-260101", "orchestrator") == "miles-run-260101-orchestrator"

    def test_a_name_leaves_room_for_every_suffix_kubernetes_appends_below_it(self):
        """A pool name grows a cell index and then a revision hash, and a label value stops at 63."""
        name = naming.component_name("a" * 200, "orchestrator")
        appended = len(naming.LONGEST_CELL_INDEX_SUFFIX) + len(naming.LONGEST_REVISION_HASH_SUFFIX)

        assert len(name) + appended <= naming.MAX_OBJECT_NAME_LENGTH

    def test_the_component_survives_a_release_long_enough_to_fill_the_budget(self):
        """Truncating the component instead of the release would render two workloads under one name."""
        assert naming.component_name("a" * 200, "orchestrator").endswith("-orchestrator")

    def test_two_components_of_one_run_never_collapse_onto_the_same_name(self):
        """Truncating a name already at the limit silently merges two workloads into one object."""
        release = "a" * 200

        assert naming.component_name(release, "leader") != naming.component_name(release, "logger")

    def test_a_truncated_prefix_never_ends_on_the_separator(self):
        """A doubled dash is legal but reads as an empty segment, and drifts from the recorded names."""
        for length in range(1, 60):
            assert "--" not in naming.component_name("b" * length, "orchestrator")

    def test_the_same_release_and_component_always_name_the_same_object(self):
        """helm upgrade replaces an object in place only while its name is unchanged."""
        assert naming.component_name("miles-run-x", "trainer-actor") == naming.component_name(
            "miles-run-x", "trainer-actor"
        )


class TestStaticWorkerHost:
    def test_a_static_cell_is_reached_through_its_own_pod_of_the_headless_service(self):
        """A pool of session servers is several addresses, and pod zero can answer only one of them."""
        assert naming.static_worker_host("myrun", "session-server", 1) == (
            "myrun-miles-run-session-server-1.myrun-miles-run-session-server"
        )
