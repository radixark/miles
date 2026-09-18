import pytest

from miles.utils.external_utils.command_utils.helm_backend.launcher.hot_restart import (
    HotRestartPlan,
    compute_orchestrator_object_key,
    plan_hot_restart,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import (
    RESTART_AT_ANNOTATION,
    STATEFUL_SET_KIND,
    Manifest,
    ManifestObjectKey,
)
from miles.utils.external_utils.command_utils.helm_backend.naming import component_name
from miles.utils.workers.types import DeployComponent, HotRestartComponent

_RELEASE = "miles-run-260101-000000-000-primary"
_STAMP = "2026-08-12T09:00:00+00:00"
_ORCHESTRATOR_OBJECT = component_name(_RELEASE, "orchestrator")
_EXECUTOR_OBJECT = component_name(_RELEASE, "rollout-executor")
_BOTH = list(HotRestartComponent)


def _plan(
    *,
    components: list[HotRestartComponent] | None = None,
    deploy_component: DeployComponent | None = None,
    installed_manifest: Manifest | None = None,
) -> HotRestartPlan:
    return plan_hot_restart(
        components=components or [],
        deploy_component=deploy_component or DeployComponent.PRIMARY,
        release=_RELEASE,
        installed_manifest=installed_manifest,
    )


def _installed_manifest(
    *,
    stamped: bool,
    stamped_components: tuple[str, ...] = ("orchestrator", "rollout-executor"),
):
    return Manifest(
        namespace="rl",
        objects=[
            dict(
                kind=STATEFUL_SET_KIND,
                metadata=dict(name=component_name(_RELEASE, component)),
                spec=dict(
                    template=dict(
                        metadata=dict(
                            annotations=(
                                {RESTART_AT_ANNOTATION: _STAMP} if stamped and component in stamped_components else {}
                            )
                        ),
                        spec=dict(containers=[dict(name=component)]),
                    )
                ),
            )
            for component in ("orchestrator", "rollout-executor")
        ],
    )


class TestPlanHotRestart:
    def test_a_launch_that_asks_for_no_hot_restart_plans_nothing(self):
        """An ordinary launch must not stamp a restart annotation onto a live run."""
        plan = _plan()

        assert plan.restart_at is None
        assert plan.stamped_components == frozenset()
        assert plan.allow_diff_object_keys == frozenset()

    def test_a_restart_stamps_a_timestamp_and_replaces_both_components(self):
        """The pods only roll because the stamp moved, and the two components are replaced together."""
        plan = _plan(components=_BOTH, installed_manifest=_installed_manifest(stamped=False))

        assert plan.restart_at is not None
        assert plan.stamped_components == {"orchestrator", "rollout-executor"}
        assert plan.allow_diff_object_keys == {
            ManifestObjectKey(kind=STATEFUL_SET_KIND, name=_ORCHESTRATOR_OBJECT),
            ManifestObjectKey(kind=STATEFUL_SET_KIND, name=_EXECUTOR_OBJECT),
        }


class TestThePreconditions:
    @pytest.mark.parametrize(
        "components", [[HotRestartComponent.ORCHESTRATION], [HotRestartComponent.ROLLOUT_EXECUTOR]]
    )
    def test_either_component_alone_is_refused(self, components: list[HotRestartComponent]):
        """A new script cannot drive the executor its predecessor initialized, nor survive its replacement."""
        with pytest.raises(AssertionError, match="only supports"):
            _plan(components=components, installed_manifest=_installed_manifest(stamped=False))

    def test_a_value_that_names_one_component_twice_is_refused(self):
        """Naming one of them twice still leaves the other one out of the restart."""
        with pytest.raises(AssertionError, match="only supports"):
            _plan(
                components=[HotRestartComponent.ORCHESTRATION, HotRestartComponent.ORCHESTRATION],
                installed_manifest=_installed_manifest(stamped=False),
            )

    @pytest.mark.parametrize("deploy_component", [DeployComponent.ALL, DeployComponent.PRIMARY])
    def test_every_release_that_carries_the_orchestration_script_may_hot_restart(
        self, deploy_component: DeployComponent
    ):
        """The requirement asks for a plain single-release run too, and the machinery does not need a split one."""
        plan = _plan(
            components=_BOTH,
            deploy_component=deploy_component,
            installed_manifest=_installed_manifest(stamped=False),
        )

        assert plan.allow_diff_object_keys == {
            ManifestObjectKey(kind=STATEFUL_SET_KIND, name=_ORCHESTRATOR_OBJECT),
            ManifestObjectKey(kind=STATEFUL_SET_KIND, name=_EXECUTOR_OBJECT),
        }

    @pytest.mark.parametrize("deploy_component", [DeployComponent.TRAINER, DeployComponent.INFERENCE])
    def test_a_release_without_an_orchestration_script_cannot_hot_restart_one(self, deploy_component: DeployComponent):
        """These releases carry neither of the two components the flag replaces, so there is nothing to restart."""
        with pytest.raises(AssertionError, match="deploys neither of them"):
            _plan(components=_BOTH, deploy_component=deploy_component)

    def test_a_release_that_is_not_installed_yet_cannot_be_hot_restarted(self):
        """Installing it here would put a second orchestration script beside the one still driving the trainers."""
        with pytest.raises(AssertionError, match="is installed"):
            _plan(components=_BOTH, installed_manifest=None)

    def test_the_preconditions_are_not_checked_without_a_hot_restart(self):
        """Every ordinary launch of every release calls this, and none of them restarts anything."""
        plan = _plan(deploy_component=DeployComponent.TRAINER)

        assert plan.restart_at is None


class TestTheStampAnOrdinaryRelaunchRenders:
    def test_a_relaunch_of_a_never_hot_restarted_run_stamps_nothing(self):
        """The annotation must not appear out of nowhere, or the first relaunch would roll the pods."""
        plan = _plan(installed_manifest=_installed_manifest(stamped=False))

        assert plan.restart_at is None
        assert plan.stamped_components == frozenset()

    def test_a_relaunch_after_a_hot_restart_renders_the_stamp_it_finds_and_replaces_nothing(self):
        """Dropping it would make the pod template differ, so the diff gate refuses an ordinary relaunch forever."""
        plan = _plan(installed_manifest=_installed_manifest(stamped=True))

        assert plan.restart_at == _STAMP
        assert plan.stamped_components == {"orchestrator", "rollout-executor"}
        assert plan.allow_diff_object_keys == frozenset()

    def test_only_the_components_that_really_carry_the_stamp_are_rendered_with_it(self):
        """Rendering it onto a pool that never got it makes an ordinary relaunch a diff the gate refuses forever."""
        plan = _plan(installed_manifest=_installed_manifest(stamped=True, stamped_components=("orchestrator",)))

        assert plan.restart_at == _STAMP
        assert plan.stamped_components == {"orchestrator"}

    def test_a_hot_restart_stamps_a_value_of_its_own_over_the_installed_one(self):
        """The pod only rolls because the value moved, so a carried-forward stamp would restart nothing."""
        plan = _plan(components=_BOTH, installed_manifest=_installed_manifest(stamped=True))

        assert plan.restart_at != _STAMP

    def test_a_first_install_has_no_manifest_to_read(self):
        """Every launch calls this, including the one that installs the release."""
        assert _plan(installed_manifest=None).restart_at is None


class TestTheOrchestratorObjectKey:
    def test_it_names_the_stateful_set_of_the_release(self):
        """The entrypoint looks this key up in a rendered manifest, so a wrong name would observe nothing."""
        assert compute_orchestrator_object_key(_RELEASE) == ManifestObjectKey(
            kind=STATEFUL_SET_KIND, name=_ORCHESTRATOR_OBJECT
        )

    def test_it_is_one_of_the_objects_a_hot_restart_replaces(self):
        """A hot restart that did not relax the gate for exactly this object could never be applied."""
        plan = _plan(components=_BOTH, installed_manifest=_installed_manifest(stamped=False))

        assert compute_orchestrator_object_key(_RELEASE) in plan.allow_diff_object_keys
