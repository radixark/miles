from __future__ import annotations

import itertools
import json
import shlex

import pytest
from pydantic import ValidationError
from tests.fast.charts.utils import REPO_ROOT
from tests.fast.ray.rollout.conftest import make_args_with_sglang_config

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.external_utils.command_utils.helm_backend.launcher import entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import MooncakeInfo, MooncakePlan
from miles.utils.external_utils.command_utils.helm_backend.naming import (
    _HELM_RELEASE_NAME_MAX,
    RUN_ID_MAX_LENGTH,
    ReleaseName,
    _deploy_instance_id_budget,
)
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.naming import component_name

RUN_ID = "260101-000000-000"
NAMESPACE = "rl"

_SPLIT_COMPONENTS = [DeployComponent.PRIMARY, DeployComponent.TRAINER, DeployComponent.INFERENCE]


def _args(tmp_path, *, component: DeployComponent):
    return make_args_with_sglang_config(
        tmp_path,
        rollout_num_gpus=8,
        use_session_server=False,
        use_critic=False,
        sglang_router_port=None,
        deploy_component=component.value,
    )


def _release(component: DeployComponent) -> str:
    return ReleaseName(run_id=RUN_ID, deploy_component=component, deploy_instance_id=None).serialize()


def _object_names(tmp_path, *, component: DeployComponent) -> set[str]:
    release = _release(component)
    return {component_name(release, spec.name) for spec in compute_specs(_args(tmp_path, component=component))}


class TestTwoReleasesOfOneRun:
    def test_no_two_releases_name_the_same_object(self, tmp_path):
        """One name shared between two releases is one launch quietly upgrading another launch's workload."""
        names_by_component = {
            component: _object_names(tmp_path, component=component) for component in _SPLIT_COMPONENTS
        }

        for first, second in itertools.combinations(_SPLIT_COMPONENTS, 2):
            assert not names_by_component[first] & names_by_component[second]

    def test_the_store_flags_it_prints_carry_every_init_kwarg_the_run_was_launched_with(self):
        """The other launch pastes this line, and a dropped kwarg leaves the deployments on different protocols."""
        primary = _release(DeployComponent.PRIMARY)
        plan = MooncakePlan(init_kwargs={"master_server_address": "0.0.0.0:50051", "protocol": "tcp"}, port=50051)

        printed = entrypoint._describe_shared_object_store(plan, release=primary, namespace=NAMESPACE)

        tokens = shlex.split(printed)
        init_kwargs = json.loads(tokens[tokens.index("--mooncake-store-init-kwargs") + 1])
        assert init_kwargs == {
            "master_server_address": f"{MooncakeInfo.master_service_host(primary, NAMESPACE)}:50051",
            "protocol": "tcp",
        }

    def test_the_object_store_master_the_trainer_release_names_is_the_primary_releases_own(self):
        """The trainer launch types this address by hand, so one place has to compute it."""
        primary = _release(DeployComponent.PRIMARY)

        master = MooncakeInfo.master_service_host(primary, NAMESPACE)

        assert master == f"{component_name(primary, 'mooncake-master')}.{NAMESPACE}.svc.cluster.local"
        assert master != MooncakeInfo.master_service_host(_release(DeployComponent.TRAINER), NAMESPACE)


def test_a_run_id_ending_in_a_component_name_collides_with_nothing() -> None:
    """Every release carries its own component suffix, so no run id can spell another run's release name."""
    named = {
        ReleaseName(run_id=run_id, deploy_component=component, deploy_instance_id=None).serialize()
        for run_id in (RUN_ID, f"{RUN_ID}-trainer")
        for component in DeployComponent
    }

    assert len(named) == 2 * len(DeployComponent)


class TestTheRunIdLeavesRoomForTheComponentSuffix:
    def test_the_longest_accepted_run_id_names_a_legal_release_for_every_component(self):
        """A run id that only fits unsplit is a trap: the split launch of it fails inside helm."""
        run_id = "a" * RUN_ID_MAX_LENGTH

        for component in DeployComponent:
            name = ReleaseName(run_id=run_id, deploy_component=component, deploy_instance_id=None)

            assert len(name.serialize()) <= _HELM_RELEASE_NAME_MAX

    def test_a_longer_run_id_is_refused_where_the_release_is_named(self):
        """helm would refuse the install itself, long after the launch computed every object name from it."""
        with pytest.raises(ValidationError, match=f"at most {RUN_ID_MAX_LENGTH}"):
            ReleaseName(
                run_id="a" * (RUN_ID_MAX_LENGTH + 1), deploy_component=DeployComponent.ALL, deploy_instance_id=None
            )


def test_the_chart_accepts_exactly_the_run_ids_the_launcher_will_name_a_release_for() -> None:
    """Two copies of one bound drift apart silently: the launcher would build values helm then rejects."""
    schema = json.loads((REPO_ROOT / "charts" / "miles-run" / "values.schema.json").read_text())

    assert schema["definitions"]["RunValues"]["properties"]["id"]["maxLength"] == RUN_ID_MAX_LENGTH


class TestReleaseNameRoundTrip:
    @pytest.mark.parametrize(
        "deploy_component, deploy_instance_id",
        [
            (DeployComponent.ALL, None),
            (DeployComponent.PRIMARY, None),
            (DeployComponent.TRAINER, "a-actor"),
            (DeployComponent.INFERENCE, "inf-east"),
        ],
    )
    def test_a_release_name_parses_back_to_what_serialized_it(self, deploy_component, deploy_instance_id):
        """The launcher names a release and later reads other launches' releases back; the two must agree."""
        name = ReleaseName(run_id=RUN_ID, deploy_component=deploy_component, deploy_instance_id=deploy_instance_id)

        assert ReleaseName.parse(name.serialize()) == name

    def test_a_run_id_carrying_a_component_name_still_parses_back(self):
        """Every release ends in its component, so the rightmost one is the separator no matter what precedes it."""
        name = ReleaseName(
            run_id=f"{RUN_ID}-trainer", deploy_component=DeployComponent.INFERENCE, deploy_instance_id=None
        )

        assert ReleaseName.parse(name.serialize()) == name

    def test_a_release_of_another_chart_is_not_ours(self):
        """The launcher lists every release in the namespace, and must not read someone else's as a run of ours."""
        assert ReleaseName.parse("something-else-260101-all") is None

    def test_a_name_naming_no_component_is_not_ours(self):
        """A release this launcher wrote always names its component, so one without is not from here."""
        assert ReleaseName.parse(f"miles-run-{RUN_ID}") is None

    def test_a_legal_helm_name_this_launcher_could_never_have_written_is_refused(self):
        """parse builds the name it read, so a run id segment longer than ours is refused rather than answered."""
        with pytest.raises(ValidationError, match=f"at most {RUN_ID_MAX_LENGTH}"):
            ReleaseName.parse(f"miles-run-{'a' * (RUN_ID_MAX_LENGTH + 1)}-all")


class TestReleaseNameRefuses:
    def test_an_instance_id_carrying_a_component_name_is_refused(self):
        """It would give the release two component tokens, and parsing back would split it at the wrong one."""
        with pytest.raises(ValidationError, match="carries the component name"):
            ReleaseName(run_id=RUN_ID, deploy_component=DeployComponent.TRAINER, deploy_instance_id="all")

    def test_a_run_id_too_long_for_a_release_is_refused(self):
        """helm would refuse the install itself, long after the launch computed every object name from it."""
        with pytest.raises(ValidationError, match=f"at most {RUN_ID_MAX_LENGTH}"):
            ReleaseName(
                run_id="a" * (RUN_ID_MAX_LENGTH + 1), deploy_component=DeployComponent.ALL, deploy_instance_id=None
            )

    def test_an_instance_id_that_names_nothing_is_refused(self):
        """It serializes to a trailing dash, which parses back as no instance at all, so the two would disagree."""
        with pytest.raises(ValidationError, match="deploy_instance_id is empty"):
            ReleaseName(run_id=RUN_ID, deploy_component=DeployComponent.TRAINER, deploy_instance_id="")

    def test_an_instance_id_that_overruns_the_release_name_is_refused(self):
        """The run id fits every component, but an instance id on top of it can still overrun helm's bound."""
        with pytest.raises(ValidationError, match="leaves -1 for it"):
            ReleaseName(
                run_id="a" * RUN_ID_MAX_LENGTH,
                deploy_component=DeployComponent.INFERENCE,
                deploy_instance_id="b" * 20,
            )


class TestTheRunIdAndTheInstanceIdShareOneReleaseName:
    @pytest.mark.parametrize("run_id_length", [10, 17, 20, RUN_ID_MAX_LENGTH - 2])
    def test_the_budget_is_the_longest_instance_id_the_release_name_still_holds(self, run_id_length):
        """Two bounds derived apart drift apart, so the number the launcher reports has to be the real one."""
        run_id = "a" * run_id_length
        budget = _deploy_instance_id_budget(run_id=run_id)

        ReleaseName(run_id=run_id, deploy_component=DeployComponent.INFERENCE, deploy_instance_id="b" * budget)
        with pytest.raises(ValidationError, match=f"leaves {budget} for it"):
            ReleaseName(
                run_id=run_id, deploy_component=DeployComponent.INFERENCE, deploy_instance_id="b" * (budget + 1)
            )

    def test_a_run_id_filling_the_whole_budget_leaves_no_room_for_an_instance_id(self):
        """The longest accepted run id fits every component exactly, which is a run that cannot be split by id."""
        assert _deploy_instance_id_budget(run_id="a" * RUN_ID_MAX_LENGTH) < 1

    def test_the_default_length_run_id_is_refused_the_length_the_flag_alone_accepts(self):
        """The per-flag bound is 17, and a 17 character run id leaves 15, which nothing else would catch."""
        run_id = "260101-000000-000"
        assert _deploy_instance_id_budget(run_id=run_id) == 15

        with pytest.raises(ValidationError, match="leaves 15 for it"):
            ReleaseName(run_id=run_id, deploy_component=DeployComponent.INFERENCE, deploy_instance_id="b" * 16)

    def test_an_instance_id_inside_the_budget_is_taken(self):
        """The check must not narrow what a split run could always name."""
        ReleaseName(
            run_id="260101-000000-000", deploy_component=DeployComponent.INFERENCE, deploy_instance_id="b" * 15
        )

    def test_a_deployment_naming_no_instance_is_bounded_by_the_run_id_alone(self):
        """`all` and `primary` carry no instance id, so there is nothing here to budget for."""
        ReleaseName(run_id="a" * RUN_ID_MAX_LENGTH, deploy_component=DeployComponent.ALL, deploy_instance_id=None)
