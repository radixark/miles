from __future__ import annotations

import itertools
import json
import shlex

import pytest
from tests.fast.charts.utils import REPO_ROOT
from tests.fast.ray.rollout.conftest import make_args_with_sglang_config

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.external_utils.command_utils.helm_backend.launcher import entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import MooncakeInfo, MooncakePlan
from miles.utils.external_utils.command_utils.helm_backend.naming import (
    _HELM_RELEASE_NAME_MAX,
    RUN_ID_MAX_LENGTH,
    RunNames,
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
    return RunNames.release(run_id=RUN_ID, deploy_component=component)


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
        RunNames.release(run_id=run_id, deploy_component=component)
        for run_id in (RUN_ID, f"{RUN_ID}-trainer")
        for component in DeployComponent
    }

    assert len(named) == 2 * len(DeployComponent)


class TestTheRunIdLeavesRoomForTheComponentSuffix:
    def test_the_longest_accepted_run_id_names_a_legal_release_for_every_component(self):
        """A run id that only fits unsplit is a trap: the split launch of it fails inside helm."""
        run_id = "a" * RUN_ID_MAX_LENGTH

        for component in DeployComponent:
            assert len(RunNames.release(run_id=run_id, deploy_component=component)) <= _HELM_RELEASE_NAME_MAX

    def test_a_longer_run_id_is_refused_where_the_release_is_named(self):
        """helm would refuse the install itself, long after the launch computed every object name from it."""
        with pytest.raises(AssertionError, match=str(_HELM_RELEASE_NAME_MAX)):
            RunNames.release(run_id="a" * (RUN_ID_MAX_LENGTH + 1))


def test_the_chart_accepts_exactly_the_run_ids_the_launcher_will_name_a_release_for() -> None:
    """Two copies of one bound drift apart silently: the launcher would build values helm then rejects."""
    schema = json.loads((REPO_ROOT / "charts" / "miles-run" / "values.schema.json").read_text())

    assert schema["definitions"]["RunValues"]["properties"]["id"]["maxLength"] == RUN_ID_MAX_LENGTH
