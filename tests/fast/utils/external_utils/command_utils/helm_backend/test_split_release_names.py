from __future__ import annotations

import json

import pytest
from tests.fast.charts.utils import REPO_ROOT

from miles.utils.external_utils.command_utils.helm_backend.naming import (
    _HELM_RELEASE_NAME_MAX,
    RUN_ID_MAX_LENGTH,
    RunNames,
)
from miles.utils.workers.types import DeployComponent


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
