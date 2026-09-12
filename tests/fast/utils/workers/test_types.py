from __future__ import annotations

import pytest

from miles.utils.workers.types import HotRestartComponent, parse_hot_restart


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
