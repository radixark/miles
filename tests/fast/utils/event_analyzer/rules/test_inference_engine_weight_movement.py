from typing import Any

import pytest
from tests.fast.utils.event_analyzer.rules.weight_event_fakes import (
    make_checksum,
    make_result,
    make_step_end,
    make_trainer_args,
)

from miles.utils.audit_utils.event_analyzer.rules import (
    inference_engine_weight_checksum_consistency,
    inference_engine_weight_checksum_coverage,
)
from miles.utils.audit_utils.event_analyzer.rules.inference_engine_weight_movement import check
from miles.utils.audit_utils.event_logger.models import Event, InferenceEngineWeightChecksumEvent


def _version(
    version: int,
    tensors: dict[str, str],
    *,
    second: float | None = None,
    model_id: str | None = None,
    load_state_timestamp: float = 0.0,
) -> InferenceEngineWeightChecksumEvent:
    return make_checksum(
        second=float(version) if second is None else second,
        update_id=f"u{version}",
        weight_version=version,
        snapshots={"a": ("hash-a", tensors)},
        model_id=model_id,
        load_state_timestamp=load_state_timestamp,
    )


def _settled(*events: Event, **arg_overrides: Any) -> list[Event]:
    return [make_trainer_args(**arg_overrides), *events, make_step_end(second=100.0)]


class TestCheck:
    def test_every_tensor_changing_between_adjacent_versions_passes(self) -> None:
        """Movement of every tensor is the healthy optimizer signature."""
        events = _settled(_version(1, {"w": "1", "b": "1"}), _version(2, {"w": "2", "b": "2"}))

        assert check(events) == []

    def test_one_unchanged_tensor_is_reported_with_its_versions(self) -> None:
        """A tensor that stayed bit-identical proves its slice of the update never arrived."""
        events = _settled(_version(1, {"w": "1", "b": "1"}), _version(2, {"w": "2", "b": "1"}))

        [issue] = check(events)

        assert issue.weight_version_before == 1
        assert issue.weight_version_after == 2
        assert issue.description == "unchanged tensor checksums: ['b']"

    def test_several_unchanged_tensors_are_all_named(self) -> None:
        """Every stale tensor is listed so one repair does not hide the next."""
        events = _settled(_version(1, {"w": "1", "b": "1", "c": "1"}), _version(2, {"w": "1", "b": "2", "c": "1"}))

        [issue] = check(events)

        assert issue.description == "unchanged tensor checksums: ['c', 'w']"

    @pytest.mark.parametrize("after", [{"w": "2"}, {"w": "2", "b": "2", "extra": "2"}, {"w": "2", "renamed": "2"}])
    def test_a_changed_tensor_set_is_reported(self, after: dict[str, str]) -> None:
        """A dropped, added or renamed tensor makes the versions incomparable and must fail."""
        events = _settled(_version(1, {"w": "1", "b": "1"}), _version(2, after))

        [issue] = check(events)

        assert issue.description == "tensor set changed"

    def test_versions_are_paired_in_numeric_order_not_arrival_order(self) -> None:
        """Out-of-order logging still compares 1-2 and 2-3, not 3-1."""
        events = _settled(
            _version(3, {"w": "3"}, second=1.0),
            _version(1, {"w": "1"}, second=2.0),
            _version(2, {"w": "2"}, second=3.0),
        )

        assert check(events) == []

    def test_only_adjacent_published_versions_are_compared(self) -> None:
        """A gap in versions pairs the neighbours that exist, so v1 and v3 are compared."""
        events = _settled(_version(1, {"w": "1"}), _version(3, {"w": "1"}))

        [issue] = check(events)

        assert (issue.weight_version_before, issue.weight_version_after) == (1, 3)

    def test_two_policies_are_never_paired_with_each_other(self) -> None:
        """Identical weights across two models are expected and must not count as a stale tensor."""
        events = _settled(
            _version(1, {"w": "same"}, model_id="solver"),
            _version(2, {"w": "same"}, model_id="verifier"),
        )

        assert check(events) == []

    def test_a_load_state_starts_a_new_lineage(self) -> None:
        """Reloading a checkpoint may legitimately restore earlier bytes, so lineages are compared separately."""
        events = _settled(
            _version(1, {"w": "1"}, load_state_timestamp=0.0),
            _version(2, {"w": "1"}, load_state_timestamp=50.0),
        )

        assert check(events) == []

    def test_a_stale_tensor_inside_one_lineage_is_still_caught_next_to_another_lineage(self) -> None:
        """Lineage isolation must not turn into ignoring each lineage's own pairs."""
        events = _settled(
            _version(1, {"w": "1"}, load_state_timestamp=0.0),
            _version(2, {"w": "1"}, load_state_timestamp=0.0),
            _version(3, {"w": "9"}, load_state_timestamp=50.0),
        )

        [issue] = check(events)

        assert issue.debug_trainer_load_state_timestamp == 0.0
        assert issue.trainer_model_id is None

    def test_a_single_version_has_nothing_to_compare(self) -> None:
        """One published version is not a movement failure."""
        assert check(_settled(_version(1, {"w": "1"}))) == []

    def test_the_unsettled_latest_version_is_not_compared(self) -> None:
        """A checksum recorded at or after the last step or result may belong to an unfinished publication."""
        events = [make_trainer_args(), _version(1, {"w": "1"}), make_step_end(second=1.5), _version(2, {"w": "1"})]

        assert check(events) == []
        assert len(check(events, include_latest=True)) == 1

    def test_nothing_is_settled_without_any_step_or_result(self) -> None:
        """No settling event means every checksum is still the latest."""
        events = [make_trainer_args(), _version(1, {"w": "1"}), _version(2, {"w": "1"})]

        assert check(events) == []

    def test_the_rule_is_silent_without_trainer_arguments(self) -> None:
        """Without the trainer env report the exclusions cannot be evaluated, so the rule stays off."""
        events = [_version(1, {"w": "1"}), _version(2, {"w": "1"}), make_step_end(second=100.0)]

        assert check(events) == []


class TestExclusions:
    @pytest.mark.parametrize(
        "overrides",
        [dict(lora_rank=8), dict(lora_adapter_path="/adapters/a"), dict(update_weights_interval=2)],
    )
    def test_each_exclusion_alone_turns_movement_off(self, overrides: dict[str, Any]) -> None:
        """LoRA, a loaded adapter or a skipped push each make unchanged tensors legitimate."""
        events = _settled(_version(1, {"w": "1"}), _version(2, {"w": "1"}), **overrides)

        assert check(events) == []

    def test_the_default_configuration_is_not_excluded(self) -> None:
        """The baseline arguments keep the check on, or the exclusions above would prove nothing."""
        events = _settled(_version(1, {"w": "1"}), _version(2, {"w": "1"}))

        assert len(check(events)) == 1

    def test_trainer_ranks_reporting_different_arguments_are_rejected(self) -> None:
        """Ranks that disagree on the exclusion arguments would make the decision depend on report order."""
        events = [
            make_trainer_args(rank=0),
            make_trainer_args(rank=1, lora_rank=8),
            _version(1, {"w": "1"}),
            make_step_end(second=100.0),
        ]

        with pytest.raises(AssertionError, match="different arguments"):
            check(events)

    def test_excluding_movement_keeps_consistency_and_coverage_reporting(self) -> None:
        """Only movement is excluded; engines disagreeing or an uncovered publication still fail."""
        disagreeing = make_checksum(
            second=1.0,
            update_id="u1",
            weight_version=1,
            snapshots={"a": ("hash-a", {"w": "1"}), "b": ("hash-b", {"w": "2"})},
        )
        uncovered = make_result(
            second=2.0, update_id="u2", published_version=2, cell_hashes={"a": "hash-a"}, updated=["a"]
        )
        events = _settled(disagreeing, uncovered, lora_rank=8)

        assert check(events) == []
        assert len(inference_engine_weight_checksum_consistency.check(events)) == 1
        assert len(inference_engine_weight_checksum_coverage.check(events)) == 1
