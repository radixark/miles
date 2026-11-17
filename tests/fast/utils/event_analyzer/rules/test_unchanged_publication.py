from tests.fast.utils.event_analyzer.rules.test_inference_engine_weight_movement import _settled, _version

from miles.utils.audit_utils.event_analyzer.rules.inference_engine_weight_movement import check


class TestUnchangedPublicationBoundary:
    def test_repeated_samples_of_one_version_are_not_a_second_publication(self) -> None:
        """Duplicate observations of the same version cannot invent unchanged-weight movement."""
        assert check(_settled(_version(1, {"w": "same"}), _version(1, {"w": "same"}, second=2))) == []

    def test_one_unchanged_tensor_is_fatal_when_it_is_the_entire_weight_set(self) -> None:
        """A single-tensor model still fails when its complete published weights never move."""
        [issue] = check(_settled(_version(1, {"w": "same"}), _version(2, {"w": "same"})))

        assert issue.kind == "unchanged_weights"
        assert not issue.is_warning_only

    def test_movement_after_an_unchanged_publication_does_not_erase_the_failure(self) -> None:
        """A later successful update cannot excuse an earlier stale publication."""
        events = _settled(_version(1, {"w": "old"}), _version(2, {"w": "old"}), _version(3, {"w": "new"}))

        [issue] = check(events)

        assert issue.kind == "unchanged_weights"
        assert (issue.weight_version_before, issue.weight_version_after) == (1, 2)
