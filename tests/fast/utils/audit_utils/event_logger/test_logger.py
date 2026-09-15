from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME


class TestEventsDirectoryName:
    def test_events_directory_name_matches_the_on_disk_contract(self) -> None:
        """The exported events directory name remains compatible with stored audit data."""
        assert EVENTS_DIRNAME == "events"
