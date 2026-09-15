from pydantic import TypeAdapter

from miles.utils.audit_utils.process_identity import ProcessIdentity, SimpleProcessIdentity


class TestProcessIdentityUnion:
    def test_registration_reporter_survives_a_union_json_roundtrip(self) -> None:
        """A registration reporter identity keeps its component and process name through the wire union."""
        adapter = TypeAdapter(ProcessIdentity)
        source = SimpleProcessIdentity(component="registration_reporter")

        parsed = adapter.validate_json(source.model_dump_json())

        assert parsed == source
        assert isinstance(parsed, SimpleProcessIdentity)
        assert parsed.to_name() == "registration_reporter"
