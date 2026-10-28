from tests.fast.fixtures.attribute_fixtures import AttributeProbe

from miles.utils.eval_config import pick_from_args


class TestPickFromArgs:
    def test_pick_from_args_keeps_its_first_non_none_contract(self) -> None:
        """The existing picker skips missing and None values while retaining False."""
        assert pick_from_args(AttributeProbe(), ("missing", "value", "false_value")) is False
        assert pick_from_args(AttributeProbe(), ("missing", "value")) is None
