import json

import pytest

from miles.utils.args.schema import BaseConfig, validate_complete_config
from miles.utils.workers.argv_utils import CONFIG_JSON_FLAG, parse_config_argv


class _DefaultedConfig(BaseConfig):
    count: int = 3


class _TupleConfig(BaseConfig):
    entries: tuple[_DefaultedConfig, ...]


class TestCompleteTransportBoundaries:
    def test_config_json_rejects_omitted_defaults(self) -> None:
        """The argv transport refuses missing defaults before consumers receive a configuration."""
        with pytest.raises(ValueError, match="Incomplete configuration.*count"):
            parse_config_argv(_DefaultedConfig, [CONFIG_JSON_FLAG, json.dumps({})])

        assert parse_config_argv(_DefaultedConfig, [CONFIG_JSON_FLAG, '{"count": 3}']).count == 3

    def test_tuple_members_cannot_acquire_unserialized_defaults(self) -> None:
        """Completeness validation descends into tuple members and reports their index."""
        with pytest.raises(ValueError, match=r"_TupleConfig.entries\[1\].*count"):
            validate_complete_config(_TupleConfig, {"entries": [{"count": 1}, {}]})

        config = validate_complete_config(_TupleConfig, {"entries": [{"count": 1}, {"count": 3}]})
        assert tuple(entry.count for entry in config.entries) == (1, 3)
