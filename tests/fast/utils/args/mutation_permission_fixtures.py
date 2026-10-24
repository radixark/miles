from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

from miles.utils.args.runtime import AllConfig


def _orchestrator_config() -> AllConfig:
    return parse_fsdp_test_config()
