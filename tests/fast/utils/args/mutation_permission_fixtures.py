from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

from miles.utils.args.runtime import OrchestratorConfig


def _orchestrator_config() -> OrchestratorConfig:
    return OrchestratorConfig.slice_from(parse_fsdp_test_config())
