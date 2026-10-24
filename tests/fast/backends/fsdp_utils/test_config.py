from tests.fast.import_isolation_utils import run_with_blocked_imports


class TestConfigurationImport:
    def test_backend_configuration_does_not_import_the_actor_or_change_logging(self) -> None:
        """Reading FSDP configuration must not initialize its actor dependencies or alter caller logging."""
        run_with_blocked_imports(
            blocked_prefixes=("miles.backends.fsdp_utils.actor",),
            script="""
import logging
logging.getLogger().setLevel(logging.DEBUG)
from miles.backends.fsdp_utils.config import FsdpArgsNamespace
config = FsdpArgsNamespace(micro_batch_size=3)
assert config.micro_batch_size == 3
assert logging.getLogger().level == logging.DEBUG
""",
        )
