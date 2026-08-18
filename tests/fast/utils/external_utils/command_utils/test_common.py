import json
import shlex

from miles.utils.external_utils.command_utils.common import MOONCAKE_INIT_KWARGS_FLAG, get_mooncake_object_store_args


class TestGetMooncakeObjectStoreArgs:
    def test_a_remote_master_host_reaches_the_serialized_store_address(self) -> None:
        """A split deployment connects to the master host supplied by its driving release."""
        argv = shlex.split(get_mooncake_object_store_args(master_port=61234, master_host="mooncake.run.svc"))

        kwargs = json.loads(argv[argv.index(MOONCAKE_INIT_KWARGS_FLAG) + 1])

        assert kwargs["master_server_address"] == "mooncake.run.svc:61234"
