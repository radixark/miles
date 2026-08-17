from miles.utils.object_store_config import compute_mooncake_init_kwargs


class TestComputeMooncakeInitKwargs:
    def test_custom_host_and_port_produce_the_complete_mooncake_configuration(self):
        """A custom endpoint produces every required Mooncake initialization setting."""
        assert compute_mooncake_init_kwargs(host="store.internal", master_port=60000) == {
            "protocol": "tcp",
            "master_server_address": "store.internal:60000",
            "global_segment_size": "2gb",
            "local_buffer_size": "2gb",
        }
