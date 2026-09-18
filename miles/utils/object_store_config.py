MOONCAKE_MASTER_PORT = 50051
MOONCAKE_MASTER_ADDRESS_KEY = "master_server_address"


def compute_mooncake_init_kwargs(*, host: str = "127.0.0.1", master_port: int = MOONCAKE_MASTER_PORT) -> dict:
    return {
        "protocol": "tcp",
        MOONCAKE_MASTER_ADDRESS_KEY: f"{host}:{master_port}",
        "global_segment_size": "2gb",
        "local_buffer_size": "2gb",
    }
