import json
from sglang.srt.server_args import ServerArgs
from miles.router.dynamo.config import Address, launch_args
from miles.utils.workers.argv_utils import _record_field_names


def sidecar_overrides(config, *, overrides: dict, ctx) -> dict:
    required = {"sidecar", "sidecar_args", "grpc_port", "incremental_streaming_output", "kv_events_config"}
    if missing := required - set(_record_field_names(ServerArgs)):
        raise ValueError(f"Build SGLang from the pinned sglang-miles revision; missing fields: {sorted(missing)}")
    managed = {
        "sidecar": "dynamo.sglang.sidecar",
        "grpc_port": ctx.self_addrs["dynamo_grpc"].port,
        # The stream adapter requires deltas; this overrides the global SGLang default.
        "incremental_streaming_output": True,
        "kv_events_config": json.dumps(
            {"publisher": "zmq", "endpoint": f"tcp://*:{ctx.self_addrs['dynamo_kv'].port}"}
        ),
        "sidecar_args": launch_args(
            config.sidecar,
            managed={
                "--namespace": config.namespace,
                "--controller-managed": None,
                "--unregister-on-pause": "true",
                "--grpc-endpoint": Address(
                    host=ctx.self_addrs["primary"].host.strip("[]"), port=ctx.self_addrs["dynamo_grpc"].port
                ).url,
            },
            reserved=("--component", "--endpoint", "--enable-rl", "--policy-version-taints", "--sglang-api-key"),
        ),
    }
    for name, value in managed.items():
        if name in overrides and overrides[name] != value:
            raise ValueError(f"SGLang {name} conflicts with the Dynamo launch contract")
    if overrides.get("grpc_mode") or overrides.get("smg_grpc_mode"):
        raise ValueError("Dynamo needs native --grpc-port alongside HTTP, not legacy gRPC mode")
    return {**overrides, **managed}
