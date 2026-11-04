import json
import os
import shlex
from sglang.srt.server_args import ServerArgs
from miles.router.dynamo.config import Address
from miles.router.dynamo.frontend import launch_args, runtime_env
from miles.utils.workers.argv_utils import _record_field_names
from miles.utils.workers.worker_spec import PortInfo


def sidecar_ports() -> list[PortInfo]:
    return [
        PortInfo(name="dynamo_grpc", static_port=14000, allow_dynamic=True),
        PortInfo(name="dynamo_system", static_port=15000, allow_dynamic=True),
        PortInfo(name="dynamo_kv", static_port=16000, allow_dynamic=True),
    ]


def sidecar_env(config, *, api_key: str | None) -> dict[str, str]:
    return runtime_env(
        config,
        inherited_env=os.environ,
        options=config.sidecar,
        managed_env={
            "DYN_SGLANG_CONTROLLER_MANAGED": "true",
            "DYN_SGLANG_UNREGISTER_ON_PAUSE": "true",
            "DYN_SGLANG_API_KEY": api_key,
            "DYN_COMPONENT": "backend",
            "DYN_ENDPOINT": "generate",
            "DYN_SYSTEM_HOST": None,
            "DYN_SYSTEM_PORT": None,
            "DYN_SYSTEM_STARTING_HEALTH_STATUS": "notready",
        },
    )


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


def sidecar_command(command: str, *, ctx) -> str:
    # Listener addresses are allocated after env_var; env exec preserves the native process lifecycle.
    return shlex.join(
        [
            "env",
            f"DYN_SYSTEM_PORT={ctx.self_addrs['dynamo_system'].port}",
            f"DYN_SYSTEM_HOST={ctx.self_addrs['dynamo_system'].host.strip('[]')}",
            *shlex.split(command),
        ]
    )
