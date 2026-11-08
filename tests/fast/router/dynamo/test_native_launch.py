import shlex
from types import SimpleNamespace

import pytest
from tests.fast.backends.sglang_utils.conftest import make_engine_args

from miles.backends.sglang_utils.server_args_utils import parse_server_args_argv
from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.backends.sglang_utils.sglang_config import ServerGroupConfig
from miles.backends.sglang_utils.sglang_engine import compute_engine_launch_cmd
from miles.router.dynamo.config import DynamoConfig
from miles.router.dynamo.sidecar import sidecar_overrides
from miles.utils.workers.worker_spec import HostAndPort


@pytest.fixture
def config():
    return DynamoConfig(
        namespace="test-run",
        discovery={"backend": "etcd", "endpoints": [{"host": "localhost", "port": 2379}]},
        pins={"dynamo": "a" * 40, "sglang": "b" * 40},
    )


@pytest.mark.parametrize("api_key", [None, "test-key"])
def test_native_sidecar_arguments_preserve_api_key(config, api_key, sidecar_ctx, group):
    ctx = sidecar_ctx
    args = make_engine_args(sglang_api_key=api_key)
    overrides = sidecar_overrides(config, sglang=args.sglang, group=group, ctx=ctx)
    command = compute_engine_launch_cmd(
        args,
        interpreter_prefix=["python"],
        node_rank=0,
        worker_type=WorkerType.REGULAR,
        base_gpu_id=0,
        sglang_overrides=overrides,
        num_gpus_per_engine=2,
        dist_init_addr="127.0.0.1:18003",
        nccl_port=18004,
        host="127.0.0.1",
        port=18000,
        disaggregation_bootstrap_port=None,
        engine_info_bootstrap_port=18005,
        gated_launch_port=18006,
        random_seed=42,
    )
    parsed = parse_server_args_argv(shlex.split(command)[3:])
    assert parsed.sidecar == "dynamo.sglang.sidecar"
    assert parsed.sidecar_args == overrides["sidecar_args"]
    assert parsed.grpc_port == 18001
    assert parsed.api_key == api_key
    assert parsed.tp_size == 2


@pytest.fixture
def sidecar_ctx():
    return SimpleNamespace(
        self_addrs={
            "primary": HostAndPort(host="127.0.0.1", port=18000),
            "dynamo_grpc": HostAndPort(host="127.0.0.1", port=18001),
            "dynamo_kv": HostAndPort(host="127.0.0.1", port=18002),
        }
    )


@pytest.mark.parametrize("scope", ["base", "group"])
@pytest.mark.parametrize(
    "name,value",
    [
        ("grpc_mode", True),
        ("smg_grpc_mode", True),
        ("sidecar", "other.sidecar"),
        ("grpc_port", 9999),
        ("kv_events_config", '{"publisher": "other"}'),
        ("sidecar_args", ["--namespace", "other"]),
    ],
)
def test_sidecar_rejects_conflicting_effective_settings(config, sidecar_ctx, scope, name, value, group):
    args = make_engine_args(**{f"sglang_{name}": value}) if scope == "base" else make_engine_args()
    if scope == "group":
        group = group.model_copy(update={"overrides": {**group.overrides, name: value}})
    with pytest.raises(ValueError, match="legacy gRPC mode" if name.endswith("grpc_mode") else name):
        sidecar_overrides(config, sglang=args.sglang, group=group, ctx=sidecar_ctx)


def test_sidecar_uses_group_precedence_without_mutating_config(config, sidecar_ctx, group):
    args = make_engine_args(sglang_grpc_mode=True, sglang_sidecar="other.sidecar")
    group = group.model_copy(
        update={"overrides": {**group.overrides, "grpc_mode": False, "sidecar": "dynamo.sglang.sidecar"}}
    )
    overrides = sidecar_overrides(config, sglang=args.sglang, group=group, ctx=sidecar_ctx)
    assert overrides["sidecar"] == "dynamo.sglang.sidecar"
    assert args.sglang.base_args["grpc_mode"] is True
    assert "grpc_port" not in group.overrides


def test_sidecar_rejects_group_streaming_opt_out(config, sidecar_ctx, group):
    args = make_engine_args()
    group = group.model_copy(update={"overrides": {"incremental_streaming_output": False}})
    with pytest.raises(ValueError, match="incremental_streaming_output"):
        sidecar_overrides(config, sglang=args.sglang, group=group, ctx=sidecar_ctx)


@pytest.fixture
def group():
    return ServerGroupConfig(worker_type=WorkerType.REGULAR, num_gpus_per_engine=2, needs_offload=False)
