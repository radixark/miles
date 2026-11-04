import shlex
from types import SimpleNamespace

import pytest
from sglang.srt.arg_groups.serving_hook import handle_deprecated_args
from tests.fast.backends.sglang_utils.conftest import make_engine_args

from miles.backends.sglang_utils.server_args_utils import parse_server_args_argv
from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.backends.sglang_utils.sglang_engine import compute_engine_launch_cmd
from miles.router.dynamo.config import DynamoConfig
from miles.router.dynamo.sidecar import sidecar_env, sidecar_overrides
from miles.utils.workers.worker_spec import HostAndPort


@pytest.fixture
def config():
    return DynamoConfig(
        namespace="test-run",
        discovery={"backend": "etcd", "endpoints": [{"host": "localhost", "port": 2379}]},
        pins={"dynamo": "a" * 40, "sglang": "b" * 40},
    )


def test_sidecar_does_not_accept_conflicting_admission_env(config):
    config = config.model_copy(
        update={"sidecar": config.sidecar.model_copy(update={"env": {"DYN_SGLANG_CONTROLLER_MANAGED": "false"}})}
    )
    with pytest.raises(ValueError, match="CONTROLLER_MANAGED"):
        sidecar_env(config, api_key=None)


@pytest.mark.parametrize("api_key", [None, "test-key"])
def test_native_sidecar_arguments_and_upstream_auth_validation(config, api_key):
    ctx = SimpleNamespace(
        self_addrs={
            "primary": HostAndPort(host="127.0.0.1", port=18000),
            "dynamo_grpc": HostAndPort(host="127.0.0.1", port=18001),
            "dynamo_kv": HostAndPort(host="127.0.0.1", port=18002),
        }
    )
    overrides = sidecar_overrides(config, overrides={}, ctx=ctx)
    command = compute_engine_launch_cmd(
        make_engine_args(sglang_api_key=api_key),
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
    handle_deprecated_args(parsed)
