import json
import sys
import types
from argparse import Namespace
from types import SimpleNamespace

import pytest
from tests.fast.fixtures.args_fixtures import parser_defaults

from miles.backends.megatron_utils import megatron_config
from miles.backends.sglang_utils import sglang_config
from miles.utils.output_store import config as output_store_config
from miles.utils.output_store.config import (
    check_output_store_args,
    connect_output_store,
    output_store_backend_config,
    output_store_reader,
    sglang_output_store_server_args,
)

MOONCAKE = {"master_server_address": "10.0.0.1:50051", "protocol": "tcp", "device_name": "mlx5_0"}


def make_args(**overrides) -> Namespace:
    return Namespace(
        **{
            **parser_defaults(),
            "use_output_store": True,
            "use_rollout_routing_replay": True,
            # Derived by miles_validate_args before it runs these checks; not a parser flag.
            "use_sampling_support_replay": False,
            "object_store_backend": "mooncake",
            "mooncake_store_init_kwargs": dict(MOONCAKE),
            "rollout_external": False,
            **overrides,
        }
    )


@pytest.fixture
def launch(monkeypatch):
    """What the checks read from the SGLang and Megatron configs, settable per test."""
    state = SimpleNamespace(pd=False, multi_policy=False, sglang_has_store=True)
    monkeypatch.setattr(
        sglang_config, "resolve_sglang_config", lambda args: SimpleNamespace(has_pd_disaggregation=state.pd)
    )
    monkeypatch.setattr(
        megatron_config, "resolve_megatron_config", lambda args: SimpleNamespace(is_multi_policy=state.multi_policy)
    )
    monkeypatch.setattr(output_store_config, "_sglang_has_output_store", lambda: state.sglang_has_store)
    return state


class TestLaunchChecks:
    def test_a_supported_launch_passes(self, launch):
        check_output_store_args(make_args())
        check_output_store_args(make_args(save_debug_train_data="/tmp/{rollout_id}.pt"))
        check_output_store_args(make_args(use_rollout_routing_replay=False, use_sampling_support_replay=True))

    def test_without_the_switch_nothing_is_checked(self, launch):
        launch.pd = True
        check_output_store_args(make_args(use_output_store=False, object_store_backend="ray"))

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"use_rollout_routing_replay": False}, "no request would use it"),
            ({"object_store_backend": "ray"}, "--object-store-backend must be mooncake"),
            ({"cluster_backend": "kubernetes"}, "--cluster-backend kubernetes"),
            ({"rollout_external": True}, "external rollout engines"),
            ({"rollout_external_router_pd": True}, "PD disaggregation"),
            ({"custom_convert_samples_to_train_data_path": "my.convert"}, "custom convert function"),
            ({"deploy_component": "trainer"}, "--deploy-component all"),
            ({"debug_rollout_only": True}, "--debug-rollout-only"),
            ({"save_debug_rollout_data": "/tmp/{rollout_id}.pt"}, "--save-debug-rollout-data"),
            ({"load_debug_rollout_data": "/tmp/{rollout_id}.pt"}, "--load-debug-rollout-data"),
            ({"ci_inject_rollout_data_path": "/tmp/{rollout_id}.pt"}, "--ci-inject-rollout-data-path"),
        ],
    )
    def test_an_unsupported_launch_setting_is_rejected(self, launch, overrides, message):
        with pytest.raises(ValueError, match=message):
            check_output_store_args(make_args(**overrides))

    def test_the_launch_facts_owned_elsewhere_are_checked_and_reported_together(self, launch, monkeypatch):
        launch.pd = launch.multi_policy = True
        launch.sglang_has_store = False
        monkeypatch.setenv("MILES_USE_LEGACY_ROLLOUT_V1", "1")

        with pytest.raises(ValueError) as error:
            check_output_store_args(make_args())

        for reason in ("PD disaggregation", "multi-policy", "ServerArgs lacks output_store_backend", "legacy"):
            assert reason in str(error.value)


class TestStoreClients:
    def test_every_client_shares_miles_mooncake_and_the_run_prefix(self):
        args = make_args(output_store_local_buffer_size="64mb", mooncake_replica_num=2, run_uuid="abcd")

        assert output_store_backend_config(args) == {
            "master_server_address": "10.0.0.1:50051",
            "local_hostname": None,
            "local_buffer_size": "64mb",
            "key_prefix": "miles-output-store/abcd",
            "protocol": "tcp",
            "metadata_server": "P2PHANDSHAKE",
            "device_name": "mlx5_0",
            "replica_num": 2,
        }

    def test_an_engine_writes_with_its_own_host(self):
        args = make_args()

        server_args = sglang_output_store_server_args(args, local_hostname="10.0.0.7")

        assert server_args["output_store_backend"] == "mooncake"
        extra = json.loads(server_args["output_store_backend_extra_config"])
        assert extra == {**output_store_backend_config(args), "local_hostname": "10.0.0.7"}

    def test_a_client_without_a_host_resolves_its_own_when_it_connects(self, monkeypatch):
        created = []
        module = types.ModuleType("radix_transfer_plugins")
        module.create_backend = lambda name, config: created.append((name, json.loads(config))) or "backend"
        module.OutputStore = lambda *, backend: SimpleNamespace(backend=backend)
        monkeypatch.setitem(sys.modules, "radix_transfer_plugins", module)
        monkeypatch.setenv("MOONCAKE_LOCAL_HOSTNAME", "10.0.0.9")

        store = connect_output_store(output_store_backend_config(make_args()))

        assert store.backend == "backend"
        [(name, config)] = created
        assert name == "mooncake" and config["local_hostname"] == "10.0.0.9"

    def test_no_reader_without_the_switch(self):
        assert output_store_reader(make_args(use_output_store=False)) is None
