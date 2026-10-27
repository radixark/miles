import json
import sys
import types
from argparse import Namespace
from types import SimpleNamespace

import pytest
from tests.fast.fixtures.args_fixtures import parser_defaults

from miles.utils.output_store import config as output_store_config
from miles.utils.output_store.config import (
    check_output_store_args,
    connect_output_store,
    output_store_backend_config,
    output_store_reader,
    sglang_output_store_server_args,
    sweep_output_store,
)

MOONCAKE = {"master_server_address": "10.0.0.1:50051", "protocol": "tcp", "device_name": "mlx5_0"}


def make_args(**overrides) -> Namespace:
    return Namespace(
        **{
            **parser_defaults(),
            "use_output_store": True,
            "use_rollout_routing_replay": True,
            # Derived by miles_validate_args, which runs before these checks; not a parser flag.
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
    state.check = lambda args: check_output_store_args(
        args,
        sglang=SimpleNamespace(has_pd_disaggregation=state.pd),
        megatron=SimpleNamespace(is_multi_policy=state.multi_policy),
    )
    monkeypatch.setattr(output_store_config, "_sglang_has_output_store", lambda: state.sglang_has_store)
    return state


class TestLaunchChecks:
    def test_a_supported_launch_passes(self, launch):
        launch.check(make_args())
        launch.check(make_args(save_debug_train_data="/tmp/{rollout_id}.pt"))
        launch.check(make_args(use_rollout_routing_replay=False, use_sampling_support_replay=True))

    def test_without_the_switch_nothing_is_checked(self, launch):
        launch.pd = True
        launch.check(make_args(use_output_store=False, object_store_backend="ray"))

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"use_rollout_routing_replay": False}, "no request would use it"),
            ({"object_store_backend": "ray"}, "--object-store-backend must be mooncake"),
            ({"rollout_external": True}, "external rollout engines"),
            ({"rollout_external_router_pd": True}, "PD disaggregation"),
            ({"custom_convert_samples_to_train_data_path": "my.convert"}, "custom convert function"),
            ({"deploy_component": "trainer"}, "--deploy-component all"),
            ({"debug_rollout_only": True}, "--debug-rollout-only"),
            ({"multimodal_keys": {"image": "images"}}, "--multimodal-keys"),
            ({"save_debug_rollout_data": "/tmp/{rollout_id}.pt"}, "--save-debug-rollout-data"),
            ({"load_debug_rollout_data": "/tmp/{rollout_id}.pt"}, "--load-debug-rollout-data"),
            ({"ci_inject_rollout_data_path": "/tmp/{rollout_id}.pt"}, "--ci-inject-rollout-data-path"),
        ],
    )
    def test_an_unsupported_launch_setting_is_rejected(self, launch, overrides, message):
        with pytest.raises(ValueError, match=message):
            launch.check(make_args(**overrides))

    def test_a_buffer_that_cannot_hold_the_indexer_top_k_of_a_full_context_is_rejected(self, launch):
        dsa = {
            "use_rollout_indexer_replay": True,
            "rollout_indexer_topk_num_streams": 61,
            "rollout_indexer_topk_width": 2048,
            "rollout_max_context_len": 32769,
        }
        # 32768 rows x 61 streams x 2048 ids x 4 bytes: under the 16gb default, over 15gb
        launch.check(make_args(**dsa))
        with pytest.raises(ValueError, match="cannot hold the 16374562816 bytes of indexer top-k"):
            launch.check(make_args(**dsa, output_store_local_buffer_size="15gb"))
        launch.check(make_args(**{**dsa, "rollout_max_context_len": None}, output_store_local_buffer_size="15gb"))

    def test_without_a_context_length_the_response_length_bounds_the_indexer_top_k(self, launch):
        dsa = {
            "use_rollout_indexer_replay": True,
            "rollout_indexer_topk_num_streams": 61,
            "rollout_indexer_topk_width": 2048,
            "rollout_max_response_len": 32768,
        }
        # A response cut at 32768 tokens stores at least 32768 rows, as a 32769-token context does.
        launch.check(make_args(**dsa))
        with pytest.raises(ValueError, match="cannot hold the 16374562816 bytes of indexer top-k"):
            launch.check(make_args(**dsa, output_store_local_buffer_size="15gb"))
        # A 2048-token prompt bound adds its rows: 34815 rows no longer fit the 16gb default.
        with pytest.raises(ValueError, match="cannot hold the 17397473280 bytes of indexer top-k"):
            launch.check(make_args(**dsa, rollout_max_prompt_len=2048))

    def test_the_launch_facts_owned_elsewhere_are_checked_and_reported_together(self, launch, monkeypatch):
        launch.pd = launch.multi_policy = True
        launch.sglang_has_store = False
        monkeypatch.setenv("MILES_USE_LEGACY_ROLLOUT_V1", "1")

        with pytest.raises(ValueError) as error:
            launch.check(make_args())

        for reason in ("PD disaggregation", "multi-policy", "ServerArgs lacks output_store_backend", "legacy"):
            assert reason in str(error.value)


class TestStoreClients:
    def test_every_client_shares_miles_mooncake_and_the_run_prefix(self):
        args = make_args(output_store_local_buffer_size="64mb", mooncake_replica_num=2, run_uuid="abcd")

        expected = {
            "master_server_address": "10.0.0.1:50051",
            "local_hostname": None,
            "local_buffer_size": 0,
            "key_prefix": "miles-output-store/abcd",
            "protocol": "tcp",
            "metadata_server": "P2PHANDSHAKE",
            "device_name": "mlx5_0",
            "replica_num": 2,
        }

        assert output_store_backend_config(args) == expected
        assert output_store_backend_config(args, puts=True) == {**expected, "local_buffer_size": "64mb"}

    def test_an_engine_writes_with_its_own_host(self):
        args = make_args()

        server_args = sglang_output_store_server_args(args, local_hostname="10.0.0.7")

        assert server_args["output_store_backend"] == "mooncake"
        extra = json.loads(server_args["output_store_backend_extra_config"])
        assert extra == {**output_store_backend_config(args, puts=True), "local_hostname": "10.0.0.7"}

    def test_an_engine_names_no_device_without_an_init_kwarg(self, monkeypatch):
        """SGLang reads no MOONCAKE_* variable, and the building process's MOONCAKE_DEVICE may be another node's."""
        monkeypatch.setenv("MOONCAKE_DEVICE", "mlx5_builder")
        args = make_args(mooncake_store_init_kwargs={"master_server_address": "10.0.0.1:50051"})

        server_args = sglang_output_store_server_args(args, local_hostname="10.0.0.7")

        assert "device_name" not in json.loads(server_args["output_store_backend_extra_config"])

    def test_a_client_resolves_its_own_host_and_device_when_it_connects(self, monkeypatch):
        """An init kwarg names the device for every client; otherwise the connecting node's MOONCAKE_DEVICE does."""
        created = []
        module = types.ModuleType("radix_transfer_plugins")
        module.create_backend = lambda name, config: created.append((name, json.loads(config))) or "backend"
        module.OutputStore = lambda *, backend: SimpleNamespace(backend=backend)
        monkeypatch.setitem(sys.modules, "radix_transfer_plugins", module)
        # Built on one node...
        monkeypatch.setenv("MOONCAKE_DEVICE", "mlx5_builder")
        named = output_store_backend_config(make_args())
        unnamed = output_store_backend_config(
            make_args(mooncake_store_init_kwargs={"master_server_address": "10.0.0.1:50051"})
        )
        # ...connected on another.
        monkeypatch.setenv("MOONCAKE_LOCAL_HOSTNAME", "10.0.0.9")
        monkeypatch.setenv("MOONCAKE_DEVICE", "mlx5_client")

        for config in (named, unnamed):
            assert connect_output_store(config).backend == "backend"

        assert [(name, config["local_hostname"], config["device_name"]) for name, config in created] == [
            ("mooncake", "10.0.0.9", "mlx5_0"),
            ("mooncake", "10.0.0.9", "mlx5_client"),
        ]

    def test_no_reader_without_the_switch(self):
        assert output_store_reader(make_args(use_output_store=False)) is None


class TestRunEndSweep:
    @staticmethod
    def fake_mooncake(monkeypatch, *, removed: int) -> list[tuple]:
        calls: list[tuple] = []

        class Store:
            def setup(self, config):
                calls.append(("setup", config))
                return 0

            def remove_by_regex(self, pattern, force):
                calls.append(("remove", pattern, force))
                return removed

            def close(self):
                calls.append(("close",))

        store_module = types.ModuleType("mooncake.store")
        store_module.MooncakeDistributedStore = Store
        package = types.ModuleType("mooncake")
        package.store = store_module
        monkeypatch.setitem(sys.modules, "mooncake", package)
        monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
        return calls

    def test_the_sweep_force_removes_the_run_prefix_through_a_remove_only_client(self, monkeypatch):
        """A read leaves a lease on its keys, and a remove without force skips them."""
        calls = self.fake_mooncake(monkeypatch, removed=6)

        assert sweep_output_store(make_args(run_uuid="0123456789abcdef")) == 6

        (_, config), remove, close = calls
        assert config["master_server_addr"] == "10.0.0.1:50051"
        assert config["global_segment_size"] == 0 and config["local_buffer_size"] == 0
        assert remove == ("remove", "^miles-output-store/0123456789abcdef/", True)
        assert close == ("close",)

    def test_a_failed_sweep_raises_and_still_closes_its_client(self, monkeypatch):
        calls = self.fake_mooncake(monkeypatch, removed=-1)

        with pytest.raises(RuntimeError, match="remove_by_regex failed: -1"):
            sweep_output_store(make_args(run_uuid="0123456789abcdef"))

        assert calls[-1] == ("close",)
