from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import msgspec
import pytest

from miles.backends.sglang_utils import sglang_engine
from miles.backends.sglang_utils.sglang_engine import _compute_server_args


def make_args(**overrides: object) -> SimpleNamespace:
    defaults = dict(
        hf_checkpoint="/fake/model",
        seed=0,
        num_gpus_per_node=8,
        rollout_num_gpus_per_engine=1,
        offload_rollout=False,
        sglang_dp_size=1,
        sglang_pp_size=1,
        sglang_ep_size=1,
        sglang_mem_fraction_static=0.7,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        fp16=False,
        lora_adapter_path=None,
        debug_rollout_only=False,
        debug_skip_weight_update=False,
        multi_lora_n_adapters=1,
        lora_adapter_targets=[f"model.layers.*.self_attn.{projection}_proj" for projection in ("q", "k", "v")],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def compute(args: SimpleNamespace, **overrides: object) -> dict:
    kwargs = dict(
        node_rank=0,
        dist_init_addr="127.0.0.1:1234",
        nccl_port=5000,
        host="127.0.0.1",
        port=30000,
        worker_type="regular",
        disaggregation_bootstrap_port=None,
        base_gpu_id=0,
        engine_info_bootstrap_port=None,
        sglang_overrides=None,
        num_gpus_per_engine=None,
        gated_launch_port=30001,
        random_seed=0,
    )
    kwargs.update(overrides)
    return _compute_server_args(args, **kwargs)


@pytest.mark.parametrize(
    "record_factory", [dataclasses.make_dataclass, msgspec.defstruct], ids=["dataclass", "msgspec"]
)
def test_server_args_representation_preserves_launch_values(monkeypatch, record_factory):
    """Both record shapes must yield the same launch values, plus the device this renderer always pins."""
    server_args_type = record_factory(
        "ServerArgs",
        [("gated_launch_port", int), ("mem_fraction_static", float), ("random_seed", int)],
    )
    monkeypatch.setattr(sglang_engine, "ServerArgs", server_args_type)

    result = compute(make_args(), random_seed=7, sglang_overrides={"random_seed": 99, "unknown_field": True})

    assert result == {"gated_launch_port": 30001, "mem_fraction_static": 0.7, "random_seed": 99, "device": "cuda"}


class TestRandomSeed:
    def test_the_caller_chosen_seed_reaches_the_engine(self):
        """A seed sglang picks for itself makes a restarted engine replay a different RNG stream."""
        server_args = compute(make_args(seed=1234), random_seed=7)

        assert server_args["random_seed"] == 7

    def test_a_group_override_still_wins_over_the_seed_the_caller_computed(self):
        """A group pinning random_seed in its sglang_overrides must keep beating the derived number."""
        server_args = compute(make_args(seed=1234), random_seed=7, sglang_overrides={"random_seed": 99})

        assert server_args["random_seed"] == 99


class TestBaseGpuId:
    def test_the_given_base_gpu_id_is_used_verbatim_under_a_visibility_mask(self, monkeypatch):
        """The id already names the engine's own device space, so remapping it here moves the engine."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

        server_args = compute(make_args(), base_gpu_id=6)

        assert server_args["base_gpu_id"] == 6

    def test_a_base_gpu_id_outside_this_processs_visibility_mask_is_not_rejected(self, monkeypatch):
        """The engine's mask differs from this process's, so judging the id here fails a valid launch."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

        server_args = compute(make_args(), base_gpu_id=7)

        assert server_args["base_gpu_id"] == 7


class TestDeviceResolution:
    def test_generic_device_arg_takes_precedence_over_the_cuda_default(self):
        """An explicit generic device remains authoritative over the controller-safe default."""
        server_args = compute(make_args(sglang_device="cpu"))

        assert server_args["device"] == "cpu"

    def test_group_device_override_takes_precedence_over_the_cuda_default(self):
        """A per-group device override remains authoritative over the controller-safe default."""
        server_args = compute(make_args(), sglang_overrides={"device": "xpu"})

        assert server_args["device"] == "xpu"


class TestSglangOverridePrecedence:
    """An override must win over every args-derived default, including the conditional ones."""

    def test_override_wins_over_conditional_args_defaults(self):
        args = make_args(fp16=True, use_rollout_routing_replay=True, use_rollout_indexer_replay=True)

        server_args = compute(args, sglang_overrides={"dtype": "bfloat16"})

        assert server_args["dtype"] == "bfloat16"

    def test_override_wins_over_lora_defaults(self):
        args = make_args(lora_rank=8)

        server_args = compute(args, sglang_overrides={"enable_lora": False})

        assert server_args["enable_lora"] is False

    @pytest.mark.parametrize("value", [0.5, 0.95])
    def test_override_wins_over_base_sglang_args(self, value):
        args = make_args(sglang_mem_fraction_static=0.7)

        server_args = compute(args, sglang_overrides={"mem_fraction_static": value})

        assert server_args["mem_fraction_static"] == value

    def test_no_overrides_keeps_args_derived_values(self):
        args = make_args(fp16=True, lora_rank=8)

        server_args = compute(args)

        assert server_args["dtype"] == "float16"
        assert server_args["enable_lora"] is True
        assert server_args["mem_fraction_static"] == 0.7


class TestAdapterOwnership:
    def test_a_trainer_run_leaves_the_adapter_to_the_first_weight_sync(self):
        """An engine that also loaded the adapter from disk would serve stale weights if the sync were skipped."""
        server_args = compute(make_args(lora_rank=8, lora_adapter_path="/fake/adapter"))

        assert server_args["enable_lora"] is True
        assert not server_args.get("lora_paths")

    @pytest.mark.parametrize(
        "flag", ["debug_rollout_only", "debug_skip_weight_update"], ids=["rollout-only", "skip-sync"]
    )
    def test_without_a_trainer_push_the_engine_loads_the_adapter_itself(self, flag):
        server_args = compute(make_args(lora_rank=8, lora_adapter_path="/fake/adapter", **{flag: True}))

        assert server_args["lora_paths"] == ["miles_lora=/fake/adapter"]


class TestOutputStoreExtraConfig:
    """Engines join the rollout executor's Mooncake cluster; the user may add only the free keys."""

    @pytest.fixture(autouse=True)
    def _server_args_with_output_store(self, monkeypatch):
        server_args_type = dataclasses.make_dataclass(
            "ServerArgs",
            [("gated_launch_port", int), ("output_store_backend", str), ("output_store_backend_extra_config", str)],
        )
        monkeypatch.setattr(sglang_engine, "ServerArgs", server_args_type)
        for name in ("MOONCAKE_MASTER", "MOONCAKE_PROTOCOL", "MOONCAKE_TE_META_DATA_SERVER", "MOONCAKE_DEVICE"):
            monkeypatch.delenv(name, raising=False)

    @staticmethod
    def _args(extra_config: str | None = None) -> SimpleNamespace:
        return make_args(
            sglang_output_store_backend="mooncake",
            sglang_output_store_backend_extra_config=extra_config,
            mooncake_store_init_kwargs={"master_server_address": "10.0.0.1:50051", "protocol": "tcp"},
            mooncake_replica_num=2,
        )

    def test_each_engine_gets_the_shared_cluster_and_its_own_address(self):
        server_args = compute(self._args(), host="10.0.0.7")

        assert json.loads(server_args["output_store_backend_extra_config"]) == {
            "local_buffer_size": "1gb",
            "master_server_address": "10.0.0.1:50051",
            "protocol": "tcp",
            "metadata_server": "P2PHANDSHAKE",
            "device_name": "",
            "key_prefix": "miles-object-store",
            "replica_num": 2,
            "local_hostname": "10.0.0.7",
        }

    def test_the_user_keeps_the_free_keys(self):
        server_args = compute(self._args('{"local_buffer_size": "4gb", "partition": "run-1"}'))

        config = json.loads(server_args["output_store_backend_extra_config"])
        assert (config["local_buffer_size"], config["partition"]) == ("4gb", "run-1")

    @pytest.mark.parametrize(
        ("extra_config", "match"),
        [('{"key_prefix": "other"}', "contradicts"), ('{"local_hostname": "10.0.0.9"}', "local_hostname")],
    )
    def test_a_key_miles_derives_cannot_be_overridden(self, extra_config, match):
        with pytest.raises(ValueError, match=match):
            compute(self._args(extra_config))

    @pytest.mark.parametrize("worker_type", ["prefill", "decode"])
    def test_pd_engines_cannot_enable_it(self, worker_type):
        with pytest.raises(ValueError, match="PD disaggregation"):
            compute(self._args(), worker_type=worker_type, disaggregation_bootstrap_port=1)

    def test_an_engine_without_the_backend_is_left_alone(self):
        args = self._args('{"partition": "run-1"}')
        args.sglang_output_store_backend = "none"

        assert compute(args)["output_store_backend_extra_config"] == '{"partition": "run-1"}'
