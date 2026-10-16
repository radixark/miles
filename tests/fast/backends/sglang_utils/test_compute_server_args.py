from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import msgspec
import pytest

from miles.backends.sglang_utils import sglang_engine
from miles.backends.sglang_utils.sglang_api_client import WorkerType
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
        sglang_output_store_backend="none",
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
        peer_host="127.0.0.1",
        port=30000,
        worker_type=WorkerType.REGULAR,
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


class TestOutputStore:
    """Miles makes every training engine a client of its own Mooncake store."""

    @pytest.fixture(autouse=True)
    def _server_args_with_output_store(self, monkeypatch):
        server_args_type = dataclasses.make_dataclass(
            "ServerArgs",
            [("gated_launch_port", int), ("output_store_backend", str), ("output_store_backend_extra_config", str)],
        )
        monkeypatch.setattr(sglang_engine, "ServerArgs", server_args_type)
        for name in ("MOONCAKE_MASTER", "MOONCAKE_PROTOCOL", "MOONCAKE_TE_META_DATA_SERVER", "MOONCAKE_DEVICE"):
            monkeypatch.delenv(name, raising=False)

    @pytest.fixture
    def output_store_config(self):
        """The class each engine parses its extra config with; Miles builds the config through it."""
        return pytest.importorskip("sglang.srt.managers.output_store").OutputStoreConfig

    @staticmethod
    def _args(local_buffer_size: object = "4gb") -> SimpleNamespace:
        return make_args(
            sglang_output_store_backend="mooncake",
            sglang_output_store_client_config={"local_buffer_size": local_buffer_size, "chunk_bytes": 1 << 20},
            mooncake_store_init_kwargs={"master_server_address": "10.0.0.1:50051", "protocol": "tcp"},
            mooncake_replica_num=2,
        )

    def test_each_engine_joins_the_miles_store_with_its_own_address(self, output_store_config):
        server_args = compute(
            self._args(),
            host="10.0.0.7",
            peer_host="10.0.0.7",
            sglang_overrides={"output_store_backend_extra_config": '{"key_prefix": "other"}'},
        )

        assert server_args["output_store_backend"] == "mooncake"
        # The derived config replaces a group override: the readers can only read what it describes.
        assert output_store_config.from_extra_config(
            server_args["output_store_backend_extra_config"]
        ) == output_store_config(
            master_server_address="10.0.0.1:50051",
            protocol="tcp",
            metadata_server="P2PHANDSHAKE",
            device_name="",
            key_prefix="miles-object-store",
            replica_num=2,
            local_hostname="10.0.0.7",
            local_buffer_size=4 * 1024**3,
            chunk_bytes=1 << 20,
        )

    def test_sglangs_parser_rejects_a_bad_user_value_when_miles_builds_the_config(self, output_store_config):
        with pytest.raises(ValueError, match="Invalid size"):
            compute(self._args(local_buffer_size=["4gb"]))

    def test_an_engine_bound_to_every_address_names_its_peer_host(self, output_store_config):
        """A bind-all host such as Kubernetes' 0.0.0.0 is no address a peer can reach."""
        server_args = compute(self._args(), host="0.0.0.0", peer_host="$(MILES_POD_IP)")

        assert json.loads(server_args["output_store_backend_extra_config"])["local_hostname"] == "$(MILES_POD_IP)"

    def test_a_group_that_turns_it_off_gets_no_extra_config(self):
        server_args = compute(self._args(), sglang_overrides={"output_store_backend": "none"})

        assert server_args["output_store_backend"] == "none"
        assert "output_store_backend_extra_config" not in server_args

    def test_engines_stay_out_of_the_store_by_default(self):
        server_args = compute(make_args())

        assert server_args["output_store_backend"] == "none"
        assert "output_store_backend_extra_config" not in server_args

    @pytest.mark.parametrize("worker_type", [WorkerType.PREFILL, WorkerType.DECODE])
    def test_pd_engines_cannot_enable_it(self, worker_type):
        with pytest.raises(AssertionError, match="PD disaggregation"):
            compute(self._args(), worker_type=worker_type, disaggregation_bootstrap_port=1)
