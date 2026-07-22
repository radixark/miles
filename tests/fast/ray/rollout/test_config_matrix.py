from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sglang_config_yaml

from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig
from miles.ray.rollout.rollout_server import _resolve_sglang_config

# ----------------------------- _resolve_sglang_config matrix -----------------------------


class TestResolveSglangConfigPaths:
    def test_default_path_when_no_yaml_or_prefill(self):
        args = make_args(rollout_num_gpus=8, sglang_config=None, prefill_num_servers=None)
        cfg = _resolve_sglang_config(args)
        assert len(cfg.models) == 1
        assert cfg.models[0].name == "default"
        assert cfg.models[0].server_groups[0].worker_type == "regular"
        assert cfg.total_num_gpus == 8

    def test_prefill_num_servers_path(self):
        args = make_args(
            rollout_num_gpus=8,
            rollout_num_gpus_per_engine=1,
            prefill_num_servers=4,
            sglang_config=None,
        )
        cfg = _resolve_sglang_config(args)
        # Two groups: prefill + decode
        groups = cfg.models[0].server_groups
        assert len(groups) == 2
        worker_types = sorted(g.worker_type for g in groups)
        assert worker_types == ["decode", "prefill"]

    def test_yaml_path_actor_only(self, tmp_path):
        cfg_path = tmp_path / "actor.yaml"
        cfg_path.write_text(make_sglang_config_yaml(name="actor"))
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=8)
        cfg = _resolve_sglang_config(args)
        assert len(cfg.models) == 1
        assert cfg.models[0].name == "actor"

    def test_yaml_path_multi_model_actor_plus_reference(self, tmp_path):
        cfg_path = tmp_path / "multi.yaml"
        # 8 gpu actor + 4 gpu ref = 12 → must match args.rollout_num_gpus
        cfg_path.write_text(
            "sglang:\n"
            "  - name: actor\n"
            "    update_weights: true\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        num_gpus_per_engine: 1\n"
            "  - name: ref\n"
            "    update_weights: false\n"
            "    model_path: /ref/model\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 1\n"
        )
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=12)
        cfg = _resolve_sglang_config(args)
        assert [m.name for m in cfg.models] == ["actor", "ref"]
        assert cfg.total_num_gpus == 12


# ----------------------------- ServerGroupConfig validation matrix ---------------


class TestServerGroupConfigValidation:
    def test_invalid_worker_type_raises(self):
        with pytest.raises(AssertionError, match="Invalid worker_type"):
            ServerGroupConfig(worker_type="invalid", num_gpus=4)

    def test_zero_or_negative_num_gpus_raises(self):
        with pytest.raises(AssertionError, match="num_gpus must be > 0"):
            ServerGroupConfig(worker_type="regular", num_gpus=0)

    @pytest.mark.parametrize("wt", ["regular", "prefill", "decode", "placeholder"])
    def test_all_valid_worker_types_accepted(self, wt):
        ServerGroupConfig(worker_type=wt, num_gpus=4)


class TestModelConfigResolve:
    def test_resolve_inherits_num_gpus_per_engine_from_args(self):
        m = ModelConfig(
            name="actor",
            server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=4)],
        )
        args = make_args(rollout_num_gpus_per_engine=2, hf_checkpoint="/x")
        m.resolve(args)
        assert m.server_groups[0].num_gpus_per_engine == 2

    def test_resolve_inherits_model_path_into_overrides(self):
        m = ModelConfig(
            name="actor",
            server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=4)],
        )
        args = make_args(rollout_num_gpus_per_engine=2, hf_checkpoint="/path/actor")
        m.resolve(args)
        assert m.server_groups[0].overrides["model_path"] == "/path/actor"

    def test_resolve_auto_infers_update_weights_false_for_diff_path(self):
        m = ModelConfig(
            name="ref",
            model_path="/ref/model",
            server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=4)],
        )
        args = make_args(rollout_num_gpus_per_engine=1, hf_checkpoint="/actor/model")
        m.resolve(args)
        assert m.update_weights is False

    def test_resolve_auto_infers_update_weights_true_for_same_path(self):
        m = ModelConfig(
            name="actor",
            model_path="/actor/model",
            server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=4)],
        )
        args = make_args(rollout_num_gpus_per_engine=1, hf_checkpoint="/actor/model")
        m.resolve(args)
        assert m.update_weights is True

    def test_resolve_explicit_update_weights_not_overridden(self):
        m = ModelConfig(
            name="ref",
            model_path="/actor/model",
            update_weights=False,  # explicit
            server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=4)],
        )
        args = make_args(hf_checkpoint="/actor/model")
        m.resolve(args)
        assert m.update_weights is False  # not flipped


# ----------------------------- has_pd_disaggregation aggregation -----------------


class TestPdDisaggregation:
    def test_pd_detected_with_prefill(self):
        m = ModelConfig(
            name="x",
            server_groups=[
                ServerGroupConfig(worker_type="prefill", num_gpus=4),
                ServerGroupConfig(worker_type="decode", num_gpus=4),
            ],
        )
        assert m.has_pd_disaggregation is True

    def test_no_pd_for_pure_regular(self):
        m = ModelConfig(
            name="x",
            server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=4)],
        )
        assert m.has_pd_disaggregation is False

    def test_sglang_config_aggregates_across_models(self):
        m1 = ModelConfig(name="a", server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=4)])
        m2 = ModelConfig(name="b", server_groups=[ServerGroupConfig(worker_type="prefill", num_gpus=4)])
        cfg = SglangConfig(models=[m1, m2])
        assert cfg.has_pd_disaggregation is True


# ----------------------------- rollout_external path -----------------------------


class TestRolloutExternalPath:
    def test_external_addrs_consumed_in_allocator(self):
        from miles.ray.rollout.addr_allocator import allocate_rollout_engine_addr_and_ports_external

        args = make_args(
            rollout_external_engine_addrs=[
                "10.0.0.1:30000",
                "10.0.0.2:30001",
                "10.0.0.3:30002",
            ]
        )
        engines = [(rank, MagicMock()) for rank in range(3)]
        result = allocate_rollout_engine_addr_and_ports_external(args=args, rollout_engines=engines)
        assert len(result) == 3
        # Verify the addr/port roundtrip is consistent
        for rank in range(3):
            assert result[rank]["dist_init_addr"].startswith(f"10.0.0.{rank+1}")
            assert result[rank]["nccl_port"] is None  # no nccl in external mode
