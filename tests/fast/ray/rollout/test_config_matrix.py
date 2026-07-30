from __future__ import annotations

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
        args = make_args(rollout_num_gpus_per_engine=2, hf_checkpoint="/x")
        m = ModelConfig.resolve(
            args=args,
            name="actor",
            server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=4)],
        )
        assert m.server_groups[0].num_gpus_per_engine == 2

    def test_resolve_inherits_model_path_into_overrides(self):
        args = make_args(rollout_num_gpus_per_engine=2, hf_checkpoint="/path/actor")
        m = ModelConfig.resolve(
            args=args,
            name="actor",
            server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=4)],
        )
        assert m.server_groups[0].overrides["model_path"] == "/path/actor"

    def test_resolve_does_not_mutate_the_passed_server_groups(self):
        """The factory deep-copies groups, so the caller's objects stay unresolved."""
        args = make_args(rollout_num_gpus_per_engine=2, hf_checkpoint="/x")
        groups = [ServerGroupConfig(worker_type="regular", num_gpus=4)]
        ModelConfig.resolve(args=args, name="actor", server_groups=groups)
        assert groups[0].num_gpus_per_engine is None
        assert "model_path" not in groups[0].overrides

    def test_resolve_auto_infers_update_weights_false_for_diff_path(self):
        args = make_args(rollout_num_gpus_per_engine=1, hf_checkpoint="/actor/model")
        m = ModelConfig.resolve(
            args=args,
            name="ref",
            model_path="/ref/model",
            server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=4)],
        )
        assert m.update_weights is False

    def test_resolve_auto_infers_update_weights_true_for_same_path(self):
        args = make_args(rollout_num_gpus_per_engine=1, hf_checkpoint="/actor/model")
        m = ModelConfig.resolve(
            args=args,
            name="actor",
            model_path="/actor/model",
            server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=4)],
        )
        assert m.update_weights is True

    def test_resolve_explicit_update_weights_not_overridden(self):
        args = make_args(hf_checkpoint="/actor/model")
        m = ModelConfig.resolve(
            args=args,
            name="ref",
            model_path="/actor/model",
            update_weights=False,  # explicit
            server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=4)],
        )
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
    async def test_starting_engines_in_external_mode_is_not_implemented(self):
        """The external allocator was removed; starting engines must fail loudly until the replacement lands."""
        from miles.ray.rollout.server_cell import ServerCell
        from miles.utils.workers.addr_allocator import PortAllocator

        cell = ServerCell(
            args=make_args(num_gpus_per_node=8, rollout_external=True), worker_type="regular", cell_id="cell-0"
        )
        with pytest.raises(NotImplementedError):
            await cell.start_engines(PortAllocator())
