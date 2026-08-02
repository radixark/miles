from __future__ import annotations

import pydantic
import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sglang_config_yaml

from miles.backends.sglang_utils.sglang_config import resolve_sglang_config

# ----------------------------- resolve_sglang_config matrix -----------------------------


def _resolve_yaml(tmp_path, yaml_text: str, **args_overrides):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_text)
    return resolve_sglang_config(make_args(sglang_config=str(cfg_path), **args_overrides))


class TestResolveSglangConfigPaths:
    def test_default_path_when_no_yaml_or_prefill(self):
        args = make_args(rollout_num_gpus=8, sglang_config=None, prefill_num_servers=None)
        cfg = resolve_sglang_config(args)
        assert len(cfg.models) == 1
        assert cfg.models[0].name == "default"
        assert cfg.models[0].server_groups[0].worker_type == "regular"
        assert sum(g.num_gpus for m in cfg.models for g in m.server_groups) == 8

    def test_prefill_num_servers_path(self):
        args = make_args(
            rollout_num_gpus=8,
            rollout_num_gpus_per_engine=1,
            prefill_num_servers=4,
            sglang_config=None,
        )
        cfg = resolve_sglang_config(args)
        # Two groups: prefill + decode
        groups = cfg.models[0].server_groups
        assert len(groups) == 2
        worker_types = sorted(g.worker_type for g in groups)
        assert worker_types == ["decode", "prefill"]

    def test_yaml_path_actor_only(self, tmp_path):
        cfg_path = tmp_path / "actor.yaml"
        cfg_path.write_text(make_sglang_config_yaml(name="actor"))
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=8)
        cfg = resolve_sglang_config(args)
        assert len(cfg.models) == 1
        assert cfg.models[0].name == "actor"

    def test_yaml_path_multi_model_actor_plus_reference(self, tmp_path):
        # 8 gpu actor + 4 gpu ref = 12 → must match args.rollout_num_gpus
        cfg = _resolve_yaml(
            tmp_path,
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
            "        num_gpus_per_engine: 1\n",
            rollout_num_gpus=12,
        )
        assert [m.name for m in cfg.models] == ["actor", "ref"]
        assert sum(g.num_gpus for m in cfg.models for g in m.server_groups) == 12


# ----------------------------- server group validation matrix ---------------


class TestServerGroupValidation:
    def test_invalid_worker_type_raises(self, tmp_path):
        """An unknown worker_type is rejected while parsing, by name."""
        with pytest.raises(pydantic.ValidationError, match="server_groups.0.worker_type"):
            _resolve_yaml(
                tmp_path,
                "sglang:\n"
                "  - name: actor\n"
                "    server_groups:\n"
                "      - worker_type: invalid\n"
                "        num_gpus: 8\n",
                rollout_num_gpus=8,
            )

    @pytest.mark.parametrize("wt", ["regular", "prefill", "decode", "placeholder"])
    def test_all_valid_worker_types_accepted(self, wt, tmp_path):
        """Every documented worker_type parses through the yaml path."""
        cfg = _resolve_yaml(
            tmp_path,
            f"sglang:\n  - name: actor\n    server_groups:\n      - worker_type: {wt}\n        num_gpus: 8\n",
            rollout_num_gpus=8,
        )
        assert cfg.models[0].server_groups[0].worker_type == wt


class TestResolveDefaults:
    def test_resolve_explicit_update_weights_not_overridden(self, tmp_path):
        """An explicit update_weights is parsed verbatim even when the paths match."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: ref\n"
            "    model_path: /actor/model\n"
            "    update_weights: false\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n",
            rollout_num_gpus=8,
            hf_checkpoint="/actor/model",
        )
        assert cfg.models[0].update_weights is False

    def test_a_model_serving_the_trained_checkpoint_receives_weight_updates(self, tmp_path):
        """Left unset this is inferred from the paths, and inferring False for the actor trains
        against a frozen policy behind nothing louder than a warning."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    model_path: /actor/model\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n",
            rollout_num_gpus=8,
            hf_checkpoint="/actor/model",
        )
        assert cfg.models[0].update_weights is True

    def test_a_model_serving_another_checkpoint_is_left_frozen(self, tmp_path):
        """Inferring True for a reference model would let weight sync overwrite the frozen
        baseline the KL term is measured against."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: ref\n"
            "    model_path: /ref/model\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n",
            rollout_num_gpus=8,
            hf_checkpoint="/actor/model",
        )
        assert cfg.models[0].update_weights is False


# ----------------------------- has_pd_disaggregation aggregation -----------------


class TestPdDisaggregation:
    def test_pd_detected_with_prefill(self, tmp_path):
        """A prefill/decode split marks the model as PD-disaggregated."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: x\n"
            "    server_groups:\n"
            "      - worker_type: prefill\n"
            "        num_gpus: 4\n"
            "      - worker_type: decode\n"
            "        num_gpus: 4\n",
            rollout_num_gpus=8,
        )
        assert cfg.models[0].has_pd_disaggregation is True

    def test_no_pd_for_pure_regular(self, tmp_path):
        """Regular-only groups do not count as PD disaggregation."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n  - name: x\n    server_groups:\n      - worker_type: regular\n        num_gpus: 8\n",
            rollout_num_gpus=8,
        )
        assert cfg.models[0].has_pd_disaggregation is False

    def test_sglang_config_aggregates_across_models(self, tmp_path):
        """One PD model makes the whole config PD-disaggregated."""
        cfg = _resolve_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: a\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "  - name: b\n"
            "    update_weights: false\n"
            "    server_groups:\n"
            "      - worker_type: prefill\n"
            "        num_gpus: 4\n",
            rollout_num_gpus=8,
        )
        assert cfg.has_pd_disaggregation is True
