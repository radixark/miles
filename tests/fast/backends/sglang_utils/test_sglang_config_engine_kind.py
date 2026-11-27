from __future__ import annotations

from argparse import Namespace

import pydantic
import pytest
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config
from tests.fast.fixtures.sglang_config_fixtures import resolve_sglang_config

from miles.backends.sglang_utils.sglang_config import EngineKind, SglangConfig, _compute_raw_sglang_config

_POLICY = """\
  - name: policy
    server_groups:
      - worker_type: regular
        num_gpus: 1
"""


def _make_args(**overrides) -> Namespace:
    defaults = dict(
        sglang_config=None,
        prefill_num_servers=None,
        rollout_num_gpus=3,
        rollout_num_gpus_per_engine=1,
        sglang_pp_size=1,
        sglang_pp_layer_partition=None,
        num_gpus_per_node=8,
        eval_num_gpus=0,
        eval_num_gpus_per_engine=1,
        hf_checkpoint="/ckpt/policy",
        offload_rollout=False,
        debug_train_only=False,
        debug_rollout_only=False,
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        multi_lora=False,
        router="miles",
        starts_inference_engines=True,
    )
    return Namespace(**(defaults | overrides))


def _resolve(tmp_path, yaml_text: str, **overrides):
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)
    return resolve_sglang_config(_make_args(sglang_config=str(path), **overrides))


def _generator(*, update_weights: str = "false", groups: str | None = None, name: str = "generator") -> str:
    groups = groups or ("      - worker_type: regular\n" "        engine_kind: diffusion\n" "        num_gpus: 2\n")
    return (
        f"  - name: {name}\n"
        "    model_path: /models/Z-Image-Turbo\n"
        f"    update_weights: {update_weights}\n"
        "    server_groups:\n" + groups
    )


class TestEngineKindParsing:
    def test_a_group_without_the_key_is_a_text_engine(self, tmp_path):
        """Every config written before the key existed must keep launching sglang.launch_server."""
        cfg = _resolve(tmp_path, "sglang:\n" + _POLICY.replace("num_gpus: 1", "num_gpus: 3"))
        assert [g.engine_kind for g in cfg.models[0].server_groups] == [EngineKind.TEXT]
        assert cfg.models[0].engine_kind == EngineKind.TEXT

    def test_a_diffusion_group_resolves_beside_the_policy(self, tmp_path):
        cfg = _resolve(tmp_path, "sglang:\n" + _POLICY + _generator())
        policy, generator = cfg.models
        assert policy.engine_kind == EngineKind.TEXT
        assert generator.engine_kind == EngineKind.DIFFUSION
        assert generator.server_groups[0].overrides == {"model_path": "/models/Z-Image-Turbo"}
        assert generator.update_weights is False

    def test_an_unknown_engine_kind_is_rejected(self, tmp_path):
        groups = "      - worker_type: regular\n        engine_kind: video\n        num_gpus: 2\n"
        with pytest.raises(pydantic.ValidationError, match="engine_kind"):
            _resolve(tmp_path, "sglang:\n" + _POLICY + _generator(groups=groups))

    def test_offload_rollout_gives_a_diffusion_group_no_text_server_arg(self, tmp_path):
        """sglang.multimodal_gen rejects enable_memory_saver, which offload_rollout adds to resident text groups."""
        cfg = _resolve(tmp_path, "sglang:\n" + _POLICY + _generator(), offload_rollout=True)
        policy, generator = cfg.models
        assert policy.server_groups[0].overrides["enable_memory_saver"] is False
        assert "enable_memory_saver" not in generator.server_groups[0].overrides

    def test_a_pipelined_policy_leaves_the_generator_one_stage(self, tmp_path):
        policy_yaml = _POLICY.replace("num_gpus: 1", "num_gpus: 2")
        groups = (
            "      - worker_type: regular\n"
            "        engine_kind: diffusion\n"
            "        num_gpus: 2\n"
            "        num_gpus_per_engine: 1\n"
        )
        cfg = _resolve(
            tmp_path,
            "sglang:\n" + policy_yaml + _generator(groups=groups),
            rollout_num_gpus=4,
            rollout_num_gpus_per_engine=2,
            sglang_pp_size=2,
        )
        policy, generator = cfg.models
        assert [g.pp_size for g in policy.server_groups] == [2]
        assert [g.pp_size for g in generator.server_groups] == [1]


_RULES = {
    "update_weights": "update_weights must be set to false",
    "first": "must not be the first model",
    "eval_name": "must not be named eval",
    "all_diffusion": "every server group must set engine_kind: diffusion",
    "router": "--router miles is required",
    "regular": "every server group must be worker_type: regular",
    "one_gpu": "num_gpus_per_engine must be 1",
    "topology": "overrides must not set what Miles sets for the engine",
    "colocate": "--colocate is not supported",
}


class TestDiffusionModelValidation:
    def test_a_valid_generator_passes(self, tmp_path):
        _resolve(tmp_path, "sglang:\n" + _POLICY + _generator())

    @pytest.mark.parametrize(
        ("rule", "yaml_text", "overrides"),
        [
            ("update_weights", "sglang:\n" + _POLICY + _generator(update_weights="true"), {}),
            ("first", "sglang:\n" + _generator() + _POLICY, {}),
            (
                "all_diffusion",
                "sglang:\n"
                + _POLICY
                + _generator(
                    groups="      - worker_type: regular\n        engine_kind: diffusion\n        num_gpus: 1\n"
                    "      - worker_type: regular\n        num_gpus: 1\n"
                ),
                {},
            ),
            ("router", "sglang:\n" + _POLICY + _generator(), {"router": "sglang"}),
            (
                "regular",
                "sglang:\n"
                + _POLICY
                + _generator(
                    groups="      - worker_type: placeholder\n        engine_kind: diffusion\n        num_gpus: 2\n"
                ),
                {},
            ),
            (
                "one_gpu",
                "sglang:\n"
                + _POLICY
                + _generator(
                    groups="      - worker_type: regular\n        engine_kind: diffusion\n        num_gpus: 2\n"
                    "        num_gpus_per_engine: 2\n"
                ),
                {},
            ),
            (
                "topology",
                "sglang:\n"
                + _POLICY
                + _generator(
                    groups="      - worker_type: regular\n        engine_kind: diffusion\n        num_gpus: 2\n"
                    "        overrides: {num_gpus: 2, master_port: 1234, scheduler_port: 5000, batching_max_size: 4}\n"
                ),
                {},
            ),
            ("colocate", "sglang:\n" + _POLICY + _generator(), {"colocate": True}),
            ("eval_name", "sglang:\n" + _POLICY + _generator(name="eval"), {}),
        ],
    )
    def test_each_rule_alone_is_named(self, tmp_path, rule, yaml_text, overrides):
        with pytest.raises(ValueError) as exc_info:
            _resolve(tmp_path, yaml_text, **overrides)
        message = str(exc_info.value)
        assert "Diffusion model '" in message
        assert [name for name, text in _RULES.items() if text in message] == [rule]

    def test_every_broken_rule_is_named_in_one_error(self, tmp_path):
        groups = (
            "      - worker_type: prefill\n        engine_kind: diffusion\n        num_gpus: 2\n"
            "        num_gpus_per_engine: 2\n        overrides: {num_gpus: 2}\n"
            "      - worker_type: regular\n        num_gpus: 1\n"
        )
        yaml_text = "sglang:\n" + _generator(update_weights="true", groups=groups, name="eval") + _POLICY
        with pytest.raises(ValueError) as exc_info:
            _resolve(tmp_path, yaml_text, rollout_num_gpus=4, router="sglang", colocate=True)
        message = str(exc_info.value)
        assert [name for name, text in _RULES.items() if text in message] == list(_RULES)
        assert "['num_gpus']" in message

    def test_a_broken_generator_is_named_before_the_run_wide_gpu_check(self, tmp_path):
        """--colocate shrinks --rollout-num-gpus to the trainer's GPUs, which would otherwise fail the GPU sum first."""
        with pytest.raises(ValueError, match="Diffusion model 'generator'.*--colocate is not supported"):
            _resolve(tmp_path, "sglang:\n" + _POLICY + _generator(), rollout_num_gpus=1, colocate=True)

    def test_rules_are_named_before_any_model_is_resolved(self, tmp_path):
        """Resolving first would raise its own model_path or gpus-per-engine error and hide these."""
        groups = (
            "      - {worker_type: regular, engine_kind: diffusion, num_gpus: 1, num_gpus_per_engine: 0}\n"
            "      - {worker_type: regular, num_gpus: 1, overrides: {model_path: /models/other}}\n"
        )
        with pytest.raises(ValueError) as exc_info:
            _resolve(tmp_path, "sglang:\n" + _POLICY + _generator(update_weights="true", groups=groups), colocate=True)
        message = str(exc_info.value)
        assert [name for name, text in _RULES.items() if text in message] == [
            "update_weights",
            "all_diffusion",
            "one_gpu",
            "colocate",
        ]

    def test_the_ports_and_gpus_miles_owns_are_named(self, tmp_path):
        groups = (
            "      - worker_type: regular\n        engine_kind: diffusion\n        num_gpus: 2\n"
            "        overrides: {num_gpus: 2, master_port: 1234, scheduler_port: 5000, batching_max_size: 4}\n"
        )
        with pytest.raises(ValueError, match=r"\['master_port', 'num_gpus', 'scheduler_port'\]"):
            _resolve(tmp_path, "sglang:\n" + _POLICY + _generator(groups=groups))

    def test_update_weights_must_be_written_out(self, tmp_path):
        yaml_text = "sglang:\n" + _POLICY + _generator().replace("    update_weights: false\n", "")
        with pytest.raises(ValueError, match="update_weights must be set to false"):
            _resolve(tmp_path, yaml_text)

    def test_a_frozen_text_model_is_not_held_to_the_diffusion_rules(self, tmp_path):
        """A frozen text ref model beside the policy predates diffusion groups and must keep resolving."""
        ref = (
            "  - name: ref\n    model_path: /models/ref\n    update_weights: false\n    server_groups:\n"
            "      - worker_type: regular\n        num_gpus: 2\n        num_gpus_per_engine: 2\n"
        )
        cfg = _resolve(tmp_path, "sglang:\n" + _POLICY + ref, router="sglang")
        assert [m.engine_kind for m in cfg.models] == [EngineKind.TEXT, EngineKind.TEXT]


class TestTextLookups:
    def test_a_diffusion_override_does_not_disagree_with_the_text_engines(self, tmp_path):
        """enable_metrics is a valid sglang.multimodal_gen arg; it must not break a lookup only text engines serve."""
        groups = (
            "      - {worker_type: regular, engine_kind: diffusion, num_gpus: 2, overrides: {enable_metrics: true}}\n"
        )
        path = tmp_path / "config.yaml"
        path.write_text("sglang:\n" + _POLICY + _generator(groups=groups))
        args = _make_args(sglang_config=str(path))
        config, _ = SglangConfig.resolve(
            raw=_compute_raw_sglang_config(args), args=args, base_args={"enable_metrics": False, "dp_size": 1}
        )
        assert config.common_value("enable_metrics") is False
        assert config.common_value("dp_size") == 1


class TestThroughTheArgumentParser:
    def test_colocate_is_named_for_the_generator(self, tmp_path):
        """The real parser rewrites --rollout-num-gpus under --colocate before the sglang config is read."""
        path = tmp_path / "config.yaml"
        path.write_text(
            "sglang:\n"
            + _POLICY
            + _generator(groups="      - {worker_type: regular, engine_kind: diffusion, num_gpus: 1}\n")
        )
        with pytest.raises(
            ValueError, match="Diffusion model 'generator' breaks these rules: --colocate is not supported"
        ):
            parse_fsdp_test_config(
                "--sglang-config",
                str(path),
                "--router",
                "miles",
                "--rollout-num-gpus",
                "2",
                "--colocate",
            )


class TestTextOnlyConfigs:
    """Behaviour main already has; these pass on main and must keep passing."""

    def test_a_dedicated_eval_group_without_gpus_per_engine_takes_the_eval_default(self, tmp_path):
        eval_model = (
            "  - name: eval\n    server_groups:\n      - {worker_type: regular, num_gpus: 1, num_gpus_per_engine: 0}\n"
        )
        cfg = _resolve(tmp_path, "sglang:\n" + _POLICY + eval_model, rollout_num_gpus=1, eval_num_gpus=1)
        assert [g.num_gpus_per_engine for g in cfg.models[1].server_groups] == [1]
