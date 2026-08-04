import argparse
import logging
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.backends.sglang_utils.arguments import add_sglang_arguments, collect_eval_sglang_overrides
from miles.backends.sglang_utils.arguments import validate_args as validate_sglang_args
from miles.utils.arguments import (
    _maybe_apply_dumper_overrides,
    _resolve_ft_components,
    _resolve_mini_ft_controller_enable,
    _resolve_rollout_functions,
    _validate_rematerialize_param_from_master_weight,
    get_miles_extra_args_provider,
    miles_validate_args,
    resolve_rollout_function_paths,
    validate_async_off_policy_correction,
    validate_skip_actor_forward_only,
)
from miles.utils.ft_utils.health_checker import SimpleHealthCheckerConfig
from miles.utils.function_registry import function_registry
from miles.utils.run_uuid import RUN_UUID_LENGTH, validate_run_uuid

PATH_ARGS = ["--rollout-function-path", "--custom-generate-function-path"]
REQUIRED_ARGS = ["--rollout-batch-size", "64"]


def make_class_with_add_arguments():
    class MyFn:
        @classmethod
        def add_arguments(cls, parser):
            parser.add_argument("--my-custom-arg", type=int, default=42)

    return MyFn


def make_function_with_add_arguments():
    def my_fn():
        pass

    my_fn.add_arguments = lambda parser: parser.add_argument("--my-custom-arg", type=int, default=42)
    return my_fn


def make_function_without_add_arguments():
    def my_fn():
        pass

    return my_fn


@pytest.mark.parametrize("path_arg", PATH_ARGS)
class TestAddArgumentsSupport:

    @pytest.mark.parametrize("fn_factory", [make_class_with_add_arguments, make_function_with_add_arguments])
    def test_add_arguments_is_called_and_arg_is_parsed(self, path_arg, fn_factory):
        fn = fn_factory()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys, "argv", ["test", path_arg, "test:fn", "--my-custom-arg", "100"] + REQUIRED_ARGS
        ):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)
            args, _ = parser.parse_known_args()
            assert args.my_custom_arg == 100

    def test_skips_function_without_add_arguments(self, path_arg):
        fn = make_function_without_add_arguments()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys, "argv", ["test", path_arg, "test:fn"] + REQUIRED_ARGS
        ):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)


class TestMaybeApplyDumperOverrides:
    def _make_args(
        self,
        *,
        dumper_enable: bool = False,
        use_fault_tolerance: bool = False,
        ft_components: list[str] | None = None,
        router_disable_health_check: bool = False,
        rollout_health_check_interval: float = 30.0,
        miles_router_health_check_failure_threshold: int = 3,
        miles_router_max_connections: int | None = 64,
        miles_router_timeout: float | None = None,
        start_rollout_id: int | None = None,
        num_rollout: int = 10,
        eval_interval: int | None = 5,
        save: str | None = "/tmp/checkpoint",
        save_interval: int | None = 5,
        save_retain_interval: int | None = 10,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            dumper_enable=dumper_enable,
            use_fault_tolerance=use_fault_tolerance,
            ft_components=ft_components if ft_components is not None else [],
            mini_ft_controller_enable=None,
            router_disable_health_check=router_disable_health_check,
            rollout_health_check_interval=rollout_health_check_interval,
            miles_router_health_check_failure_threshold=miles_router_health_check_failure_threshold,
            miles_router_max_connections=miles_router_max_connections,
            miles_router_timeout=miles_router_timeout,
            start_rollout_id=start_rollout_id,
            num_rollout=num_rollout,
            eval_interval=eval_interval,
            save=save,
            save_interval=save_interval,
            save_retain_interval=save_retain_interval,
        )

    def test_noop_when_dumper_disabled(self) -> None:
        args = self._make_args(
            dumper_enable=False,
            use_fault_tolerance=True,
        )
        _maybe_apply_dumper_overrides(args)

        assert args.use_fault_tolerance is True
        assert args.router_disable_health_check is False
        assert args.num_rollout == 10
        assert args.eval_interval == 5
        assert args.save == "/tmp/checkpoint"
        assert args.save_interval == 5
        assert args.save_retain_interval == 10

    def test_disables_fault_tolerance_and_sglang_router_heartbeats(self) -> None:
        """Dumper mode turns off fault tolerance and the SGLang router health check."""
        args = self._make_args(
            dumper_enable=True,
            use_fault_tolerance=True,
        )
        _maybe_apply_dumper_overrides(args)

        assert args.use_fault_tolerance is False
        assert args.router_disable_health_check is True

    def test_no_healing_loop_survives_dumper_mode(self) -> None:
        """It is resolved from ft_components, which dumper mode clears, so resolving it first
        would leave the loop polling a registry with nothing in it for the whole run."""
        args = self._make_args(dumper_enable=True, use_fault_tolerance=True, ft_components=["rollout"])

        _maybe_apply_dumper_overrides(args)

        assert _resolve_mini_ft_controller_enable(args) is False

    def test_the_selected_ft_components_go_with_the_flag(self) -> None:
        """ft_components is resolved from the flag long before this runs, so clearing the flag
        alone would leave every component selected and its probes still firing."""
        args = self._make_args(dumper_enable=True, use_fault_tolerance=True, ft_components=["rollout", "train"])

        _maybe_apply_dumper_overrides(args)

        assert args.ft_components == []

    def test_forces_single_rollout(self) -> None:
        args = self._make_args(dumper_enable=True, num_rollout=100)
        _maybe_apply_dumper_overrides(args)

        assert args.start_rollout_id == 0
        assert args.num_rollout == 1
        assert args.eval_interval is None
        assert args.save is None
        assert args.save_interval is None
        assert args.save_retain_interval is None

    def test_respects_start_rollout_id(self) -> None:
        args = self._make_args(dumper_enable=True, start_rollout_id=5, num_rollout=100)
        _maybe_apply_dumper_overrides(args)

        assert args.num_rollout == 6


def test_fully_async_eval_resolves_to_the_producer_itself():
    """Only the producer's own instance pauses on eval, and RolloutManager reuses one
    instance only when both paths match."""
    path = "miles.rollout.fully_async_rollout.FullyAsyncRolloutFn"
    default = SimpleNamespace(rollout_function_path=None, eval_function_path=None, fully_async=True)
    assert resolve_rollout_function_paths(default) == (path, path)

    override = SimpleNamespace(rollout_function_path=None, eval_function_path="pkg.CustomEval", fully_async=True)
    assert resolve_rollout_function_paths(override) == (path, "pkg.CustomEval")


def test_fully_async_rejects_abort_pause_mode():
    """Generation is always in flight, so aborting on every weight update would kill it."""
    args = SimpleNamespace(
        fully_async=True,
        multi_lora=False,
        rollout_function_path=None,
        eval_function_path=None,
        colocate=False,
        partial_rollout=False,
        pause_generation_mode="abort",
        recompute_logprobs_via_prefill=False,
        rollout_all_samples_process_path=None,
        eval_num_gpus=0,
    )

    with pytest.raises(AssertionError, match="pause-generation-mode abort"):
        _resolve_rollout_functions(args)

    args.pause_generation_mode = "retract"
    _resolve_rollout_functions(args)


def test_recompute_logprobs_via_prefill_flag_is_parsed():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    args = parser.parse_args(["--recompute-logprobs-via-prefill"] + REQUIRED_ARGS)

    assert args.recompute_logprobs_via_prefill is True


def test_sglang_parallel_sizes_keep_server_args_destinations():
    parser = add_sglang_arguments(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--sglang-tp-size",
            "6",
            "--sglang-data-parallel-size",
            "2",
            "--sglang-pipeline-parallel-size",
            "3",
            "--sglang-expert-parallel-size",
            "4",
            "--sglang-attention-context-parallel-size",
            "5",
        ]
    )
    args.rollout_num_gpus_per_engine = 8
    args.true_on_policy_mode = False
    args.sglang_enable_dp_attention = True
    args.use_session_server = False

    validate_sglang_args(args)

    assert args.sglang_tp_size == 8
    assert args.sglang_dp_size == 2
    assert args.sglang_pp_size == 3
    assert args.sglang_ep_size == 4
    assert args.sglang_attn_cp_size == 5


class TestEvalSglangOverrides:
    """Unset means "inherit --sglang-*", so an unset flag must leave no attribute at all."""

    def _parse(self, argv):
        return add_sglang_arguments(argparse.ArgumentParser()).parse_args(argv)

    def test_unset_flags_produce_no_overrides(self):
        args = self._parse(["--sglang-mem-fraction-static", "0.7"])

        assert collect_eval_sglang_overrides(args) == {}
        assert not hasattr(args, "eval_sglang_mem_fraction_static")

    def test_set_flag_becomes_an_override_without_touching_the_base_family(self):
        args = self._parse(["--sglang-mem-fraction-static", "0.7", "--eval-sglang-mem-fraction-static", "0.9"])

        assert collect_eval_sglang_overrides(args) == {"mem_fraction_static": 0.9}
        assert args.sglang_mem_fraction_static == 0.7

    def test_boolean_can_be_turned_back_off(self):
        args = self._parse(["--sglang-enable-dp-attention", "--no-eval-sglang-enable-dp-attention"])

        assert args.sglang_enable_dp_attention is True
        assert collect_eval_sglang_overrides(args) == {"enable_dp_attention": False}

    def test_parallel_sizes_keep_server_args_destinations(self):
        args = self._parse(["--eval-sglang-data-parallel-size", "2", "--eval-sglang-expert-parallel-size", "4"])

        assert collect_eval_sglang_overrides(args) == {"dp_size": 2, "ep_size": 4}

    def test_tp_size_is_not_exposed(self):
        """A second TP knob could move tp_size off the bundles --eval-num-gpus-per-engine placed."""
        with pytest.raises(SystemExit):
            self._parse(["--eval-sglang-tp-size", "2"])


def test_custom_megatron_post_save_hook_path_is_parsed():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    args = parser.parse_args(["--custom-megatron-post-save-hook-path", "pkg.module.hook"] + REQUIRED_ARGS)

    assert args.custom_megatron_post_save_hook_path == "pkg.module.hook"


def test_custom_megatron_post_save_hook_path_requires_save():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(
        ["--custom-megatron-post-save-hook-path", "pkg.module.hook", "--num-rollout", "1"] + REQUIRED_ARGS
    )

    with pytest.raises(
        AssertionError,
        match="'--save' is required when custom_megatron_post_save_hook_path is set.",
    ):
        miles_validate_args(args)


def test_dynamic_global_batch_size_requires_dynamic_batch_size():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(["--use-dynamic-global-batch-size", "--num-rollout", "1"] + REQUIRED_ARGS)

    with pytest.raises(AssertionError, match="requires --use-dynamic-batch-size"):
        miles_validate_args(args)


def test_shared_actor_critic_ppo_rejects_indep_dp():
    """Multi-cell PPO used to pass validation and fail only at the first training step's external-data assert."""
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(["--advantage-estimator", "ppo", "--indep-dp", "--num-rollout", "1"] + REQUIRED_ARGS)
    _set_megatron_parallel_sizes(args)

    with pytest.raises(AssertionError, match="does not support --indep-dp"):
        miles_validate_args(args)


def test_rollout_fault_tolerance_rejects_a_dedicated_eval_fleet():
    """The eval fleet pins engine addresses once, so a healed eval cell would be skipped silently."""
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(
        ["--use-fault-tolerance", "--ft-components", "rollout", "--eval-num-gpus", "8", "--num-rollout", "1"]
        + REQUIRED_ARGS
    )

    with pytest.raises(AssertionError, match="dedicated eval fleet"):
        miles_validate_args(args)
class TestCriticSaveDerivation:
    def _validate(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        args = parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)
        miles_validate_args(args)
        return args

    def test_derives_sibling_dir_from_save(self):
        args = self._validate(["--advantage-estimator", "ppo", "--save", "/ckpts/run1"])
        assert args.critic_save == "/ckpts/run1_critic"

    def test_trailing_slash_is_stripped(self):
        args = self._validate(["--advantage-estimator", "ppo", "--save", "/ckpts/run1/"])
        assert args.critic_save == "/ckpts/run1_critic"

    def test_explicit_critic_save_is_respected(self):
        args = self._validate(
            ["--advantage-estimator", "ppo", "--save", "/ckpts/run1", "--critic-save", "/elsewhere/critic"]
        )
        assert args.critic_save == "/elsewhere/critic"

    def test_stays_none_without_save(self):
        args = self._validate(["--advantage-estimator", "ppo"])
        assert args.critic_save is None


class TestSessionServerV2Validation:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)

    @pytest.mark.parametrize(
        ("extra", "flag"),
        [
            (["--group-rm"], "--group-rm"),
            (
                ["--true-on-policy-mode", "--recompute-logprobs-via-prefill"],
                "--recompute-logprobs-via-prefill",
            ),
        ],
    )
    def test_rejects_unsupported_list_consumers(self, extra, flag):
        args = self._parse(["--use-session-server", "v2", *extra])

        with pytest.raises(ValueError) as exc_info:
            miles_validate_args(args)

        assert str(exc_info.value) == (f"--use-session-server v2 does not support {flag}; v2 returns list[Sample]")


class TestSessionServerScalingArguments:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)

    def test_defaults_to_32_instances_and_an_auto_port(self):
        args = self._parse([])

        assert args.session_server_port is None
        assert args.session_server_workers == 32

    def test_parses_starting_port_and_instance_count(self):
        args = self._parse(["--session-server-port", "30000", "--session-server-workers", "4"])

        assert args.session_server_port == 30000
        assert args.session_server_workers == 4

    def test_rejects_the_removed_end_port_form(self):
        with pytest.raises(SystemExit):
            self._parse(["--session-server-port", "30000", "30004"])


class TestSessionMessageMatcherArgument:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)

    def test_defaults_to_strict(self):
        assert self._parse([]).session_message_matcher == "strict"

    @pytest.mark.parametrize(
        "selector",
        [
            "strict",
            "loose_tool_call",
            "role_content_only",
            "not_installed.matchers.same_message",
        ],
    )
    def test_preserves_selector_without_importing(self, selector):
        args = self._parse(["--session-message-matcher", selector])

        assert args.session_message_matcher == selector


class TestSessionServerPauseGenerationMode:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)

    @pytest.mark.parametrize("colocate", [False, True])
    def test_session_server_accepts_abort(self, colocate):
        extra = ["--use-session-server", "--pause-generation-mode", "abort"]
        if colocate:
            extra.append("--colocate")
        args = self._parse(extra)

        miles_validate_args(args)

    def test_abort_without_session_server_passes(self):
        miles_validate_args(self._parse(["--pause-generation-mode", "abort"]))

    @pytest.mark.parametrize("mode", ["retract", "in_place"])
    def test_session_server_accepts_non_abort_modes(self, mode):
        miles_validate_args(self._parse(["--use-session-server", "--pause-generation-mode", mode]))

    @pytest.mark.parametrize(
        "session_server_args",
        [["--use-session-server"], ["--use-session-server", "v1"], ["--use-session-server", "v2"]],
    )
    def test_session_server_rejects_partial_rollout(self, session_server_args):
        args = self._parse([*session_server_args, "--partial-rollout"])

        with pytest.raises(AssertionError, match="does not support --partial-rollout"):
            miles_validate_args(args)

    @pytest.mark.parametrize(
        ("extra", "expect_warning"),
        [
            (
                ["--use-session-server", "--use-rollout-routing-replay", "--pause-generation-mode", "retract"],
                True,
            ),
            (["--use-session-server", "--pause-generation-mode", "retract"], False),
            (["--use-rollout-routing-replay", "--pause-generation-mode", "retract"], False),
            (
                [
                    "--use-session-server",
                    "--use-rollout-routing-replay",
                    "--colocate",
                    "--pause-generation-mode",
                    "abort",
                ],
                False,
            ),
            (
                ["--use-session-server", "--use-rollout-routing-replay", "--pause-generation-mode", "in_place"],
                False,
            ),
        ],
    )
    def test_retract_r3_warning(self, caplog, extra, expect_warning):
        args = self._parse(extra)

        with caplog.at_level(logging.WARNING, logger="miles.utils.arguments"):
            miles_validate_args(args)

        warned = any("R3 payloads can become very large" in record.message for record in caplog.records)
        assert warned is expect_warning


class TestTitoFixedTemplateConfiguration:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)

    def test_removed_role_flag_is_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["--tito-allowed-append-roles", "tool"])

    @pytest.mark.parametrize(
        ("extra", "expect_warning"),
        [
            (["--use-session-server"], True),
            ([], False),
            (["--use-session-server", "--tito-model", "qwen3"], False),
        ],
    )
    def test_warns_only_for_default_model_session(self, caplog, extra, expect_warning):
        args = self._parse(extra)

        with caplog.at_level(logging.WARNING, logger="miles.utils.arguments"):
            miles_validate_args(args)

        target_records = [
            record
            for record in caplog.records
            if record.getMessage().startswith("--tito-model=default uses a best-effort four-role append surface.")
        ]
        assert len(target_records) == int(expect_warning)

    def test_named_family_requires_session_server(self):
        args = self._parse(["--tito-model", "qwen3"])
        with pytest.raises(ValueError, match="--tito-model=qwen3 requires --use-session-server"):
            miles_validate_args(args)

    def test_named_family_resolves_registered_template_and_kwargs(self):
        args = self._parse(["--use-session-server", "--tito-model", "qwen3"])
        miles_validate_args(args)
        assert args.chat_template_path.endswith("/qwen3_fixed.jinja")
        assert args.apply_chat_template_kwargs == {"clear_thinking": False}

    @pytest.mark.parametrize(
        ("family", "template"),
        [("qwen35", "qwen3.5_fixed.jinja"), ("qwen36", "qwen3.6_fixed.jinja")],
    )
    def test_qwen35_and_qwen36_resolve_family_template(self, family, template):
        args = self._parse(["--use-session-server", "--tito-model", family])
        miles_validate_args(args)
        assert args.chat_template_path.endswith(f"/{template}")
        assert args.apply_chat_template_kwargs == {"preserve_thinking": True}

    @pytest.mark.parametrize("family", ["qwen38small", "qwen4exp"])
    def test_qwen38_families_resolve_default_template(self, family):
        args = self._parse(["--use-session-server", "--tito-model", family])
        miles_validate_args(args)
        assert args.chat_template_path.endswith("/qwen3.8_small_and_flash_next_fixed.jinja")
        assert args.apply_chat_template_kwargs == {"preserve_thinking": True, "reasoning_effort": "xhigh"}

    def test_named_family_rejects_custom_template(self):
        args = self._parse(
            [
                "--use-session-server",
                "--tito-model",
                "qwen3",
                "--chat-template-path",
                "/tmp/custom.jinja",
            ]
        )
        with pytest.raises(ValueError, match="cannot override the template registered"):
            miles_validate_args(args)

    def test_named_family_rejects_conflicting_registered_kwarg(self):
        args = self._parse(
            [
                "--use-session-server",
                "--tito-model",
                "qwen3",
                "--apply-chat-template-kwargs",
                '{"clear_thinking": true}',
            ]
        )
        with pytest.raises(ValueError, match="clear_thinking=True conflicts"):
            miles_validate_args(args)

    def test_named_family_accepts_same_registered_and_additional_kwargs(self):
        args = self._parse(
            [
                "--use-session-server",
                "--tito-model",
                "qwen3",
                "--apply-chat-template-kwargs",
                '{"clear_thinking": false, "enable_thinking": true}',
            ]
        )
        miles_validate_args(args)
        assert args.apply_chat_template_kwargs == {
            "clear_thinking": False,
            "enable_thinking": True,
        }


def test_bridge_mode_accepts_critic(tmp_path):
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(
        [
            "--advantage-estimator",
            "ppo",
            "--megatron-to-hf-mode",
            "bridge",
            "--hf-checkpoint",
            str(tmp_path),
            "--num-rollout",
            "1",
        ]
        + REQUIRED_ARGS
    )

    miles_validate_args(args)
    assert args.use_critic is True


def test_critic_rejects_experimental_ft_trainer(tmp_path, monkeypatch):
    monkeypatch.setenv("MILES_EXPERIMENTAL_FT_TRAINER", "1")
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(
        ["--advantage-estimator", "ppo", "--hf-checkpoint", str(tmp_path), "--num-rollout", "1"] + REQUIRED_ARGS
    )

    with pytest.raises(AssertionError, match="MILES_EXPERIMENTAL_FT_TRAINER"):
        miles_validate_args(args)


def test_critic_rejects_reward_level_kl(tmp_path):
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(
        [
            "--advantage-estimator",
            "ppo",
            "--kl-coef",
            "0.05",
            "--ref-load",
            str(tmp_path),
            "--hf-checkpoint",
            str(tmp_path),
            "--num-rollout",
            "1",
        ]
        + REQUIRED_ARGS
    )

    with pytest.raises(AssertionError, match="does not support reward-level KL"):
        miles_validate_args(args)


class TestMultiLoRAValidation:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(
            [
                "--multi-lora-n-adapters",
                "2",
                "--lora-rank",
                "8",
                "--target-modules",
                "linear_qkv",
                "--num-rollout",
                "1",
            ]
            + extra
            + REQUIRED_ARGS
        )

    def test_rejects_multiple_tokenizer_workers(self):
        # Each sglang tokenizer worker holds its own LoRA registry, so per-step
        # upserts fail non-deterministically; fail at launch, not first push.
        args = self._parse(["--sglang-tokenizer-worker-num", "2"])

        with pytest.raises(AssertionError, match="sglang-tokenizer-worker-num 1"):
            miles_validate_args(args)

    def test_accepts_default_single_tokenizer_worker(self):
        args = self._parse([])

        miles_validate_args(args)

        assert args.multi_lora is True

    def test_defaults_rollout_fn_and_data_source_to_multi_lora(self):
        args = self._parse([])

        miles_validate_args(args)

        assert args.rollout_function_path == "miles.rollout.multi_lora.async_rollout.generate_rollout_multi_lora"
        assert args.data_source_path == "miles.rollout.multi_lora.data_source.MultiLoRAAsyncDataSource"
        assert args.rollout_global_dataset is True

    def test_keeps_user_supplied_rollout_fn_and_data_source(self):
        args = self._parse(
            ["--rollout-function-path", "my.custom.rollout_fn", "--data-source-path", "my.custom.DataSource"]
        )

        miles_validate_args(args)

        assert args.rollout_function_path == "my.custom.rollout_fn"
        assert args.data_source_path == "my.custom.DataSource"

    def test_empty_wait_is_a_registered_argument(self):
        assert self._parse([]).multi_lora_max_empty_wait_s == 30.0
        assert self._parse(["--multi-lora-max-empty-wait-s", "5"]).multi_lora_max_empty_wait_s == 5.0

    def test_rejects_non_adam_optimizer(self):
        # Per-slot optimizer isolation (state init, retirement cleanup, step
        # clocks) only implements Adam semantics. Muon has its own dedicated
        # rejection; anything else non-Adam trips the generic guard.
        args = self._parse([])
        args.optimizer = "muon"
        with pytest.raises(AssertionError, match="does not support Muon"):
            miles_validate_args(args)

        args = self._parse([])
        args.optimizer = "sgd"
        with pytest.raises(AssertionError, match="requires --optimizer adam"):
            miles_validate_args(args)

    def test_accepts_experimental_ft_trainer(self, monkeypatch):
        """The v2 train group implements reconcile_adapters, so multi-LoRA may use it."""
        monkeypatch.setenv("MILES_EXPERIMENTAL_FT_TRAINER", "1")
        args = self._parse([])

        miles_validate_args(args)

    def test_rejects_pipeline_parallelism(self):
        # Adapter routing is not recompute-safe under a pipelined schedule.
        args = self._parse([])
        args.pipeline_model_parallel_size = 2
        with pytest.raises(AssertionError, match="pipeline-model-parallel-size 1"):
            miles_validate_args(args)

    def test_rejects_bshd_qkv_format(self):
        # bshd interleaves samples in the sequence-major flattening the spans assume.
        args = self._parse([])
        args.qkv_format = "bshd"
        with pytest.raises(AssertionError, match="qkv-format thd"):
            miles_validate_args(args)

    def test_rejects_shared_outer_expert_loras(self):
        # Per-expert layout only; the flag would switch sglang to a layout training never produces.
        args = self._parse([])
        args.experts_shared_outer_loras = True
        with pytest.raises(AssertionError, match="experts-shared-outer-loras"):
            miles_validate_args(args)

    def test_accepts_expert_leaf_targets_without_expert_tp_flag(self):
        # --expert-tensor-parallel-size stays None until Megatron's own validate_args;
        # comparing the raw value here rejected every run that omitted the flag.
        args = self._parse(["--target-modules", "gate_proj,up_proj,down_proj"])
        args.expert_tensor_parallel_size = None

        miles_validate_args(args)

        assert args.multi_lora is True


class TestResolveFtComponents:
    def test_disabled_with_no_components_returns_empty_without_warning(self, caplog) -> None:
        """use_fault_tolerance off and no ft_components yields an empty list and no warning."""
        args = SimpleNamespace(use_fault_tolerance=False, ft_components=None)
        with caplog.at_level(logging.WARNING, logger="miles.utils.arguments"):
            result = _resolve_ft_components(args)

        assert result == []
        assert not any("--ft-components is ignored" in record.message for record in caplog.records)

    def test_disabled_with_components_returns_empty_and_warns(self, caplog) -> None:
        """use_fault_tolerance off but ft_components set returns empty list and logs an ignore warning."""
        args = SimpleNamespace(use_fault_tolerance=False, ft_components=["train"])
        with caplog.at_level(logging.WARNING, logger="miles.utils.arguments"):
            result = _resolve_ft_components(args)

        assert result == []
        assert any(
            "--ft-components is ignored without --use-fault-tolerance" in record.message for record in caplog.records
        )

    def test_enabled_with_no_components_returns_default(self) -> None:
        """use_fault_tolerance on with no ft_components falls back to the default ['rollout']."""
        args = SimpleNamespace(use_fault_tolerance=True, ft_components=None)
        result = _resolve_ft_components(args)

        assert result == ["rollout"]

    def test_enabled_with_components_returns_distinct_copy(self) -> None:
        """use_fault_tolerance on with ft_components returns an equal but distinct list copy."""
        components = ["train", "rollout"]
        args = SimpleNamespace(use_fault_tolerance=True, ft_components=components)
        result = _resolve_ft_components(args)

        assert result == ["train", "rollout"]
        assert result is not components


@pytest.mark.parametrize(
    ("parallel_args", "expected"),
    [
        ([], (1, 1, 1, 1)),
        (
            [
                "--sglang-tensor-parallel-size",
                "2",
                "--sglang-data-parallel-size",
                "3",
                "--sglang-pipeline-parallel-size",
                "4",
                "--sglang-expert-parallel-size",
                "5",
                "--sglang-enable-dp-attention",
            ],
            (2, 3, 4, 5),
        ),
        (
            [
                "--sglang-tp-size",
                "2",
                "--sglang-dp-size",
                "3",
                "--sglang-pp-size",
                "4",
                "--sglang-ep-size",
                "5",
                "--sglang-enable-dp-attention",
            ],
            (2, 3, 4, 5),
        ),
    ],
)
def test_sglang_parallel_sizes_use_short_namespace_fields(parallel_args, expected):
    parser = argparse.ArgumentParser()
    add_sglang_arguments(parser)
    args = parser.parse_args(parallel_args)

    assert (args.sglang_tp_size, args.sglang_dp_size, args.sglang_pp_size, args.sglang_ep_size) == expected
    assert not hasattr(args, "sglang_tensor_parallel_size")
    assert not hasattr(args, "sglang_data_parallel_size")
    assert not hasattr(args, "sglang_pipeline_parallel_size")
    assert not hasattr(args, "sglang_expert_parallel_size")

    args.rollout_num_gpus_per_engine = 8
    args.true_on_policy_mode = False
    args.recompute_logprobs_via_prefill = False
    args.sglang_router_policy = None
    args.use_session_server = False

    validate_sglang_args(args)

    assert args.sglang_tp_size == 8
    assert (args.sglang_dp_size, args.sglang_pp_size, args.sglang_ep_size) == expected[1:]


def test_sglang_parallel_size_aliases_keep_last_value():
    parser = argparse.ArgumentParser()
    add_sglang_arguments(parser)

    args = parser.parse_args(["--sglang-data-parallel-size", "2", "--sglang-dp-size", "3"])

    assert args.sglang_dp_size == 3


def _make_async_ppo_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        use_critic=True,
        use_rollout_logprobs=False,
        use_tis=False,
        keep_old_actor=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestValidateAsyncOffPolicyCorrection:
    def test_ppo_without_correction_is_rejected(self):
        with pytest.raises(AssertionError, match="behavior-policy correction"):
            validate_async_off_policy_correction(_make_async_ppo_args())

    @pytest.mark.parametrize("flag", ["use_rollout_logprobs", "use_tis", "keep_old_actor"])
    def test_ppo_with_any_correction_passes(self, flag):
        validate_async_off_policy_correction(_make_async_ppo_args(**{flag: True}))

    def test_non_ppo_estimators_are_unaffected(self):
        validate_async_off_policy_correction(_make_async_ppo_args(use_critic=False))


class TestValidateRematerializeParamFromMasterWeight:
    def _make_args(self, **overrides) -> SimpleNamespace:
        args = SimpleNamespace(
            rematerialize_param_from_master_weight=True,
            train_backend="megatron",
            lora_rank=0,
            lora_adapter_path=None,
            debug_disable_optimizer=False,
            indep_dp=False,
            colocate=True,
            offload_train=True,
            offload_train_target="cpu",
            use_distributed_optimizer=True,
            keep_old_actor=False,
            kl_coef=0,
            use_kl_loss=False,
            opd_teacher_load=None,
            use_precision_aware_optimizer=False,
            optimizer_cpu_offload=False,
            overlap_param_gather=False,
            compute_advantages_and_returns=True,
            num_critic_only_steps=0,
            debug_train_only=False,
            ci_test=False,
            check_rematerialize_param_from_master_weight=False,
            disable_param_buffers_cpu_backup=False,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_valid_config_forces_no_param_buffer_cpu_backup(self):
        args = self._make_args()
        _validate_rematerialize_param_from_master_weight(args)
        assert args.disable_param_buffers_cpu_backup is True

    def test_accepts_precision_aware_with_cpu_offload(self):
        args = self._make_args(use_precision_aware_optimizer=True, optimizer_cpu_offload=True)
        _validate_rematerialize_param_from_master_weight(args)
        assert args.disable_param_buffers_cpu_backup is True

    def test_ci_test_auto_enables_the_check(self):
        args = self._make_args(ci_test=True)
        _validate_rematerialize_param_from_master_weight(args)
        assert args.check_rematerialize_param_from_master_weight is True

    def test_check_stays_off_outside_ci(self):
        args = self._make_args()
        _validate_rematerialize_param_from_master_weight(args)
        assert args.check_rematerialize_param_from_master_weight is False

    def test_accepts_ref_and_teacher_tags(self):
        for overrides in ({"use_kl_loss": True}, {"kl_coef": 0.1}, {"opd_teacher_load": "/path/to/teacher"}):
            _validate_rematerialize_param_from_master_weight(self._make_args(**overrides))

    def test_debug_train_only_silently_disables(self):
        args = self._make_args(debug_train_only=True, colocate=False)
        _validate_rematerialize_param_from_master_weight(args)
        assert args.rematerialize_param_from_master_weight is False
        assert args.disable_param_buffers_cpu_backup is False

    def test_noop_when_disabled(self):
        args = self._make_args(rematerialize_param_from_master_weight=False, colocate=False)
        _validate_rematerialize_param_from_master_weight(args)
        assert args.disable_param_buffers_cpu_backup is False

    @pytest.mark.parametrize(
        "overrides",
        [
            {"train_backend": "fsdp"},
            {"lora_rank": 8},
            {"lora_adapter_path": "/path/to/adapter"},
            {"debug_disable_optimizer": True},
            {"indep_dp": True},
            {"colocate": False},
            {"offload_train": False},
            {"offload_train_target": "disk"},
            {"use_distributed_optimizer": False},
            {"keep_old_actor": True},
            {"use_precision_aware_optimizer": True},
            {"overlap_param_gather": True},
            {"compute_advantages_and_returns": False},
            {"num_critic_only_steps": 2},
        ],
    )
    def test_rejects_unsupported_config(self, overrides):
        with pytest.raises(AssertionError):
            _validate_rematerialize_param_from_master_weight(self._make_args(**overrides))

    def test_backend_is_checked_before_megatron_only_args(self):
        # An fsdp Namespace has none of the megatron args the later asserts read.
        args = SimpleNamespace(
            rematerialize_param_from_master_weight=True,
            train_backend="fsdp",
            debug_train_only=False,
        )
        with pytest.raises(AssertionError, match="Megatron"):
            _validate_rematerialize_param_from_master_weight(args)


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ([], False),
        (["--skip-actor-forward-only"], True),
    ],
)
def test_skip_actor_forward_only_flag_is_parsed(extra_args, expected):
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    args = parser.parse_args(extra_args + REQUIRED_ARGS)

    assert args.skip_actor_forward_only is expected


def test_skip_actor_forward_only_is_gated_during_miles_validation():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(
        ["--skip-actor-forward-only", "--global-batch-size", "32", "--num-rollout", "1"] + REQUIRED_ARGS
    )
    vars(args).update(
        hidden_dropout=0.0,
        attention_dropout=0.0,
        lora_dropout=0.0,
        moe_input_jitter_eps=None,
        moe_router_force_biased=None,
        moe_router_force_load_balancing=False,
        moe_router_load_balancing_type="aux_loss",
    )

    with pytest.raises(AssertionError, match="--skip-actor-forward-only"):
        miles_validate_args(args)


def _make_skip_actor_forward_only_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        compute_advantages_and_returns=True,
        custom_megatron_before_log_prob_hook_path=None,
        custom_megatron_before_train_step_hook_path=None,
        custom_model_provider_path=None,
        dumper_enable=False,
        dumper_fwd_only=None,
        dumper_source_patcher_config_train=None,
        dump_details=None,
        get_mismatch_metrics=False,
        global_batch_size=64,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        keep_old_actor=False,
        kl_coef=0.0,
        lora_dropout=0.0,
        log_correct_samples=False,
        loss_type="policy_loss",
        moe_input_jitter_eps=None,
        moe_router_force_biased=None,
        moe_router_force_load_balancing=False,
        moe_router_load_balancing_type="aux_loss",
        multi_lora=False,
        n_samples_per_prompt=8,
        num_steps_per_rollout=None,
        rollout_batch_size=8,
        rollout_data_postprocess_path=None,
        save_debug_train_data=None,
        train_backend="megatron",
        true_on_policy_mode=False,
        use_dynamic_global_batch_size=False,
        use_indexer_replay=False,
        use_opd=False,
        use_rollout_entropy=False,
        use_rollout_indexer_replay=False,
        use_rollout_logprobs=False,
        use_rollout_routing_replay=False,
        use_routing_replay=False,
        use_tis=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestValidateSkipActorForwardOnly:
    def test_valid_single_step_configuration_passes(self):
        validate_skip_actor_forward_only(_make_skip_actor_forward_only_args())

    def test_zero_moe_input_jitter_passes(self):
        validate_skip_actor_forward_only(_make_skip_actor_forward_only_args(moe_input_jitter_eps=0.0))

    def test_tis_configuration_passes(self):
        validate_skip_actor_forward_only(_make_skip_actor_forward_only_args(use_tis=True))

    def test_rollout_logprobs_configuration_passes(self):
        validate_skip_actor_forward_only(_make_skip_actor_forward_only_args(use_rollout_logprobs=True))

    def test_rollout_logprobs_with_mismatch_metrics_passes(self):
        validate_skip_actor_forward_only(
            _make_skip_actor_forward_only_args(
                get_mismatch_metrics=True,
                use_rollout_logprobs=True,
            )
        )

    @pytest.mark.parametrize(
        "overrides",
        [
            {"dumper_enable": True},
            {"dumper_fwd_only": ["enable=true"]},
            {"dumper_enable": True, "dumper_fwd_only": ["enable=false"]},
            {"dump_details": "/tmp/details"},
            {
                "dump_details": "/tmp/details",
                "save_debug_train_data": "/tmp/details/train_data/{rollout_id}_{rank}.pt",
            },
        ],
    )
    def test_dumper_configuration_passes(self, overrides):
        validate_skip_actor_forward_only(_make_skip_actor_forward_only_args(**overrides))

    @pytest.mark.parametrize(
        "overrides",
        [
            {"train_backend": "fsdp"},
            {"loss_type": "custom_loss"},
            {"compute_advantages_and_returns": False},
            {"keep_old_actor": True},
            {"kl_coef": 0.1},
            {"use_opd": True},
            {"hidden_dropout": 0.1},
            {"attention_dropout": 0.1},
            {"lora_dropout": 0.1},
            {"moe_input_jitter_eps": 0.1},
            {"moe_router_force_load_balancing": True},
            {"moe_router_force_biased": 0.0},
            {"moe_router_load_balancing_type": ["sinkhorn"]},
            {"use_rollout_entropy": True},
            {"true_on_policy_mode": True},
            {"log_correct_samples": True},
            {"rollout_data_postprocess_path": "pkg.hook"},
            {"custom_megatron_before_log_prob_hook_path": "pkg.hook"},
            {"custom_megatron_before_train_step_hook_path": "pkg.hook"},
            {"custom_model_provider_path": "pkg.model_provider"},
            {"dumper_source_patcher_config_train": "patcher.yaml"},
            {"save_debug_train_data": "train-{rollout_id}.pt"},
            {"use_routing_replay": True},
            {"use_indexer_replay": True},
            {"num_steps_per_rollout": 2},
            {"global_batch_size": 32},
        ],
    )
    def test_incompatible_configuration_is_rejected(self, overrides):
        with pytest.raises(AssertionError, match="--skip-actor-forward-only"):
            validate_skip_actor_forward_only(_make_skip_actor_forward_only_args(**overrides))

    @pytest.mark.parametrize(
        ("base_flag", "rollout_flag"),
        [
            ("use_routing_replay", "use_rollout_routing_replay"),
            ("use_indexer_replay", "use_rollout_indexer_replay"),
        ],
    )
    def test_rollout_replay_is_compatible(self, base_flag, rollout_flag):
        validate_skip_actor_forward_only(
            _make_skip_actor_forward_only_args(
                **{
                    base_flag: True,
                    rollout_flag: True,
                }
            )
        )

    def test_dynamic_global_batch_size_defers_step_count_to_runtime(self):
        validate_skip_actor_forward_only(
            _make_skip_actor_forward_only_args(
                global_batch_size=32,
                use_dynamic_global_batch_size=True,
            )
        )


class TestRunUuidResolution:
    def _parse(self, extra: list[str]):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(["--num-rollout", "1"] + extra + REQUIRED_ARGS)

    def test_unset_run_uuid_is_generated(self):
        """Every launch gets an identifier, so nothing has to cope with it being absent."""
        args = self._parse([])
        miles_validate_args(args)

        assert validate_run_uuid(args.run_uuid)

    def test_two_launches_do_not_share_a_run_uuid(self):
        """A colliding identifier would attribute one run's artifacts to another."""
        first, second = self._parse([]), self._parse([])
        miles_validate_args(first)
        miles_validate_args(second)

        assert first.run_uuid != second.run_uuid

    def test_an_explicit_run_uuid_is_kept(self):
        """Reproducing a run means being able to pin its identifier."""
        pinned = ("ab12cd34ef5678ab" * 4)[:RUN_UUID_LENGTH]
        args = self._parse(["--run-uuid", pinned])
        miles_validate_args(args)

        assert args.run_uuid == pinned

    def test_a_run_uuid_from_the_custom_config_file_is_validated_too(self, tmp_path):
        """The config file overwrites args after the flags are parsed, so it must not skip the check."""
        config = tmp_path / "override.yaml"
        config.write_text("run_uuid: my-experiment\n")
        args = self._parse(["--custom-config-path", str(config)])

        with pytest.raises(ValueError, match="invalid run uuid"):
            miles_validate_args(args)

    def test_a_run_uuid_blanked_by_the_custom_config_file_is_regenerated(self, tmp_path):
        """A null in the config file must not leave the identifier unset for the whole run."""
        config = tmp_path / "override.yaml"
        config.write_text("run_uuid: null\n")
        args = self._parse(["--custom-config-path", str(config)])
        miles_validate_args(args)

        assert validate_run_uuid(args.run_uuid)

    def test_a_malformed_explicit_run_uuid_fails_at_launch(self):
        """Rejecting it here beats corrupting every string that embeds it hours into a run."""
        args = self._parse(["--run-uuid", "my-experiment"])

        with pytest.raises(ValueError, match="invalid run uuid"):
            miles_validate_args(args)


class TestRolloutHealthCheckArguments:
    def _parse(self, extra: list[str]):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(extra + REQUIRED_ARGS)

    def test_the_rollout_defaults_survive_the_move_onto_the_shared_config(self):
        """The shared config carries the trainer's defaults, which are not the rollout ones."""
        args = self._parse([])

        assert args.rollout_health_check_interval == 30.0
        assert args.rollout_health_check_timeout == 30.0
        assert args.rollout_health_check_first_wait == 0.0

    def test_the_first_wait_grace_period_is_still_tunable(self):
        """A first launch compiling deepgemm kernels needs a grace period, or it is killed while warming up."""
        assert self._parse(["--rollout-health-check-first-wait", "600"]).rollout_health_check_first_wait == 600.0

    def test_the_resolved_rollout_config_matches_the_parsed_arguments(self):
        """The config is what the checker actually runs on, so it must not diverge from the flags."""
        config = SimpleHealthCheckerConfig.from_args(
            self._parse(["--rollout-health-check-first-wait", "600"]), prefix="rollout_health_check"
        )

        assert (config.interval, config.timeout, config.first_wait) == (30.0, 30.0, 600.0)

    def test_a_rollout_cell_is_reported_unhealthy_on_the_very_first_failed_probe(self):
        """At a 30s interval the shared three-failure debounce would hide a dead engine for 90s."""
        assert self._parse([]).rollout_health_check_failure_threshold == 1

    def test_the_trainer_heartbeat_keeps_its_own_debounce(self):
        """The rollout default must not be pushed down into the shared config: a trainer heartbeat
        shares an RPC channel with the train step, so one slow reply is a blip, not a dead cell."""
        assert self._parse([]).trainer_heartbeat_checker_failure_threshold == 3
