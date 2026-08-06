import argparse
import logging
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.backends.sglang_utils.arguments import add_sglang_arguments
from miles.backends.sglang_utils.arguments import validate_args as validate_sglang_args
from miles.router.config import MilesRouterConfig, compute_miles_router_config
from miles.utils.arguments import (
    _maybe_apply_dumper_overrides,
    _resolve_ft_components,
    get_miles_extra_args_provider,
    miles_validate_args,
    validate_async_off_policy_correction,
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

    def test_leaves_miles_router_heartbeat_enabled(self) -> None:
        """Dumper mode does not suppress MilesRouter probing: its health check interval is unchanged."""
        args = self._make_args(
            dumper_enable=True,
            use_fault_tolerance=True,
            rollout_health_check_interval=30.0,
        )
        _maybe_apply_dumper_overrides(args)

        config: MilesRouterConfig = compute_miles_router_config(args, host="10.0.0.1", port=1234)

        assert args.rollout_health_check_interval == 30.0
        assert config.health_check_interval == 30.0
        assert config.health_check_failure_threshold == 3

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


def test_bridge_mode_rejects_critic(tmp_path):
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

    with pytest.raises(
        AssertionError,
        match="Critic models are not supported with --megatron-to-hf-mode bridge",
    ):
        miles_validate_args(args)


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

    def test_rejects_experimental_ft_trainer(self, monkeypatch):
        # The v2 train group has no reconcile_adapters.
        monkeypatch.setenv("MILES_EXPERIMENTAL_FT_TRAINER", "1")
        args = self._parse([])

        with pytest.raises(AssertionError, match="MILES_EXPERIMENTAL_FT_TRAINER"):
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

    def test_the_failure_threshold_knob_is_now_available(self):
        """Debouncing was previously pinned to a single failed probe with no way to tune it."""
        assert self._parse([]).rollout_health_check_failure_threshold == 3
        assert (
            self._parse(["--rollout-health-check-failure-threshold", "5"]).rollout_health_check_failure_threshold == 5
        )
