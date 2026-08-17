import argparse
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.backends.sglang_utils.arguments import add_sglang_arguments, collect_eval_sglang_overrides
from miles.backends.sglang_utils.arguments import validate_args as validate_sglang_args
from miles.utils.arguments import (
    _compute_custom_inference_engine_provider_path,
    _compute_rollout_external,
    _maybe_apply_dumper_overrides,
    _resolve_api_server_port,
    _resolve_ft_components,
    _resolve_mini_ft_controller_enable,
    _resolve_rollout_functions,
    _resolve_run_uuid,
    _validate_deploy_component,
    _validate_rematerialize_param_from_master_weight,
    get_miles_extra_args_provider,
    miles_validate_args,
    resolve_rollout_function_paths,
    validate_async_off_policy_correction,
    validate_skip_actor_forward_only,
)
from miles.utils.env_report.redaction import _SECRET_ARG_NAMES, _SECRET_ENV_VAR_PATTERN
from miles.utils.ft_utils.health_checker import SimpleHealthCheckerConfig
from miles.utils.function_registry import function_registry
from miles.utils.run_uuid import RUN_UUID_LENGTH, validate_run_uuid

PATH_ARGS = ["--rollout-function-path", "--custom-generate-function-path", "--custom-inference-engine-provider-path"]
REQUIRED_ARGS = ["--rollout-batch-size", "64"]

_MEGATRON_PARALLEL_SIZES: dict[str, int] = {
    "world_size": 8,
    "tensor_model_parallel_size": 2,
    "pipeline_model_parallel_size": 1,
    "context_parallel_size": 1,
}


def _set_megatron_parallel_sizes(args: argparse.Namespace) -> None:
    for name, size in _MEGATRON_PARALLEL_SIZES.items():
        setattr(args, name, size)


# These name a dataset column, a metric or a prompt field, not a credential.
_NOT_ACTUALLY_SECRET_ARG_NAMES = frozenset(
    {
        "ci_metric_checker_key",
        "eval_input_key",
        "eval_label_key",
        "eval_reward_key",
        "eval_tool_key",
        "input_key",
        "label_key",
        "metadata_key",
        "opd_teacher_key",
        "reward_key",
        "tool_key",
    }
)
_SGLANG_ARG_PREFIXES = ("sglang_", "eval_sglang_")
_INHERITED_CREDENTIAL_PATTERN = re.compile(r"^(eval_)?(sglang|router)_(.*_)?(api_keys?|password)$")


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


class TestFullyAsyncDataBufferFlags:
    def test_a_fully_async_run_reaches_the_data_buffer_flags_through_its_rollout_function(self):
        """The flag sits beside --custom-async-data-buffer-path, so a fully async run parses it like any other."""
        argv = [
            "test",
            "--fully-async",
            "--custom-async-data-buffer-path-per-model",
            "solver=pkg.SolverBuffer",
        ] + REQUIRED_ARGS
        with patch.object(sys, "argv", argv):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)
            args, _ = parser.parse_known_args()

        assert args.custom_async_data_buffer_path_per_model == ["solver=pkg.SolverBuffer"]

    def test_a_run_that_is_not_fully_async_never_declares_the_flag(self):
        """The flag belongs to the fully async rollout function, so no other run should accept or expose it."""
        with patch.object(sys, "argv", ["test"] + REQUIRED_ARGS):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)
            args, _ = parser.parse_known_args()

        assert not hasattr(args, "custom_async_data_buffer_path_per_model")


class TestRolloutExternalDerivation:
    def test_static_addrs_imply_external_rollout(self):
        """Giving engine addresses is the whole point of external mode, so no separate flag is needed."""
        args = SimpleNamespace(
            rollout_external_engine_addrs=["host1:8000"], custom_inference_engine_provider_path=None
        )

        assert _compute_rollout_external(args) is True

    def test_a_custom_provider_path_implies_external_rollout(self):
        """A user-supplied provider means miles must not launch engines of its own."""
        args = SimpleNamespace(
            rollout_external_engine_addrs=None, custom_inference_engine_provider_path="my_pkg.my_provider"
        )

        assert _compute_rollout_external(args) is True

    def test_without_either_arg_rollout_stays_internal(self):
        """The default run keeps launching its own engines."""
        args = SimpleNamespace(rollout_external_engine_addrs=None, custom_inference_engine_provider_path=None)

        assert _compute_rollout_external(args) is False

    def test_the_standalone_external_flag_no_longer_exists(self):
        """--rollout-external was replaced by derivation, so the parser must not define it anymore."""
        with patch.object(sys, "argv", ["test"] + REQUIRED_ARGS):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)

        option_strings = {s for action in parser._actions for s in action.option_strings}
        assert "--rollout-external" not in option_strings
        assert "--rollout-external-engine-addrs" in option_strings
        assert "--custom-inference-engine-provider-path" in option_strings


class TestEngineProviderPathAutofill:
    def test_a_user_given_path_is_never_overwritten(self):
        """The custom hook is the escape hatch, so validation must not replace it with a builtin."""
        args = SimpleNamespace(
            rollout_external_engine_addrs=["host1:8000"],
            custom_inference_engine_provider_path="my_pkg.my_provider",
        )

        assert _compute_custom_inference_engine_provider_path(args) == "my_pkg.my_provider"

    def test_static_addrs_fill_in_the_discovery_provider(self):
        """Static addresses mean the built-in discovery provider, chosen once in arg validation."""
        args = SimpleNamespace(
            rollout_external_engine_addrs=["host1:8000"], custom_inference_engine_provider_path=None
        )

        assert _compute_custom_inference_engine_provider_path(args) == (
            "miles.ray.rollout.external_engine_provider.static_inference_engine_provider"
        )

    def test_an_internal_run_fills_in_the_backend_provider(self):
        """Without external args the backend keeps announcing the engines it launches itself."""
        args = SimpleNamespace(rollout_external_engine_addrs=None, custom_inference_engine_provider_path=None)

        assert _compute_custom_inference_engine_provider_path(args) == (
            "miles.ray.specs.inference.backend_inference_engine_provider"
        )


EXTERNAL_ARGS = [
    "--rollout-external-engine-addrs",
    "host1:8000",
    "--rollout-num-gpus",
    "1",
    "--rollout-num-gpus-per-engine",
    "1",
    "--num-rollout",
    "1",
]


class TestExternalRolloutValidation:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(extra + REQUIRED_ARGS)

    def test_static_addrs_derive_both_external_and_the_discovery_provider(self):
        """The helpers are unit tested in isolation, so only the real chain proves the order they run in."""
        args = self._parse(EXTERNAL_ARGS)

        miles_validate_args(args)

        assert args.rollout_external is True
        assert args.custom_inference_engine_provider_path == (
            "miles.ray.rollout.external_engine_provider.static_inference_engine_provider"
        )

    def test_an_internal_run_derives_the_backend_provider(self):
        """Every existing run takes this path, and it must reach the provider the backend announces."""
        args = self._parse(["--num-rollout", "1"])

        miles_validate_args(args)

        assert args.rollout_external is False
        assert args.custom_inference_engine_provider_path == (
            "miles.ray.specs.inference.backend_inference_engine_provider"
        )

    def test_a_custom_provider_path_alone_is_external_and_is_kept(self):
        """A user-supplied provider means miles launches no engines, and its path must survive autofill."""
        args = self._parse(["--custom-inference-engine-provider-path", "my_pkg.my_provider", "--num-rollout", "1"])

        miles_validate_args(args)

        assert args.rollout_external is True
        assert args.custom_inference_engine_provider_path == "my_pkg.my_provider"

    def test_static_addrs_do_not_overrule_a_custom_provider_path(self):
        """Both args together are how a custom provider reads the address book miles parsed."""
        args = self._parse(EXTERNAL_ARGS + ["--custom-inference-engine-provider-path", "my_pkg.my_provider"])

        miles_validate_args(args)

        assert args.custom_inference_engine_provider_path == "my_pkg.my_provider"

    @pytest.mark.parametrize(
        "extra, message",
        [
            (["--prefill-num-servers", "1"], "prefill_num_servers cannot be set"),
            (["--eval-num-gpus", "1"], "eval_num_gpus cannot be set"),
        ],
    )
    def test_an_arg_that_declares_a_second_topology_is_rejected(self, extra, message):
        """Two topologies would size the placement group, the router and the weight-update group
        against different fleets."""
        args = self._parse(EXTERNAL_ARGS + extra)

        with pytest.raises(AssertionError, match=message):
            miles_validate_args(args)

    def test_an_sglang_config_is_rejected_with_external_engines(self, tmp_path):
        """The external topology comes from discovery, so a declared one could only disagree with it."""
        config = tmp_path / "sglang.yaml"
        config.write_text("sglang:\n  - name: default\n    server_groups:\n      - num_gpus: 1\n")
        args = self._parse(EXTERNAL_ARGS + ["--sglang-config", str(config)])

        with pytest.raises(AssertionError, match="sglang_config cannot be set"):
            miles_validate_args(args)

    def test_the_external_pd_router_flag_is_rejected_on_an_internal_run(self):
        """An internal run reads PD off its own config, so the flag could only contradict it."""
        args = self._parse(["--rollout-external-router-pd", "--num-rollout", "1"])

        with pytest.raises(AssertionError, match="rollout-external-router-pd"):
            miles_validate_args(args)

    def test_the_external_pd_router_flag_is_accepted_with_external_engines(self):
        """This is the only channel external PD has, so the guard must not close it."""
        args = self._parse(EXTERNAL_ARGS + ["--rollout-external-router-pd"])

        miles_validate_args(args)

        assert args.rollout_external_router_pd is True

    def test_the_same_args_stay_legal_on_an_internal_run(self):
        """The guards are about the combination, so each half alone must keep working."""
        args = self._parse(["--prefill-num-servers", "1", "--num-rollout", "1"])

        miles_validate_args(args)

        assert args.rollout_external is False


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


class TestClusterBackend:

    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(extra + REQUIRED_ARGS)

    def test_defaults_to_ray(self):
        """Runs that do not mention the flag keep the ray-launched worker behaviour."""
        assert self._parse([]).cluster_backend == "ray"

    @pytest.mark.parametrize("backend", ["ray", "kubernetes"])
    def test_accepts_supported_backends(self, backend):
        """Both supported backends parse into the raw string."""
        assert self._parse(["--cluster-backend", backend]).cluster_backend == backend

    def test_rejects_unknown_backend(self):
        """An unsupported backend name fails at parse time instead of later."""
        with pytest.raises(SystemExit):
            self._parse(["--cluster-backend", "slurm"])

    def test_validation_accepts_kubernetes_now_that_it_provisions_workers(self):
        """The kubernetes backend observes platform-created workers, so validation must let a run reach it."""
        args = self._parse(["--cluster-backend", "kubernetes", "--num-rollout", "1"])

        miles_validate_args(args)

        assert args.cluster_backend == "kubernetes"

    def test_the_custom_config_file_still_decides_the_backend(self, tmp_path):
        """The config file overwrites args after the flags are parsed, so its backend must be the one that survives."""
        config = tmp_path / "override.yaml"
        config.write_text("cluster_backend: kubernetes\n")
        args = self._parse(["--custom-config-path", str(config), "--num-rollout", "1"])

        miles_validate_args(args)

        assert args.cluster_backend == "kubernetes"

    def test_a_kubernetes_run_is_moved_onto_the_mooncake_object_store(self):
        """A ray store reference can only be redeemed by a ray driver, and this run has none."""
        args = self._parse(["--cluster-backend", "kubernetes", "--object-store-backend", "ray", "--num-rollout", "1"])

        miles_validate_args(args)

        assert args.object_store_backend == "mooncake"

    def test_the_override_outlives_the_custom_config_file(self, tmp_path):
        """That file is applied late, so a ray store named there would otherwise survive the override."""
        config = tmp_path / "override.yaml"
        config.write_text("cluster_backend: kubernetes\nobject_store_backend: ray\n")
        args = self._parse(["--custom-config-path", str(config), "--num-rollout", "1"])

        miles_validate_args(args)

        assert args.object_store_backend == "mooncake"

    def test_the_store_this_backend_chose_is_also_configured_by_it(self):
        """The launcher asserts these kwargs exist and rewrites their host to the master it starts, so
        a run that never asked for mooncake in the first place must not have to name them itself: with
        them unset, every kubernetes run using the defaults died before a single pod did any work."""
        args = self._parse(["--cluster-backend", "kubernetes", "--num-rollout", "1"])

        miles_validate_args(args)

        assert ":" in args.mooncake_store_init_kwargs["master_server_address"]
        assert set(args.mooncake_store_init_kwargs) == set(compute_mooncake_init_kwargs())

    def test_a_named_store_configuration_is_left_alone(self):
        """A run that configured the store itself knows something the default cannot."""
        named = '{"master_server_address": "10.0.0.2:60000", "protocol": "rdma"}'
        args = self._parse(
            ["--cluster-backend", "kubernetes", "--mooncake-store-init-kwargs", named, "--num-rollout", "1"]
        )

        miles_validate_args(args)

        assert args.mooncake_store_init_kwargs == {"master_server_address": "10.0.0.2:60000", "protocol": "rdma"}

    def test_a_ray_run_is_not_given_a_store_it_does_not_use(self):
        """The ray store needs none of this, and inventing kwargs would misreport what the run uses."""
        args = self._parse(["--cluster-backend", "ray", "--num-rollout", "1"])

        miles_validate_args(args)

        assert args.mooncake_store_init_kwargs is None

    def test_a_ray_run_may_keep_the_ray_object_store(self):
        """Every existing run takes this path, and nothing about it changed."""
        args = self._parse(["--cluster-backend", "ray", "--object-store-backend", "ray", "--num-rollout", "1"])

        miles_validate_args(args)

        assert args.object_store_backend == "ray"


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


_SHARED_STORE_ARGS = [
    "--object-store-backend",
    "mooncake",
    "--mooncake-store-init-kwargs",
    '{"master_server_address": "the-master:50051"}',
]

_PRIMARY_ARGS = ["--deploy-component", "primary", "--trainer-controller-addrs", "actor=10.0.0.1:8000"]

_SPLIT_RUN_UUID_ARGS = ["--run-uuid", "0123456789abcdef"]

_RAY_RPC_ARGS = ["--cluster-backend", "ray", "--worker-comm-backend", "rpc"]

_RAY_ACTOR_ARGS = ["--cluster-backend", "ray", "--worker-comm-backend", "ray"]


class TestDeployComponent:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        args = parser.parse_args(["--cluster-backend", "kubernetes", *extra, *REQUIRED_ARGS, "--num-rollout", "1"])
        args.ft_components = []
        args.mini_ft_controller_enable = False
        args.use_critic = False
        return args

    def _parse_validated(self, extra):
        return _parse_deploy_args(extra, resolve_fault_tolerance=True)

    def test_defaults_to_deploying_the_whole_run(self):
        """A run that does not mention the flag is one deployment, exactly as before the flag existed."""
        assert self._parse([]).deploy_component == "all"

    def test_rejects_a_component_that_is_not_one_of_the_four(self):
        """The four values partition the run, so a fifth name would deploy an undefined subset."""
        with pytest.raises(SystemExit):
            self._parse(["--deploy-component", "rollout"])

    def test_an_unsplit_run_is_validated_exactly_as_it_was_before_the_flag(self):
        """`all` must stay free of every split-only requirement, or it would break every existing launch."""
        _validate_deploy_component(self._parse([]))

    def test_a_trainer_deployment_needs_no_addresses(self):
        """It calls nobody: the orchestration script calls it, so it has nothing to be told."""
        _validate_deploy_component(self._parse(["--deploy-component", "trainer", *_SHARED_STORE_ARGS]))

    def test_an_inference_deployment_needs_no_addresses_either(self):
        """It serves the engines the run calls into, so nothing addresses anything from here."""
        _validate_deploy_component(self._parse(["--deploy-component", "inference", *_SHARED_STORE_ARGS]))

    def test_an_inference_deployment_has_to_share_an_object_store_too(self):
        """Its engines write rollout data the orchestration script of another deployment reads."""
        with pytest.raises(AssertionError, match="--object-store-backend"):
            _validate_deploy_component(
                self._parse(["--deploy-component", "inference", "--object-store-backend", "ray"])
            )

    def test_a_trainer_deployment_has_to_share_an_object_store(self):
        """A ray reference is redeemable only inside the deployment that made it, and the data crosses deployments."""
        with pytest.raises(AssertionError, match="--object-store-backend"):
            _validate_deploy_component(self._parse(["--deploy-component", "trainer", "--object-store-backend", "ray"]))

    def test_a_trainer_deployment_has_to_be_told_where_the_store_master_is(self):
        """It runs no master of its own, so an unnamed one leaves it writing into a store nobody else reads."""
        with pytest.raises(AssertionError, match="master_server_address"):
            _validate_deploy_component(
                self._parse(["--deploy-component", "trainer", "--object-store-backend", "mooncake"])
            )

    def test_the_store_master_address_has_to_carry_a_port(self):
        """A host without a port cannot be dialed, and the failure would surface as a hang much later."""
        with pytest.raises(AssertionError, match="master_server_address"):
            _validate_deploy_component(
                self._parse(
                    [
                        "--deploy-component",
                        "trainer",
                        "--object-store-backend",
                        "mooncake",
                        "--mooncake-store-init-kwargs",
                        '{"master_server_address": "the-master"}',
                    ]
                )
            )

    def test_a_trainer_deployment_is_still_asked_where_the_store_master_is(self):
        """The kubernetes default that spares a whole run from naming its own store must not answer for a
        deployment that runs no master, or the assert above never fires again and the trainer quietly dials
        a master that is not there."""
        args = _parse_deploy_args(
            ["--deploy-component", "trainer", "--object-store-backend", "mooncake", *_SPLIT_RUN_UUID_ARGS]
        )

        with pytest.raises(AssertionError, match="master_server_address"):
            miles_validate_args(args)

    def test_a_primary_deployment_is_still_given_the_store_it_starts(self):
        """It runs the master itself, so the launcher's assert on these kwargs must still find them."""
        args = _parse_deploy_args([*_PRIMARY_ARGS, *_SPLIT_RUN_UUID_ARGS])

        miles_validate_args(args)

        assert set(args.mooncake_store_init_kwargs) == set(compute_mooncake_init_kwargs())

    def test_a_primary_deployment_has_to_be_told_where_the_trainer_is(self):
        """Nothing derives another release's pod names, so an unnamed trainer is unreachable."""
        with pytest.raises(AssertionError, match="--trainer-controller-addrs"):
            _validate_deploy_component(self._parse(["--deploy-component", "primary", *_SHARED_STORE_ARGS]))

    def test_a_fully_addressed_primary_deployment_validates(self):
        """The address is what makes an orchestration script able to run without its own trainer."""
        _validate_deploy_component(self._parse([*_PRIMARY_ARGS, *_SHARED_STORE_ARGS]))

    def test_a_primary_deployment_shares_an_object_store_too(self):
        """It writes the rollout data the trainer deployment reads, which its own store alone cannot carry."""
        with pytest.raises(AssertionError, match="--object-store-backend"):
            _validate_deploy_component(self._parse([*_PRIMARY_ARGS, "--object-store-backend", "ray"]))

    @pytest.mark.parametrize("component", ["trainer", "all"])
    def test_refuses_a_static_address_for_the_trainer_this_launch_deploys_itself(self, component):
        """A static address describes what another launch deploys, so one for our own release is a contradiction."""
        with pytest.raises(AssertionError, match="--trainer-controller-addrs"):
            _validate_deploy_component(
                self._parse(
                    [
                        "--deploy-component",
                        component,
                        "--trainer-controller-addrs",
                        "10.0.0.1:8000",
                        *_SHARED_STORE_ARGS,
                    ]
                )
            )

    def test_a_split_run_keeps_the_api_server_of_the_half_that_drives_it(self):
        """Its api server answers for the cells it deploys, which is a smaller set, not an empty one."""
        args = self._parse_validated([*_PRIMARY_ARGS, "--use-fault-tolerance", *_SHARED_STORE_ARGS])

        assert args.api_server_port

    def test_a_trainer_deployment_keeps_the_fault_tolerance_of_its_own_cells(self):
        """Its controller watches its own ranks, and it serves them from an api server of its own."""
        _validate_deploy_component(
            self._parse_validated(
                [
                    "--deploy-component",
                    "trainer",
                    "--use-fault-tolerance",
                    "--ft-components",
                    "train",
                    *_SHARED_STORE_ARGS,
                ]
            )
        )

    def test_refuses_a_trainer_deployment_asked_to_answer_for_cells_it_does_not_deploy(self):
        """rollout defaults on, and its engines live in another release, which this launch cannot suspend."""
        with pytest.raises(AssertionError, match="--ft-components train"):
            _validate_deploy_component(
                self._parse_validated(["--deploy-component", "trainer", "--use-fault-tolerance", *_SHARED_STORE_ARGS])
            )

    def test_refuses_to_split_a_colocated_run(self):
        """Colocated trainers and engines share gpus, so they can only be installed as one unit."""
        with pytest.raises(AssertionError, match="--colocate"):
            _validate_deploy_component(self._parse(["--deploy-component", "trainer", "--colocate"]))

    @pytest.mark.parametrize("component", ["trainer", "primary"])
    def test_refuses_to_split_a_ray_run_whose_workers_are_actor_handles(self, component):
        """A ray-comm worker is reached by a handle of its own launch, so the other half could never dial it."""
        with pytest.raises(AssertionError, match="--worker-comm-backend ray"):
            _validate_deploy_component(
                self._parse([*_RAY_ACTOR_ARGS, "--deploy-component", component, *_SHARED_STORE_ARGS])
            )

    def test_splits_a_ray_run_whose_workers_speak_rpc(self):
        """Its trainer controllers listen on rpc ports, which a launch against another ray cluster can dial."""
        _validate_deploy_component(self._parse([*_RAY_RPC_ARGS, "--deploy-component", "trainer", *_SHARED_STORE_ARGS]))

    def test_the_dialing_half_of_a_ray_run_splits_too(self):
        """The primary half runs the script, and rpc is what lets it call a trainer it never deployed."""
        _validate_deploy_component(self._parse([*_RAY_RPC_ARGS, *_PRIMARY_ARGS, *_SHARED_STORE_ARGS]))

    def test_refuses_a_primary_deployment_that_leaves_one_of_its_roles_unaddressed(self):
        """Installing the release first and finding the critic missing at init leaves a broken run running."""
        args = self._parse([*_PRIMARY_ARGS, *_SHARED_STORE_ARGS])
        args.use_critic = True

        with pytest.raises(AssertionError, match=r"--trainer-controller-addrs must name each of .* exactly once"):
            _validate_deploy_component(args)

    def test_refuses_an_address_that_is_not_a_host_and_port(self):
        """An unparseable address is found by the launch that installs the release, not by the one that dials it."""
        with pytest.raises(AssertionError, match="host:port"):
            _validate_deploy_component(
                self._parse(
                    [
                        "--deploy-component",
                        "primary",
                        "--trainer-controller-addrs",
                        "actor=10.0.0.1",
                        *_SHARED_STORE_ARGS,
                    ]
                )
            )


class TestRunUuidOfASplitRun:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args([*extra, *REQUIRED_ARGS, "--num-rollout", "1"])

    def test_a_ray_split_launch_has_to_be_told_the_run_uuid(self):
        """Each launch would otherwise invent its own, and the handshake would refuse the other deployment."""
        args = self._parse([*_RAY_RPC_ARGS, "--deploy-component", "trainer"])

        with pytest.raises(AssertionError, match="--run-uuid"):
            _resolve_run_uuid(args)

    def test_a_ray_split_launch_that_was_told_one_keeps_it(self):
        """The two launches are joined by nothing else, so the value has to survive verbatim."""
        args = self._parse([*_RAY_RPC_ARGS, "--deploy-component", "trainer", "--run-uuid", "0123456789abcdef"])

        assert _resolve_run_uuid(args) == "0123456789abcdef"

    def test_an_unsplit_run_still_invents_its_own(self):
        """One launch is the whole run, so nothing else has to agree with it."""
        args = self._parse([])

        assert len(_resolve_run_uuid(args)) == RUN_UUID_LENGTH

    def test_a_kubernetes_split_launch_has_to_be_told_it_too(self):
        """Its launcher is given the uuid and stamps it on every part, so no backend mints one of its own."""
        args = self._parse(["--cluster-backend", "kubernetes", "--deploy-component", "trainer"])

        with pytest.raises(AssertionError, match="--run-uuid"):
            _resolve_run_uuid(args)


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


class TestFaultToleranceResolutionOrder:
    def _validate(self, tmp_path: Path, extra: list[str], yaml_body: str) -> argparse.Namespace:
        config_path = tmp_path / "custom.yaml"
        config_path.write_text(yaml_body)
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        args = parser.parse_args(
            extra + ["--custom-config-path", str(config_path), "--num-rollout", "1"] + REQUIRED_ARGS
        )
        _set_megatron_parallel_sizes(args)
        miles_validate_args(args)
        return args

    def test_the_config_file_can_turn_fault_tolerance_off(self, tmp_path):
        """Resolving ft_components before the file override left implicit healing on for a run that asked for none."""
        args = self._validate(tmp_path, ["--use-fault-tolerance"], "use_fault_tolerance: false\n")

        assert (args.ft_components, args.mini_ft_controller_enable, args.api_server_port) == ([], False, 0)

    def test_the_config_file_can_turn_fault_tolerance_on(self, tmp_path):
        """A file-only opt-in must reach the same defaults the flag would have produced."""
        args = self._validate(tmp_path, [], "use_fault_tolerance: true\n")

        assert (args.ft_components, args.mini_ft_controller_enable) == (["rollout"], True)

    def test_the_config_file_can_select_the_train_component(self, tmp_path):
        """Picking the component list after the derivations left a train-side run without the state it needs."""
        args = self._validate(tmp_path, ["--use-fault-tolerance"], "ft_components: [train]\n")

        assert args.ft_components == ["train"]
        assert (args.indep_dp, args.enable_witness, args.non_persistent_ckpt_type) == (True, True, "local")
        assert args.world_size == 2

    def test_the_config_file_can_cancel_the_train_component(self, tmp_path):
        """A file that switches fault tolerance off must also undo the per-cell world the train component implies."""
        args = self._validate(
            tmp_path, ["--use-fault-tolerance", "--ft-components", "train"], "use_fault_tolerance: false\n"
        )

        assert (args.ft_components, args.indep_dp, args.enable_witness) == ([], False, False)
        assert args.world_size == 8

    def test_the_config_file_keeps_an_explicit_api_server_port(self, tmp_path):
        """The implicit healing defaults must follow the port the file asks for, not the one the flag implied."""
        args = self._validate(tmp_path, [], "use_fault_tolerance: true\napi_server_port: 0\n")

        assert (args.ft_components, args.api_server_port, args.mini_ft_controller_enable) == (["rollout"], 0, False)


class TestCustomConfigAppliedBeforeDerivedArgs:
    def _parse(self, tmp_path: Path, extra: list[str], yaml_body: str) -> argparse.Namespace:
        config_path = tmp_path / "custom.yaml"
        config_path.write_text(yaml_body)
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(
            extra + ["--custom-config-path", str(config_path), "--num-rollout", "1"] + REQUIRED_ARGS
        )

    def test_a_dashboard_switched_on_by_the_config_file_is_still_checked(self, tmp_path):
        """Checking the dashboard before the file override let a file-only opt-in start without a dump directory."""
        args = self._parse(tmp_path, [], "use_miles_dashboard: true\n")

        with pytest.raises(AssertionError, match="--dump-details is required"):
            miles_validate_args(args)

    def test_a_dashboard_switched_off_by_the_config_file_drops_its_requirement(self, tmp_path):
        """A run whose file turns the dashboard off must not be rejected for telemetry it will never write."""
        args = self._parse(tmp_path, ["--use-miles-dashboard"], "use_miles_dashboard: false\n")
        miles_validate_args(args)

        assert args.use_miles_dashboard is False

    def test_eval_prompt_data_from_the_config_file_is_expanded_into_datasets(self, tmp_path):
        """Building the dataset list before the file override left the file's paths as an unparsed raw list."""
        args = self._parse(tmp_path, [], "eval_prompt_data: [/data/aime.jsonl]\n")
        miles_validate_args(args)

        assert [(dataset.name, dataset.path) for dataset in args.eval_datasets] == [("aime", "/data/aime.jsonl")]
        assert args.eval_prompt_data == ["aime", "/data/aime.jsonl"]


def test_stream_optimizer_state_to_disk_rejects_fault_tolerant_training():
    """Deriving indep_dp after the disk-stream asserts would have let an unsupported pair through."""
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(
        [
            "--stream-optimizer-state-to-disk",
            "--use-fault-tolerance",
            "--ft-components",
            "train",
            "--num-rollout",
            "1",
        ]
        + REQUIRED_ARGS
    )
    args.optimizer = "adam"
    args.use_distributed_optimizer = True
    args.multi_lora = False
    args.optimizer_cpu_offload = False
    args.offload_optimizer_states = False
    args.use_precision_aware_optimizer = False
    _set_megatron_parallel_sizes(args)

    with pytest.raises(AssertionError, match="does not support --indep-dp"):
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


class TestCheckpointLoadFallbackWiring:
    def _validate(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        args = parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)
        miles_validate_args(args)
        return args

    def test_a_fresh_ppo_run_starts_the_actor_and_its_critic_from_the_reference_weights(self, tmp_path):
        """The fallback has to run before critic_load is derived, or the critic resumes from a dir nobody wrote."""
        ref_load = tmp_path / "ref"
        ref_load.mkdir()

        args = self._validate(
            ["--advantage-estimator", "ppo", "--load", str(tmp_path / "absent"), "--ref-load", str(ref_load)]
        )

        assert args.load == str(ref_load)
        assert args.critic_load == str(ref_load)

    def test_an_existing_checkpoint_is_left_alone(self, tmp_path):
        """A real resume must keep --load, which is also what the critic inherits."""
        load = tmp_path / "save"
        load.mkdir()
        (load / "latest_checkpointed_iteration.txt").write_text("10")

        args = self._validate(["--advantage-estimator", "ppo", "--load", str(load), "--ref-load", str(tmp_path)])

        assert args.load == str(load)
        assert args.critic_load == str(load)


class TestSessionServerV2Validation:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)

    @pytest.mark.parametrize(
        ("extra", "flag"),
        [
            (["--group-rm"], "--group-rm"),
            (["--partial-rollout"], "--partial-rollout"),
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

    def test_session_server_rejects_abort(self):
        args = self._parse(["--use-session-server", "--pause-generation-mode", "abort"])

        with pytest.raises(
            AssertionError, match="--use-session-server is incompatible with --pause-generation-mode=abort"
        ):
            miles_validate_args(args)

    def test_abort_without_session_server_passes(self):
        miles_validate_args(self._parse(["--pause-generation-mode", "abort"]))

    @pytest.mark.parametrize("mode", ["retract", "in_place"])
    def test_session_server_accepts_non_abort_modes(self, mode):
        miles_validate_args(self._parse(["--use-session-server", "--pause-generation-mode", mode]))


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


def test_critic_is_accepted_on_the_only_trainer(tmp_path):
    """Shared actor/critic PPO used to be rejected on the cell based trainer, which is now the only one."""
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(
        ["--advantage-estimator", "ppo", "--hf-checkpoint", str(tmp_path), "--num-rollout", "1"] + REQUIRED_ARGS
    )

    miles_validate_args(args)

    assert args.use_critic is True


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

    def test_is_accepted_on_the_only_trainer(self):
        """Multi-LoRA used to be rejected on the cell based trainer, which is now the only one."""
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

    def test_a_tuned_rollout_debounce_reaches_the_shared_config(self):
        """The failure threshold is what debounces transient blips, so the flag must reach the checker's config."""
        config = SimpleHealthCheckerConfig.from_args(
            self._parse(["--rollout-health-check-failure-threshold", "7"]), prefix="rollout_health_check"
        )

        assert config.failure_threshold == 7

    def test_the_trainer_heartbeat_keeps_its_own_debounce(self):
        """The rollout default must not be pushed down into the shared config: a trainer heartbeat
        shares an RPC channel with the train step, so one slow reply is a blip, not a dead cell."""
        assert self._parse([]).trainer_heartbeat_checker_failure_threshold == 3


class TestMiniFtControllerArguments:
    def _validate(self, extra: list[str]):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        args = parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)
        miles_validate_args(args)
        return args

    def test_fault_tolerance_alone_turns_the_healing_loop_on(self):
        """Asking for fault tolerance heals on its own, so the loop must come up without a second flag."""
        assert self._validate(["--use-fault-tolerance"]).mini_ft_controller_enable is True

    def test_the_negative_flag_turns_the_healing_loop_back_off(self):
        """A run that drives healing from outside needs a way to keep the health reporting without the loop."""
        args = self._validate(["--use-fault-tolerance", "--no-mini-ft-controller-enable"])

        assert args.mini_ft_controller_enable is False

    def test_asking_for_the_loop_without_a_port_is_rejected_at_launch(self):
        """The loop drives cells over the api server port, so a disabled port would fail every poll instead."""
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        args = parser.parse_args(
            ["--mini-ft-controller-enable", "--api-server-port", "0", "--num-rollout", "1"] + REQUIRED_ARGS
        )

        with pytest.raises(ValueError, match="requires --api-server-port to be set"):
            miles_validate_args(args)


class TestSessionServerArguments:
    def _parse(self, extra: list[str]):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(extra + REQUIRED_ARGS)

    def test_the_instance_count_and_base_port_come_from_the_flags(self):
        """A network policy whitelists a known range, so the base port is one scalar the instances offset from."""
        args = self._parse(["--session-server-workers", "3", "--session-server-port", "41000"])

        assert args.session_server_workers == 3
        assert args.session_server_port == 41000

    def test_an_unset_base_port_leaves_the_default_instances_dynamically_placed(self):
        """Without the flag the port stays unset so the placement allocates one, and the instance count is the default."""
        args = self._parse([])

        assert args.session_server_workers == 32
        assert args.session_server_port is None


class TestSecretArgumentsAreClassified:
    def _declared_names(self) -> set[str]:
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        # The eval sglang flags default to SUPPRESS, so parsing alone would not materialise them.
        return {action.dest for action in parser._actions}

    def test_every_secret_looking_miles_flag_is_either_redacted_or_declared_harmless(self):
        """The env report hashes args by an explicit list, so a new credential flag would leak until listed."""
        suspicious = {
            name
            for name in self._declared_names()
            if _SECRET_ENV_VAR_PATTERN.search(name) and not name.startswith(_SGLANG_ARG_PREFIXES)
        }

        assert suspicious - _SECRET_ARG_NAMES == _NOT_ACTUALLY_SECRET_ARG_NAMES, (
            "an argument's name looks like a credential; add it to _SECRET_ARG_NAMES in env_report/redaction.py so the env "
            "report hashes it, or to _NOT_ACTUALLY_SECRET_ARG_NAMES here to say it names something else"
        )

    def test_every_credential_inherited_from_sglang_and_the_router_is_redacted(self):
        """sglang and the router contribute api keys and key passwords that land in the args dump verbatim."""
        credentials = {name for name in self._declared_names() if _INHERITED_CREDENTIAL_PATTERN.search(name)}

        assert credentials >= {"sglang_api_key", "eval_sglang_api_key", "router_api_key"}
        assert credentials <= _SECRET_ARG_NAMES


def test_a_run_without_a_policy_id_still_carries_the_attribute():
    """megatron declares no --trainer-model-id, so every log point relies on miles defaulting the attribute."""
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    args = parser.parse_args(REQUIRED_ARGS)

    assert args.trainer_model_id is None


def test_the_megatron_config_flag_defaults_to_none(tmp_path):
    """Without the flag a run is single policy, and the flag takes a path the whole run can read."""
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    assert parser.parse_args(REQUIRED_ARGS).megatron_config is None
    assert parser.parse_args(["--megatron-config", str(tmp_path / "x.yaml")] + REQUIRED_ARGS).megatron_config == str(
        tmp_path / "x.yaml"
    )


class TestMilesValidateArgsCheckpointResolution:
    @staticmethod
    def _parse(extra, tmp_path):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        # megatron owns --finetune and the fallback only ever turns it on, so the run that
        # leaves it alone has to start from the default megatron would have given it
        parser.set_defaults(finetune=False)
        return parser.parse_args(
            ["--hf-checkpoint", str(tmp_path), "--ref-load", str(tmp_path), "--num-rollout", "1"]
            + extra
            + REQUIRED_ARGS
        )

    def test_a_single_policy_run_still_resolves_its_checkpoint_fallback(self, tmp_path):
        """The fallback is what lets a fresh run start from --ref-load, and it must survive the multi policy fork."""
        args = self._parse([], tmp_path)

        miles_validate_args(args)

        assert (args.load, args.finetune, args.start_rollout_id) == (str(tmp_path), True, 0)

    def test_a_multi_policy_run_leaves_the_global_load_and_save_untouched(self, tmp_path):
        """Each trainer resolves its own fallback later; settling it globally would point every policy at one dir."""
        args = self._parse(["--megatron-config", encode_megatron_config("a", "b")], tmp_path)

        miles_validate_args(args)

        assert (args.load, args.finetune, args.start_rollout_id) == (None, False, None)

    def test_a_single_trainer_config_also_defers_the_fallback_to_the_overlay(self, tmp_path):
        """That trainer may override --ref-load, and a fallback settled before the overlay would ignore it."""
        args = self._parse(["--megatron-config", encode_megatron_config("a")], tmp_path)

        miles_validate_args(args)

        assert (args.load, args.finetune, args.start_rollout_id) == (None, False, None)
