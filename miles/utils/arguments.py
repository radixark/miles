import argparse
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
from typing import Any

import yaml

from miles.backends.fsdp_utils.config import FsdpArgsNamespace
from miles.backends.megatron_utils.megatron_config import (
    ACTOR_ROLE,
    CRITIC_ROLE,
    has_megatron_checkpoint,
    resolve_args_checkpoint_load,
    resolve_megatron_config,
)
from miles.backends.sglang_utils.arguments import collect_eval_sglang_overrides
from miles.backends.sglang_utils.arguments import validate_args as sglang_validate_args
from miles.backends.sglang_utils.sglang_config import SglangConfig
from miles.dashboard.args import validate_dashboard_args
from miles.ray.specs.train import external_trainer_controller_addrs
from miles.rollout.checkpoint_eval import is_checkpoint_eval_fn
from miles.utils.args.configs.algo import AlgoConfig
from miles.utils.args.configs.backend_fields import TrainerBackendTraitConfig
from miles.utils.args.configs.ci import CiConfig
from miles.utils.args.configs.cluster import ClusterConfig
from miles.utils.args.configs.custom_megatron_plugins import CustomMegatronPluginsConfig, Dsv4MegatronPluginsConfig
from miles.utils.args.configs.dashboard import DashboardConfig
from miles.utils.args.configs.data import DataConfig
from miles.utils.args.configs.debug import DebugConfig
from miles.utils.args.configs.eval import EvalConfig
from miles.utils.args.configs.fault_tolerance import _DEFAULT_FT_API_SERVER_PORT, FaultToleranceConfig
from miles.utils.args.configs.lora import LoraConfig
from miles.utils.args.configs.mlflow import MlflowConfig
from miles.utils.args.configs.mtp_training import MtpTrainingConfig
from miles.utils.args.configs.network import NetworkConfig
from miles.utils.args.configs.on_policy_distillation import OnPolicyDistillationConfig
from miles.utils.args.configs.prefill_decode_disaggregation import PrefillDecodeDisaggregationConfig
from miles.utils.args.configs.prometheus import PrometheusConfig
from miles.utils.args.configs.reward_model import RewardModelConfig
from miles.utils.args.configs.rollout import RolloutRelatedConfig
from miles.utils.args.configs.rollout_buffer import RolloutBufferConfig
from miles.utils.args.configs.router import RouterConfig
from miles.utils.args.configs.run_uuid import RunUuidConfig
from miles.utils.args.configs.session import SessionConfig
from miles.utils.args.configs.tensorboard import TensorboardConfig
from miles.utils.args.configs.train import TrainConfig
from miles.utils.args.configs.wandb import WandbConfig
from miles.utils.args.custom_function import add_user_provided_function_arguments, resolve_custom_function_configs
from miles.utils.args.runtime import AllConfig, OrchestratorConfig
from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType
from miles.utils.environ import use_legacy_rollout_v1
from miles.utils.eval_config import EvalDatasetConfig, build_eval_dataset_configs, ensure_dataset_list
from miles.utils.file_arg_utils import resolve_file_arg
from miles.utils.hf_config import is_dsa, load_hf_config
from miles.utils.logging_utils import configure_logger_raw
from miles.utils.lora import is_lora_enabled
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp
from miles.utils.object_store import ObjectStoreBackend
from miles.utils.object_store_config import (
    MOONCAKE_MASTER_ADDRESS_KEY,
    compute_mooncake_init_kwargs_from_env,
    compute_mooncake_init_kwargs_vanilla,
)
from miles.utils.run_uuid import generate_run_uuid, validate_run_uuid
from miles.utils.tracking_utils.ci_history import RECORD_DIR_ENV
from miles.utils.workers.naming import DEPLOY_INSTANCE_ID_MAX_LENGTH, DNS_LABEL_PATTERN
from miles.utils.workers.types import ClusterBackend, DeployComponent, WorkerCommBackend, resolve_worker_comm_backend
from miles.utils.workers.worker_provider.static import parse_host_and_port

logger = logging.getLogger(__name__)


def resolve_rollout_function_paths(args: argparse.Namespace) -> None:
    """The (rollout, eval) function paths the arguments select."""
    if use_legacy_rollout_v1():
        standard_path = "miles.rollout.sglang_rollout.generate_rollout"
    else:
        standard_path = "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn"
    rollout_path = args.rollout_function_path or standard_path
    if args.fully_async:
        rollout_path = "miles.rollout.fully_async_rollout.FullyAsyncRolloutFn"
    # Resolved after the override: shared-engine eval must reach the producer it pauses.
    eval_path = args.eval_function_path or rollout_path
    args.rollout_function_path = rollout_path
    args.eval_function_path = eval_path


def driver_owns_generation_pause(args) -> bool:
    return args.fully_async and args.colocate


def _resolve_rollout_functions(args) -> None:
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and not use_legacy_rollout_v1():
        raise ValueError(
            "--mask-offpolicy-in-partial-rollout does not re-extend the loss mask on the "
            "class-based rollout path yet; set MILES_USE_LEGACY_ROLLOUT_V1=1"
        )
    if args.fully_async:
        assert (
            not use_legacy_rollout_v1()
        ), "--fully-async needs the class-based rollout API; unset MILES_USE_LEGACY_ROLLOUT_V1"
        # Runs after validate_multi_lora_args, which selects a rollout function of its own.
        assert not args.multi_lora, "--fully-async and multi-LoRA select different rollout functions"
        assert (
            args.rollout_function_path is None
        ), "--fully-async and --rollout-function-path both select a rollout function; pass only one"
        if args.colocate:
            assert args.train_backend != "fsdp", (
                "--fully-async --colocate needs the megatron IPC weight updater; the FSDP updater still "
                "pauses and resumes generation on its own"
            )
            assert args.pause_generation_mode != "in_place", (
                "--fully-async --colocate releases the KV cache to make room for training, so the "
                "in_place promise to keep it cannot hold: use --pause-generation-mode retract"
            )
            assert "rollout" not in args.ft_components, (
                "--fully-async --colocate does not support rollout fault tolerance: a cell replaced while "
                "generation is paused for training would neither inherit the pause nor get its KV cache back "
                "before serving"
            )
        assert not args.partial_rollout, "--fully-async does not support --partial-rollout"
        assert args.pause_generation_mode != "abort", (
            "--fully-async cannot use --pause-generation-mode abort: generation is always in flight, "
            "so every weight update would kill it and force a full regeneration"
        )
        assert (
            not args.recompute_logprobs_via_prefill
        ), "--fully-async does not support --recompute-logprobs-via-prefill"
        assert (
            args.rollout_all_samples_process_path is None
        ), "--fully-async does not support --rollout-all-samples-process-path"

    user_eval_path = args.eval_function_path
    resolve_rollout_function_paths(args)
    # An inherited eval path is the rollout fn serving eval itself, never a checkpoint
    # backend: skip the resolve so custom rollout modules are not imported on the driver.
    checkpoint_backend = user_eval_path is not None and is_checkpoint_eval_fn(args.eval_function_path)
    assert not (args.eval_num_gpus > 0 and checkpoint_backend), (
        "--eval-num-gpus and a CheckpointEvalFn --eval-function-path each select an eval "
        "backend; the fleet would boot and then hand the work to the other one."
    )
    assert not (
        args.eval_num_gpus > 0 and _compute_rollout_external(args)
    ), "eval_num_gpus cannot be set with external rollout engines."
    args.eval_uses_snapshots = args.eval_num_gpus > 0 or checkpoint_backend


def reset_arg(parser: argparse.ArgumentParser, name: str, **kwargs: Any) -> None:
    """
    Reset the default value of a Megatron argument.
    :param parser: The argument parser.
    :param name: The name of the argument to reset.
    :param default: The new default value.
    """
    for action in parser._actions:
        if name in action.option_strings:
            _assert_reset_arg_compatible(parser=parser, action=action, name=name, kwargs=kwargs)
            if "default" in kwargs:
                action.default = kwargs["default"]
            break
    else:
        parser.add_argument(name, **kwargs)


def _assert_reset_arg_compatible(
    *, parser: argparse.ArgumentParser, action: argparse.Action, name: str, kwargs: dict[str, Any]
) -> None:
    action_type = kwargs.get("action", "store")
    expected_action = parser._registry_get("action", action_type, action_type)
    assert (
        type(action) is expected_action
    ), f"Cannot reset {name}: action {type(action)} does not match {expected_action}"
    expected = {"type": None, **kwargs}
    for key, value in expected.items():
        if key in {"action", "default", "help"}:
            continue
        actual = vars(action)[key]
        assert actual == value, f"Cannot reset {name}: {key}={actual!r} does not match {value!r}"


def get_miles_extra_args_provider(
    add_custom_arguments: Callable[[argparse.ArgumentParser], argparse.ArgumentParser] | None = None,
) -> Callable[[argparse.ArgumentParser], argparse.ArgumentParser]:
    def add_miles_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Add custom arguments in front to prevent overwritten some miles arguments.
        if add_custom_arguments is not None:
            parser = add_custom_arguments(parser)

        RunUuidConfig.add_arguments(parser=parser)
        ClusterConfig.add_arguments(parser=parser)
        TrainConfig.add_arguments(parser=parser)
        RolloutRelatedConfig.add_arguments(parser=parser)
        FaultToleranceConfig.add_arguments(parser=parser)
        DataConfig.add_arguments(parser=parser)
        EvalConfig.add_arguments(parser=parser)
        AlgoConfig.add_arguments(parser=parser)
        TrainerBackendTraitConfig.add_arguments(parser=parser)
        reset_arg(parser=parser, name="--lr", type=float, default=1e-6)
        reset_arg(parser=parser, name="--clip-grad", type=float, default=1.0)
        reset_arg(parser=parser, name="--calculate-per-token-loss", action="store_true")
        reset_arg(
            parser=parser,
            name="--no-save-optim",
            action="store_true",
            default=False,
            help=(
                "If set, do not save the optimizer state when saving checkpoints. "
                "This reduces checkpoint size but disables training resumption from the saved checkpoint."
            ),
        )
        OnPolicyDistillationConfig.add_arguments(parser=parser)
        LoraConfig.add_arguments(parser=parser)
        WandbConfig.add_arguments(parser=parser)
        MlflowConfig.add_arguments(parser=parser)
        TensorboardConfig.add_arguments(parser=parser)
        PrometheusConfig.add_arguments(parser=parser)
        DashboardConfig.add_arguments(parser=parser.add_argument_group("miles dashboard"))
        RouterConfig.add_arguments(parser=parser)
        DebugConfig.add_arguments(parser=parser)
        SglangConfig.add_arguments(parser)
        # required whenever expert projections are LoRA targets, inert otherwise
        # (sglang's own default is False)
        parser.set_defaults(sglang_lora_use_virtual_experts=True)
        SessionConfig.add_arguments(parser=parser)
        NetworkConfig.add_arguments(parser=parser)
        RewardModelConfig.add_arguments(parser=parser)
        RolloutBufferConfig.add_arguments(parser=parser)
        MtpTrainingConfig.add_arguments(parser=parser)
        reset_arg(parser=parser, name="--mtp-num-layers", type=int, default=None)
        reset_arg(parser=parser, name="--mtp-loss-scaling-factor", type=float, default=0.2)
        PrefillDecodeDisaggregationConfig.add_arguments(parser=parser)
        CiConfig.add_arguments(parser=parser)
        CustomMegatronPluginsConfig.add_arguments(parser=parser)
        Dsv4MegatronPluginsConfig.add_arguments(parser=parser)
        parser = add_user_provided_function_arguments(parser, modify_args=resolve_rollout_function_paths)

        reset_arg(
            parser,
            "--custom-config-path",
            type=str,
            default=None,
            help="Path to the YAML config for custom function arguments, or an inline `base64:<payload>`.",
        )
        reset_arg(
            parser,
            "--megatron-config",
            type=str,
            default=None,
            help=(
                "Path to a YAML config naming every trainer (or an inline `base64:<payload>`), "
                "symmetric to --sglang-config. Format: "
                "`trainers: [{model_id: ..., role: ..., trainer_id: ..., overrides: {lr: ...}}]`. "
                "Each `model_id` is the policy model id: it is what a custom rollout function writes into "
                "Sample.trainer_model_id, and it must match a --sglang-config model with update_weights: true. "
                "Each `role` defaults to 'actor', and each `trainer_id` addresses one trainer controller and "
                "its engine pool, defaulting to `<model_id>-<role>`. "
                "Each `overrides` mapping overrides the base CLI arguments for that trainer only. Omitting the flag "
                "is a single policy run. Several policies require train_multi_policy.py."
            ),
        )
        parser.set_defaults(trainer_id=ACTOR_ROLE, trainer_model_id=None)

        return parser

    return add_miles_arguments


def parse_args(
    add_custom_arguments: Callable[[argparse.ArgumentParser], argparse.ArgumentParser] | None = None,
) -> AllConfig | OrchestratorConfig:
    args, _ = parse_args_and_get_parser(add_custom_arguments=add_custom_arguments)
    return args


def parse_args_and_get_parser(
    add_custom_arguments: Callable[[argparse.ArgumentParser], argparse.ArgumentParser] | None = None,
) -> tuple[AllConfig | OrchestratorConfig, argparse.ArgumentParser]:
    # Users may call `parse_args` very early, thus we ensure logger is configured here
    configure_logger_raw("main")

    transport_parser = argparse.ArgumentParser(add_help=False)
    transport_parser.add_argument("--config-json")
    transport, remaining = transport_parser.parse_known_args()
    if transport.config_json is not None:
        if remaining:
            raise ValueError("Serialized orchestration config cannot be combined with CLI arguments")
        payload = json.loads(transport.config_json)
        if not isinstance(payload, dict):
            raise ValueError("Serialized orchestration configuration must be a JSON object")
        return OrchestratorConfig.model_validate(payload), transport_parser

    add_miles_arguments = get_miles_extra_args_provider(add_custom_arguments)
    parser: argparse.ArgumentParser | None = None
    # TODO: Revisit this after Zhichen's training backend refactor.
    training_backend_arg_names: set[str] = set()

    def add_miles_arguments_and_capture_parser(value: argparse.ArgumentParser) -> argparse.ArgumentParser:
        nonlocal parser
        training_backend_arg_names.update(action.dest for action in value._actions)
        parser = add_miles_arguments(value)
        return parser

    backend = parse_args_train_backend()
    if backend == "megatron":
        from miles.backends.megatron_utils.arguments import parse_args as megatron_parse_args
        from miles.backends.megatron_utils.arguments import set_default_megatron_args
        from miles.backends.megatron_utils.arguments import validate_args as megatron_validate_args

        args = megatron_parse_args(extra_args_provider=add_miles_arguments_and_capture_parser)
        previous_arg_names = set(vars(args))
        args.compress_ratios = None
        if args.hf_checkpoint:
            hf_config = load_hf_config(args.hf_checkpoint)
            args.compress_ratios = getattr(
                hf_config, "compress_ratios", None
            )  # config-access-exempt: model-family schemas differ in optional compress_ratios metadata
            hf_validate_args(args, hf_config)

            if is_dsa(hf_config):
                args.indexer_rope_interleave = bool(
                    getattr(hf_config, "indexer_rope_interleave", False)
                )  # config-access-exempt: model-family schemas differ in optional indexer_rope_interleave metadata
                logger.info(f"Setting indexer_rope_interleave: {args.indexer_rope_interleave} into args")

        # TODO: unify this .rank and .world_size w/ indep_dp logics
        args.rank = 0
        args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
        args = set_default_megatron_args(args)
        training_backend_arg_names.update(vars(args).keys() - previous_arg_names)
    else:
        from miles.backends.fsdp_utils.arguments import load_fsdp_args

        args = load_fsdp_args(extra_args_provider=add_miles_arguments_and_capture_parser)
        # TODO: unify this .rank and .world_size w/ indep_dp logics
        args.rank = 0  # Primary process rank for wandb initialization
        args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node

        if args.hf_checkpoint:
            args.num_layers = resolve_fsdp_num_layers(load_hf_config(args.hf_checkpoint))
        else:
            args.num_layers = None

        assert args.context_parallel_size == 1, "Context parallelism is not supported for FSDP backend."

    # On iff the CI harness injected MILES_CI_GATE_RECORD_DIR (the same env var
    # locates the per-test record). No CLI flag: non-CI runs always stay False.
    args.ci_enable_metrics_capture = bool(os.environ.get(RECORD_DIR_ENV))

    miles_validate_args(args)

    if backend == "megatron":
        previous_arg_names = set(vars(args))
        megatron_validate_args(args)

        # always use varlen
        args.variable_seq_lengths = True
        if args.moe_token_dispatcher_type == "allgather":
            logger.info(
                "--moe-token-dispatcher-type allgather does not support variable sequence length, "
                "please use alltoall dispatcher instead."
            )
            args.moe_token_dispatcher_type = "alltoall"

        if args.pipeline_model_parallel_size == 1:
            assert args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None, (
                "decoder_first_pipeline_num_layers and decoder_last_pipeline_num_layers should be None when "
                "pipeline_model_parallel_size is 1."
            )
        training_backend_arg_names.update(vars(args).keys() - previous_arg_names)
    else:
        from miles.backends.fsdp_utils.arguments import validate_hybrid_shard_args

        validate_hybrid_shard_args(args)
        training_backend_arg_names.update(
            {
                "bf16",
                "calculate_per_token_loss",
                "ckpt_step",
                "clip_grad",
                "load",
                "rank",
                "world_size",
            }
        )

    sglang_validate_args(args)

    if backend == "fsdp" and args.fsdp_cpu_offload:
        args.offload_train = False

    vars(args).setdefault("ckpt_step", None)
    vars(args).setdefault("lora_A_init_method", "xavier")
    vars(args).setdefault("lora_B_init_method", "zero")

    assert parser is not None
    resolve_custom_function_configs(args)
    backend_values = {name: value for name, value in vars(args).items() if name in training_backend_arg_names}
    _validate_argument_ownership(args, parser=parser, training_backend_arg_names=training_backend_arg_names)
    values = {name: value for name, value in vars(args).items() if name in AllConfig.model_fields} | {
        "raw_megatron": resolve_megatron_config(args, base_args=backend_values if backend == "megatron" else {}),
        "raw_fsdp": FsdpArgsNamespace(**backend_values) if backend == "fsdp" else None,
        "sglang": SglangConfig.parse_args(args),
        "sglang_model_routers": None,
    }
    values.update(RouterConfig.from_args(args))
    return AllConfig.model_validate(values), parser


def _validate_argument_ownership(
    args: argparse.Namespace, *, parser: argparse.ArgumentParser, training_backend_arg_names: set[str]
) -> None:
    sglang_arg_names = {
        action.dest for action in parser._actions if action.dest.startswith(("sglang_", "eval_sglang_"))
    }
    owned_arg_names = (
        AllConfig.model_fields.keys()
        | training_backend_arg_names
        | sglang_arg_names
        | RouterConfig.arg_names()
        | {"custom_config_path", "megatron_config"}
    )
    if unknown_arg_names := vars(args).keys() - owned_arg_names:
        raise ValueError(f"Parsed arguments have no configuration owner: {sorted(unknown_arg_names)}")


def parse_args_train_backend():
    if os.environ.get("MILES_BACKEND") is not None:
        raise Exception("`MILES_BACKEND` is deprecated, please use --train-backend directly.")

    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args_partial, _ = parser.parse_known_args()
    return args_partial.train_backend


def _resolve_eval_datasets(args) -> list[EvalDatasetConfig]:
    """
    Build evaluation dataset configurations from either --eval-config or --eval-prompt-data.
    """
    datasets_config = []
    defaults: dict[str, Any] = {}

    if args.eval_config:
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(resolve_file_arg(args.eval_config))
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            raise ValueError("--eval-config must contain a mapping at the root.")

        eval_cfg = cfg_dict.get("eval", cfg_dict)
        if not isinstance(eval_cfg, dict):
            raise ValueError("--eval-config must define an `eval` mapping or be a mapping itself.")

        defaults = dict(eval_cfg.get("defaults") or {})
        datasets_config = ensure_dataset_list(eval_cfg.get("datasets"))
        if not datasets_config:
            raise ValueError("--eval-config does not define any datasets under `eval.datasets`.")
    elif args.eval_prompt_data:
        values = list(args.eval_prompt_data)
        if len(values) == 1:
            logger.info("[legacy] only one eval_prompt_data detected, will assume it is data for aime")
            values = ["aime", values[0]]
        if len(values) % 2 != 0:
            raise ValueError("eval prompt data must be provided as name/path pairs.")
        datasets_config = [{"name": values[i], "path": values[i + 1]} for i in range(0, len(values), 2)]
    else:
        datasets_config = []

    eval_datasets = build_eval_dataset_configs(args, datasets_config, defaults)
    if eval_datasets:
        args.eval_prompt_data = [item for dataset in eval_datasets for item in (dataset.name, dataset.path)]
    else:
        args.eval_prompt_data = None

    return eval_datasets


def _compute_rollout_external(args: argparse.Namespace) -> bool:
    return args.rollout_external_engine_addrs is not None or args.custom_inference_engine_provider_path is not None


_BACKEND_ENGINE_PROVIDER_PATH = "miles.ray.specs.inference.backend_inference_engine_provider"
_STATIC_EXTERNAL_ENGINE_PROVIDER_PATH = "miles.ray.rollout.external_engine_provider.static_inference_engine_provider"


def _compute_custom_inference_engine_provider_path(args: argparse.Namespace) -> str:
    if (path := args.custom_inference_engine_provider_path) is not None:
        return path
    if args.rollout_external_engine_addrs is not None:
        return _STATIC_EXTERNAL_ENGINE_PROVIDER_PATH
    return _BACKEND_ENGINE_PROVIDER_PATH


_DEPLOY_INSTANCE_ID_PATTERN = re.compile(DNS_LABEL_PATTERN)


def _validate_deploy_component(args: argparse.Namespace) -> None:
    component = DeployComponent(args.deploy_component)

    _validate_deploy_instance_id(args, component=component)
    _validate_static_addrs_external_launch(args, component=component)
    _validate_registration(args, component=component)
    _validate_single_engine_source(args, component=component)

    if not component.is_split():
        return

    cluster_backend = ClusterBackend(args.cluster_backend)
    assert cluster_backend is ClusterBackend.KUBERNETES or (
        cluster_backend is ClusterBackend.RAY and WorkerCommBackend(args.worker_comm_backend) is WorkerCommBackend.RPC
    ), (
        f"--deploy-component {component.value} needs --cluster-backend {ClusterBackend.KUBERNETES.value}, or "
        f"{ClusterBackend.RAY.value} with --worker-comm-backend {WorkerCommBackend.RPC.value} and a ray cluster "
        f"per deployment; got --cluster-backend {args.cluster_backend} --worker-comm-backend "
        f"{args.worker_comm_backend}"
    )

    assert (
        not args.colocate
    ), f"--deploy-component {component.value} cannot be combined with --colocate, which shares gpus across the two"

    if component.deploys_orchestration_script():
        assert (
            args.trainer_controller_addrs is not None
        ), f"--deploy-component {component.value} deploys no trainer, so it needs --trainer-controller-addrs"
        _validate_trainer_controller_addrs(args)

    if component is DeployComponent.TRAINER:
        _validate_single_deployed_trainer(args)
        assert not (
            args.debug_rollout_only and cluster_backend is ClusterBackend.RAY
        ), f"--debug-rollout-only needs an inference side, which --deploy-component {component.value} has none"

    if component is not DeployComponent.INFERENCE:
        _validate_shared_object_store(args, component=component)
    _validate_watched_cells_deployed_locally(args, component=component)


def _validate_deploy_instance_id(args: argparse.Namespace, *, component: DeployComponent) -> None:
    if (instance_id := args.deploy_instance_id) is None:
        return

    assert component.takes_instance_id(), (
        f"--deploy-instance-id {instance_id!r} names one deployment of {component.value} apart from the others, and "
        f"a run has exactly one {component.value}; only "
        f"{[one.value for one in DeployComponent if one.takes_instance_id()]} are deployed more than once"
    )

    if component is not DeployComponent.INFERENCE:
        return

    assert _DEPLOY_INSTANCE_ID_PATTERN.fullmatch(instance_id), (
        f"--deploy-instance-id {instance_id!r} names the release this launch installs and the pool ids of the "
        f"engines it deploys, so it has to match {_DEPLOY_INSTANCE_ID_PATTERN.pattern}"
    )
    assert len(instance_id) <= DEPLOY_INSTANCE_ID_MAX_LENGTH, (
        f"--deploy-instance-id {instance_id!r} is {len(instance_id)} characters, and it is carried inside every "
        f"engine pool id this deployment names, which kubernetes bounds; it takes at most "
        f"{DEPLOY_INSTANCE_ID_MAX_LENGTH}"
    )


def _validate_single_deployed_trainer(args: argparse.Namespace) -> None:
    trainers = resolve_megatron_config(args, base_args={}).trainers
    assert len(trainers) == 1, (
        f"--deploy-component trainer deploys one trainer and its arguments describe {len(trainers)} "
        f"({[t.trainer_id for t in trainers]}); give this deployment the config of the one trainer it carries, "
        f"and launch every other trainer as a deployment of its own"
    )
    assert not args.use_critic, (
        "--use-critic grows this run by a critic trainer, and --deploy-component trainer carries exactly the "
        "one trainer its config describes; deploy the critic separately with an explicit single-trainer config"
    )
    assert trainers[0].role != CRITIC_ROLE, (
        f"--deploy-component trainer carries the trainer {trainers[0].trainer_id!r}, whose role is "
        f"{CRITIC_ROLE!r}; a critic is deployed together with the run that drives it, and deploying one on its own "
        f"is not supported yet"
    )
    assert args.deploy_instance_id is None or args.deploy_instance_id == trainers[0].trainer_id, (
        f"--deploy-instance-id {args.deploy_instance_id!r} names this deployment, but its config describes trainer "
        f"{trainers[0].trainer_id!r}; the run reaches a trainer by the id its config declares, so the two must "
        f"agree"
    )


def _validate_static_addrs_external_launch(args: argparse.Namespace, *, component: DeployComponent) -> None:
    assert component.deploys_orchestration_script() or args.trainer_controller_addrs is None, (
        f"--trainer-controller-addrs describes the trainer side the orchestration script drives, but "
        f"--deploy-component {component.value} carries no orchestration script, so nothing here would call those "
        f"addresses"
    )
    assert not (
        component.selects(DeployComponent.TRAINER) and args.trainer_controller_addrs is not None
    ), f"--deploy-component {component.value} deploys the trainer itself, so drop --trainer-controller-addrs"


def _validate_registration(args: argparse.Namespace, *, component: DeployComponent) -> None:
    if (init_expected := args.init_expected_num_cells) is not None:
        assert component is DeployComponent.PRIMARY, (
            f"--init-expected-num-cells needs --deploy-component {DeployComponent.PRIMARY.value}, not "
            f"{component.value}"
        )
        assert (
            init_expected >= 1
        ), f"--init-expected-num-cells {init_expected} lets the run start before a single engine registered into it"

    if component is DeployComponent.INFERENCE:
        assert args.deploy_instance_id is not None, (
            f"--deploy-component {component.value} needs --deploy-instance-id: it names the engine pools this "
            f"deployment reports and tells it apart from the other engine deployments of the run"
        )
        assert args.inference_controller_addr is not None, (
            f"--deploy-component {component.value} deploys engines and nothing that drives them, so the one "
            f"inference controller of the run has to be named by --inference-controller-addr"
        )
        parse_host_and_port(args.inference_controller_addr)
    else:
        assert args.inference_controller_addr is None, (
            f"--deploy-component {component.value} holds the one inference controller of the run, so it reaches it "
            f"in its own process rather than through --inference-controller-addr"
        )


def _validate_single_engine_source(args: argparse.Namespace, *, component: DeployComponent) -> None:
    if component is not DeployComponent.PRIMARY:
        return

    assert args.rollout_external_engine_addrs is None, (
        f"--deploy-component {component.value} serves the engines that register into it, so "
        f"--rollout-external-engine-addrs would be dropped and the run would wait for registrations forever"
    )
    assert (path := args.custom_inference_engine_provider_path) in (None, _BACKEND_ENGINE_PROVIDER_PATH), (
        f"--deploy-component {component.value} serves the engines that register into it, so "
        f"--custom-inference-engine-provider-path {path!r} would be dropped and never asked for an engine"
    )


def _validate_watched_cells_deployed_locally(args: argparse.Namespace, *, component: DeployComponent) -> None:
    if not component.deploys_orchestration_script():
        unservable = sorted(set(args.ft_components) - {"train"})
        assert not unservable, (
            f"--deploy-component {component.value} installs no inference engines, and {unservable} cells are "
            f"suspended and resumed through the controller of the deployment that owns them, so this launch cannot "
            f"answer for them; pass --ft-components train"
        )
        return

    assert (
        not args.api_server_port
    ), f"--deploy-component {component.value} watches cells it does not deploy; pass --api-server-port 0"


def _validate_trainer_controller_addrs(args: argparse.Namespace) -> None:
    trainer_ids = [trainer.trainer_id for trainer in resolve_megatron_config(args, base_args={}).trainers]
    external_trainer_controller_addrs(args, trainer_ids=trainer_ids)


def _validate_shared_object_store(args: argparse.Namespace, *, component: DeployComponent) -> None:
    assert ObjectStoreBackend(args.object_store_backend) == ObjectStoreBackend.MOONCAKE, (
        f"--deploy-component {component.value} needs --object-store-backend "
        f"{ObjectStoreBackend.MOONCAKE.value}, shared by every deployment of the run"
    )

    if component.deploys_orchestration_script():
        return

    address = (args.mooncake_store_init_kwargs or {}).get(MOONCAKE_MASTER_ADDRESS_KEY)
    assert isinstance(address, str) and ":" in address, (
        f"--deploy-component {component.value} runs no object store master, so it needs "
        f'--mooncake-store-init-kwargs \'{{"{MOONCAKE_MASTER_ADDRESS_KEY}": "<host>:<port>"}}\' '
        f"(got {address!r})"
    )


_FT_DEFAULT_COMPONENTS: list[str] = ["rollout"]


def _resolve_ft_components(args: argparse.Namespace) -> list[str]:
    if not args.use_fault_tolerance:
        if args.ft_components is not None:
            logger.warning("--ft-components is ignored without --use-fault-tolerance")
        return []
    if args.ft_components is None:
        return list(_FT_DEFAULT_COMPONENTS)
    return list(args.ft_components)


def _validate_rematerialize_param_from_master_weight(args):
    if not args.rematerialize_param_from_master_weight:
        return
    if args.debug_train_only:
        # update_weights never runs, so the param buffer would never be paused.
        args.rematerialize_param_from_master_weight = False
        return
    assert (
        args.train_backend == "megatron"
    ), "--rematerialize-param-from-master-weight reads Megatron's distributed-optimizer main params"
    from miles.backends.megatron_utils.lora_utils import is_lora_enabled

    assert not is_lora_enabled(args), "--rematerialize-param-from-master-weight does not support LoRA"
    assert not args.debug_disable_optimizer, "--debug-disable-optimizer leaves no main params to rematerialize from"
    assert not args.indep_dp, (
        "--rematerialize-param-from-master-weight drops the backup inside update_weights, which "
        "RayTrainGroup runs on the first alive cell only. Every other cell would keep it for the whole "
        "run. Lift this once all cells update weights."
    )
    assert args.colocate and args.offload_train
    assert args.offload_train_target == "cpu", (
        "--offload-train-target=disk streams the weights to NVMe and reads them back from GPU after "
        "resume, so there is no backup for the rebuild to replace"
    )
    assert args.use_distributed_optimizer
    assert not args.keep_old_actor
    assert not args.use_precision_aware_optimizer or args.optimizer_cpu_offload, (
        "--use-precision-aware-optimizer on GPU keeps the master weights inside TE FusedAdam, stored as "
        "int16 remainders of the params. There is nothing standalone to rebuild from. Add "
        "--optimizer-cpu-offload, which holds standalone masters instead."
    )
    assert (
        not args.overlap_param_gather
    ), "the rebuild calls DDP.start_param_sync outside the training step; overlap-param-gather does not support that"
    assert (
        args.compute_advantages_and_returns
    ), "the per-cycle rebuild runs in the compute_advantages_and_returns block; without it training would run on dropped weights"
    assert (
        args.num_critic_only_steps == 0
    ), "critic-only steps run update_weights repeatedly without an intervening actor wake_up"
    args.disable_param_buffers_cpu_backup = True
    if args.ci_test:
        args.check_rematerialize_param_from_master_weight = True


def _resolve_api_server_port(args: argparse.Namespace) -> int:
    if (port := args.api_server_port) is not None:
        return port
    return _DEFAULT_FT_API_SERVER_PORT if args.ft_components else 0


def _resolve_mini_ft_controller_enable(args: argparse.Namespace) -> bool:
    if (enable := args.mini_ft_controller_enable) is not None:
        return enable
    return bool(args.ft_components) and args.api_server_port != 0


def _resolve_run_uuid(args: argparse.Namespace) -> str:
    if (given := args.run_uuid) is not None:
        return validate_run_uuid(given)

    component = DeployComponent(args.deploy_component)
    assert not component.is_split(), (
        f"--deploy-component {component.value} installs one part of a run whose other parts are installed by other "
        f"launches, and nothing but the run uuid joins them, so the layer that deploys them all has to name it "
        f"with --run-uuid"
    )
    return generate_run_uuid()


def _resolve_sample_ownership_check(args: argparse.Namespace) -> None:
    if args.sample_ownership_grace_steps is None:
        args.sample_ownership_grace_steps = 2 if args.ci_test else 10
    if args.enable_sample_ownership_checker is None:
        args.enable_sample_ownership_checker = args.ci_test
    if not args.enable_sample_ownership_checker:
        return

    assert (
        args.custom_convert_samples_to_train_data_path is None
    ), "--enable-sample-ownership-checker is incompatible with --custom-convert-samples-to-train-data-path"

    multi_policy = (
        args.train_backend == "megatron"
        and args.megatron_config is not None
        and len(
            [config for config in resolve_megatron_config(args, base_args={}).trainers if config.role == ACTOR_ROLE]
        )
        > 1
    )
    unsupported = [
        reason
        for condition, reason in (
            (args.train_backend != "megatron", "the FSDP backend has no model companion info"),
            (is_lora_enabled(args), "LoRA training has no model companion info"),
            (args.multi_lora, "multi-LoRA training can replay samples"),
            (multi_policy, "multi-policy training has separate model companion lineages"),
            (args.debug_train_only, "train-only mode has no issuing data source"),
            (args.debug_rollout_only, "rollout-only mode has no trainer model companion"),
            (args.debug_disable_optimizer, "a disabled optimizer trains nothing"),
            (args.num_critic_only_steps > 0, "critic-only warmup steps drop actor samples"),
        )
        if condition
    ]
    if unsupported:
        raise ValueError(f"--enable-sample-ownership-checker is not supported here: {'; '.join(unsupported)}")

    if args.sample_ownership_grace_steps < 0:
        raise ValueError("--sample-ownership-grace-steps must be non-negative")

    if args.save_debug_event_data is None:
        raise ValueError(
            "--enable-sample-ownership-checker needs an event directory: "
            "pass --save-debug-event-data, --save, or --dump-details"
        )


def miles_validate_args(args):
    if args.custom_config_path:
        logger.warning("--custom-config-path is deprecated; use the custom function's declared CLI arguments instead.")
        data = yaml.safe_load(resolve_file_arg(args.custom_config_path)) or {}
        for k, v in data.items():
            if hasattr(args, k):  # config-access-exempt: attribute selected at runtime from k
                logger.info(
                    f"Warning: Argument {k} is already set to {getattr(args, k)}, will override with {v}."
                )  # config-access-exempt: attribute selected at runtime from k
            setattr(args, k, v)

    validate_dashboard_args(args)

    args.ft_components = _resolve_ft_components(args)
    assert not ("rollout" in args.ft_components and args.eval_num_gpus > 0), (
        "rollout fault tolerance does not support a dedicated eval fleet (--eval-num-gpus > 0): "
        "the eval fleet pins engine addresses once at startup, so a healed eval cell would make "
        "every later eval skip silently"
    )
    args.eval_datasets = _resolve_eval_datasets(args)

    if "train" in args.ft_components:
        args.indep_dp = True
        args.delay_split_train_data_by_dp = True
        args.save_local_weight_checksum = True
        args.enable_event_analyzer = True
        args.enable_witness = True
        args.non_persistent_ckpt_type = "local"
        if args.non_persistent_local_ckpt_dir is None:
            args.non_persistent_local_ckpt_dir = "/tmp/miles_local_ckpt"
        # atomic: each rank saves independently, no collective communication.
        # fully_parallel needs all_gather_object which hangs after ncclCommAbort in healing.
        args.non_persistent_local_ckpt_algo = "atomic"
        logger.info(
            "train in ft_components. Auto set indep_dp=True, delay_split_train_data_by_dp=True, save_local_weight_checksum=True, enable_event_analyzer=True, enable_witness=True, non_persistent_ckpt_type='local', non_persistent_local_ckpt_algo=%r",
            args.non_persistent_local_ckpt_algo,
        )

    if args.indep_dp:
        assert (
            args.train_backend == "megatron"
        ), f"indep_dp requires train_backend='megatron', got '{args.train_backend}'"
        assert args.use_dynamic_batch_size, (
            "--indep-dp requires --use-dynamic-batch-size (with --max-tokens-per-gpu): "
            "the live cell count after a fault need not divide global_batch_size"
        )
        assert not args.use_dynamic_global_batch_size, (
            "--indep-dp does not support --use-dynamic-global-batch-size: "
            "independent cells do not expose a DP size to the rollout side"
        )
        per_replica_size = compute_megatron_world_size_except_dp(args)
        logger.info(f"indep_dp: adjusting args.world_size from {args.world_size} to {per_replica_size} (per-cell)")
        args.world_size = per_replica_size

    if args.recompute_logprobs_via_prefill:
        assert args.true_on_policy_mode, "--recompute-logprobs-via-prefill requires --true-on-policy-mode"

    if args.use_session_server not in (False, True, "v1", "v2"):
        raise ValueError(
            f"--use-session-server={args.use_session_server!r} is not a known session server "
            "version; pass it bare (or 'v1') for the append-only linear server, or 'v2' for "
            "tree serving."
        )

    assert not (
        args.use_session_server and args.partial_rollout
    ), "--use-session-server does not support --partial-rollout"

    if args.use_session_server == "v2":
        unsupported = [
            flag
            for enabled, flag in (
                (args.group_rm, "--group-rm"),
                (args.recompute_logprobs_via_prefill, "--recompute-logprobs-via-prefill"),
            )
            if enabled
        ]
        if unsupported:
            raise ValueError(
                f"--use-session-server v2 does not support {', '.join(unsupported)}; v2 returns list[Sample]"
            )

    if args.use_session_server and args.use_rollout_routing_replay and args.pause_generation_mode == "retract":
        logger.warning(
            "--use-session-server with --use-rollout-routing-replay and "
            "--pause-generation-mode=retract returns full R3 data on every turn; "
            "R3 payloads can become very large. TODO: Retract-mode weight updates R3 "
            "have known issues in SGLang and need to be fixed."
        )

    if not args.use_session_server and args.tito_model != TITOTokenizerType.DEFAULT.value:
        raise ValueError(
            f"--tito-model={args.tito_model} requires --use-session-server; "
            "this flag only configures the session-server TITO middleware."
        )

    # DEFAULT uses the checkpoint's native or caller-provided template. Its
    # maximal four-role surface is best-effort rather than a Miles-verified
    # FixedTemplate contract.
    if args.use_session_server and args.tito_model == TITOTokenizerType.DEFAULT.value:
        logger.warning(
            "--tito-model=default uses a best-effort four-role append surface. "
            "Incremental tokenization assumes appended messages do not change how "
            "earlier turns render, which may not hold for user messages on "
            "context-sensitive chat templates (e.g. last_query_index logic, "
            "thinking-token trimming). This can cause input_ids to diverge from "
            "the canonical template output. Use at your own risk."
        )

    if args.debug_disable_optimizer:
        args.no_load_optim = True
        args.no_save_optim = True

    # Normalize the deprecated ``--chat-template-path=autofix`` alias to None
    # up-front so the rest of this block treats it as "no path given".
    if args.chat_template_path == "autofix":
        logger.warning(
            "--chat-template-path=autofix is deprecated; remove the flag and rely "
            "on --tito-model to auto-resolve. The "
            "alias will be removed in a future release."
        )
        args.chat_template_path = None

    # A named family is one fixed renderer contract.  Letting a custom path or
    # conflicting required kwarg through would detach its declared role
    # capability from the renderer that actually runs.
    if args.tito_model != TITOTokenizerType.DEFAULT.value:
        tito_model = TITOTokenizerType(args.tito_model)
        from miles.utils.chat_template_utils import resolve_fixed_chat_template

        if args.chat_template_path is not None:
            raise ValueError(
                f"--chat-template-path cannot override the template registered for "
                f"--tito-model={tito_model.value}; use --tito-model=default for a custom template"
            )

        resolved_path, resolved_kwargs = resolve_fixed_chat_template(tito_model)
        if resolved_path is not None:
            args.chat_template_path = resolved_path
        user_kwargs = dict(args.apply_chat_template_kwargs or {})
        for key, value in resolved_kwargs.items():
            if key in user_kwargs and user_kwargs[key] != value:
                raise ValueError(
                    f"--apply-chat-template-kwargs {key}={user_kwargs[key]!r} conflicts "
                    f"with the value registered for --tito-model={tito_model.value}: {value!r}"
                )
            user_kwargs[key] = value
        args.apply_chat_template_kwargs = user_kwargs

    if args.chat_template_path is not None:
        if not os.path.isfile(args.chat_template_path):
            raise FileNotFoundError(f"--chat-template-path file not found: {args.chat_template_path}")
        args.sglang_chat_template = args.chat_template_path

    if args.kl_coef != 0 or args.use_kl_loss:
        if not os.path.exists(args.ref_load):
            raise FileNotFoundError(f"ref_load {args.ref_load} does not exist, please check the path.")

        if not os.path.exists(os.path.join(args.ref_load, "latest_checkpointed_iteration.txt")):
            logger.info(
                f"ref_load {args.ref_load} does not have latest_checkpointed_iteration.txt, "
                "please make sure it is a valid megatron checkpoint directory."
            )

    # Validate on-policy distillation (OPD) arguments
    if args.use_opd:
        if args.opd_type is None:
            raise ValueError("--opd-type must be specified when --use-opd is enabled. Choose 'sglang' or 'megatron'.")
        if args.opd_log_prob_top_k < 0:
            raise ValueError("--opd-log-prob-top-k must be non-negative.")
        if args.opd_log_prob_top_k > 0 and args.opd_type != "sglang":
            raise ValueError("--opd-log-prob-top-k is currently supported only with --opd-type=sglang.")
        if args.opd_log_prob_top_k > 0 and args.opd_top_k_strategy != "only-teacher" and not use_legacy_rollout_v1():
            raise ValueError(
                "--opd-log-prob-top-k with a student-side strategy needs opd_student_top_logprobs, "
                "which only the v1 rollout produces; set MILES_USE_LEGACY_ROLLOUT_V1=1"
            )
        if args.opd_teacher_urls:
            if args.opd_type != "sglang":
                raise ValueError("--opd-teacher-urls is only supported with --opd-type=sglang.")
            # Local import to keep miles.utils free of rollout imports at module load.
            from miles.rollout.on_policy_distillation import parse_teacher_urls

            parse_teacher_urls(args.opd_teacher_urls)  # fail fast on malformed/duplicate entries

        if args.opd_type == "megatron":
            if args.opd_teacher_load is None:
                raise ValueError(
                    "--opd-teacher-load is required when --opd-type=megatron. "
                    "Please provide the path to the teacher model checkpoint."
                )
            if not os.path.exists(args.opd_teacher_load):
                raise FileNotFoundError(
                    f"opd_teacher_load {args.opd_teacher_load} does not exist, please check the path."
                )
            if not os.path.exists(os.path.join(args.opd_teacher_load, "latest_checkpointed_iteration.txt")):
                logger.info(
                    f"opd_teacher_load {args.opd_teacher_load} does not have latest_checkpointed_iteration.txt, "
                    "please make sure it is a valid megatron checkpoint directory."
                )

        elif args.opd_type == "sglang":
            if args.opd_teacher_load is not None:
                raise ValueError(
                    "--opd-teacher-load should not be set when --opd-type=sglang. "
                    "In sglang mode, teacher log-probs are obtained from external server during rollout."
                )
    else:
        if args.opd_teacher_load is not None:
            raise ValueError("--opd-teacher-load is set but --use-opd is not enabled. Please add --use-opd flag.")
        if args.opd_teacher_urls:
            raise ValueError("--opd-teacher-urls is set but --use-opd is not enabled. Please add --use-opd flag.")

    # TODO: refactor
    args.requested_load = args.load
    if args.megatron_config is None:
        resolve_args_checkpoint_load(args)

    if args.eval_interval is not None:
        assert args.eval_datasets, "Evaluation datasets must be configured when eval_interval is set."

    if args.eval_num_gpus > 0:
        assert args.eval_num_gpus % args.eval_num_gpus_per_engine == 0, (
            f"eval_num_gpus ({args.eval_num_gpus}) must be divisible by "
            f"eval_num_gpus_per_engine ({args.eval_num_gpus_per_engine})."
        )
    else:
        overrides = collect_eval_sglang_overrides(args)
        assert not overrides, (
            f"--eval-sglang-* configures the dedicated eval fleet, which needs --eval-num-gpus > 0. "
            f"Got {sorted(overrides)} with --eval-num-gpus 0."
        )

    if args.save_interval is not None:
        assert args.save is not None, "'--save' is required when save_interval is set."

    if args.save_trigger_sentinel is not None:
        assert args.save is not None, "'--save' is required when save_trigger_sentinel is set."

    if args.custom_megatron_post_save_hook_path is not None:
        assert args.save is not None, "'--save' is required when custom_megatron_post_save_hook_path is set."

    # Parse LoRA target modules
    if args.lora_rank > 0:
        assert args.target_modules is not None, "'--target-modules' is required when LoRA is enabled."

        if args.target_modules == "all-linear":
            # MLA projections are HF-config-gated (SGLang sizes LoRA buffers per module name;
            # listing them on a dense model crashes the engine). The DSA indexer stays excluded.
            modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            hf_config = load_hf_config(args.hf_checkpoint)
            if getattr(
                hf_config, "kv_lora_rank", None
            ):  # config-access-exempt: model-family schemas differ in optional kv_lora_rank metadata
                modules += ["kv_a_proj_with_mqa", "kv_b_proj"]
                if getattr(
                    hf_config, "q_lora_rank", None
                ):  # config-access-exempt: model-family schemas differ in optional q_lora_rank metadata
                    modules += ["q_a_proj", "q_b_proj"]
        elif "," in args.target_modules:
            modules = [m.strip() for m in args.target_modules.split(",")]
        else:
            modules = [args.target_modules]

        if args.exclude_modules:
            exclude_set = (
                set(m.strip() for m in args.exclude_modules.split(","))
                if "," in args.exclude_modules
                else {args.exclude_modules}
            )
            modules = [m for m in modules if m not in exclude_set]

        args.target_modules = modules

        # Training and serving must agree on shared-outer grouped-expert LoRA
        # (expert_dim=1 buffers in SGLang).
        if args.experts_shared_outer_loras and hasattr(
            args, "sglang_experts_shared_outer_loras"
        ):  # config-access-exempt: older SGLang parsers omit the expert-LoRA switch
            args.sglang_experts_shared_outer_loras = True
        assert args.experts_shared_outer_loras == bool(
            getattr(
                args, "sglang_experts_shared_outer_loras", args.experts_shared_outer_loras
            )  # config-access-exempt: older SGLang parsers omit the expert-LoRA switch
        ), "experts_shared_outer_loras and sglang_experts_shared_outer_loras must agree"

        # the two MoE-expert adapter layouts are not checkpoint-compatible; say which one runs
        _expert_leaves = ("linear_fc1", "linear_fc2", "gate_proj", "up_proj", "down_proj")
        if any(leaf in str(tm) for tm in modules for leaf in _expert_leaves):
            logger.warning(
                "MoE-expert LoRA layout: %s (--experts-shared-outer-loras).",
                "shared-outer" if args.experts_shared_outer_loras else "per-expert",
            )

    # Sets args.multi_lora, then validates/defaults the multi-LoRA arg surface
    # (adapter configs themselves are loaded later by the controller).
    from miles.utils.multi_lora import validate_multi_lora_args

    validate_multi_lora_args(args)
    _resolve_data_source_path(args)

    assert not (args.kl_coef != 0 and args.kl_loss_coef != 0), "Only one of kl_coef and kl_loss_coef can be set"

    if args.advantage_estimator in ["reinforce_plus_plus", "reinforce_plus_plus_baseline"]:
        assert args.normalize_advantages, (
            "The 'reinforce_plus_plus' and 'reinforce_plus_plus_baseline' advantage estimators "
            "require advantage normalization. Please add `--normalize-advantages` to your command."
        )

    if args.use_rollout_logprobs:
        assert not args.use_tis, "use_rollout_logprobs and use_tis cannot be set at the same time."

    if args.get_mismatch_metrics:
        assert (
            args.custom_tis_function_path is not None
        ), "custom_tis_function_path must be set when get_mismatch_metrics is set"

        if args.use_rollout_logprobs and not args.skip_actor_forward_only:
            logger.info(
                "get_mismatch_metrics is set; For metrics calculation, the log probs will still be recomputed by training engine. One more forward pass will be applied."
            )

    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None, "max_tokens_per_gpu must be set when use_dynamic_batch_size is set"
        if args.log_probs_max_tokens_per_gpu is None:
            args.log_probs_max_tokens_per_gpu = args.max_tokens_per_gpu

    # --use-dynamic-global-batch-size has two motivations:
    # 1. compaction/subagent rollouts: static micro-batching cannot guarantee alignment
    #    when the physical sample count is data-dependent, so --use-dynamic-batch-size
    #    is required.
    # 2. multi-LoRA (auto-enabled, no compaction/subagent): the per-round sample count is
    #    a config-shaped multiple of dp_size trained as exactly one step on the legacy
    #    training-side schedule; static micro-batching stays valid there.
    if args.use_dynamic_global_batch_size and not args.multi_lora:
        assert args.use_dynamic_batch_size, (
            "--use-dynamic-global-batch-size requires --use-dynamic-batch-size (with --max-tokens-per-gpu): "
            "static micro-batching cannot guarantee dp_size * mb_group alignment when the physical sample count "
            "is data-dependent; this configuration is not supported."
        )

    if args.balance_by_flops:
        assert args.use_dynamic_batch_size, "--balance-by-flops requires --use-dynamic-batch-size"

    if args.eps_clip_high is None:
        args.eps_clip_high = args.eps_clip

    if args.eval_reward_key is None:
        args.eval_reward_key = args.reward_key

    if args.dump_details is not None:
        args.save_debug_rollout_data = f"{args.dump_details}/rollout_data/{{rollout_id}}.pt"
        args.save_debug_train_data = f"{args.dump_details}/train_data/{{rollout_id}}_{{rank}}.pt"
        args.save_debug_trajectory_data = f"{args.dump_details}/trajectory/{{rollout_id}}.jsonl"
        args.save_debug_event_data = f"{args.dump_details}/{EVENTS_DIRNAME}"

    if args.save_debug_event_data is None and args.save is not None:
        args.save_debug_event_data = f"{args.save}/{EVENTS_DIRNAME}"

    if args.load_debug_rollout_data is not None:
        logger.info(
            f"load_debug_rollout_data {args.load_debug_rollout_data} is set, "
            "will not instantiate sglang servers and will only run the training process."
        )
        args.debug_train_only = True

    assert (args.ci_inject_rollout_data_path is None) == (args.ci_inject_rollout_data_start_rollout_id is None), (
        "--ci-inject-rollout-data-path and --ci-inject-rollout-data-start-rollout-id " "must be set together."
    )
    if args.ci_inject_rollout_data_path is not None:
        assert args.load_debug_rollout_data is None, (
            "--ci-inject-rollout-data-path replaces data of individual rollouts while engines "
            "stay alive; it cannot be combined with --load-debug-rollout-data (debug_train_only)."
        )

    args.use_critic = args.advantage_estimator == "ppo"
    if args.use_critic:
        assert not args.indep_dp, (
            "Shared Actor/Critic PPO hands the critic outputs to a single trainer cell as external data; "
            "it does not support --indep-dp, which train fault tolerance also implies"
        )
        if args.train_backend != "megatron":
            raise ValueError("Shared Actor/Critic PPO requires the Megatron backend")
        assert args.kl_coef == 0, (
            "Shared Actor/Critic PPO does not support reward-level KL (--kl-coef): the critic "
            "trains before the actor and never sees ref log probs, so its value targets would "
            "silently exclude the KL penalty applied to the actor's rewards. Use --use-kl-loss "
            "for KL regularization instead."
        )
        args.critic_num_gpus_per_node = args.actor_num_gpus_per_node
        args.critic_num_nodes = args.actor_num_nodes
    if args.critic_load is None:
        args.critic_load = args.load
    if args.critic_lr is None:
        args.critic_lr = args.lr
    if args.critic_save is None and args.save is not None:
        # a sibling dir, not args.save itself: sharing a dir would clobber the actor's iteration tracker
        args.critic_save = args.save.rstrip("/") + "_critic"

    if args.offload:
        args.offload_train = True
        args.offload_rollout = True
    del args.offload

    if args.debug_rollout_only:
        if args.colocate and (not args.rollout_num_gpus):
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        else:
            args.actor_num_gpus_per_node = min(8, args.rollout_num_gpus)
            args.actor_num_nodes = args.rollout_num_gpus // args.actor_num_gpus_per_node
        args.colocate = False
        args.offload_train = args.offload_rollout = False
        if args.train_memory_margin_bytes > 0:
            logger.warning("Force train_memory_margin_bytes=0 since debug_rollout_only does not support it")
            args.train_memory_margin_bytes = 0

    assert not (args.debug_rollout_only and args.debug_train_only), (
        "debug_rollout_only and debug_train_only cannot be set at the same time, " "please set only one of them."
    )

    if (
        args.ci_test
        and not args.debug_rollout_only
        and not args.debug_train_only
        and not args.ci_disable_weight_update_checker
    ):
        args.check_weight_update_equal = True

    # always true on offload for colocate at the moment.
    if args.update_weight_transfer_mode == "p2p":
        assert not args.colocate, (
            "P2P weight transfer mode is not compatible with --colocate. "
            "Please use broadcast mode or disable colocate."
        )
        assert args.prefill_num_servers is None, "P2P weight transfer mode has not been tested when PD is enabled."
        assert args.lora_rank <= 0, "LoRA weight sync is not supported for p2p (RDMA) weight transfer."
        assert (
            args.megatron_to_hf_mode != "bridge"
        ), f"{args.update_weight_transfer_mode} mode is not supported when use megatron-bridge"

    if args.update_weight_transfer_mode == "disk-delta":
        assert not args.colocate, (
            "Disk-delta weight transfer mode is not compatible with --colocate. Colocate transfers "
            "weights via CUDA IPC (only a handle crosses processes), so the delta bookkeeping "
            "(snapshot + diff + encode) is pure overhead."
        )
        assert (
            args.prefill_num_servers is None
        ), "Disk-delta weight transfer mode has not been tested when PD is enabled."
        assert args.lora_rank <= 0, "LoRA weight sync is not supported for disk-delta weight transfer."
        assert args.update_weight_disk_dir, (
            "--update-weight-transfer-mode=disk-delta requires --update-weight-disk-dir to point at "
            "a filesystem shared between the trainer and the rollout engines."
        )
        assert args.update_weight_local_checkpoint_dir, (
            "--update-weight-transfer-mode=disk-delta requires --update-weight-local-checkpoint-dir "
            "(a rollout-host-local directory, e.g. NVMe)."
        )
        assert os.path.isdir(args.hf_checkpoint), (
            "--update-weight-transfer-mode=disk-delta requires --hf-checkpoint to be a local directory: "
            "the baseline snapshot is seeded from its safetensors bytes."
        )
        if has_megatron_checkpoint(args.requested_load):
            raise ValueError(
                "--update-weight-transfer-mode=disk-delta cannot resume from a training checkpoint: "
                "the first sync only captures a baseline from --hf-checkpoint and never transfers the "
                f"weights restored from --load={args.requested_load}."
            )

    if args.colocate:
        if args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True
        if args.sglang_cuda_graph_backend_prefill is None:
            args.sglang_cuda_graph_backend_prefill = "disabled"
            logger.info(
                "Colocate mode: defaulting --sglang-cuda-graph-backend-prefill=disabled to avoid NVLS OOM. "
                "Set --sglang-cuda-graph-backend-prefill explicitly to override."
            )
        elif args.sglang_cuda_graph_backend_prefill != "disabled":
            logger.warning(
                f"Warning: colocate mode with --sglang-cuda-graph-backend-prefill="
                f"{args.sglang_cuda_graph_backend_prefill} may trigger NVLS OOM."
            )
        if args.rollout_num_gpus != args.actor_num_gpus_per_node * args.actor_num_nodes:
            logger.info(
                f"rollout_num_gpus {args.rollout_num_gpus} != actor_num_gpus_per_node {args.actor_num_gpus_per_node} "
                f"* actor_num_nodes {args.actor_num_nodes}, overriding rollout_num_gpus to match actor_num_gpus_per_node * actor_num_nodes."
            )
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes

    if args.debug_train_only:
        args.rollout_num_gpus = 0
    args.starts_inference_engines = not args.debug_train_only or args.eval_num_gpus > 0

    if args.use_critic and not args.debug_rollout_only:
        if args.offload_train is None:
            args.offload_train = True
        elif not args.offload_train:
            logger.warning(
                "--no-offload-train with shared Actor/Critic PPO is reserved for offload debugging: "
                "both models stay resident on the shared train GPUs, so make sure they fit."
            )

    if args.offload_train is None:
        args.offload_train = False
    if args.offload_rollout is None:
        args.offload_rollout = False

    if args.offload_train:
        args.disable_grad_buffers_cpu_backup = True
        args.disable_param_buffers_cpu_backup = True

    _validate_rematerialize_param_from_master_weight(args)

    if (args.offload_train_target == "disk" or args.stream_optimizer_state_to_disk) and (
        args.offload_train_disk_dir is None
    ):
        uid = os.getuid() if hasattr(os, "getuid") else 0  # config-access-exempt: os.getuid is platform-dependent
        args.offload_train_disk_dir = os.path.join(os.environ.get("SCRATCH", "/scratch"), f"miles_train_offload_{uid}")

    if args.offload_train_target == "disk":
        assert args.offload_train, "--offload-train-target=disk requires --offload-train"
        assert (
            args.train_backend == "megatron"
        ), "--offload-train-target=disk is only supported on the megatron backend"
        assert args.offload_train_disk_chunk_mb > 0, "--offload-train-disk-chunk-mb must be positive"
        logger.info(
            f"Train offload target=disk, dir={args.offload_train_disk_dir}, "
            f"chunk={args.offload_train_disk_chunk_mb}MB"
        )

    if args.stream_optimizer_state_to_disk:
        assert args.offload_train_target == "disk" or not args.offload_train, (
            "--stream-optimizer-state-to-disk with --offload-train requires "
            "--offload-train-target=disk: a run that cannot hold the optimizer state on GPU for "
            "the duration of the step will not hold a pinned host copy of the whole actor either. "
            "Disaggregated runs do not offload the trainer at all, and the target is unused there."
        )
        assert not args.indep_dp, (
            "--stream-optimizer-state-to-disk does not support --indep-dp: each cell has its own "
            "process group, so torch.distributed.get_rank() restarts at 0 per cell and two cells "
            "on one node would share a store directory"
        )
        _muon_disk_state = "muon" in (args.optimizer or "").lower()
        if _muon_disk_state:
            # Megatron's validate_args has not run yet, so gate on the dist_ prefix rather than
            # use_layer_wise_distributed_optimizer.
            assert args.optimizer.lower().startswith("dist_"), (
                "--stream-optimizer-state-to-disk with Muon requires the layer-wise distributed "
                f"optimizer; pass --optimizer dist_muon, got {args.optimizer}"
            )
            assert args.chunked_optimizer_state_offload and args.optimizer_state_offload_fraction > 0.0, (
                "--stream-optimizer-state-to-disk with Muon is the disk backend for the chunked "
                "offloader; pass --chunked-optimizer-state-offload and a non-zero "
                "--optimizer-state-offload-fraction"
            )
        else:
            assert (
                args.use_distributed_optimizer
            ), "--stream-optimizer-state-to-disk requires the distributed optimizer"
            assert (
                args.optimizer == "adam"
            ), f"--stream-optimizer-state-to-disk requires --optimizer adam, got {args.optimizer}"
        assert not (args.multi_lora or is_lora_enabled(args)), (
            "--stream-optimizer-state-to-disk does not support LoRA: the LoRA checkpoint path "
            "persists optimizer.state_dict(), which the store leaves empty, and restores the "
            "adapter into the model params without refreshing the streamed main params"
        )
        assert not args.optimizer_cpu_offload, "--stream-optimizer-state-to-disk excludes --optimizer-cpu-offload"
        assert (
            _muon_disk_state or not args.offload_optimizer_states
        ), "--stream-optimizer-state-to-disk excludes --offload-optimizer-states"
        assert (
            not args.use_precision_aware_optimizer
        ), "--stream-optimizer-state-to-disk requires mcore to hold the fp32 main params"
        assert not args.reset_optimizer_states, (
            "--reset-optimizer-states walks the master optimizer's state, which the NVMe store "
            "leaves empty, so the reset would silently do nothing"
        )
        assert not args.save_local_weight_checksum, (
            "--save-local-weight-checksum reads param.main_param, whose storage the NVMe store "
            "resizes to 0 between steps"
        )
        assert (
            not args.enable_witness
        ), "--enable-witness reads the master optimizer's per-param state, which the NVMe store owns"
        logger.info(
            f"Streaming optimizer state to disk, dir={args.offload_train_disk_dir}, "
            f"chunk={args.offload_train_disk_chunk_mb}MB, moments={args.stream_optimizer_state_moment_dtype}"
        )

    if args.async_max_concurrent_samples is not None:
        assert args.async_max_concurrent_samples >= args.n_samples_per_prompt, (
            f"--async-max-concurrent-samples ({args.async_max_concurrent_samples}) must be at least "
            f"--n-samples-per-prompt ({args.n_samples_per_prompt}): the worker submits whole groups, "
            f"so one group already puts n_samples_per_prompt trajectories in flight"
        )

    if args.namespaced_radix_cache is None:
        args.namespaced_radix_cache = args.fully_async and args.pause_generation_mode == "in_place"
        if args.namespaced_radix_cache:
            logger.info(
                "--fully-async with --pause-generation-mode in_place never flushes the engine cache: "
                "defaulting to --namespaced-radix-cache so prefix KV computed under old "
                "weights cannot serve a later rollout call. Pass "
                "--no-namespaced-radix-cache to keep one shared cache."
            )

    if args.namespaced_radix_cache:
        assert not use_legacy_rollout_v1(), (
            "--namespaced-radix-cache requires the class-based rollout API; "
            "unset MILES_USE_LEGACY_ROLLOUT_V1 or pass --no-namespaced-radix-cache"
        )

    _resolve_rollout_functions(args)

    # Both snapshot postures drive the same RolloutManager._eval_checkpoint path.
    # (The fleet-vs-CheckpointEvalFn conflict is asserted where the posture is derived.)
    if args.eval_uses_snapshots:
        assert (
            not use_legacy_rollout_v1()
        ), "Snapshot eval requires the class-based rollout API; unset MILES_USE_LEGACY_ROLLOUT_V1"
        assert args.eval_interval is not None, "Snapshot eval requires --eval-interval."
        assert args.eval_hf_dir is not None or args.save_hf is not None, (
            "Snapshot eval requires a snapshot source: set --eval-hf-dir (staging exports) "
            "or --save-hf (reuse periodic HF checkpoints)."
        )
        assert not args.colocate, "Snapshot eval is not supported with --colocate."
        assert not args.debug_rollout_only, "Snapshot eval is not supported with debug_rollout_only."
        assert (
            args.load_debug_rollout_data is None
        ), "Snapshot eval is not supported with --load-debug-rollout-data: no rollout functions are loaded."
        if args.debug_train_only:
            assert args.eval_function_path != args.rollout_function_path, (
                "Snapshot eval during --debug-train-only requires an explicit --eval-function-path; "
                "the training rollout function cannot evaluate snapshots."
            )
        if args.eval_hf_dir is None:
            assert args.save_interval is not None and args.eval_interval % args.save_interval == 0, (
                "Reusing --save-hf checkpoints for eval requires eval_interval to be a "
                f"multiple of save_interval (got eval_interval={args.eval_interval}, "
                f"save_interval={args.save_interval}). Set --eval-hf-dir for independent snapshots."
            )

    if args.num_steps_per_rollout is not None:
        global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt // args.num_steps_per_rollout
        if args.global_batch_size is not None:
            assert args.global_batch_size == global_batch_size, (
                f"global_batch_size {args.global_batch_size} is not equal to "
                f"rollout_batch_size {args.rollout_batch_size} * n_samples_per_prompt {args.n_samples_per_prompt} "
                f"// num_steps_per_rollout {args.num_steps_per_rollout}"
            )
        args.global_batch_size = global_batch_size

    # Multi-LoRA adapters carry their own n_samples_per_prompt; the per-group
    # normalization path already skips std for singleton groups.
    if args.n_samples_per_prompt == 1 and not args.multi_lora:
        args.grpo_std_normalization = False
        logger.info("n_samples_per_prompt is set to 1, grpo_std_normalization will be set to False.")

    if args.over_sampling_batch_size is None:
        args.over_sampling_batch_size = args.rollout_batch_size

    assert args.over_sampling_batch_size >= args.rollout_batch_size, (
        f"over_sampling_batch_size {args.over_sampling_batch_size} should be greater than or equal to "
        f"rollout_batch_size {args.rollout_batch_size}"
    )

    if args.num_epoch is not None:
        if args.num_rollout is not None:
            logger.info("Both num_epoch and num_rollout are set, num_epoch will be ignored.")
        else:
            assert args.rollout_global_dataset, (
                "num_epoch is set, but rollout_global_dataset is not set, "
                "please remove --disable-rollout-global-dataset to use num_epoch"
            )
    else:
        # if num_epoch is not set, we should set num_rollout
        assert args.num_rollout is not None, (
            "num_epoch is not set, but num_rollout is not set, " "please set --num-rollout or --num-epoch"
        )

    if args.enable_mtp_training:
        assert args.mtp_num_layers, "mtp_num_layers must be set when enable_mtp_training is set"

    if args.use_rollout_routing_replay:
        args.use_routing_replay = True

    args.rollout_external = _compute_rollout_external(args)
    args.custom_inference_engine_provider_path = _compute_custom_inference_engine_provider_path(args)

    args.worker_comm_backend = resolve_worker_comm_backend(
        cluster_backend=ClusterBackend(args.cluster_backend), requested=args.worker_comm_backend
    ).value

    if ClusterBackend(args.cluster_backend) == ClusterBackend.KUBERNETES:
        assert (
            not args.use_miles_dashboard
        ), "--use-miles-dashboard creates a Ray actor, which --cluster-backend kubernetes has no Ray cluster for"
        assert (
            not args.use_distributed_post
        ), "--use-distributed-post reads ray.nodes(), which --cluster-backend kubernetes has no Ray cluster for"
        assert (
            args.multi_lora_n_adapters == 0
        ), "--multi-lora-n-adapters drives RayWorkerManager, which --cluster-backend kubernetes does not use"
        if ObjectStoreBackend(args.object_store_backend) != ObjectStoreBackend.MOONCAKE:
            logger.info(
                f"Overriding --object-store-backend {args.object_store_backend} with "
                f"{ObjectStoreBackend.MOONCAKE.value} under --cluster-backend {ClusterBackend.KUBERNETES.value}."
            )
            args.object_store_backend = ObjectStoreBackend.MOONCAKE.value
        if (
            not args.mooncake_store_init_kwargs
            and DeployComponent(args.deploy_component).deploys_orchestration_script()
        ):
            args.mooncake_store_init_kwargs = (
                compute_mooncake_init_kwargs_vanilla() | compute_mooncake_init_kwargs_from_env()
            )

    args.run_uuid = _resolve_run_uuid(args)

    if args.save_debug_event_data is None and args.ci_test:
        args.save_debug_event_data = os.path.join(tempfile.gettempdir(), "miles-ci", args.run_uuid, EVENTS_DIRNAME)

    _resolve_sample_ownership_check(args)

    if args.use_rollout_indexer_replay:
        args.use_indexer_replay = True
        assert args.context_parallel_size == 1, "indexer replay does not support context parallelism yet"

    if args.eval_max_context_len is None:
        logger.info(
            f"args.eval_max_context_len is not set. Use args.rollout_max_context_len {args.rollout_max_context_len} as default value."
        )
        args.eval_max_context_len = args.rollout_max_context_len

    if args.rollout_max_context_len is not None:
        if args.rollout_max_prompt_len is None:
            args.rollout_max_prompt_len = args.rollout_max_context_len - 1
            logger.info(
                f"args.rollout_max_prompt_len is not set. Use args.rollout_max_context_len - 1 ({args.rollout_max_context_len} - 1) as default value so that there is at least one generated token to compute loss."
            )
        assert (
            args.rollout_max_prompt_len <= args.rollout_max_context_len - 1
        ), f"args.rollout_max_prompt_len ({args.rollout_max_prompt_len}) must be smaller than args.rollout_max_context_len ({args.rollout_max_context_len}) so that there is at least one generated token to compute loss."

    assert not (
        args.prefill_num_servers is not None and args.rollout_external
    ), "prefill_num_servers cannot be set with external rollout engines; use --rollout-external-router-pd."

    assert not (
        args.sglang_config is not None and args.rollout_external
    ), "sglang_config cannot be set with external rollout engines; the topology comes from discovery."

    assert not (
        args.rollout_external_router_pd and not args.rollout_external
    ), "--rollout-external-router-pd only applies to external rollout engines; internally launched engines infer PD from the sglang config."

    assert not (
        args.sglang_config is not None and args.prefill_num_servers is not None
    ), "sglang_config and prefill_num_servers are mutually exclusive. Use server_groups in the YAML config instead."

    if args.qkv_format == "bshd":
        assert args.train_backend == "megatron", "bshd format is only supported for megatron backend."
        assert (
            args.use_dynamic_batch_size is False
        ), "Dynamic batch size is not supported for bshd format. Please specify --micro-batch-size instead."

    if args.skip_actor_forward_only:
        validate_skip_actor_forward_only(args)

    _maybe_apply_dumper_overrides(args)

    args.api_server_port = _resolve_api_server_port(args)
    args.mini_ft_controller_enable = _resolve_mini_ft_controller_enable(args)

    if args.mini_ft_controller_enable and args.api_server_port == 0:
        raise ValueError("--mini-ft-controller-enable requires --api-server-port to be set (non-zero)")

    _validate_deploy_component(args)


def validate_skip_actor_forward_only(args) -> None:
    option = "--skip-actor-forward-only"
    assert args.train_backend == "megatron", f"{option} only supports --train-backend megatron"
    assert args.loss_type == "policy_loss", f"{option} only supports --loss-type policy_loss"
    assert args.compute_advantages_and_returns, f"{option} requires actor advantage computation"

    incompatible_options = [
        name
        for name, enabled in (
            ("--keep-old-actor", args.keep_old_actor),
            ("--kl-coef", args.kl_coef != 0),
            ("--use-opd", args.use_opd),
            ("--hidden-dropout", args.hidden_dropout != 0),
            ("--attention-dropout", args.attention_dropout != 0),
            ("--lora-dropout", args.lora_dropout != 0),
            ("--moe-input-jitter-eps", args.moe_input_jitter_eps not in (None, 0)),
            ("--moe-router-force-load-balancing", args.moe_router_force_load_balancing),
            ("--moe-router-force-biased", args.moe_router_force_biased is not None),
            (
                "--moe-router-load-balancing-type sinkhorn",
                "sinkhorn" in args.moe_router_load_balancing_type,
            ),
            ("--use-rollout-entropy", args.use_rollout_entropy),
            ("--true-on-policy-mode", args.true_on_policy_mode),
            ("--log-correct-samples", args.log_correct_samples),
            ("--rollout-data-postprocess-path", args.rollout_data_postprocess_path is not None),
            (
                "--custom-megatron-before-log-prob-hook-path",
                args.custom_megatron_before_log_prob_hook_path is not None,
            ),
            (
                "--custom-megatron-before-train-step-hook-path",
                args.custom_megatron_before_train_step_hook_path is not None,
            ),
            ("--custom-model-provider-path", args.custom_model_provider_path is not None),
            (
                "--dumper-source-patcher-config-train",
                args.dumper_source_patcher_config_train is not None,
            ),
            ("--save-debug-train-data", args.save_debug_train_data is not None and args.dump_details is None),
            (
                "--use-routing-replay",
                args.use_routing_replay and not args.use_rollout_routing_replay,
            ),
            (
                "--use-indexer-replay",
                args.use_indexer_replay and not args.use_rollout_indexer_replay,
            ),
        )
        if enabled
    ]
    assert not incompatible_options, f"{option} is incompatible with: {', '.join(incompatible_options)}"

    assert args.num_steps_per_rollout in (None, 1), (
        f"{option} requires exactly one optimizer step per rollout; "
        f"got --num-steps-per-rollout {args.num_steps_per_rollout}"
    )
    if not args.use_dynamic_global_batch_size and not args.multi_lora:
        samples_per_rollout = args.rollout_batch_size * args.n_samples_per_prompt
        assert args.global_batch_size == samples_per_rollout, (
            f"{option} requires exactly one optimizer step for {samples_per_rollout} rollout samples; "
            f"got --global-batch-size {args.global_batch_size}"
        )


def validate_async_off_policy_correction(args) -> None:
    """Require an explicit behavior-policy choice for async PPO training.

    In the async train loop the next rollout is generated before the current
    weight update is published, so samples can come from a stale policy. With
    the default flags the PPO ratio denominator (``log_probs``) is recomputed
    by the *current* actor, silently anchoring clipping (and KL-shaped
    advantages) to a policy that never generated the trajectory; the recorded
    ``weight_versions`` are a metric, not an enforcement mechanism.
    """
    if not args.use_critic:
        return
    assert args.use_rollout_logprobs or args.use_tis or args.keep_old_actor, (
        "Async PPO training requires an explicit behavior-policy correction, because rollouts are "
        "generated before the current weight update while log probs are recomputed by the current "
        "actor by default. Pass one of: --use-rollout-logprobs (use the rollout engine's log probs "
        "as the ratio denominator), --use-tis (truncated importance sampling correction), or "
        "--keep-old-actor (recompute the denominator with the weights the rollout engines used)."
    )


def _maybe_apply_dumper_overrides(args) -> None:
    if not args.dumper_enable:
        return

    if args.use_fault_tolerance:
        logger.info("Dumper mode: disabling --use-fault-tolerance to suppress fault tolerance heartbeats")
        args.use_fault_tolerance = False
        args.ft_components = []

    logger.info("Dumper mode: all heartbeat mechanisms disabled")
    args.router_disable_health_check = True

    if args.start_rollout_id is None:
        args.start_rollout_id = 0

    args.num_rollout = (args.start_rollout_id or 0) + 1
    logger.info(
        "Dumper mode: forced rollout range [%d, %d), disabled eval and save",
        args.start_rollout_id,
        args.num_rollout,
    )
    args.eval_interval = None
    args.save = None
    args.save_interval = None
    args.save_retain_interval = None


def resolve_fsdp_num_layers(hf_config) -> int | None:
    """Decoder-layer count for the FSDP path.

    ``num_layers`` comes from the Megatron parser, but backend-agnostic code reads it:
    ``sglang_rollout`` reshapes the R3 routing buffer as ``[num_tokens, num_layers, topk]``. The
    text config wins when present, since a top-level ``num_hidden_layers`` may describe a vision
    tower instead.
    """
    getter = getattr(
        hf_config, "get_text_config", None
    )  # config-access-exempt: model-family schemas differ in optional get_text_config metadata
    text_config = (
        getter() if callable(getter) else getattr(hf_config, "text_config", None)
    ) or hf_config  # config-access-exempt: model-family schemas differ in optional text_config metadata

    num_layers = getattr(
        text_config, "num_hidden_layers", None
    )  # config-access-exempt: model-family schemas differ in optional num_hidden_layers metadata
    if num_layers is None:
        num_layers = getattr(
            hf_config, "num_hidden_layers", None
        )  # config-access-exempt: model-family schemas differ in optional num_hidden_layers metadata
    return num_layers


def hf_validate_args(args, hf_config):
    def equal(x, y):
        return x == y

    errors = []

    # multimodal models have different config structure
    if hasattr(
        hf_config, "text_config"
    ):  # config-access-exempt: model-family schemas differ in optional text_config metadata
        hf_config = hf_config.text_config

    if hasattr(hf_config, "rope_parameters") and isinstance(
        hf_config.rope_parameters, dict
    ):  # config-access-exempt: model-family schemas differ in optional rope_parameters metadata
        if "rope_theta" in hf_config.rope_parameters:
            hf_config.rope_theta = hf_config.rope_parameters["rope_theta"]
        else:
            # Gemma-4 nests rope_theta per attention type; take the first.
            for _entry in hf_config.rope_parameters.values():
                if isinstance(_entry, dict) and "rope_theta" in _entry:
                    hf_config.rope_theta = _entry["rope_theta"]
                    break

    model_name = (args.model_name or "").lower().replace("-", "").replace("_", "")
    if (hf_config.model_type == "deepseek_v4" or "deepseekv4" in model_name) and args.context_parallel_size > 1:
        assert args.allgather_cp, "zigzag CP is not supported for DeepSeek V4."

    for hf_config_name, megatron_config_name, compare_fn in [
        ("hidden_size", "hidden_size", equal),
        ("num_attention_heads", "num_attention_heads", equal),
        ("num_hidden_layers", "num_layers", equal),
        ("intermediate_size", "ffn_hidden_size", equal),
        ("moe_intermediate_size", "moe_ffn_hidden_size", equal),
        ("tie_word_embeddings", "untie_embeddings_and_output_weights", lambda x, y: not x == y),
        (
            "rms_norm_eps",
            "norm_epsilon" if os.getenv("DEPRECATED_MEGATRON_COMPATIBLE", "0") == "1" else "layernorm_epsilon",
            equal,
        ),
        ("rope_theta", "rotary_base", equal),
    ]:
        # FIXME: Qwen3.5 transfomers has bug.
        if (
            getattr(hf_config, "model_type", "") == "qwen3_5_moe_text" and hf_config_name == "intermediate_size"
        ):  # config-access-exempt: model-family schemas differ in optional model_type metadata
            continue
        if (
            getattr(hf_config, "model_type", "") == "deepseek_v4" and hf_config_name == "intermediate_size"
        ):  # config-access-exempt: model-family schemas differ in optional model_type metadata
            continue
        if hasattr(
            hf_config, hf_config_name
        ):  # config-access-exempt: attribute selected at runtime from hf_config_name
            if not compare_fn(
                getattr(hf_config, hf_config_name), getattr(args, megatron_config_name)
            ):  # config-access-exempt: attribute selected at runtime from hf_config_name; attribute selected at runtime from megatron_config_name
                errors.append(
                    f"{hf_config_name} in hf config {getattr(hf_config, hf_config_name)} is not equal to "  # config-access-exempt: attribute selected at runtime from hf_config_name
                    f"{megatron_config_name} {getattr(args, megatron_config_name)}, please check the config."  # config-access-exempt: attribute selected at runtime from megatron_config_name
                )

    if len(errors) > 0:
        raise AssertionError("hf_validate_args failed: " + "; ".join(errors))


def _resolve_data_source_path(args: argparse.Namespace) -> None:
    if args.partial_rollout and args.data_source_path == "miles.rollout.data_source.RolloutDataSource":
        args.data_source_path = "miles.rollout.data_source.LegacyRolloutDataSourceWithBuffer"
        logger.info("Partial rollout uses the legacy buffered data source by default")
