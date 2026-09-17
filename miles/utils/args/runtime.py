from typing import Self

from pydantic import ConfigDict, model_validator

from miles.backends.fsdp_utils.config import FsdpArgsNamespace
from miles.backends.megatron_utils.megatron_config import MegatronConfig
from miles.utils.args.component_multi_lora import MultiLoraOnlyConfig
from miles.utils.args.component_orchestrator import OrchestratorOnlyConfig
from miles.utils.args.component_rollout import InferenceControllerOnlyConfig, RolloutOnlyConfig
from miles.utils.args.component_shared import SglangFieldsConfig
from miles.utils.args.component_trainer import TrainerOnlyConfig
from miles.utils.args.configs.algo import AlgoConfig
from miles.utils.args.configs.backend_fields import TrainerBackendTraitConfig
from miles.utils.args.configs.ci import CiConfig
from miles.utils.args.configs.cluster import ClusterConfig
from miles.utils.args.configs.custom_megatron_plugins import CustomMegatronPluginsConfig
from miles.utils.args.configs.dashboard import DashboardConfig
from miles.utils.args.configs.data import DataConfig
from miles.utils.args.configs.debug import DebugConfig
from miles.utils.args.configs.eval import EvalConfig
from miles.utils.args.configs.fault_tolerance import FaultToleranceConfig
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
from miles.utils.args.runtime_base import BaseLeafConfig


class OrchestratorConfig(
    BaseLeafConfig,
    TrainerBackendTraitConfig,
    OrchestratorOnlyConfig,
    RunUuidConfig,
    ClusterConfig,
    TrainConfig,
    RolloutRelatedConfig,
    FaultToleranceConfig,
    DataConfig,
    EvalConfig,
    AlgoConfig,
    OnPolicyDistillationConfig,
    LoraConfig,
    RouterConfig,
    DebugConfig,
    NetworkConfig,
    RewardModelConfig,
    RolloutBufferConfig,
    CustomMegatronPluginsConfig,
    MtpTrainingConfig,
    PrefillDecodeDisaggregationConfig,
    CiConfig,
    SessionConfig,
    MlflowConfig,
    PrometheusConfig,
    TensorboardConfig,
    WandbConfig,
    DashboardConfig,
    SglangFieldsConfig,
):
    pass


class TrainerConfig(
    BaseLeafConfig,
    TrainerOnlyConfig,
    RunUuidConfig,
    ClusterConfig,
    TrainConfig,
    RolloutRelatedConfig,
    FaultToleranceConfig,
    DataConfig,
    EvalConfig,
    AlgoConfig,
    OnPolicyDistillationConfig,
    LoraConfig,
    RouterConfig,
    DebugConfig,
    NetworkConfig,
    RewardModelConfig,
    RolloutBufferConfig,
    CustomMegatronPluginsConfig,
    MtpTrainingConfig,
    PrefillDecodeDisaggregationConfig,
    CiConfig,
    SessionConfig,
    MlflowConfig,
    PrometheusConfig,
    TensorboardConfig,
    WandbConfig,
    DashboardConfig,
    SglangFieldsConfig,
):
    @model_validator(mode="after")
    def _validate_backend_name(self) -> Self:
        assert self.train_backend == self.backend.backend_name, "train_backend must match backend.backend_name"
        return self

    @model_validator(mode="after")
    def _validate_no_duplicated_backend_fields(self) -> Self:
        duplicated_fields = type(self).model_fields.keys() & vars(self.backend).keys()
        assert not duplicated_fields, f"Duplicated trainer backend fields: {sorted(duplicated_fields)}"
        return self


class InferenceControllerConfig(
    BaseLeafConfig,
    TrainerBackendTraitConfig,
    InferenceControllerOnlyConfig,
    RunUuidConfig,
    ClusterConfig,
    TrainConfig,
    RolloutRelatedConfig,
    FaultToleranceConfig,
    DataConfig,
    EvalConfig,
    AlgoConfig,
    OnPolicyDistillationConfig,
    LoraConfig,
    RouterConfig,
    DebugConfig,
    NetworkConfig,
    RewardModelConfig,
    RolloutBufferConfig,
    CustomMegatronPluginsConfig,
    MtpTrainingConfig,
    PrefillDecodeDisaggregationConfig,
    CiConfig,
    SessionConfig,
    MlflowConfig,
    PrometheusConfig,
    TensorboardConfig,
    WandbConfig,
    DashboardConfig,
    SglangFieldsConfig,
):
    pass


class RolloutConfig(
    BaseLeafConfig,
    TrainerBackendTraitConfig,
    RolloutOnlyConfig,
    RunUuidConfig,
    ClusterConfig,
    TrainConfig,
    RolloutRelatedConfig,
    FaultToleranceConfig,
    DataConfig,
    EvalConfig,
    AlgoConfig,
    OnPolicyDistillationConfig,
    LoraConfig,
    RouterConfig,
    DebugConfig,
    NetworkConfig,
    RewardModelConfig,
    RolloutBufferConfig,
    CustomMegatronPluginsConfig,
    MtpTrainingConfig,
    PrefillDecodeDisaggregationConfig,
    CiConfig,
    SessionConfig,
    MlflowConfig,
    PrometheusConfig,
    TensorboardConfig,
    WandbConfig,
    DashboardConfig,
    SglangFieldsConfig,
):
    pass


class MultiLoraConfig(
    BaseLeafConfig,
    TrainerBackendTraitConfig,
    MultiLoraOnlyConfig,
    RunUuidConfig,
    ClusterConfig,
    TrainConfig,
    RolloutRelatedConfig,
    FaultToleranceConfig,
    DataConfig,
    EvalConfig,
    AlgoConfig,
    OnPolicyDistillationConfig,
    LoraConfig,
    RouterConfig,
    DebugConfig,
    NetworkConfig,
    RewardModelConfig,
    RolloutBufferConfig,
    CustomMegatronPluginsConfig,
    MtpTrainingConfig,
    PrefillDecodeDisaggregationConfig,
    CiConfig,
    SessionConfig,
    MlflowConfig,
    PrometheusConfig,
    TensorboardConfig,
    WandbConfig,
    DashboardConfig,
    SglangFieldsConfig,
):
    pass


class AllConfig(
    BaseLeafConfig,
    TrainerBackendTraitConfig,
    RunUuidConfig,
    ClusterConfig,
    TrainConfig,
    RolloutRelatedConfig,
    FaultToleranceConfig,
    DataConfig,
    EvalConfig,
    AlgoConfig,
    OnPolicyDistillationConfig,
    LoraConfig,
    RouterConfig,
    DebugConfig,
    NetworkConfig,
    RewardModelConfig,
    RolloutBufferConfig,
    CustomMegatronPluginsConfig,
    MtpTrainingConfig,
    PrefillDecodeDisaggregationConfig,
    CiConfig,
    SessionConfig,
    MlflowConfig,
    PrometheusConfig,
    TensorboardConfig,
    WandbConfig,
    DashboardConfig,
    SglangFieldsConfig,
    OrchestratorOnlyConfig,
    RolloutOnlyConfig,
    InferenceControllerOnlyConfig,
    MultiLoraOnlyConfig,
):
    # TODO: Remove extra="allow" after backend, custom, and derived fields have explicit config owners.
    model_config = ConfigDict(extra="allow")

    # TODO: Unify trainer descriptions after zhichen's training backend refactor; FSDP also uses these descriptions.
    raw_megatron: MegatronConfig
    raw_fsdp: FsdpArgsNamespace | None
