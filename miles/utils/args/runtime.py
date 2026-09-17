from pydantic import ConfigDict

from miles.utils.args.configs.algo import AlgoConfig
from miles.utils.args.configs.ci import CiConfig
from miles.utils.args.configs.cluster import ClusterConfig
from miles.utils.args.configs.custom_megatron_plugins import CustomMegatronPluginsConfig
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


class AllConfig(
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
):
    # TODO: Remove extra="allow" after backend, custom, and derived fields have explicit config owners.
    model_config = ConfigDict(extra="allow")

    # TODO: Remove this temporary override after separating CLI input types from normalized config types.
    target_modules: str | list[str] | None = None
