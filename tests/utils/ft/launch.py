import os

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.base_backend import LaunchGuard

MEGATRON_PATH: str = os.environ.get("MILES_SCRIPT_MEGATRON_PATH", "/root/Megatron-LM")
DEFAULT_TRAIN_SCRIPT: str = "train.py"
FULLY_ASYNC_TRAIN_SCRIPT: str = "train_async.py"


def get_train_script(*, fully_async: bool) -> str:
    return FULLY_ASYNC_TRAIN_SCRIPT if fully_async else DEFAULT_TRAIN_SCRIPT


def get_fully_async_args(*, fully_async: bool) -> str:
    if not fully_async:
        return ""
    return "--fully-async --pause-generation-mode in_place "


# Required for reproducibility (ref: https://github.com/THUDM/slime/pull/370)
DETERMINISTIC_ENV_VARS: dict[str, str] = {
    "NCCL_ALGO": "Ring",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    # The default 4096 split overflows FlashInfer's fixed 2 GiB deterministic workspace
    # while capturing the 8192-token prefill graph for the 5-layer Qwen3 MoE model.
    "SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE": "8192",
}


def launch_training(
    *,
    train_args: str,
    num_gpus_per_node: int,
    megatron_model_type: str | None,
    config: command_utils.ExecuteTrainConfig | None = None,
    train_script: str = DEFAULT_TRAIN_SCRIPT,
    extra_env_vars: dict[str, str] | None = None,
    guard: LaunchGuard | None = None,
) -> None:
    config = resolve_config(config)
    U = config.create_backend()
    merged_env_vars = {
        **DETERMINISTIC_ENV_VARS,
        # Run eager (no torch.compile). A cell respawned after a crash cold-recompiles its first
        # forward; under dynamic batch sizes that is a per-shape Inductor compile that is slow
        # (observed 124s..1510s, growing) and memory-heavy enough to OOM-kill the actor. That
        # recompile-on-respawn is a torch.compile + FT infra limitation orthogonal to what these
        # tests assert (FT crash recovery + baseline-vs-target metric equivalence); both runs are
        # eager so the comparison stays valid.
        #
        # TODO: this only sidesteps the respawn recompile cost, it does not fix it. Investigate
        # keeping torch.compile under FT respawn (warm/shared Inductor cache survivor->respawn, or
        # bounded recompile) so the tests can exercise the compiled path again.
        "TORCHDYNAMO_DISABLE": "1",
        "RAY_DEDUP_LOGS": "0",
        "SGLANG_LOG_MS": "1",
        **(extra_env_vars or {}),
    }
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        extra_env_vars=merged_env_vars,
        megatron_path=MEGATRON_PATH,
        train_script=train_script,
        guard=guard,
    )


def resolve_config(config: command_utils.ExecuteTrainConfig | None) -> command_utils.ExecuteTrainConfig:
    return config or command_utils.default_config()
