# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import json
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from tests.e2e.conftest_dumper import MEGATRON_PATCHER_YAMLS
from tests.e2e.ft.conftest_ft.modes import DEBUG_ROLLOUT_DATA_HF_REPO, FTTestMode
from tests.utils.cluster_backends import create_backend_for_run
from tests.utils.ft.launch import (
    DEFAULT_TRAIN_SCRIPT,
    DETERMINISTIC_ENV_VARS,
    MEGATRON_PATH,
    launch_training,
    resolve_config,
)
from tests.utils.soak.core.utils import DATA_DIR, MODEL_DIR, get_dumps_root

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.external_utils import command_utils

_LAUNCH_ID: str = uuid4().hex
_DEBUG_ROLLOUT_DATA_DIR: str = f"{DATA_DIR}/{DEBUG_ROLLOUT_DATA_HF_REPO.split('/')[-1]}"


def materialize_cyclic_debug_rollout_data(count: int) -> str:
    src = Path(_DEBUG_ROLLOUT_DATA_DIR)
    available = sorted(int(p.stem) for p in src.glob("*.pt") if p.stem.isdigit())
    if not available:
        raise FileNotFoundError(f"No debug rollout data files found in {src}")
    dst = Path(tempfile.mkdtemp(prefix="ft_cyclic_rollout_", dir=DATA_DIR))
    dst.chmod(0o755)
    for i in range(count):
        (dst / f"{i}.pt").symlink_to(src / f"{available[i % len(available)]}.pt")
    return str(dst)


def _get_hf_num_layers(model_path: str) -> int:
    with open(f"{model_path}/config.json") as f:
        return json.load(f)["num_hidden_layers"]


def prepare(mode: FTTestMode, *, config: command_utils.ExecuteTrainConfig | None = None) -> None:
    config = resolve_config(config)
    patcher_path = _source_patcher_path()

    patcher_path.parent.mkdir(parents=True, exist_ok=True)

    U = create_backend_for_run(config)
    U.exec_command_cpu(f"mkdir -p {MODEL_DIR} {DATA_DIR}")
    U.exec_command_cpu(f"hf download {mode.model_hf_repo} --local-dir {MODEL_DIR}/{mode.model_name}")

    hf_model_path = f"{MODEL_DIR}/{mode.model_name}"
    num_layers = _get_hf_num_layers(hf_model_path)
    convert_gpus = min(mode.train_gpus_per_node, num_layers)

    U.convert_checkpoint(
        model_name=mode.model_name,
        megatron_model_type=mode.megatron_model_type,
        num_gpus_per_node=convert_gpus,
        megatron_path=MEGATRON_PATH,
        hf_checkpoint=hf_model_path,
        dir_dst=MODEL_DIR,
    )
    if not mode.has_real_rollout:
        U.hf_download_dataset(DEBUG_ROLLOUT_DATA_HF_REPO, data_dir=DATA_DIR)
    U.hf_download_dataset("zhuzilin/gsm8k", data_dir=DATA_DIR)

    megatron_yaml: str = MEGATRON_PATCHER_YAMLS["thd"]
    patcher_path.write_text(megatron_yaml)


def get_common_train_args(
    mode: FTTestMode,
    *,
    dump_dir: str,
    num_steps: int | None = None,
    enable_dumper: bool = True,
    debug_rollout_data_dir: str | None = None,
) -> str:
    ckpt_args = f"--hf-checkpoint {MODEL_DIR}/{mode.model_name} --ref-load {MODEL_DIR}/{mode.model_name}_torch_dist "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--lr-warmup-fraction 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
    )

    rollout_args: str
    rollout_data_path = Path(dump_dir) / "rollout_data" / "{rollout_id}.pt"
    if not mode.has_real_rollout:
        rollout_dir = debug_rollout_data_dir or _DEBUG_ROLLOUT_DATA_DIR
        rollout_args = (
            f"--prompt-data {DATA_DIR}/gsm8k/train.parquet "
            f"--load-debug-rollout-data {rollout_dir}/{{rollout_id}}.pt "
            "--debug-train-only "
            "--rollout-batch-size 32 "
            "--n-samples-per-prompt 8 "
        )
    else:
        rollout_args = (
            f"--prompt-data {DATA_DIR}/gsm8k/train.parquet "
            "--input-key messages "
            "--label-key label "
            "--apply-chat-template "
            "--rollout-shuffle "
            "--rm-type deterministic_random "
            "--rollout-max-response-len 200 "
            "--rollout-temperature 0.8 "
            "--rollout-batch-size 32 "
            "--n-samples-per-prompt 8 "
            # Required for reproducibility (ref: https://github.com/THUDM/slime/pull/370)
            + DETERMINISTIC_ROLLOUT_ARGS + f"--save-debug-rollout-data {shlex.quote(str(rollout_data_path))} "
            f"--rollout-num-gpus {mode.total_rollout_gpus} "
            f"--rollout-num-gpus-per-engine {mode.rollout_gpus_per_engine} "
        )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {mode.train_num_nodes} "
        f"--actor-num-gpus-per-node {mode.train_gpus_per_node} "
        f"--global-batch-size 256 "
        "--delay-split-train-data-by-dp "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
        "--moe-token-dispatcher-type alltoall "
        "--advantage-estimator grpo "
        "--eps-clip 0.2 "
        f"--num-rollout {num_steps if num_steps is not None else mode.num_steps} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{optimizer_args} "
        f"{rollout_args} "
        f"{get_debug_dump_args(dump_dir=dump_dir, enable_dumper=enable_dumper)} "
        f"{mode.parallel_args} "
        f"{misc_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
    )

    return train_args


def get_debug_dump_args(*, dump_dir: str, enable_dumper: bool) -> str:
    dumper_args: str = ""
    if enable_dumper:
        dumper_args = (
            f"--dumper-dir {shlex.quote(str(Path(dump_dir) / 'dumps'))} "
            f"--dumper-fwd-bwd enable=1 enable_model_value=1 enable_model_grad=1 include_parallel_rank_in_filename=1 "
            f"--dumper-source-patcher-config-train {shlex.quote(str(_source_patcher_path()))} "
        )

    return f"--save-debug-event-data {shlex.quote(str(Path(dump_dir) / EVENTS_DIRNAME))} {dumper_args}"


def get_ft_args(mode: FTTestMode, *, api_server_args: str = "--api-server-port 0 ") -> str:
    checksum_args = "--save-inference-engine-weight-checksum " if mode.has_real_rollout else ""
    return f"--use-fault-tolerance --ft-components {' '.join(mode.ft_components)} {api_server_args}{checksum_args}"


DETERMINISTIC_ROLLOUT_ARGS: str = (
    "--sglang-enable-deterministic-inference --sglang-attention-backend flashinfer --deterministic-mode "
)


def get_train_env_vars_arg(
    mode: FTTestMode, *, deterministic: bool, extra_env_vars: dict[str, str] | None = None
) -> str:
    env_vars: dict[str, str] = {}
    if deterministic:
        env_vars.update(DETERMINISTIC_ENV_VARS)
    if mode.has_real_rollout:
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if extra_env_vars is not None:
        env_vars.update(extra_env_vars)
    if not env_vars:
        return ""
    return f"--train-env-vars '{json.dumps(env_vars)}' "


def run_training(
    train_args: str,
    mode: FTTestMode,
    *,
    dump_dir: str | None = None,
    extra_env_vars: dict[str, str] | None = None,
    config: command_utils.ExecuteTrainConfig | None = None,
    train_script: str = DEFAULT_TRAIN_SCRIPT,
) -> None:
    if dump_dir is not None and os.path.exists(dump_dir):
        shutil.rmtree(dump_dir)
    launch_training(
        train_args=train_args,
        num_gpus_per_node=mode.total_node_gpus,
        megatron_model_type=mode.megatron_model_type,
        config=config,
        train_script=train_script,
        extra_env_vars=extra_env_vars,
    )


def _source_patcher_path() -> Path:
    return get_dumps_root() / "launch-config" / _LAUNCH_ID / "megatron_source_patcher.yaml"
