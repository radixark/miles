#!/usr/bin/env python3
"""
Unified prepare + run script for P2P weight transfer examples.

Replaces 9 prepare-*.sh + 8 run-*-profile.sh scripts with a single Python
script driven by a model registry.

Usage:
    python run.py prepare <MODEL> [--download-only] [--ckpt-dir DIR]
    python run.py run     <MODEL> [--mode p2p] [--node-rank 0] [--head-ip IP]

Examples:
    python run.py prepare GLM-4.7-Flash
    python run.py run     GLM-4.7-Flash --mode p2p --node-rank 0 --head-ip 10.0.0.1
    python run.py prepare GLM-5_4layer --download-only
    python run.py run     GLM-5_4layer --mode broadcast --node-rank 1 --head-ip 10.0.0.1
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass
class PrepareConfig:
    """Configuration for the `prepare` subcommand."""

    hf_repo: str
    model_type: str  # megatron model type (maps to scripts/models/<type>.sh)
    datasets: list[str] = field(default_factory=lambda: ["zhuzilin/dapo-math-17k"])
    convert_gpus_per_node: int = 8
    convert_multinode: bool = False
    convert_num_nodes: int | None = None
    convert_extra_args: str = ""
    ckpt_dir: str = "/root/multinode"  # default dir_dst for convert_checkpoint
    needs_transformers_install: str | None = None  # pip install URL
    needs_config_patch: bool = False  # GLM-5 deepseek_v32 patch
    use_snapshot_download: bool = False  # use huggingface_hub.snapshot_download


@dataclass
class RunConfig:
    """Configuration for the `run` subcommand."""

    model_type: str
    nnodes: int
    num_train_gpus: int
    num_rollout_gpus: int
    gpus_per_node: int = 8

    # Training parallelism
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1
    etp: int = 1
    sequence_parallel: bool = True
    decoder_last_pipeline_num_layers: int | None = None
    max_tokens_per_gpu: int = 2048

    # Rollout
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 4
    rollout_temperature: float = 1.0
    global_batch_size: int = 16
    balance_data: bool = True

    # Eval
    has_eval: bool = False
    eval_name: str = "aime"
    eval_dataset: str = "/root/datasets/aime-2024/aime-2024.jsonl"
    eval_temperature: float | None = None
    eval_top_p: float = 0.7

    # GRPO
    advantage_estimator: str = "grpo"
    use_kl_loss: bool = False
    kl_loss_coef: str = "0.00"
    kl_coef: str | None = None
    entropy_coef: str = "0.00"
    eps_clip: str = "0.2"
    eps_clip_high: str | None = "0.28"

    # Optimizer
    optimizer_cpu_offload: bool = False

    # SGLang
    sglang_gpus_per_engine: int = 2
    sglang_mem_fraction: float = 0.8
    sglang_ep_size: int | None = None
    sglang_dp_size: int | None = None
    sglang_moe_dense_tp_size: int | None = None
    sglang_cuda_graph_bs: str | None = None  # e.g. "1 2 4 8 16"
    sglang_enable_dp_attention: bool = False
    sglang_enable_dp_lm_head: bool = False
    sglang_server_concurrency: int | None = None
    sglang_moe_runner_backend: str | None = None
    sglang_fp8_gemm_backend: str | None = None
    # GLM-5 NSA-specific
    sglang_nsa: bool = False
    sglang_page_size: int | None = None
    sglang_cuda_graph_max_bs: int | None = None
    sglang_max_running_requests: int | None = None
    sglang_chunked_prefill_size_factor: int | None = None  # multiplied by sglang_world_size
    sglang_watchdog_timeout: int | None = None
    sglang_disable_cuda_graph: bool = False

    # Misc
    enable_nccl_nvls: bool = False
    buffer_size_gb: float = 1.0
    buffer_size_broadcast_gb: float | None = None  # separate for broadcast mode
    mc_transfer_timeout: int = 300
    allgather_cp: bool = False
    moe_token_dispatcher_type: str | None = None
    data_pad_size_multiplier: int | None = None
    log_probs_chunk_size: int | None = None

    # Checkpoint location
    ckpt_dir: str = "/root/multinode"
    use_ckpt_save_dir: bool = False  # use CKPT_SAVE_DIR env var

    # Extra runtime env vars
    extra_env_vars: dict[str, str] = field(default_factory=dict)

    # Rotary base override
    rotary_base: int | None = None


# ---------------------------------------------------------------------------
# Model registry: prepare configs
# ---------------------------------------------------------------------------

PREPARE_CONFIGS: dict[str, PrepareConfig] = {
    "Qwen3-4B": PrepareConfig(
        hf_repo="Qwen/Qwen3-4B",
        model_type="qwen3-4B",
    ),
    "GLM-Z1-9B-0414": PrepareConfig(
        hf_repo="zai-org/GLM-Z1-9B-0414",
        model_type="glm4-9B",
    ),
    "Moonlight-16B-A3B-Instruct": PrepareConfig(
        hf_repo="moonshotai/Moonlight-16B-A3B-Instruct",
        model_type="moonlight",
    ),
    "GLM-4.7-Flash": PrepareConfig(
        hf_repo="zai-org/GLM-4.7-Flash",
        model_type="glm4.7-flash",
        datasets=["zhuzilin/dapo-math-17k", "zhuzilin/aime-2024"],
        needs_transformers_install=(
            "git+https://github.com/huggingface/transformers.git" "@76732b4e7120808ff989edbd16401f61fa6a0afa"
        ),
    ),
    "GLM-5_4layer": PrepareConfig(
        hf_repo="Pinaster/GLM-5_4layer",
        model_type="glm5-744B-A40B_4layer",
        convert_gpus_per_node=4,
        convert_extra_args=(
            "--pipeline-model-parallel-size 1 "
            "--expert-model-parallel-size 1 "
            "--tensor-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1"
        ),
        needs_config_patch=True,
        use_snapshot_download=True,
    ),
    "GLM-5_20layer": PrepareConfig(
        hf_repo="Pinaster/GLM-5_20layer",
        model_type="glm5-744B-A40B_20layer",
        convert_multinode=True,
        convert_num_nodes=2,
        convert_extra_args=(
            "--tensor-model-parallel-size 1 " "--expert-tensor-parallel-size 1 " "--expert-model-parallel-size 4"
        ),
        needs_config_patch=True,
        use_snapshot_download=True,
    ),
    "GLM-5": PrepareConfig(
        hf_repo="zai-org/GLM-5",
        model_type="glm5-744B-A40B",
        convert_multinode=True,
        convert_extra_args=(
            "--pipeline-model-parallel-size 4 "
            "--expert-model-parallel-size 32 "
            "--tensor-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
            "--decoder-last-pipeline-num-layers 18"
        ),
        needs_config_patch=True,
        use_snapshot_download=True,
    ),
    "Qwen3-30B-A3B": PrepareConfig(
        hf_repo="Qwen/Qwen3-30B-A3B",
        model_type="qwen3-30B-A3B",
        convert_gpus_per_node=4,
        datasets=["zhuzilin/dapo-math-17k", "zhuzilin/aime-2024"],
    ),
    "GLM-4.5-Air": PrepareConfig(
        hf_repo="zai-org/GLM-4.5-Air",
        model_type="glm4.5-106B-A12B",
        datasets=["zhuzilin/dapo-math-17k", "zhuzilin/aime-2024"],
        convert_multinode=True,
        use_snapshot_download=True,
    ),
    "Qwen3-235B-A22B-Instruct-2507": PrepareConfig(
        hf_repo="Qwen/Qwen3-235B-A22B-Instruct-2507",
        model_type="qwen3-235B-A22B",
        convert_gpus_per_node=4,
        datasets=["zhuzilin/dapo-math-17k", "zhuzilin/aime-2024"],
    ),
    "Kimi-K2-Instruct": PrepareConfig(
        hf_repo="moonshotai/Kimi-K2-Instruct",
        model_type="kimi-k2",
        datasets=["zhuzilin/dapo-math-17k", "zhuzilin/aime-2024"],
        convert_extra_args="--expert-model-parallel-size 8 --decoder-last-pipeline-num-layers 5",
    ),
}


# ---------------------------------------------------------------------------
# Model registry: run configs
# ---------------------------------------------------------------------------

RUN_CONFIGS: dict[str, RunConfig] = {
    "Qwen3-4B": RunConfig(
        model_type="qwen3-4B",
        nnodes=1,
        num_train_gpus=4,
        num_rollout_gpus=4,
        tp=2,
        cp=2,
        rollout_batch_size=8,
        n_samples_per_prompt=8,
        rollout_temperature=0.8,
        global_batch_size=32,
        sglang_gpus_per_engine=2,
        sglang_mem_fraction=0.8,
    ),
    "GLM-Z1-9B-0414": RunConfig(
        model_type="glm4-9B",
        nnodes=1,
        num_train_gpus=4,
        num_rollout_gpus=4,
        tp=2,
        cp=2,
        ep=1,
        use_kl_loss=True,
        rollout_batch_size=8,
        n_samples_per_prompt=8,
        global_batch_size=32,
        sglang_gpus_per_engine=2,
        sglang_mem_fraction=0.8,
    ),
    "Moonlight-16B-A3B-Instruct": RunConfig(
        model_type="moonlight",
        nnodes=2,
        num_train_gpus=8,
        num_rollout_gpus=8,
        tp=2,
        ep=8,
        use_kl_loss=True,
        sglang_gpus_per_engine=8,
        sglang_mem_fraction=0.7,
        sglang_ep_size=8,
        sglang_cuda_graph_bs="1 2 4 8 16",
        sglang_enable_dp_attention=True,
        sglang_enable_dp_lm_head=True,
    ),
    "GLM-4.7-Flash": RunConfig(
        model_type="glm4.7-flash",
        nnodes=2,
        num_train_gpus=8,
        num_rollout_gpus=8,
        tp=4,
        ep=8,
        use_kl_loss=True,
        optimizer_cpu_offload=True,
        has_eval=True,
        eval_name="aime24",
        eval_temperature=0.6,
        eval_top_p=0.95,
        sglang_gpus_per_engine=4,
        sglang_mem_fraction=0.7,
        sglang_ep_size=4,
        sglang_cuda_graph_bs="1 2 4 8 16",
        sglang_enable_dp_attention=True,
        sglang_enable_dp_lm_head=True,
    ),
    "GLM-5_4layer": RunConfig(
        model_type="glm5-744B-A40B_4layer",
        nnodes=2,
        num_train_gpus=8,
        num_rollout_gpus=8,
        tp=4,
        ep=8,
        kl_coef="0.00",
        enable_nccl_nvls=True,
        mc_transfer_timeout=600,
        allgather_cp=True,
        moe_token_dispatcher_type="alltoall",
        data_pad_size_multiplier=4096,
        log_probs_chunk_size=1024,
        use_ckpt_save_dir=True,
        # SGLang NSA
        sglang_gpus_per_engine=8,
        sglang_mem_fraction=0.70,
        sglang_ep_size=8,
        sglang_dp_size=8,
        sglang_moe_dense_tp_size=1,
        sglang_enable_dp_attention=True,
        sglang_enable_dp_lm_head=True,
        sglang_nsa=True,
        sglang_page_size=64,
        sglang_cuda_graph_max_bs=8,
        sglang_max_running_requests=512,
        sglang_chunked_prefill_size_factor=2048,
        sglang_watchdog_timeout=3600,
        sglang_disable_cuda_graph=True,
        extra_env_vars={
            "INDEXER_ROPE_NEOX_STYLE": "0",
            "NVSHMEM_DISABLE_NCCL": "1",
        },
    ),
    "GLM-5_20layer": RunConfig(
        model_type="glm5-744B-A40B_20layer",
        nnodes=12,
        num_train_gpus=48,
        num_rollout_gpus=48,
        tp=4,
        pp=3,
        ep=16,
        decoder_last_pipeline_num_layers=6,
        max_tokens_per_gpu=1024,
        kl_coef="0.00",
        optimizer_cpu_offload=True,
        enable_nccl_nvls=True,
        mc_transfer_timeout=600,
        allgather_cp=True,
        moe_token_dispatcher_type="alltoall",
        data_pad_size_multiplier=4096,
        log_probs_chunk_size=1024,
        use_ckpt_save_dir=True,
        sglang_gpus_per_engine=16,
        sglang_mem_fraction=0.70,
        sglang_ep_size=16,
        sglang_dp_size=16,
        sglang_moe_dense_tp_size=1,
        sglang_enable_dp_attention=True,
        sglang_enable_dp_lm_head=True,
        sglang_nsa=True,
        sglang_page_size=64,
        sglang_cuda_graph_max_bs=8,
        sglang_max_running_requests=512,
        sglang_chunked_prefill_size_factor=2048,
        sglang_watchdog_timeout=3600,
        sglang_disable_cuda_graph=True,
        extra_env_vars={
            "INDEXER_ROPE_NEOX_STYLE": "0",
            "NVSHMEM_DISABLE_NCCL": "1",
        },
    ),
    "GLM-5": RunConfig(
        model_type="glm5-744B-A40B",
        nnodes=32,
        num_train_gpus=128,
        num_rollout_gpus=128,
        tp=4,
        pp=8,
        cp=2,
        ep=16,
        decoder_last_pipeline_num_layers=8,
        max_tokens_per_gpu=256,
        kl_coef="0.00",
        optimizer_cpu_offload=True,
        enable_nccl_nvls=True,
        mc_transfer_timeout=600,
        allgather_cp=True,
        moe_token_dispatcher_type="alltoall",
        data_pad_size_multiplier=4096,
        log_probs_chunk_size=1024,
        use_ckpt_save_dir=True,
        sglang_gpus_per_engine=64,
        sglang_mem_fraction=0.90,
        sglang_ep_size=64,
        sglang_dp_size=64,
        sglang_moe_dense_tp_size=1,
        sglang_enable_dp_attention=True,
        sglang_enable_dp_lm_head=True,
        sglang_nsa=True,
        sglang_page_size=64,
        sglang_cuda_graph_max_bs=8,
        sglang_max_running_requests=512,
        sglang_chunked_prefill_size_factor=2048,
        sglang_watchdog_timeout=3600,
        sglang_disable_cuda_graph=True,
        extra_env_vars={
            "INDEXER_ROPE_NEOX_STYLE": "0",
            "NVSHMEM_DISABLE_NCCL": "1",
        },
    ),
    "Qwen3-30B-A3B": RunConfig(
        model_type="qwen3-30B-A3B",
        nnodes=4,
        num_train_gpus=16,
        num_rollout_gpus=16,
        tp=4,
        ep=8,
        advantage_estimator="gspo",
        eps_clip="4e-4",
        eps_clip_high=None,
        rollout_temperature=0.8,
        has_eval=True,
        optimizer_cpu_offload=True,
        enable_nccl_nvls=True,
        rotary_base=1000000,
        sglang_gpus_per_engine=8,
        sglang_mem_fraction=0.8,
        sglang_ep_size=8,
        sglang_cuda_graph_bs="1 2 4 8 16",
        sglang_enable_dp_attention=True,
        sglang_enable_dp_lm_head=True,
    ),
    "GLM-4.5-Air": RunConfig(
        model_type="glm4.5-106B-A12B",
        nnodes=8,
        num_train_gpus=32,
        num_rollout_gpus=32,
        tp=1,
        pp=4,
        ep=8,
        decoder_last_pipeline_num_layers=10,
        advantage_estimator="gspo",
        eps_clip="4e-4",
        eps_clip_high=None,
        rollout_temperature=0.8,
        has_eval=True,
        optimizer_cpu_offload=True,
        use_ckpt_save_dir=True,
        buffer_size_broadcast_gb=4.0,
        rotary_base=1000000,
        sglang_gpus_per_engine=8,
        sglang_mem_fraction=0.8,
        sglang_ep_size=8,
        sglang_cuda_graph_bs="1 2 4 8 16",
        sglang_enable_dp_attention=True,
        sglang_enable_dp_lm_head=True,
    ),
    "Qwen3-235B-A22B-Instruct-2507": RunConfig(
        model_type="qwen3-235B-A22B",
        nnodes=16,
        num_train_gpus=64,
        num_rollout_gpus=64,
        tp=4,
        pp=4,
        cp=2,
        ep=16,
        decoder_last_pipeline_num_layers=22,
        max_tokens_per_gpu=8192,
        advantage_estimator="gspo",
        eps_clip="4e-4",
        eps_clip_high=None,
        rollout_batch_size=8,
        n_samples_per_prompt=8,
        rollout_temperature=0.8,
        global_batch_size=64,
        has_eval=True,
        optimizer_cpu_offload=True,
        enable_nccl_nvls=True,
        rotary_base=5000000,
        sglang_gpus_per_engine=32,
        sglang_mem_fraction=0.75,
        sglang_ep_size=32,
        sglang_dp_size=1,
        sglang_cuda_graph_bs="1 2 4 8 16",
        sglang_enable_dp_attention=True,
        sglang_enable_dp_lm_head=True,
    ),
    "Kimi-K2-Instruct": RunConfig(
        model_type="kimi-k2",
        nnodes=64,
        num_train_gpus=256,
        num_rollout_gpus=256,
        tp=8,
        pp=8,
        cp=4,
        ep=32,
        decoder_last_pipeline_num_layers=5,
        max_tokens_per_gpu=16384,
        use_kl_loss=True,
        rollout_batch_size=8,
        n_samples_per_prompt=8,
        global_batch_size=64,
        has_eval=True,
        optimizer_cpu_offload=True,
        enable_nccl_nvls=True,
        moe_token_dispatcher_type="alltoall",
        sglang_gpus_per_engine=32,
        sglang_mem_fraction=0.7,
        sglang_ep_size=32,
        sglang_dp_size=8,
        sglang_moe_dense_tp_size=1,
        sglang_cuda_graph_bs="1 2 4 8 16",
        sglang_enable_dp_attention=True,
        sglang_enable_dp_lm_head=True,
        sglang_server_concurrency=1024,
        sglang_moe_runner_backend="triton",
        sglang_fp8_gemm_backend="triton",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MILES_ROOT = Path(__file__).resolve().parents[2]


def run_cmd(cmd: str, check: bool = True) -> int:
    """Run a shell command, streaming output."""
    print(f"+ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, executable="/bin/bash")
    if check and result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


def list_models():
    """Print available model names."""
    print("Available models for `prepare`:")
    for name in sorted(PREPARE_CONFIGS):
        print(f"  {name}")
    print("\nAvailable models for `run`:")
    for name in sorted(RUN_CONFIGS):
        print(f"  {name}")


# ---------------------------------------------------------------------------
# GLM-5 config patch
# ---------------------------------------------------------------------------


def patch_glm5_config(model_dir: str):
    """Patch config.json for DeepseekV32 and create stub Python files."""
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found")

    with open(config_path) as f:
        config = json.load(f)

    # Patch model_type and architectures
    if config.get("model_type") != "deepseek_v32":
        config["architectures"] = ["DeepseekV32ForCausalLM"]
        config["auto_map"] = {
            "AutoConfig": "configuration_deepseek_v32.DeepseekV32Config",
            "AutoModelForCausalLM": "modeling_deepseek_v32.DeepseekV32ForCausalLM",
        }
        config["model_type"] = "deepseek_v32"
        if "rope_theta" not in config:
            rp = config.get("rope_parameters", {})
            if isinstance(rp, dict) and "rope_theta" in rp:
                config["rope_theta"] = rp["rope_theta"]
            else:
                config["rope_theta"] = 1000000
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Patched {config_path}")
    elif "auto_map" not in config:
        config["auto_map"] = {
            "AutoConfig": "configuration_deepseek_v32.DeepseekV32Config",
            "AutoModelForCausalLM": "modeling_deepseek_v32.DeepseekV32ForCausalLM",
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Added auto_map to {config_path}")
    else:
        print("Checkpoint already patched, skipping")

    # Always ensure rope_theta at top level
    if "rope_theta" not in config:
        rp = config.get("rope_parameters", {})
        if isinstance(rp, dict) and "rope_theta" in rp:
            config["rope_theta"] = rp["rope_theta"]
        else:
            config["rope_theta"] = 1000000
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Added rope_theta={config['rope_theta']} to {config_path}")

    # Create stub Python files
    config_py = model_dir / "configuration_deepseek_v32.py"
    config_py.write_text(
        "from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config\n"
        "\n"
        "\n"
        "class DeepseekV32Config(DeepseekV3Config):\n"
        '    model_type = "deepseek_v32"\n'
        "\n"
        "    def __init__(self, index_topk=2048, **kwargs):\n"
        "        super().__init__(**kwargs)\n"
        "        self.index_topk = index_topk\n"
        "        # Promote rope_theta from rope_parameters to top-level for mbridge\n"
        '        if not hasattr(self, "rope_theta") or self.rope_theta is None:\n'
        '            rp = getattr(self, "rope_parameters", None) or {}\n'
        '            if isinstance(rp, dict) and "rope_theta" in rp:\n'
        '                self.rope_theta = rp["rope_theta"]\n'
        "            else:\n"
        "                self.rope_theta = 1000000  # GLM-5 default\n"
    )
    print(f"Wrote {config_py}")

    modeling_py = model_dir / "modeling_deepseek_v32.py"
    modeling_py.write_text(
        "from transformers import PreTrainedModel\n"
        "from .configuration_deepseek_v32 import DeepseekV32Config\n"
        "\n"
        "\n"
        "class DeepseekV32ForCausalLM(PreTrainedModel):\n"
        "    config_class = DeepseekV32Config\n"
    )
    print(f"Wrote {modeling_py}")


# ---------------------------------------------------------------------------
# Subcommand: prepare
# ---------------------------------------------------------------------------


def cmd_prepare(model_name: str, download_only: bool = False, ckpt_dir: str | None = None):
    """Download model, datasets, patch config, and convert checkpoint."""
    if model_name not in PREPARE_CONFIGS:
        print(f"ERROR: Unknown model '{model_name}'.")
        list_models()
        sys.exit(1)

    cfg = PREPARE_CONFIGS[model_name]
    effective_ckpt_dir = ckpt_dir or os.environ.get("CKPT_SAVE_DIR") or cfg.ckpt_dir

    print(f"=== Preparing {model_name} ===")
    print(f"HF repo       : {cfg.hf_repo}")
    print(f"Model type    : {cfg.model_type}")
    print(f"Ckpt dir      : {effective_ckpt_dir}")
    print(f"Download-only : {download_only}")

    # Step 0: Install custom transformers if needed
    if cfg.needs_transformers_install:
        run_cmd(f"pip install {cfg.needs_transformers_install}")

    # Step 1: Download model
    os.makedirs("/root/models", exist_ok=True)
    os.makedirs("/root/datasets", exist_ok=True)
    model_dir = f"/root/models/{model_name}"
    if cfg.use_snapshot_download:
        run_cmd(
            f'python3 -c "'
            f"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{cfg.hf_repo}', local_dir='{model_dir}')"
            f'"'
        )
    else:
        run_cmd(f'hf download "{cfg.hf_repo}" --local-dir "{model_dir}"')

    # Step 2: Download datasets
    for ds in cfg.datasets:
        run_cmd(
            f'python3 -c "'
            f"from miles.utils.external_utils.command_utils import hf_download_dataset; "
            f"hf_download_dataset('{ds}')"
            f'"'
        )

    # Step 3: Patch config (GLM-5 only)
    if cfg.needs_config_patch:
        patch_glm5_config(model_dir)

    # Step 4: Convert checkpoint
    if not download_only:
        os.makedirs(effective_ckpt_dir, exist_ok=True)

        num_nodes_arg = ""
        if cfg.convert_num_nodes is not None:
            num_nodes_arg = f", num_nodes={cfg.convert_num_nodes}"

        run_cmd(
            f'python3 -c "'
            f"from miles.utils.external_utils.command_utils import convert_checkpoint; "
            f"convert_checkpoint("
            f"model_name='{model_name}', "
            f"megatron_model_type='{cfg.model_type}', "
            f"num_gpus_per_node={cfg.convert_gpus_per_node}, "
            f"multinode={cfg.convert_multinode}, "
            f"extra_args='{cfg.convert_extra_args}', "
            f"dir_dst='{effective_ckpt_dir}'"
            f"{num_nodes_arg})"
            f'"'
        )
        print("Prepare done (full: download + convert).")
    else:
        print("Prepare done (download-only: skipped checkpoint conversion).")


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------


def cmd_run(
    model_name: str,
    mode: str = "p2p",
    node_rank: int = 0,
    head_ip: str = "",
):
    """Launch training with P2P or broadcast weight transfer."""
    if model_name not in RUN_CONFIGS:
        print(f"ERROR: Unknown model '{model_name}'.")
        list_models()
        sys.exit(1)

    cfg = RUN_CONFIGS[model_name]
    is_single_node = cfg.nnodes == 1
    num_train_nodes = max(1, cfg.num_train_gpus // cfg.gpus_per_node)

    # Resolve checkpoint directory
    if cfg.use_ckpt_save_dir:
        ckpt_save_dir = os.environ.get("CKPT_SAVE_DIR", "/root")
    else:
        ckpt_save_dir = cfg.ckpt_dir

    skip_validation = int(os.environ.get("SKIP_VALIDATION", "0"))
    if model_name == "Kimi-K2-Instruct" and not skip_validation:
        raise NotImplementedError(
            "Kimi-K2-Instruct does not support --check-weight-update-equal without hard-coded "
            "workarounds. Set SKIP_VALIDATION=1 to bypass this check and run without weight "
            "validation. Check `./docs/en/advanced/p2p-weight-transfer.md` for more details."
        )
    bucket_size_gb = float(os.environ.get("BUCKET_SIZE_GB", str(cfg.buffer_size_gb)))

    print()
    print("=" * 60)
    print(f"  Model      : {model_name} ({cfg.model_type})")
    print(f"  Mode       : {mode}")
    if is_single_node:
        print(f"  GPUs       : {cfg.num_train_gpus} train + {cfg.num_rollout_gpus} rollout (single node)")
    else:
        print(
            f"  Nodes      : {cfg.nnodes} total " f"({num_train_nodes} train + {cfg.nnodes - num_train_nodes} rollout)"
        )
    print(f"  Parallelism: TP={cfg.tp} PP={cfg.pp} CP={cfg.cp} EP={cfg.ep}")
    if not is_single_node:
        print(f"  Node rank  : {node_rank}, Head: {head_ip}")
    print("=" * 60)
    print()

    # --- Cleanup stale processes (exclude self) ---
    my_pid = os.getpid()
    my_ppid = os.getppid()
    # Kill stale python/ray but spare this process tree
    pkill_python = f"pgrep -x 'python|python3' | grep -v -w {my_pid} | grep -v -w {my_ppid} | xargs -r kill -9 || true"
    run_cmd("pkill -9 sglang || true", check=False)
    run_cmd("sleep 3", check=False)
    run_cmd("ray stop --force || true", check=False)
    run_cmd("pkill -9 ray || true", check=False)
    run_cmd(pkill_python, check=False)
    run_cmd("sleep 3", check=False)
    run_cmd("pkill -9 ray || true", check=False)
    run_cmd(pkill_python, check=False)
    run_cmd("pkill -9 redis || true", check=False)

    # --- Source model args ---
    model_args_source = f'source "{MILES_ROOT}/scripts/models/{cfg.model_type}.sh"'

    # --- Worker sleep ---
    if not is_single_node and node_rank > 0:
        time.sleep(20)

    # --- Launch Ray ---
    if is_single_node:
        run_cmd(
            f"ray start --head --num-gpus {cfg.gpus_per_node} "
            f"--disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265"
        )
    elif node_rank == 0:
        run_cmd(
            f"RAY_memory_monitor_refresh_ms=0 "
            f"ray start --head --node-ip-address {head_ip} --num-gpus {cfg.gpus_per_node} "
            f"--disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265"
        )
    else:
        run_cmd(
            f"RAY_memory_monitor_refresh_ms=0 "
            f"ray start --address={head_ip}:6379 --num-gpus {cfg.gpus_per_node} "
            f"--disable-usage-stats"
        )

    # --- Wait for all GPUs (head node only, multi-node) ---
    if not is_single_node and node_rank == 0:
        expected_gpus = cfg.nnodes * cfg.gpus_per_node
        print(f"Waiting for {expected_gpus} GPUs in Ray cluster...")
        while True:
            try:
                result = subprocess.run(
                    "python3 -c \"import ray; ray.init(address='auto', ignore_reinit_error=True); "
                    "print(int(ray.cluster_resources().get('GPU', 0))); ray.shutdown()\"",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                available = int(result.stdout.strip()) if result.returncode == 0 else 0
            except Exception:
                available = 0
            print(f"  ... detected {available}/{expected_gpus} GPUs")
            if available >= expected_gpus:
                break
            time.sleep(5)
        print(f"All {expected_gpus} GPUs available. Submitting job.")

    # --- Build runtime env JSON ---
    nccl_nvls_val = "1" if cfg.enable_nccl_nvls else "0"
    env_vars = {
        "RAY_DEBUG": "1",
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": nccl_nvls_val,
        "MILES_LOG_DIR": os.environ.get("MILES_LOG_DIR", ""),
    }
    if not is_single_node:
        env_vars["MC_TRANSFER_TIMEOUT"] = str(cfg.mc_transfer_timeout)
    if cfg.rotary_base is not None:
        env_vars["MODEL_ARGS_ROTARY_BASE"] = str(cfg.rotary_base)
    env_vars.update(cfg.extra_env_vars)
    runtime_env_json = json.dumps({"env_vars": env_vars})

    # --- Build train.py arguments ---
    # Checkpoint
    args = []
    args.extend(["--hf-checkpoint", f"/root/models/{model_name}"])
    args.extend(["--ref-load", f"{ckpt_save_dir}/{model_name}_torch_dist"])

    # Rollout
    args.extend(
        [
            "--prompt-data",
            "/root/datasets/dapo-math-17k/dapo-math-17k.jsonl",
            "--input-key",
            "prompt",
            "--label-key",
            "label",
            "--apply-chat-template",
            "--rollout-shuffle",
            "--rm-type",
            "deepscaler",
            "--num-rollout",
            "13",
            "--rollout-batch-size",
            str(cfg.rollout_batch_size),
            "--n-samples-per-prompt",
            str(cfg.n_samples_per_prompt),
            "--rollout-max-response-len",
            "100",
            "--rollout-temperature",
            str(cfg.rollout_temperature),
            "--global-batch-size",
            str(cfg.global_batch_size),
        ]
    )
    if cfg.balance_data:
        args.append("--balance-data")

    # Eval
    if cfg.has_eval:
        args.extend(
            [
                "--eval-prompt-data",
                cfg.eval_name,
                cfg.eval_dataset,
                "--n-samples-per-eval-prompt",
                "16",
                "--eval-max-response-len",
                "16384",
            ]
        )
        if cfg.eval_temperature is not None:
            args.extend(["--eval-temperature", str(cfg.eval_temperature)])
        args.extend(["--eval-top-p", str(cfg.eval_top_p)])

    # Training parallelism
    args.extend(["--tensor-model-parallel-size", str(cfg.tp)])
    if cfg.sequence_parallel:
        args.append("--sequence-parallel")
    args.extend(["--pipeline-model-parallel-size", str(cfg.pp)])
    args.extend(["--context-parallel-size", str(cfg.cp)])
    args.extend(["--expert-model-parallel-size", str(cfg.ep)])
    args.extend(["--expert-tensor-parallel-size", str(cfg.etp)])
    if cfg.decoder_last_pipeline_num_layers is not None:
        args.extend(["--decoder-last-pipeline-num-layers", str(cfg.decoder_last_pipeline_num_layers)])
    args.extend(
        [
            "--recompute-granularity",
            "full",
            "--recompute-method",
            "uniform",
            "--recompute-num-layers",
            "1",
            "--use-dynamic-batch-size",
            "--max-tokens-per-gpu",
            str(cfg.max_tokens_per_gpu),
        ]
    )
    if cfg.data_pad_size_multiplier is not None:
        args.extend(["--data-pad-size-multiplier", str(cfg.data_pad_size_multiplier)])
    if cfg.log_probs_chunk_size is not None:
        args.extend(["--log-probs-chunk-size", str(cfg.log_probs_chunk_size)])

    # GRPO / GSPO
    args.extend(["--advantage-estimator", cfg.advantage_estimator])
    if cfg.use_kl_loss:
        args.append("--use-kl-loss")
    args.extend(["--kl-loss-coef", cfg.kl_loss_coef])
    args.extend(["--kl-loss-type", "low_var_kl"])
    if cfg.kl_coef is not None:
        args.extend(["--kl-coef", cfg.kl_coef])
    args.extend(["--entropy-coef", cfg.entropy_coef])
    args.extend(["--eps-clip", cfg.eps_clip])
    if cfg.eps_clip_high is not None:
        args.extend(["--eps-clip-high", cfg.eps_clip_high])

    # Optimizer
    args.extend(
        [
            "--optimizer",
            "adam",
            "--lr",
            "1e-6",
            "--lr-decay-style",
            "constant",
            "--weight-decay",
            "0.1",
            "--adam-beta1",
            "0.9",
            "--adam-beta2",
            "0.98",
        ]
    )
    if cfg.optimizer_cpu_offload:
        args.extend(
            [
                "--optimizer-cpu-offload",
                "--overlap-cpu-optimizer-d2h-h2d",
                "--use-precision-aware-optimizer",
            ]
        )

    # SGLang
    args.extend(["--rollout-num-gpus-per-engine", str(cfg.sglang_gpus_per_engine)])
    args.extend(["--rollout-num-gpus", str(cfg.num_rollout_gpus)])
    args.extend(["--sglang-mem-fraction-static", str(cfg.sglang_mem_fraction)])
    if cfg.sglang_ep_size is not None:
        args.extend(["--sglang-ep-size", str(cfg.sglang_ep_size)])
    if cfg.sglang_dp_size is not None:
        args.extend(["--sglang-dp-size", str(cfg.sglang_dp_size)])
    if cfg.sglang_moe_dense_tp_size is not None:
        args.extend(["--sglang-moe-dense-tp-size", str(cfg.sglang_moe_dense_tp_size)])
    if cfg.sglang_cuda_graph_bs:
        args.extend(["--sglang-cuda-graph-bs"] + cfg.sglang_cuda_graph_bs.split())
    if cfg.sglang_enable_dp_attention:
        args.append("--sglang-enable-dp-attention")
    if cfg.sglang_enable_dp_lm_head:
        args.append("--sglang-enable-dp-lm-head")
    if cfg.sglang_server_concurrency is not None:
        args.extend(["--sglang-server-concurrency", str(cfg.sglang_server_concurrency)])
    if cfg.sglang_moe_runner_backend:
        args.extend(["--sglang-moe-runner-backend", cfg.sglang_moe_runner_backend])
    if cfg.sglang_fp8_gemm_backend:
        args.extend(["--sglang-fp8-gemm-backend", cfg.sglang_fp8_gemm_backend])

    # NSA (GLM-5)
    if cfg.sglang_nsa:
        if cfg.sglang_page_size is not None:
            args.extend(["--sglang-page-size", str(cfg.sglang_page_size)])
        args.extend(
            [
                "--sglang-nsa-decode-backend",
                "flashmla_sparse",
                "--sglang-nsa-prefill-backend",
                "flashmla_sparse",
                "--sglang-attention-backend",
                "nsa",
            ]
        )
        if cfg.sglang_cuda_graph_max_bs is not None:
            args.extend(["--sglang-cuda-graph-max-bs", str(cfg.sglang_cuda_graph_max_bs)])
        if cfg.sglang_max_running_requests is not None:
            args.extend(["--sglang-max-running-requests", str(cfg.sglang_max_running_requests)])
        if cfg.sglang_chunked_prefill_size_factor is not None:
            chunked_size = cfg.sglang_chunked_prefill_size_factor * cfg.sglang_gpus_per_engine
            args.extend(["--sglang-chunked-prefill-size", str(chunked_size)])
        if cfg.sglang_watchdog_timeout is not None:
            args.extend(["--sglang-watchdog-timeout", str(cfg.sglang_watchdog_timeout)])
        if cfg.sglang_disable_cuda_graph:
            args.append("--sglang-disable-cuda-graph")

    # P2P-specific SGLang args
    if mode == "p2p":
        args.append("--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine")

    # Skip validation / model loader
    if skip_validation:
        args.append("--sglang-load-format dummy")
    else:
        # Only add multithread loader for models that have it in the original scripts
        if cfg.nnodes >= 4 or cfg.sglang_nsa:
            args.extend(
                [
                    "--sglang-model-loader-extra-config",
                    '{"enable_multithread_load":true,"num_threads":8}',
                ]
            )

    # Misc
    args.extend(
        [
            "--attention-dropout",
            "0.0",
            "--hidden-dropout",
            "0.0",
            "--accumulate-allreduce-grads-in-fp32",
            "--attention-softmax-in-fp32",
            "--attention-backend",
            "flash",
        ]
    )
    if cfg.allgather_cp:
        args.append("--allgather-cp")
    if cfg.moe_token_dispatcher_type:
        args.extend(["--moe-token-dispatcher-type", cfg.moe_token_dispatcher_type])

    args.extend(["--actor-num-nodes", str(num_train_nodes)])
    args.extend(["--actor-num-gpus-per-node", str(cfg.num_train_gpus // num_train_nodes)])

    # Buffer size
    if mode == "p2p":
        buffer_size = int(bucket_size_gb * 1024 * 1024 * 1024)
    elif cfg.buffer_size_broadcast_gb is not None:
        buffer_size = int(cfg.buffer_size_broadcast_gb * 1024 * 1024 * 1024)
    else:
        buffer_size = int(bucket_size_gb * 1024 * 1024 * 1024)
    buffer_size = int(os.environ.get("BUFFER_SIZE", str(buffer_size)))
    args.extend(["--update-weight-buffer-size", str(buffer_size)])

    if not skip_validation:
        args.append("--check-weight-update-equal")

    args.extend(["--update-weight-transfer-mode", mode])

    # --- Submit Ray job (head node only, or single-node) ---
    if is_single_node or node_rank == 0:
        import shlex

        args_str = " ".join(shlex.quote(a) for a in args)
        run_cmd(
            f"{model_args_source} && "
            f"ray job submit --address='http://127.0.0.1:8265' "
            f"--runtime-env-json='{runtime_env_json}' "
            f'-- python3 "{MILES_ROOT}/train.py" '
            f"${{MODEL_ARGS[@]}} "
            f"{args_str}",
            check=False,
        )

        # Signal workers
        if not is_single_node:
            signal_dir = os.environ.get("MILES_LOG_DIR", "/data/ray/signals")
            os.makedirs(signal_dir, exist_ok=True)
            done_file = os.path.join(signal_dir, f"job_done_{mode}")
            with open(done_file, "w") as f:
                f.write("0")
    else:
        # Worker: wait for head to finish
        signal_dir = os.environ.get("MILES_LOG_DIR", "/data/ray/signals")
        done_file = os.path.join(signal_dir, f"job_done_{mode}")
        print(f"Worker node {node_rank}: Ray joined, waiting for head to finish...")
        while not os.path.exists(done_file):
            time.sleep(10)
        print(f"Worker node {node_rank}: head finished, exiting.")

    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """Simple CLI dispatcher (no external dependencies)."""
    if len(sys.argv) < 2:
        print("Usage: python run.py <prepare|run|list> [args...]")
        sys.exit(1)

    subcmd = sys.argv[1]

    if subcmd == "list":
        list_models()
        return

    if subcmd == "prepare":
        import argparse

        parser = argparse.ArgumentParser(description="Prepare model for training")
        parser.add_argument("model", help="Model name (see `python run.py list`)")
        parser.add_argument(
            "--download-only", action="store_true", help="Skip checkpoint conversion (for worker nodes)"
        )
        parser.add_argument("--ckpt-dir", default=None, help="Override checkpoint save directory")
        # Skip first 2 args (run.py prepare)
        parsed = parser.parse_args(sys.argv[2:])
        cmd_prepare(parsed.model, download_only=parsed.download_only, ckpt_dir=parsed.ckpt_dir)

    elif subcmd == "run":
        import argparse

        parser = argparse.ArgumentParser(description="Run training with weight transfer")
        parser.add_argument("model", help="Model name (see `python run.py list`)")
        parser.add_argument(
            "--mode", default="p2p", choices=["p2p", "broadcast"], help="Weight transfer mode (default: p2p)"
        )
        parser.add_argument("--node-rank", type=int, default=0, help="Node rank (0=head, default: 0)")
        parser.add_argument("--head-ip", default="", help="Head node IP (auto-detect for single-node)")
        parsed = parser.parse_args(sys.argv[2:])
        cmd_run(parsed.model, mode=parsed.mode, node_rank=parsed.node_rank, head_ip=parsed.head_ip)

    else:
        print(f"Unknown subcommand: {subcmd}")
        print("Usage: python run.py <prepare|run|list> [args...]")
        sys.exit(1)


if __name__ == "__main__":
    main()
