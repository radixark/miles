#!/usr/bin/env python3
"""
Nemotron-3-Ultra-550B-A55B GRPO RL launcher (16 nodes x 8 H200 = 128 GPU, colocate).

Self-contained Python sibling of scripts/run-nemotron-3-ultra-550b-a55b.sh:
starts/join Ray and submits train.py with the Ultra arg set.

Prereq: run scripts/convert-nemotron-3-ultra-550b-hf-to-dist.sh once to produce
the torch_dist checkpoint (--load uses it to skip the slow per-run 512-expert HF
bridge; --hf-checkpoint still points at HF for tokenizer + SGLang rollout).

Usage (run on each pod):
    head:   python scripts/run_nemotron_3_ultra_550b_a55b.py head   <head_ip>
    worker: python scripts/run_nemotron_3_ultra_550b_a55b.py worker <head_ip>

Key knobs (env or flags):
    MODELS_DIR / DATASETS_DIR   base dirs (default /cluster_public/miles_data/{models,datasets})
    --num-rollout / --response-len / --rollout-batch-size

Notes
- Mamba n_groups=8 caps attention/mamba TP at 8. The 550B (~1.1TB bf16) does not
  fit one 8-GPU SGLang engine, so rollout uses 32-GPU engines with EP=32 +
  DP-attention (dp=4) so attention/mamba run at attn_tp = 32/4 = 8.
- Rollout routing-replay (--use-miles-router/--use-rollout-routing-replay) is NOT
  enabled for the 108-layer Ultra yet (capturer shape fix pending); logprob diff
  is still ~0.01 without it.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

NUM_NODES = 16
GPUS_PER_NODE = 8
TOTAL_GPUS = NUM_NODES * GPUS_PER_NODE  # 128


def sh(cmd: str, check: bool = True):
    print(f"+ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def model_args() -> list[str]:
    # NVIDIA Nemotron-3-Ultra-550B-A55B (nemotron_h: hybrid Mamba2 + Attention + latent-MoE).
    # 108 layers, hidden 8192, 512 experts top-22, moe_latent 2048, sigmoid router + expert bias.
    return [
        "--disable-bias-linear",
        "--group-query-attention",
        "--num-attention-heads",
        "64",
        "--num-query-groups",
        "2",
        "--kv-channels",
        "128",
        "--num-layers",
        "108",
        "--hidden-size",
        "8192",
        "--ffn-hidden-size",
        "5120",
        "--normalization",
        "RMSNorm",
        "--position-embedding-type",
        "none",
        "--vocab-size",
        "131072",
        "--make-vocab-size-divisible-by",
        "128",
        "--untie-embeddings-and-output-weights",
        "--num-experts",
        "512",
        "--moe-router-topk",
        "22",
        "--moe-ffn-hidden-size",
        "5120",
        "--moe-shared-expert-intermediate-size",
        "10240",
        "--moe-latent-size",
        "2048",
        "--moe-router-score-function",
        "sigmoid",
        "--moe-router-enable-expert-bias",
        "--moe-grouped-gemm",
        "--moe-router-dtype",
        "fp32",
        "--moe-router-num-groups",
        "1",
        "--moe-router-group-topk",
        "1",
        "--moe-router-topk-scaling-factor",
        "5.0",
        "--moe-router-pre-softmax",
        "--moe-router-load-balancing-type",
        "seq_aux_loss",
        "--moe-router-bias-update-rate",
        "0",
        "--moe-aux-loss-coeff",
        "0",
    ]


def build_train_argv(a) -> list[str]:
    hf = a.hf_checkpoint or f"{a.models_dir}/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
    dist = a.load or f"{a.models_dir}/nemotron-3-ultra-550b-a55b_torch_dist"
    data = a.prompt_data or f"{a.datasets_dir}/dapo-math-17k/dapo-math-17k.jsonl"

    ckpt = [
        "--hf-checkpoint",
        hf,  # tokenizer + SGLang rollout
        "--load",
        dist,  # Megatron native load (skip per-run HF bridge)
        "--ref-load",
        dist,
        "--save",
        f"{a.models_dir}/nemotron-3-ultra-550b-a55b_miles",
        "--save-interval",
        str(a.save_interval),
        "--no-save-optim",
        "--megatron-to-hf-mode",
        "bridge",
    ]
    rollout = [
        "--prompt-data",
        data,
        "--input-key",
        "prompt",
        "--label-key",
        "label",
        "--apply-chat-template",
        "--rollout-shuffle",
        "--rm-type",
        "deepscaler",
        "--num-rollout",
        str(a.num_rollout),
        "--rollout-batch-size",
        str(a.rollout_batch_size),
        "--n-samples-per-prompt",
        str(a.n_samples_per_prompt),
        "--rollout-max-response-len",
        str(a.response_len),
        "--rollout-temperature",
        "1",
        "--global-batch-size",
        str(a.global_batch_size),
        "--balance-data",
    ]
    perf = [
        # TP8 PP4 EP32 ETP1 (DP4) = 128. Mamba n_groups=8 -> attention/mamba TP<=8.
        "--tensor-model-parallel-size",
        "8",
        "--sequence-parallel",
        "--pipeline-model-parallel-size",
        "4",
        "--context-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "32",
        "--expert-tensor-parallel-size",
        "1",
        "--recompute-granularity",
        "full",
        "--recompute-method",
        "uniform",
        "--recompute-num-layers",
        "1",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu",
        str(a.max_tokens_per_gpu),
        "--log-probs-chunk-size",
        "128",
    ]
    grpo = [
        "--advantage-estimator",
        "grpo",
        "--use-kl-loss",
        "--kl-loss-coef",
        "0.00",
        "--kl-loss-type",
        "low_var_kl",
        "--entropy-coef",
        "0.00",
        "--eps-clip",
        "0.2",
        "--eps-clip-high",
        "0.28",
    ]
    optim = [
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
        "--optimizer-cpu-offload",
        "--overlap-cpu-optimizer-d2h-h2d",
        "--use-precision-aware-optimizer",
    ]
    sglang = [
        # 32-GPU engine + EP=32 + DP-attention (dp=4 -> attn_tp=8, divides n_groups=8).
        "--rollout-num-gpus-per-engine",
        "32",
        "--sglang-ep-size",
        "32",
        "--sglang-dp-size",
        "4",
        "--sglang-enable-dp-attention",
        "--sglang-mem-fraction-static",
        str(a.mem_fraction),
    ]
    misc = [
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend",
        "auto",
    ]
    wandb = []
    if a.wandb_project:
        wandb = [
            "--use-wandb",
            "--wandb-project",
            a.wandb_project,
            "--wandb-group",
            a.wandb_group or "nemotron-3-ultra-550b-a55b",
        ]
        if os.environ.get("WANDB_KEY"):
            wandb += ["--wandb-key", os.environ["WANDB_KEY"]]

    return (
        [
            "python3",
            "train.py",
            "--actor-num-nodes",
            str(NUM_NODES),
            "--actor-num-gpus-per-node",
            str(GPUS_PER_NODE),
            "--rollout-num-gpus",
            str(TOTAL_GPUS),
            "--colocate",
        ]
        + model_args()
        + ckpt
        + rollout
        + optim
        + grpo
        + wandb
        + perf
        + sglang
        + misc
    )


def main():
    p = argparse.ArgumentParser(description="Nemotron-3-Ultra-550B-A55B RL launcher (128 GPU).")
    p.add_argument("role", choices=["head", "worker"])
    p.add_argument("head_ip")
    p.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "/cluster_public/miles_data/models"))
    p.add_argument("--datasets-dir", default=os.environ.get("DATASETS_DIR", "/cluster_public/miles_data/datasets"))
    p.add_argument("--hf-checkpoint", default=os.environ.get("HF", ""))
    p.add_argument("--load", default=os.environ.get("DIST", ""))
    p.add_argument("--prompt-data", default="")
    p.add_argument("--num-rollout", type=int, default=30)
    p.add_argument("--rollout-batch-size", type=int, default=32)
    p.add_argument("--n-samples-per-prompt", type=int, default=8)
    p.add_argument("--response-len", type=int, default=8192)
    p.add_argument("--global-batch-size", type=int, default=128)
    p.add_argument("--max-tokens-per-gpu", type=int, default=1024)
    p.add_argument("--mem-fraction", type=float, default=0.7)
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-group", default="")
    a = p.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    # clean any prior run on this pod
    for c in ["pkill -9 sglang", "ray stop --force", "pkill -9 ray", "pkill -9 python3"]:
        sh(c, check=False)
    time.sleep(3)

    has_nvlink = (
        "1"
        if subprocess.run("nvidia-smi topo -m 2>/dev/null | grep -qo 'NV[0-9]'", shell=True).returncode == 0
        else "0"
    )

    if a.role == "worker":
        # wait for head, then join and block
        for _ in range(60):
            if subprocess.run(f"nc -z {a.head_ip} 6379", shell=True).returncode == 0:
                break
            print(f"waiting for head {a.head_ip}:6379 ...", flush=True)
            time.sleep(5)
        sh(f"ray start --address={a.head_ip}:6379 --num-gpus={GPUS_PER_NODE} --disable-usage-stats --block")
        return

    # head: start ray, wait for all 128 GPUs, submit
    sh(
        f"ray start --head --node-ip-address {a.head_ip} --num-gpus {GPUS_PER_NODE} "
        f"--disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265"
    )
    print(f"Waiting for {TOTAL_GPUS} GPUs in the Ray cluster ...", flush=True)
    for _ in range(240):
        r = subprocess.run("ray status 2>/dev/null", shell=True, capture_output=True, text=True)
        if f"{float(TOTAL_GPUS)} GPU" in r.stdout:
            print(f"[ray] cluster ready: {TOTAL_GPUS} GPUs", flush=True)
            break
        time.sleep(5)

    runtime_env = {
        "env_vars": {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": has_nvlink,
            # nemotron DP-attention uses existing kernels; skip the blanket sgl-kernel guard.
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        }
    }
    train_argv = build_train_argv(a)
    cmd = [
        "ray",
        "job",
        "submit",
        "--address=http://127.0.0.1:8265",
        f"--runtime-env-json={json.dumps(runtime_env)}",
        "--",
    ] + train_argv
    print("+ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
