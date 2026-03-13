from typing import Any

COMMON_ENV_VARS = {
    "PYTHONUNBUFFERED": "1",
    "RAY_DEDUP_LOGS": "0",
    "RAY_memory_monitor_refresh_ms": "0",
    "NCCL_DEBUG": "WARN",
    "WANDB_API_KEY": "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",
}


def flavor_miles_sunrise(hardware: str) -> dict[str, Any]:
    """Miles Sunrise environment for DeepSeek-V4 RL training (H200/B200, CUDA 12)."""
    return {
        "image": "radixark/miles@sha256:497cebb0995928bf093b149519564b3dabf2d7424b806ac8b2552fe5270272fe",
        "build_commands": [
            "pip uninstall -y miles megatron sglang || true",
            "pip install flashinfer-jit-cache==0.6.2 --index-url https://flashinfer.ai/whl/cu129",
            "apt remove -y libgtest-dev || true",
            "pip install z3-solver cython polars einops",
            "cd /tmp && rm -rf fast-hadamard-transform && git clone https://github.com/Dao-AILab/fast-hadamard-transform.git && cd fast-hadamard-transform && pip install -e . -v --no-build-isolation",
            "cd /tmp && rm -rf flash-mla && git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla && cd flash-mla && git submodule update --init --recursive && pip install --no-build-isolation -v .",
            "cd /tmp && rm -rf transformers && git clone https://github.com/huggingface/transformers.git && cd transformers && git checkout 8cb5963cc22174954e7dca2c0a3320b7dc2f4edc && pip install -e .",
        ],
        "setup_commands": [
            "pip install -e /workspace/sglang/python --no-deps",
            "pip install -e /workspace/miles --no-deps",
            "pip install -e /workspace/Megatron-LM --no-deps",
        ],
        "repos": [
            "NightFall->sglang",
            "miles-sunrise->miles",
            "megatron-sunrise->Megatron-LM",
            "/Users/yueming.yuan/radixark/cluster_scripts->cluster_scripts",
        ],
        "ray": False,  # managed manually with custom ports (6399/8266) to avoid conflicts on shared host
        "env_vars": COMMON_ENV_VARS,
    }
