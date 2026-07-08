import os

from scripts.run_qwen3_5_35b_a3b_lora import ScriptArgs, _prepare_download, _train
from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

# Smoke test for scripts/run_qwen3_5_35b_a3b_lora.py on the full Qwen3.5-35B-A3B
# (40 layers, hybrid GDN + full attention, 256-expert MoE), matching the other
# Qwen3.5 e2e tests which also run the full checkpoint. LoRA covers attention,
# MoE routed + shared experts, and the GDN in_proj/out_proj projections. Runs the
# MoE-expert LoRA matrix — {shared-outer + virtual-experts, per-expert +
# no-virtual-experts} — and every combination must pass. Functionality, not
# accuracy; 8 GPUs (TP2 per num_query_groups=2, EP=8).


register_cuda_ci(est_time=3600, suite="stage-c-8-gpu-h100", labels=["model-scripts"])

# (name, experts_shared_outer_loras, virtual_experts_serving)
_CONFIGS = [
    ("shared-outer + virtual-experts", True, True),
    ("per-expert + no-virtual-experts", False, False),
]


def _args(shared_outer: bool, virtual_experts: bool) -> ScriptArgs:
    return ScriptArgs(
        model_name="Qwen3.5-35B-A3B",
        num_nodes=1,
        num_gpus_per_node=8,
        num_rollout=1,
        experts_shared_outer_loras=shared_outer,
        enable_wandb=False,
        extra_args=(
            "--ci-test --ci-disable-logprobs-checker --disable-weights-backuper "
            + ("" if virtual_experts else "--no-sglang-lora-use-virtual-experts ")
        ),
    )


def prepare(args: ScriptArgs):
    _prepare_download(args)


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(_args(*_CONFIGS[0][1:]))
    for name, shared_outer, virtual_experts in _CONFIGS:
        print(f"[qwen3.5-lora-ci] ===== combo: {name} =====", flush=True)
        # fresh ray/sglang between combos
        U.exec_command("ray stop --force || true; pkill -9 sglang || true; sleep 10")
        execute(_args(shared_outer, virtual_experts))
        print(f"[qwen3.5-lora-ci] ===== combo PASSED: {name} =====", flush=True)
