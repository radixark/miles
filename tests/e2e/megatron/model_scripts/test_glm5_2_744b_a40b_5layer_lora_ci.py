import os

from scripts.run_glm5_2_744b_a40b_lora import ScriptArgs, _prepare_download, _train
from tests.ci.ci_register import register_cuda_ci

from miles.utils.external_utils import command_utils


# Smoke test for scripts/run_glm5_2_744b_a40b_lora.py on the 5-layer toy (DSA cross-layer
# path, full rollout -> train -> save loop). Runs one diagonal of the MoE-expert LoRA
# matrix plus sglang's fused shared expert (engine EP=1), which no other LoRA test serves;
# every combination must pass. Functionality, not accuracy; 4 GPUs (TP=EP=4).


# weekly only: `long` is outside run-ci-lora and the nightly scope
register_cuda_ci(
    est_time=1800,
    suite="stage-c-4-gpu-h200",
    labels=["long"],
    hardware=["hopper", "blackwell"],
    num_gpus=4,
)

# skip the engine-side stacked params a frozen-base LoRA run cannot re-ship
# (they keep their correct checkpoint values; everything else is verified)
_BASE_EXTRA = (
    "--ci-test --no-enable-sample-ownership-checker "
    "--ci-disable-logprobs-checker "
    "--check-weight-update-skip-list fused_qkv_a_proj_with_mqa indexer. "
)

# (name, dsa_attention_backend, experts_shared_outer_loras, virtual_experts_serving, shared_experts_fusion)
_CONFIGS = [
    ("megatron + shared-outer + virtual-experts", "megatron", True, True, False),
    ("tilelang + per-expert + no-virtual-experts", "tilelang", False, False, False),
    ("megatron + per-expert + no-virtual-experts + fused-shared-experts", "megatron", False, False, True),
]


def _args(dsa: str, shared_outer: bool, virtual_experts: bool, shared_experts_fusion: bool) -> ScriptArgs:
    return ScriptArgs.from_env(
        model_name="GLM-5.2_5layer",
        num_nodes=1,
        num_gpus_per_node=4,
        num_rollout=2,
        enable_wandb=False,
        dsa_attention_backend=dsa,
        experts_shared_outer_loras=shared_outer,
        sglang_shared_experts_fusion=shared_experts_fusion,
        extra_args=_BASE_EXTRA + ("" if virtual_experts else "--no-sglang-lora-use-virtual-experts "),
    )


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.output_dir}")
    _prepare_download(args)


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(_args(*_CONFIGS[0][1:]))
    for name, *config in _CONFIGS:
        print(f"[glm5.2-lora-ci] ===== combo: {name} =====", flush=True)
        # fresh ray/sglang between combos
        command_utils.default_config().create_backend().exec_command_cpu(
            "ray stop --force || true; pkill -9 sglang || true; sleep 10"
        )
        execute(_args(*config))
        print(f"[glm5.2-lora-ci] ===== combo PASSED: {name} =====", flush=True)
