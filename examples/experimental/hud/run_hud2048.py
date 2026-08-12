"""HUD 2048 computer-use GRPO launcher (Qwen3-VL-4B-Instruct, FSDP, single node).

Based on the CI-verified qwen3_vl FSDP recipe (tests/e2e/fsdp/
test_qwen3_vl_4B_fsdp.py); swaps in the HUD multi-turn rollout, the env-side
dense reward, and text-only prompts (screenshots arrive at runtime, so no
--multimodal-keys). Runs the legacy rollout path — the multi-turn interaction
engine this builds on is verified there.

Env knobs:
  MILES_SCRIPT_MODEL_NAME   default Qwen3-VL-4B-Instruct
  MILES_SCRIPT_NUM_GPUS     default 8
  MILES_SCRIPT_NUM_ROLLOUT  default 40
  MILES_SCRIPT_MODE         normal | debug_rollout_only (skips training)
  MILES_SCRIPT_OUTPUT_DIR   checkpoints and rollout dumps; point at storage
                            that outlives the node (default /root/hud2048-rl)
"""

import os

import miles.utils.external_utils.command_utils as U

MODEL_NAME = os.environ.get("MILES_SCRIPT_MODEL_NAME", "Qwen3-VL-4B-Instruct")
NUM_GPUS = int(os.environ.get("MILES_SCRIPT_NUM_GPUS", "8"))
NUM_ROLLOUT = int(os.environ.get("MILES_SCRIPT_NUM_ROLLOUT", "40"))
MODE = os.environ.get("MILES_SCRIPT_MODE", "normal")
OUTPUT_DIR = os.environ.get("MILES_SCRIPT_OUTPUT_DIR", "/root/hud2048-rl")


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    rollout_args = (
        "--prompt-data /root/hud2048_train.jsonl "
        "--input-key prompt "
        "--label-key label "
        # carries the HUD task row (image, setup, grade) onto Sample.metadata
        "--metadata-key metadata "
        "--apply-chat-template "
        "--rollout-shuffle "
        f"--num-rollout {NUM_ROLLOUT} "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 12288 "
        "--rollout-max-context-len 16384 "
        "--rollout-temperature 1.0 "
        "--global-batch-size 32 "
    )

    custom_args = (
        "--custom-generate-function-path examples.experimental.hud.rollout.generate "
        "--custom-rm-path examples.experimental.hud.rollout.reward_func "
        "--custom-config-path examples/experimental/hud/hud2048_config.yaml "
    )

    fsdp_args = "--train-backend fsdp " "--gradient-checkpointing " "--update-weight-buffer-size 536870912 "

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    # Telemetry. The dashboard collector is what makes a multi-hour run
    # diagnosable after the fact: it records per-phase timing, GPU utilisation
    # and engine metrics, which is how the ViT-prefill stall in this recipe was
    # eventually understood. log-multi-turn adds per-turn counts and lengths,
    # which for a computer-use episode is the difference between "the reward
    # moved" and knowing how many actions produced it.
    telemetry_args = "--use-miles-dashboard --dashboard-gpu-sample-interval 5 --log-multi-turn --log-passrate "

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        "--sglang-attention-backend fa3 "
        "--attn-implementation flash_attention_3 "
    )

    debug_args = "--debug-rollout-only " if MODE == "debug_rollout_only" else ""

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        # Point MILES_SCRIPT_OUTPUT_DIR at storage that outlives the node --
        # without --save a multi-hour run leaves nothing but a metrics curve
        # behind, and a config change means relearning from the base model.
        f"--save {OUTPUT_DIR}/ckpt "
        "--save-interval 10 "
        f"--dump-details {OUTPUT_DIR}/dump "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{custom_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{fsdp_args} "
        f"{sglang_args} "
        f"{telemetry_args} "
        f"{debug_args} "
        f"{misc_args} "
    )

    extra_env_vars = {"CUDA_DEVICE_MAX_CONNECTIONS": "1"}
    if os.environ.get("WANDB_API_KEY"):
        extra_env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    if os.environ.get("DAYTONA_API_KEY"):
        extra_env_vars["DAYTONA_API_KEY"] = os.environ["DAYTONA_API_KEY"]

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
        extra_env_vars=extra_env_vars,
    )


if __name__ == "__main__":
    execute()
