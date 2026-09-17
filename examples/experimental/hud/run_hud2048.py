"""HUD v6 2048 computer-use GRPO launcher (Qwen3-VL-4B-Instruct, FSDP, single node).

Based on the CI-verified qwen3_vl FSDP recipe (tests/e2e/fsdp/
test_qwen3_vl_4B_fsdp.py); swaps in the HUD rollout and the env-side reward.
Prompts are task slugs (the real prompt is the task template's first yield,
delivered inside the environment), so no --multimodal-keys: screenshots arrive
at runtime through the harness.

--sglang-tool-call-parser is load-bearing: the policy acts through OpenAI
function calling, and without the parser its tool calls stay unparsed text,
every episode ends at turn 1, and training silently optimizes nothing.

Env knobs:
  MILES_SCRIPT_MODEL_NAME   default Qwen3-VL-4B-Instruct
  MILES_SCRIPT_NUM_GPUS     default 8
  MILES_SCRIPT_NUM_ROLLOUT  default 40
  MILES_SCRIPT_MODE         normal | debug_rollout_only (skips training)
                            | smoke (2 episodes, rollout only)
  MILES_SCRIPT_OUTPUT_DIR   checkpoints and rollout dumps; point at storage
                            that outlives the node (default /root/hud2048-rl)

MODE=smoke is the rung to use when checking that the plumbing works: the bugs
a rollout batch surfaces are per-episode ones, so two episodes find them in
four minutes where a full batch takes half an hour. What a bigger *rollout*
batch cannot tell you -- reward spread within a group, weight sync, hours-long
stability -- needs real training, so go there next rather than widening this.
"""

import os
from pathlib import Path

from miles.utils.external_utils import command_utils

MODEL_NAME = os.environ.get("MILES_SCRIPT_MODEL_NAME", "Qwen3-VL-4B-Instruct")
NUM_GPUS = int(os.environ.get("MILES_SCRIPT_NUM_GPUS", "8"))
NUM_ROLLOUT = int(os.environ.get("MILES_SCRIPT_NUM_ROLLOUT", "40"))
MODE = os.environ.get("MILES_SCRIPT_MODE", "normal")
OUTPUT_DIR = os.environ.get("MILES_SCRIPT_OUTPUT_DIR", "/root/hud2048-rl")

SMOKE = MODE == "smoke"
# 4 prompts x 8 samples: the group size is what gives GRPO its within-group
# reward spread, so it is a training parameter, not a throughput one.
BATCH_SIZE, SAMPLES_PER_PROMPT = (1, 2) if SMOKE else (4, 8)


def preflight() -> dict[str, str]:
    """Check what every episode needs, before any GPU is claimed.

    Both failures this catches -- a missing SDK, a missing Daytona credential --
    would otherwise first appear inside an episode, where the sample is dropped
    and the rollout loop simply refills: a silent churn burning GPUs instead of
    an error. (The credential contract follows the openenv example's: workers
    read the key from their own environment or from a file, and the launcher
    forwards only the *path*, because worker env rides Ray's runtime_env, which
    is echoed into driver logs and persisted in job metadata in plaintext.)
    """
    import importlib.util  # noqa: PLC0415 - launch-time only

    for module, hint in (("hud", "pip install hud"), ("daytona", "pip install daytona")):
        if importlib.util.find_spec(module) is None:
            raise SystemExit(f"the rollout needs the {module} SDK: {hint}")

    key_file = Path(os.environ.get("DAYTONA_API_KEY_FILE", "~/.config/daytona/api_key")).expanduser()
    try:
        has_file = bool(key_file.read_text().strip())
    except OSError:
        has_file = False
    if has_file:
        print(f"hud: Daytona credential: file {key_file} (forwarding the path, workers read it)")
        return {"DAYTONA_API_KEY_FILE": str(key_file)}
    if os.environ.get("DAYTONA_API_KEY", "").strip():
        print("hud: Daytona credential: worker environment (DAYTONA_API_KEY assumed present there)")
        return {}
    raise SystemExit(
        "the HUD sandbox needs Daytona credentials: put the key in a file "
        f"({key_file}; DAYTONA_API_KEY_FILE overrides) or in the workers' environment "
        "as DAYTONA_API_KEY. Provision the file with:\n"
        "  mkdir -p ~/.config/daytona && echo dtn_... > ~/.config/daytona/api_key"
    )


def execute():
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    rollout_args = (
        "--prompt-data /root/hud2048_train.jsonl "
        "--input-key prompt "
        "--label-key label "
        # carries the HUD task row (env, template id, args) onto Sample.metadata
        "--metadata-key metadata "
        "--apply-chat-template "
        "--rollout-shuffle "
        f"--num-rollout {1 if SMOKE else NUM_ROLLOUT} "
        f"--rollout-batch-size {BATCH_SIZE} "
        f"--n-samples-per-prompt {SAMPLES_PER_PROMPT} "
        "--rollout-max-response-len 12288 "
        "--rollout-max-context-len 16384 "
        "--rollout-temperature 1.0 "
        f"--global-batch-size {BATCH_SIZE * SAMPLES_PER_PROMPT} "
    )

    custom_args = (
        "--custom-generate-function-path examples.experimental.hud.rollout.generate "
        # Reward = which task template to train on. Of the starter's three 2048
        # graders, reach_score is the only always-informative one: reach_tile is a
        # step function that goes flat when a population converges on one tile
        # (zeroing GRPO's within-group variance), and near_win is binary. Its target
        # is retuned to 1024 via make_hud_data --args-json: a strong episode on this
        # move budget scores ~900, which keeps the best episodes inside the linear
        # part instead of all clipping at 1.0.
        "--custom-rm-path examples.experimental.hud.rollout.reward_func "
        # HUD 2048 (v6 browser starter). Sizing comes from measurements on this task;
        # the numbers and their reasons are annotated where they are set.
        #
        # The env package: `hud init v6browser --preset browser` (no API key; public
        # starter with 2048 + a todo app in real Chromium). Point hud_env_dir at it;
        # the snapshot is built cloud-side by Daytona from its Dockerfile.hud on first
        # use and rebuilt automatically whenever the package's content hash changes --
        # which also means a laptop and a training node each build their own unless
        # the directory content is byte-identical (mind stray __pycache__).
        #
        # That package needs one edit before it is trainable (README has the sed): run
        # the browser in --kiosk on a 1280x1200 desktop. As scaffolded, the browser's
        # chrome costs twice -- it crops the 4x4 board out of the viewport, and it
        # gives the policy a sidebar to click, which takes keyboard focus away from the
        # game so arrow keys do nothing. A capable model reasons its way out (zoom,
        # click the page); a small one cannot, and with every episode then scoring zero
        # there is no reward spread for RL to learn the escape from either.
        "--hud-env-dir /root/v6browser "
        "--hud-snapshot-name miles-hud-v6browser "
        # Turns are the expensive axis -- every turn re-prefills the whole screenshot
        # history through the vision encoder -- while keys within one press() call are
        # nearly free. Vision tokens go with screenshot *area*, so width is the lever:
        # at 512 a frame of this desktop costs ~240 of them, at 640 ~380, and the
        # difference compounds over 24 turns. 512 still resolves the tiles.
        "--hud-max-steps 24 "
        "--hud-shot-width 512 "
        # Room for a short thought plus a complete tool call. Too tight and a turn is
        # cut mid-JSON, which the harness answers by rewriting that turn's recorded
        # arguments -- ending the stitch early (see rollout.py) and wasting the turns
        # after it.
        "--hud-max-tokens-per-turn 512 "
        # One episode = one Daytona sandbox; the gate caps them process-wide,
        # independent of rollout batch geometry. The timeout bounds an episode wedged
        # on a dead sandbox or a stalled engine -- at batch granularity one wedged
        # episode otherwise stalls the whole step.
        "--hud-max-sandboxes 32 "
        "--hud-episode-timeout-s 1800 "
    )

    # A rollout batch of long multi-image episodes is ~2GB of pixel tensors;
    # dispatching it to the train actors keeps rank 0 busy past the default
    # 10-minute NCCL timeout while the other ranks sit in their first
    # collective, and the watchdog then kills the job at the end of step one.
    fsdp_args = (
        "--train-backend fsdp --gradient-checkpointing --update-weight-buffer-size 536870912 "
        "--distributed-timeout-minutes 60 "
    )

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
    # diagnosable after the fact; log-multi-turn adds per-turn counts and
    # lengths, the difference between "the reward moved" and knowing how many
    # actions produced it.
    telemetry_args = "--use-miles-dashboard --dashboard-gpu-sample-interval 5 --log-multi-turn --log-passrate "

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        "--sglang-attention-backend fa3 "
        "--sglang-tool-call-parser qwen25 "
        "--attn-implementation flash_attention_3 "
    )

    debug_args = "--debug-rollout-only " if MODE in ("debug_rollout_only", "smoke") else ""

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        # Point MILES_SCRIPT_OUTPUT_DIR at storage that outlives the node --
        # without --save a multi-hour run leaves nothing but a metrics curve
        # behind, and a config change means relearning from the base model.
        f"--save {OUTPUT_DIR}/ckpt "
        "--save-interval 5 "
        f"--dump-details {OUTPUT_DIR}/dump "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{custom_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{fsdp_args} "
        f"{sglang_args} "
        f"{telemetry_args} "
        f"{debug_args} "
        f"{misc_args} "
    )

    extra_env_vars = {"CUDA_DEVICE_MAX_CONNECTIONS": "1", **preflight()}
    for key in ("WANDB_API_KEY", "HUD_API_KEY"):
        if os.environ.get(key):
            extra_env_vars[key] = os.environ[key]

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
        extra_env_vars=extra_env_vars,
    )


if __name__ == "__main__":
    execute()
