"""Shared part of the Kimi-K3.1 W4A16 single-LoRA RL recipes, `run_kimi_k31_w4a16_lora_<gpus>x<hardware>.py`.

W4A16: the trainer keeps the frozen routed experts as the checkpoint's MXFP4 bytes and dequantizes them per layer;
the rollout serves the same checkpoint on Marlin MXFP4 with the BF16 adapter as a bypass. Both sides read the
native checkpoint, so there is no BF16 or torch_dist conversion. Each recipe file pins the hardware and the
parallel layout; everything else (workload, loss, rollout checks, saving) lives here.
"""

import dataclasses
from dataclasses import dataclass
from typing import Literal

from scripts.run_kimi_k3 import ScriptArgs, _train

# routing-replay check threshold for --ci-test; NCCL 2.30.7's diagnostic RAS thread segfaulted in
# rasMsgHandlePeersUpdate on every node right after the engine's first memory release
_RECIPE_ENV = "MILES_TEST_R3_THRESHOLD=0.1 NCCL_RAS_ENABLE=0"
# prompt tokens per request when sizing the KV cache; dapo-math under the K3.1 chat template has median 215,
# p99.9 853 and max 1546 tokens, and the cache holds a DP group's sum, so this only has to cover the mean
_PROMPT_TOKEN_BUDGET = 1024


@dataclass
class W4A16LoraArgs(ScriptArgs):
    mode: Literal["normal"] = "normal"
    model_name: Literal["Kimi-K3.1"] = "Kimi-K3.1"
    train_mode: Literal["lora"] = "lora"
    task: Literal["dapo-math"] = "dapo-math"

    lora_rank: int = 32
    lora_alpha: int = 64

    reward_model: Literal["deterministic_random", "deepscaler", "math"] | None = "deepscaler"
    rollout_batch_size: int | None = 32
    n_samples_per_prompt: int | None = 16
    global_batch_size: int | None = 512
    rollout_max_response_len: int | None = 3072
    # a decode step costs about the same at any batch, so throughput scales with concurrency; finished requests
    # are refilled from the step's queue
    rollout_max_concurrency: int = 256
    max_tokens_per_gpu: int | None = 8192
    # 256 MiB buckets cost 0.7 s each for the ~46 GiB r=32 adapter
    update_weight_buffer_size: int | None = 2 * 1024**3
    distributed_timeout_minutes: int = 60
    sglang_mem_fraction_static: float = 0.9
    # drop groups whose 16 rewards are all equal and keep sampling prompts until rollout_batch_size groups remain
    dynamic_sampling_filter: bool = False
    # groups taken from the dataset per submission, at the start and on every refill; None means rollout_batch_size
    over_sampling_batch_size: int | None = None

    # adapter shards every save_interval rollouts and after the last; --skip-saving turns saving off
    save_interval: int = 5

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"
        if self.ref_load is None:
            # raw bridge mode loads the trainer straight from the MXFP4 checkpoint
            self.ref_load = self.hf_checkpoint
        if self.sglang_max_total_tokens is None:
            # per attention-DP group (the flag's unit there): its share of the running requests, each with a full
            # response, admitted at once without retraction
            requests_per_group = self.rollout_max_concurrency // self.rollout_attn_dp_size
            self.sglang_max_total_tokens = requests_per_group * (self.rollout_max_response_len + _PROMPT_TOKEN_BUDGET)
        super().__post_init__()


def _recipe_train_args(args: W4A16LoraArgs) -> str:
    recipe_args = (
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-kv-cache-dtype bfloat16 "
        "--sglang-chunked-prefill-size 4096 "
        # score centering: REINFORCE with the sampler's top-128 candidates centering the score, so rollout/trainer
        # drift cannot push a constant-reward update; rollout log-probs replace the trainer's forward-only pass
        "--loss-type score_centering "
        "--rollout-top-logprobs-num 128 "
        "--score-centering-is none "
        "--use-rollout-logprobs "
        "--disable-grpo-std-normalization "
        "--calculate-per-token-loss "
        # routing replay only: indexer replay ships ~196 KB of top-k per token and cost ~30 min a step
        "--use-rollout-routing-replay "
        "--no-enable-sample-ownership-checker "
        "--ci-test "
        "--ci-disable-logprobs-checker "
        # the weight-update checker snapshots the whole engine to host memory and re-sends the base every LoRA sync
        "--ci-disable-weight-update-checker "
        "--update-weights-timeout 3600 "
        "--update-weight-engine-request-timeout 1800 "
    )
    if args.dynamic_sampling_filter:
        recipe_args += (
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter "
        )
    if args.over_sampling_batch_size is not None:
        recipe_args += f"--over-sampling-batch-size {args.over_sampling_batch_size} "
    if not args.skip_saving:
        # raw bridge mode saves only the adapter shards; the Adam state (~294 GB at r=32) is left out
        recipe_args += f"--save {args.save_dir}/{args.run_id} --save-interval {args.save_interval} --no-save-optim "
    return recipe_args


def train_w4a16_lora(args: W4A16LoraArgs) -> None:
    _train(
        dataclasses.replace(
            args,
            extra_args=f"{_recipe_train_args(args)}{args.extra_args}",
            extra_env_vars=f"{_RECIPE_ENV} {args.extra_env_vars}".strip(),
            # the recipe's own --save above replaces the launcher's --save-interval 50
            skip_saving=True,
        )
    )
