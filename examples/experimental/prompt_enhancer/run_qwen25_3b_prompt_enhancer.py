"""Prompt enhancement with a frozen SGLang Diffusion generator: GRPO on Qwen2.5-3B-Instruct rewriting short
text-to-image prompts, rewarded by HPSv3 (or ImageReward, RePrompt's own reward, for parity) of the image Z-Image-Turbo
renders from the rewrite, scored against the user's prompt, plus RePrompt's format and length terms.

The settings are RePrompt's RL stage (https://arxiv.org/abs/2505.17540), as our earlier reproduction of it in Miles
ran them: 1108 prompts for 3 epochs of 34 steps, 32 prompts x 4
completions per step, lr 2e-6 decaying linearly to 0, AdamW without weight decay, gradient clip 1, temperature 0.9 and
256 new tokens, and a k3 KL loss at 0.04. Z-Image-Turbo renders at 512x512 with 9 steps and no CFG. Run with
ImageReward (`--scorer-urls`) and `--seed-per-sample`, as that reproduction was, the one difference from it, which
rendered with diffusers outside the job, is who renders: here the job launches the generator as a frozen model in
`--sglang-config`.

The job's GPUs hold the trainer, the policy engine, one diffusion engine per generator GPU and, with `--scorer hpsv3`
(the default), one HPSv3 reward pool worker per scorer GPU. With `--scorer imagereward` the image is scored by
`scorer_server.py` processes outside the job, as the earlier reproduction ran it; start them first and pass their URLs.

Args:
    --train-gpus, --policy-gpus, --generator-gpus, --scorer-gpus: the job's GPUs; --num-gpus-per-node is their sum.
    --scorer: hpsv3 (a reward pool in the job) or imagereward (scorer_server.py at --scorer-urls, no scorer GPUs).
    --scorer-urls: comma-separated scorer_server.py URLs; given alone, they select --scorer imagereward.
    Unset, --scorer-gpus is 1 for hpsv3 and 0 for imagereward, and --num-gpus-per-node is the sum of the job's GPUs.
    --seed-per-sample: draw each image's seed per sample, as RePrompt's own run does, instead of one per group.
    --training-seed: Megatron and rollout seed; unset keeps Miles' defaults.
    --model-dir, --data-dir: where the models and the prompt data live; checkpoints go under --output-dir.

Example:
    python examples/experimental/prompt_enhancer/run_qwen25_3b_prompt_enhancer.py
    python examples/experimental/prompt_enhancer/run_qwen25_3b_prompt_enhancer.py --scorer-urls http://127.0.0.1:18200
"""

from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils

POLICY = "Qwen/Qwen2.5-3B-Instruct"
GENERATOR = "Tongyi-MAI/Z-Image-Turbo"
HPSV3 = "MizzenAI/HPSv3"
HPSV3_BASE = "Qwen/Qwen2-VL-7B-Instruct"
# The image term's weight: RePrompt's 0.5 * IR / 2.0 for ImageReward; HPSv3 spreads 3.1x wider on Z-Image-Turbo's
# images (std 2.44 against ImageReward's 0.78), so its weight is 0.25 / 3.1, keeping the image term's spread.
IMAGE_WEIGHTS = {"hpsv3": 0.08, "imagereward": 0.25}
MEGATRON_MODEL_TYPE = "qwen2.5-3B"
REPROMPT_REPO = "https://github.com/microsoft/DKI_LLM.git"
REPROMPT_COMMIT = "c090b7d57e3996c8900e86fecd16085b27e07f29"
RECIPE = "examples.experimental.prompt_enhancer"

_SGLANG_CONFIG = """\
sglang:
  - name: policy
    server_groups:
      - {{worker_type: regular, num_gpus: {policy_gpus}}}
  - name: generator
    model_path: {generator_path}
    update_weights: false
    server_groups:
      - {{worker_type: regular, engine_kind: diffusion, num_gpus: {generator_gpus}}}
"""

_REWARD_POOL_CONFIG = """\
- name: hpsv3
  scorer_path: miles.rollout.rm_hub.image.hpsv3.HPSv3Scorer
  scorer_kwargs: {{checkpoint_path: {checkpoint_path}, base_model_path: {base_model_path}}}
  num_workers: {num_workers}
  uses_gpu: true
  batch_size: 16
  max_wait_ms: 50
"""


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    num_gpus_per_node: int | None = None
    train_gpus: int = 1
    policy_gpus: int = 1
    generator_gpus: int = 5
    scorer_gpus: int | None = None
    scorer: str | None = None
    scorer_urls: str | None = None
    seed_per_sample: bool = False
    training_seed: int | None = None
    num_epoch: int = 3
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 4
    save_interval: int = 50
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""

    def __post_init__(self):
        assert self.num_nodes == 1, "the recipe places its trainer and engines on one node"
        # Scorer URLs alone select ImageReward, so the parity command from before HPSv3 keeps its scorer and layout.
        if self.scorer is None:
            self.scorer = "imagereward" if self.scorer_urls is not None else "hpsv3"
        assert self.scorer in IMAGE_WEIGHTS, f"--scorer is one of {sorted(IMAGE_WEIGHTS)}, got {self.scorer}"
        if self.scorer == "imagereward":
            self.scorer_urls = self.scorer_urls or "http://127.0.0.1:18200"
        assert self.scorer_urls is None or self.scorer == "imagereward", "--scorer-urls are ImageReward's scorers"
        if self.scorer_gpus is None:
            self.scorer_gpus = 1 if self.scorer == "hpsv3" else 0
        assert (self.scorer_gpus > 0) == (
            self.scorer == "hpsv3"
        ), "--scorer hpsv3 scores on --scorer-gpus in the job; --scorer imagereward scores outside it, on none"
        job_gpus = self.train_gpus + self.policy_gpus + self.generator_gpus + self.scorer_gpus
        if self.num_gpus_per_node is None:
            self.num_gpus_per_node = job_gpus
        assert job_gpus == self.num_gpus_per_node, (
            f"--train-gpus {self.train_gpus}, --policy-gpus {self.policy_gpus}, --generator-gpus "
            f"{self.generator_gpus} and --scorer-gpus {self.scorer_gpus} must add up to --num-gpus-per-node "
            f"{self.num_gpus_per_node}"
        )

    @property
    def policy_path(self) -> str:
        return f"{self.model_dir}/{POLICY.split('/')[1]}"

    @property
    def generator_path(self) -> str:
        return f"{self.model_dir}/{GENERATOR.split('/')[1]}"

    @property
    def hpsv3_path(self) -> str:
        return f"{self.model_dir}/{HPSV3.split('/')[1]}"

    @property
    def hpsv3_base_path(self) -> str:
        return f"{self.model_dir}/{HPSV3_BASE.split('/')[1]}"

    @property
    def recipe_data_dir(self) -> str:
        return f"{self.data_dir}/prompt_enhancer"

    @property
    def run_dir(self) -> str:
        return f"{self.output_dir}/{self.run_id}"


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.recipe_data_dir}")
    U.exec_command_cpu(f"hf download {POLICY} --local-dir {args.policy_path}")
    U.exec_command_cpu(f"hf download {GENERATOR} --local-dir {args.generator_path}")
    if args.scorer == "hpsv3":
        U.exec_command_cpu(f"hf download {HPSV3} --local-dir {args.hpsv3_path}")
        U.exec_command_cpu(f"hf download {HPSV3_BASE} --local-dir {args.hpsv3_base_path}")
    U.exec_command_cpu(
        f"[ -d {args.recipe_data_dir}/DKI_LLM ] || git clone {REPROMPT_REPO} {args.recipe_data_dir}/DKI_LLM; "
        f"git -C {args.recipe_data_dir}/DKI_LLM checkout {REPROMPT_COMMIT}"
    )
    U.exec_command_cpu(
        f"cd {command_utils.repo_base_dir} && python -m {RECIPE}.make_data "
        f"{args.recipe_data_dir}/DKI_LLM/RePrompt/data/generation_prompts_1k_filterd.txt {args.policy_path} "
        f"{args.recipe_data_dir}/train.jsonl"
    )
    U.convert_checkpoint(
        model_name=POLICY.split("/")[1],
        megatron_model_type=MEGATRON_MODEL_TYPE,
        # The trainer's GPUs: Qwen2.5-3B's 36 layers do not split into pipeline stages over the whole job's 7.
        num_gpus_per_node=args.train_gpus,
        dir_dst=args.model_dir,
        hf_checkpoint=args.policy_path,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    U = args.create_backend()
    U.execute_train(
        train_args=_train_args(args),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        megatron_path=args.megatron_path,
    )


def _train_args(args: ScriptArgs) -> str:
    ckpt_args = (
        f"--hf-checkpoint {args.policy_path} "
        f"--ref-load {args.policy_path}_torch_dist "
        f"--load {args.run_dir}/checkpoints --save {args.run_dir}/checkpoints --save-interval {args.save_interval} "
        f"--save-hf {args.run_dir}/hf/iter_{{rollout_id}} "
        "--megatron-to-hf-mode raw "
    )

    rollout_args = (
        f"--prompt-data {args.recipe_data_dir}/train.jsonl "
        "--input-key text "
        "--rollout-shuffle "
        f"--num-epoch {args.num_epoch} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--rollout-max-response-len 256 "
        "--rollout-temperature 0.9 "
        "--rollout-skip-special-tokens "
    )

    if args.scorer == "hpsv3":
        reward_pool_config = _REWARD_POOL_CONFIG.format(
            checkpoint_path=f"{args.hpsv3_path}/HPSv3.safetensors",
            base_model_path=args.hpsv3_base_path,
            num_workers=args.scorer_gpus,
        )
        scorer_args = (
            f"--reward-pool-config {command_utils.encode_pseudo_file(reward_pool_config)} --pe-scorer-pool hpsv3 "
        )
    else:
        scorer_args = f"--pe-scorer-urls {args.scorer_urls} "
    reward_args = (
        f"--custom-rm-path {RECIPE}.reward.prompt_enhancer_rm "
        "--pe-generator-model generator --pe-width 512 --pe-height 512 "
        "--pe-num-inference-steps 9 --pe-guidance-scale 0.0 "
        f"{scorer_args}--pe-image-weight {IMAGE_WEIGHTS[args.scorer]} "
        f"{'--pe-seed-per-sample ' if args.seed_per_sample else ''}"
        f"--custom-rollout-log-function-path {RECIPE}.reward.log_rollout_data "
        f"--pe-sample-log {args.run_dir}/samples "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss --kl-loss-coef 0.04 --kl-loss-type k3 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--use-tis "
    )

    optimizer_args = (
        "--optimizer adam --lr 2e-6 --lr-decay-style linear --min-lr 0 --lr-warmup-iters 0 "
        "--weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --clip-grad 1.0 "
    )

    sglang_config = _SGLANG_CONFIG.format(
        policy_gpus=args.policy_gpus, generator_gpus=args.generator_gpus, generator_path=args.generator_path
    )
    sglang_args = (
        f"--sglang-config {command_utils.encode_pseudo_file(sglang_config)} "
        # The image route exists only on Miles' router.
        "--router miles "
        f"--rollout-num-gpus {args.policy_gpus + args.generator_gpus} "
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
        # Qwen2.5 trains without dropout; Megatron's default is 0.1.
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
    )

    seed_args = (
        "" if args.training_seed is None else f"--seed {args.training_seed} --rollout-seed {args.training_seed} "
    )

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.train_gpus} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )

    return (
        f"{ckpt_args} {rollout_args} {reward_args} {grpo_args} {optimizer_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{sglang_args} {perf_args} {seed_args} {misc_args} {args.extra_args} "
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
