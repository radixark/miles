"""Inkling-Small (6-layer slice) LoRA on the AVQA audio recipe, in token space.

Runs `scripts/run_inkling.py --task avqa_audio --train-mode lora` as the AVQA example does, cut to two
rollouts of 4 prompts on one 4-GPU engine. The data is one of AVQA's four clip directories (2.7 GB, 8.5K clips)
through the example's own `prepare_avqa_audio.py`, which builds the prompts from the clips it finds.
Functionality, not accuracy.
"""

from scripts.run_inkling import _MODEL_REGISTRY, ScriptArgs, _train
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

# The first run on a runner fetches and converts the 8.5K clips, about 17 minutes without a Hub token; later runs
# reuse them from the runner's dataset volume.
register_cuda_ci(
    est_time=2000,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "multimodal", "lora"],
    hardware=["hopper"],
    num_gpus=4,
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")
register_ci_gate(metric_key="perf/actor_train_mfu")

_MODEL_ORG = "pb09204048"
_AVQA_CLIP_DIR = "VGG10000"


def _args() -> ScriptArgs:
    return ScriptArgs.from_env(
        model_name="Inkling-Small-6layer",
        train_mode="lora",
        task="avqa_audio",
        num_nodes=1,
        num_gpus_per_node=4,
        rollout_num_gpus_per_engine=4,
        num_rollout=2,
        rollout_batch_size=4,
        n_samples_per_prompt=4,
        global_batch_size=16,
        rollout_max_response_len=512,
        sglang_context_length=4096,
        extra_args=(
            "--ci-test --no-enable-sample-ownership-checker "
            "--ci-disable-kl-checker "
            # frozen towers and the engine-derived adapter buffers never match the snapshot
            "--check-weight-update-skip-list visual. audio. ._w1_delta ._a_cat "
            "--ci-disable-logprobs-checker "
            "--check-lora-weight-equal "
        ),
    )


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download {_MODEL_ORG}/{args.model_name} --local-dir {args.hf_checkpoint}")
    # The question file and one clip directory; `--include` takes one pattern.
    avqa_dir = f"{args.data_dir}/avqa"
    U.exec_command_cpu(f"hf download --repo-type dataset Joysw909/AVQA train_r1aqa_line.json --local-dir {avqa_dir}")
    U.exec_command_cpu(
        f'hf download --repo-type dataset Joysw909/AVQA --include "{_AVQA_CLIP_DIR}/*" --local-dir {avqa_dir}'
    )
    U.exec_command_cpu(f"python3 examples/avqa_audio/prepare_avqa_audio.py {args.data_dir}")
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=_MODEL_REGISTRY[args.model_name],
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    execute(args)
