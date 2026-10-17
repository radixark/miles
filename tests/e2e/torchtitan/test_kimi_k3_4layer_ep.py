"""Two-rollout smoke for K3 BF16 text training and SGLang weight sync.

Numerical parity also requires SGLang's K3 attention-residual score cache to be
refreshed after hot weight updates; successful execution alone does not check it.
"""

from pathlib import Path

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.torchtitan._common import CaseConfig, execute

import miles.utils.external_utils.command_utils.legacy as U

register_cuda_ci(est_time=1800, suite="stage-c-8-gpu-h100", num_gpus=8, labels=["torchtitan"], hardware=["hopper"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")


class KimiK3Case(CaseConfig):
    @property
    def model_dir(self) -> str:
        return f"{super().model_dir}-bf16"


CASE = KimiK3Case(
    model_repo="Pinaster/Kimi-K3-4layer-64experts",
    titan_model_name="kimi_k3",
    titan_model_flavor="hf",
    num_gpus=8,
    ep_size=8,
    rollout_num_gpus_per_engine=4,
    seq_len=4096,
    max_response_len=256,
    mem_fraction_static=0.5,
    extra_args=(
        "--model-name kimi_k3 "
        # extra_args follow the shared defaults; keep the partial-model batch at 16 samples.
        "--rollout-batch-size 4 --n-samples-per-prompt 4 --global-batch-size 16 "
        # Hash-based rewards exercise RL updates even when the pruned model cannot solve GSM8K.
        "--prompt-data /root/datasets/gsm8k/train.parquet --input-key messages "
        "--rm-type deterministic_random "
        "--sglang-ep-size 4 --sglang-moe-runner-backend triton "
        "--sglang-disable-shared-experts-fusion --sglang-disable-radix-cache "
        "--sglang-kv-cache-dtype bfloat16 "
        "--sglang-server-concurrency 8 --sglang-max-running-requests 8 "
        "--sglang-max-mamba-cache-size 32 --sglang-max-total-tokens 32768 "
        # The pinned SGLang K3 fails parity after hot updates with captured decode graphs.
        "--sglang-context-length 4096 --sglang-disable-cuda-graph "
        "--check-weight-update-equal "
        # The text trainer does not own the frozen vision modules.
        "--check-weight-update-skip-list vision_tower. mm_projector. "
        # A four-layer slice has near-uniform logits, outside the convergence checker's entropy bound.
        "--ci-disable-logprobs-checker "
    ),
)


def prepare() -> None:
    source = CASE.model_dir.removesuffix("-bf16")
    U.exec_command_cpu(f"hf download {CASE.model_repo} --local-dir {source}")
    if not Path(f"{CASE.model_dir}/model.safetensors.index.json").exists():
        U.exec_command_gpu(
            f"python {U.repo_base_dir}/tools/convert_mxfp4_to_bf16.py "
            f"--model-dir {source} --save-dir {CASE.model_dir} --device cuda"
        )
    U.hf_download_dataset("zhuzilin/gsm8k")


if __name__ == "__main__":
    prepare()
    execute(CASE, wandb_file=__file__)
