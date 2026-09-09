"""Pure SFT (--debug-train-only) with a snapshot-eval fleet.

No rollout engines exist in this mode; the only engine is the eval fleet, which loads
the HF snapshots the trainer exports. The metric checker demands exactly three eval
points (pre-train on --hf-checkpoint, rollout 1, rollout 3), so a train-only
short-circuit anywhere on the engine or eval path fails the test.
"""

import os

import pandas as pd
from tests.ci.ci_register import register_cuda_ci
from transformers import AutoTokenizer

import miles.utils.external_utils.command_utils as U

register_cuda_ci(est_time=600, suite="stage-c-2-gpu-h200", labels=["short", "eval", "megatron"])

MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 2
SFT_DATA = "/root/datasets/gsm8k_sft_qwen3/train.parquet"
EVAL_DATA = "/root/datasets/gsm8k_sft_qwen3/eval.parquet"


def _write_datasets():
    os.makedirs(os.path.dirname(SFT_DATA), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(f"/root/models/{MODEL_NAME}")

    train = pd.read_parquet("/root/datasets/gsm8k/train.parquet")
    messages = [
        [*(dict(m) for m in row["messages"]), {"role": "assistant", "content": f"\\boxed{{{row['label']}}}"}]
        for row in train.to_dict("records")
    ]
    pd.DataFrame({"messages": messages, "label": train["label"]}).to_parquet(SFT_DATA)

    test = pd.read_parquet("/root/datasets/gsm8k/test.parquet").head(256)
    prompts = [
        tokenizer.apply_chat_template(
            [dict(m) for m in row["messages"]], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for row in test.to_dict("records")
    ]
    pd.DataFrame({"prompt": prompts, "label": test["label"]}).to_parquet(EVAL_DATA)


def prepare():
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    _write_datasets()


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} " f"--ref-load /root/models/{MODEL_NAME} "

    sft_args = (
        "--debug-train-only "
        "--rollout-function-path miles.rollout.sft_rollout.generate_rollout "
        f"--prompt-data {SFT_DATA} "
        "--input-key messages "
        "--rollout-shuffle "
        "--num-rollout 4 "
        "--rollout-batch-size 16 "
        "--global-batch-size 16 "
        "--rollout-max-response-len 512 "
        "--loss-type sft_loss "
        "--calculate-per-token-loss "
        "--disable-compute-advantages-and-returns "
    )

    eval_args = (
        "--eval-interval 2 "
        "--eval-function-path miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn "
        f"--eval-prompt-data gsm8k {EVAL_DATA} "
        "--eval-input-key prompt "
        "--eval-label-key label "
        "--rm-type math "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 512 "
        "--eval-top-k 1 "
        "--eval-num-gpus 1 "
        "--eval-num-gpus-per-engine 1 "
        "--eval-hf-dir /dev/shm/miles_e2e_sft_eval_hf "
        "--eval-keep-snapshots 2 "
        "--sglang-mem-fraction-static 0.65 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    ci_args = (
        "--ci-test --ci-metric-checker-key eval/gsm8k --ci-metric-checker-threshold 0.3 "
        "--ci-metric-checker-expect-num 3 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 1 "
        "--megatron-to-hf-mode bridge "
    )

    train_args = (
        f"{ckpt_args} "
        f"{sft_args} "
        f"{optimizer_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py",
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
