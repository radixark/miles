import unittest
import pytest
import torch
import miles.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-3B"
MODEL_TYPE = "qwen2.5-3B"
NUM_GPUS = 2
BASE_DIR = "/root"


@pytest.mark.skipif(torch.cuda.device_count() < NUM_GPUS, reason=f"Need at least {NUM_GPUS} GPUs")
class TestWeightUpdateCorrectness(unittest.TestCase):
    def setUp(self):
        U.exec_command(f"mkdir -p {BASE_DIR}/models {BASE_DIR}/datasets")
        U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {BASE_DIR}/models/{MODEL_NAME}")
        U.exec_command(
            f"hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir {BASE_DIR}/datasets/dapo-math-17k"
        )

        U.convert_checkpoint(
            model_name=MODEL_NAME,
            megatron_model_type=MODEL_TYPE,
            num_gpus_per_node=NUM_GPUS,
            dir_dst=f"{BASE_DIR}/models",
            hf_checkpoint=f"{BASE_DIR}/models/{MODEL_NAME}",
        )

    def test_weight_correctness(self):
        ckpt_args = (
            f"--hf-checkpoint {BASE_DIR}/models/{MODEL_NAME}/ --ref-load {BASE_DIR}/models/{MODEL_NAME}_torch_dist "
        )

        rollout_args = (
            f"--prompt-data {BASE_DIR}/datasets/dapo-math-17k/dapo-math-17k.jsonl "
            "--input-key prompt "
            "--label-key label "
            "--apply-chat-template "
            "--num-rollout 2 "
            "--rollout-batch-size 4 "
            "--n-samples-per-prompt 1 "
            "--rollout-max-response-len 128 "
            "--global-batch-size 4 "
        )

        ppo_args = "--advantage-estimator grpo --rm-type math "
        sglang_args = (
            f"--rollout-num-gpus-per-engine {NUM_GPUS} --rollout-num-gpus {NUM_GPUS} --sglang-mem-fraction-static 0.6 "
        )
        checker_args = "--enable-weight-checker "

        misc_args = (
            "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {NUM_GPUS} " "--colocate " "--update-weights-interval 1"
        )

        train_args = (
            f"{ckpt_args} " f"{rollout_args} " f"{ppo_args} " f"{sglang_args} " f"{checker_args} " f"{misc_args}"
        )

        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=MODEL_TYPE,
        )


if __name__ == "__main__":
    unittest.main()
