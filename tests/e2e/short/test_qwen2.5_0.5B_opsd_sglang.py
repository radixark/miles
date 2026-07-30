import os

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.sglang.utils.sglang_server import SGLangServer, start_sglang_server

import miles.utils.external_utils.command_utils as U

register_cuda_ci(est_time=500, suite="stage-c-2-gpu-h200", labels=["short"])

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
NUM_GPUS = 2
NUM_TRAIN_GPUS = 1
TEACHER_HOST = "127.0.0.1"
TEACHER_PORT = 13141


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(NUM_GPUS))).split(",")
    assert len(visible_gpus) >= NUM_GPUS
    teacher_gpu = visible_gpus[NUM_TRAIN_GPUS]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpus[:NUM_TRAIN_GPUS])
    teacher_server: SGLangServer | None = None

    def launch_teacher():
        nonlocal teacher_server
        train_gpus = os.environ["CUDA_VISIBLE_DEVICES"]
        os.environ["CUDA_VISIBLE_DEVICES"] = teacher_gpu
        try:
            teacher_server = start_sglang_server(
                model_path=f"/root/models/{MODEL_NAME}",
                host=TEACHER_HOST,
                port=TEACHER_PORT,
                enable_deterministic_inference=False,
                extra_args=["--tp", "1", "--mem-fraction-static", "0.6"],
            )
        finally:
            os.environ["CUDA_VISIBLE_DEVICES"] = train_gpus

    train_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 1 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 1 "
        "--global-batch-size 4 "
        "--loss-type opsd_loss "
        "--disable-compute-advantages-and-returns "
        "--opsd-type sglang "
        "--opsd-teacher-top-k 8 "
        "--opsd-pointwise-kl-clip 0.05 "
        f"--opsd-teacher-url http://{TEACHER_HOST}:{TEACHER_PORT}/generate "
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--eval-interval 1 "
        "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 512 "
        "--eval-top-k 1 "
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        "--use-miles-router "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_TRAIN_GPUS} "
        "--colocate "
        "--train-backend fsdp "
        "--ci-test "
        "--ci-disable-kl-checker "
        f"{U.get_default_wandb_args(__file__)} "
    )

    try:
        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=NUM_TRAIN_GPUS,
            megatron_model_type=None,
            before_ray_job_submit=launch_teacher,
            extra_env_vars={"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"},
        )
    finally:
        if teacher_server is not None:
            teacher_server.stop()


if __name__ == "__main__":
    prepare()
    for proxy_variable in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_variable, None)
    execute()
