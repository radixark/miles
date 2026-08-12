import os

from tests.ci.ci_register import register_cuda_ci

from miles.utils.external_utils import command_utils

register_cuda_ci(est_time=700, suite="stage-c-4-gpu-h200", labels=["short"])

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_TRAIN_GPUS = 2
NUM_ENGINES = 2
ENGINE_PORTS = [32001, 32002]
ENGINE_HOST = "127.0.0.1"


def compute_train_and_engine_devices(visible_devices: str | None) -> tuple[list[str], list[str]]:
    devices = (visible_devices or "").split(",") if visible_devices else []
    if not devices:
        devices = [str(i) for i in range(NUM_TRAIN_GPUS + NUM_ENGINES)]
    assert len(devices) == NUM_TRAIN_GPUS + NUM_ENGINES, (
        f"this test trains on {NUM_TRAIN_GPUS} gpus and pins one engine to each of {NUM_ENGINES} more, "
        f"but the runner offered {devices}"
    )
    return devices[:NUM_TRAIN_GPUS], devices[NUM_TRAIN_GPUS:]


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_TRAIN_GPUS)


def _external_engines_launch_cmd(engine_devices: list[str]) -> str:
    launches = " ".join(
        f"CUDA_VISIBLE_DEVICES={device} nohup python3 -m sglang.launch_server "
        f"--model-path /root/models/{MODEL_NAME} --host 0.0.0.0 --port {port} "
        f"--tp 1 --mem-fraction-static 0.7 --trust-remote-code "
        f"> /tmp/miles_external_engine_{port}.log 2>&1 &"
        for device, port in zip(engine_devices, ENGINE_PORTS, strict=True)
    )
    probes = " && ".join(f"curl -sf http://{ENGINE_HOST}:{port}/server_info >/dev/null" for port in ENGINE_PORTS)
    wait = (
        f"ok=0; for _ in $(seq 1 120); do {probes} && ok=1 && break; sleep 5; done; "
        f"[ \"$ok\" -eq 1 ] || {{ echo 'external sglang engines failed to start' >&2; "
        f"tail -n 100 /tmp/miles_external_engine_*.log >&2; exit 1; }}"
    )
    return f"{launches} {wait}"


def execute():
    U = command_utils.default_config().create_backend()
    _train_devices, engine_devices = compute_train_and_engine_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/{MODEL_NAME}_torch_dist/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--over-sampling-batch-size 16 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 32 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
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

    external_args = (
        "--rollout-external-engine-addrs "
        + " ".join(f"{ENGINE_HOST}:{port}" for port in ENGINE_PORTS)
        + " --rollout-num-gpus 2 "
        + "--rollout-num-gpus-per-engine 1 "
    )

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_TRAIN_GPUS} "
        "--megatron-to-hf-mode bridge "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{external_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_TRAIN_GPUS,
        megatron_model_type=MODEL_TYPE,
        prepare_cmd={"trainer": _external_engines_launch_cmd(engine_devices)},
        extra_env_vars={"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"},
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
