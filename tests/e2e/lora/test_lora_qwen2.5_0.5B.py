"""E2E test for LoRA training with Qwen2.5-0.5B on GSM8K.

Uses the Megatron backend with bridge mode.  Runs a short GRPO training loop
with LoRA enabled (rank=32, all-linear) to validate:
  - LoRA model setup via Bridge
  - LoRA weight sync to SGLang rollout engines
  - LoRA checkpoint save (native + HF PEFT format)
  - Training completes without errors

Requires: 8 GPUs, Qwen2.5-0.5B-Instruct model, GSM8K dataset.
Triggered by label: run-ci-lora
"""

import base64
import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

import miles.utils.external_utils.command_utils as U

register_cuda_ci(est_time=400, suite="stage-c-4-gpu-h200", labels=["lora"])
register_rocm_ci(est_time=300, suite="stage-c-4-gpu-mi350", labels=["lora"])


ENABLE_EVAL = bool(int(os.environ.get("MILES_TEST_ENABLE_EVAL", "1")))

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4


# TEMPORARY diagnostic (revert once the cuDNN BAD_PARAM in fused_attn_bwd is understood).
# Everything here is read-only: it dumps the host/driver/cuDNN/env facts that cannot be
# recovered from an ordinary CI log, so the CI container can be compared field-by-field
# against a devbox reproduction that passes on the same image digest.
_PROBE = r"""
set +e
echo '===== DIAG begin ====='
echo '--- [1] host / driver / kernel ---'
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,persistence_mode \
           --format=csv 2>&1
nvidia-smi | sed -n '1,12p' 2>&1
uname -a; cat /proc/version
echo '--- [2] gpu memory already in use BEFORE this test (prior-test residue) ---'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1
echo '--- [3] cuDNN on disk ---'
ldconfig -p | grep -i cudnn || echo '(none in ldconfig)'
for d in $(echo "${LD_LIBRARY_PATH}" | tr ':' ' ') /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64 \
         /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib; do
  echo "  dir $d -> $(ls "$d"/libcudnn* 2>/dev/null | wc -l) libcudnn file(s)"
done
dpkg -l 2>/dev/null | grep -i cudnn || echo '(no cudnn apt package)'
echo '--- [4] package versions ---'
python -m pip list 2>/dev/null | grep -iE 'cudnn|^torch |transformer.engine|flash|nvidia-cublas'
echo '--- [5] what a torch+TE process actually loads ---'
python - <<'PY' 2>&1
import ctypes, torch
try:
    print("torch", torch.__version__, "| torch.backends.cudnn.version() =", torch.backends.cudnn.version())
except Exception as e:
    print("torch cudnn init FAILED:", e)
import transformer_engine, transformer_engine.pytorch  # noqa: F401
print("transformer_engine", transformer_engine.__version__)
try:
    import flash_attn_interface
    print("flash_attn_interface import OK")
except Exception as e:
    print("flash_attn_interface import FAILED:", type(e).__name__, e)
maps = sorted({l.split()[-1] for l in open("/proc/self/maps") if "libcudnn" in l})
for m in maps:
    print("  mapped:", m)
for m in maps:
    if "libcudnn.so" in m:
        lib = ctypes.CDLL(m); lib.cudnnGetVersion.restype = ctypes.c_size_t
        v = lib.cudnnGetVersion()
        print("  cudnnGetVersion() =", v, "->", "%d.%d.%d" % (v // 10000, (v % 10000) // 100, v % 100))
print("device", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
print("cuda driver/runtime:", torch.version.cuda, torch.cuda_version if hasattr(torch, "cuda_version") else "")
PY
echo '--- [6] model snapshot identity (CI mounts a host cache; a devbox downloads fresh) ---'
cat /root/models/Qwen2.5-0.5B-Instruct/config.json 2>&1
ls -la /root/models/Qwen2.5-0.5B-Instruct/ 2>&1
echo '--- [7] container limits ---'
ulimit -a; df -h /dev/shm
echo '--- [8] full container env ---'
env | sort
echo '===== DIAG end ====='
"""


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command("hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/datasets/gsm8k")
    U.exec_command(f"echo {base64.b64encode(_PROBE.encode()).decode()} | base64 -d | bash")


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " "--megatron-to-hf-mode bridge "

    lora_args = "--lora-rank 32 " "--lora-alpha 32 " "--lora-dropout 0.0 " '--target-modules "all-linear" '

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1.0 "
        "--global-batch-size 32 "
    )

    eval_args = (
        f"{'--eval-interval 2 ' if ENABLE_EVAL else ''}"
        "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
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
        "--lr 1e-5 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.4 "

    ci_args = "--ci-test "

    save_args = "--save-interval 2 " "--save /root/checkpoints/lora-qwen2.5-0.5B-ci "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--calculate-per-token-loss "
        "--use-miles-router "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{lora_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{save_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        # NVTE_DEBUG here is the other half of the TEMPORARY diagnostic above: it makes TE
        # log the available backends and the selected (sub-)backend for every attention
        # call, which is the only way to line the CI run up against a devbox run that
        # takes the same code path and passes.
        extra_env_vars={
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            "NVTE_DEBUG": "1",
            "NVTE_DEBUG_LEVEL": "2",
        },
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
