import base64
import os

from scripts.run_qwen3_5_35b_a3b_lora import ScriptArgs, _prepare_download, _train
from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils as U

# Smoke test for scripts/run_qwen3_5_35b_a3b_lora.py on the full Qwen3.5-35B-A3B
# checkpoint, like the other Qwen3.5 e2e tests (full rollout -> train -> save loop;
# LoRA targets include the GDN projections). Runs the MoE-expert LoRA matrix —
# {shared-outer + virtual-experts, per-expert + no-virtual-experts} — and every
# combination must pass. Functionality, not accuracy; 8 GPUs (TP2, EP=8).


register_cuda_ci(est_time=3600, suite="stage-c-8-gpu-h100", labels=["model-scripts"])

# (name, experts_shared_outer_loras, virtual_experts_serving)
_CONFIGS = [
    ("shared-outer + virtual-experts", True, True),
    ("per-expert + no-virtual-experts", False, False),
]


def _args(shared_outer: bool, virtual_experts: bool) -> ScriptArgs:
    return ScriptArgs(
        model_name="Qwen3.5-35B-A3B",
        num_nodes=1,
        num_gpus_per_node=8,
        num_rollout=1,
        experts_shared_outer_loras=shared_outer,
        enable_wandb=False,
        extra_args=(
            "--ci-test --ci-disable-logprobs-checker --disable-weights-backuper "
            + ("" if virtual_experts else "--no-sglang-lora-use-virtual-experts ")
        ),
    )


# TEMPORARY diagnostic (revert with the one in tests/e2e/lora/test_lora_qwen2.5_0.5B.py).
# This shard runs on the H100 hosts, so it gives the second host's driver/cuDNN facts to
# compare against both the H200 shard and the devbox reproduction.
_PROBE = r"""
set +e
echo '===== DIAG begin (h100 shard) ====='
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv 2>&1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1
uname -a
ldconfig -p | grep -i cudnn || echo '(none in ldconfig)'
dpkg -l 2>/dev/null | grep -i cudnn || echo '(no cudnn apt package)'
python -m pip list 2>/dev/null | grep -iE 'cudnn|^torch |transformer.engine|flash'
python - <<'PY' 2>&1
import ctypes, torch
try:
    print("torch", torch.__version__, "| cudnn.version() =", torch.backends.cudnn.version())
except Exception as e:
    print("torch cudnn init FAILED:", e)
import transformer_engine.pytorch  # noqa: F401
maps = sorted({l.split()[-1] for l in open("/proc/self/maps") if "libcudnn" in l})
for m in maps:
    print("  mapped:", m)
    if "libcudnn.so" in m:
        lib = ctypes.CDLL(m); lib.cudnnGetVersion.restype = ctypes.c_size_t
        print("  cudnnGetVersion() =", lib.cudnnGetVersion())
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
env | sort
echo '===== DIAG end (h100 shard) ====='
"""


def prepare(args: ScriptArgs):
    _prepare_download(args)
    U.exec_command(f"echo {base64.b64encode(_PROBE.encode()).decode()} | base64 -d | bash")


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(_args(*_CONFIGS[0][1:]))
    for name, shared_outer, virtual_experts in _CONFIGS:
        print(f"[qwen3.5-lora-ci] ===== combo: {name} =====", flush=True)
        # fresh ray/sglang between combos
        U.exec_command("ray stop --force || true; pkill -9 sglang || true; sleep 10")
        execute(_args(shared_outer, virtual_experts))
        print(f"[qwen3.5-lora-ci] ===== combo PASSED: {name} =====", flush=True)
