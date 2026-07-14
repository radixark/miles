import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.e2e.megatron.test_qwen3_30B_A3B._common import CaseConfig, execute, prepare

# Limited by host memory
register_cuda_ci(est_time=900, suite="stage-c-8-gpu-h100", labels=["megatron"])
register_rocm_ci(est_time=900, suite="stage-c-8-gpu-mi300x", labels=["megatron"])

CASE = CaseConfig(
    use_deepep=True,
    use_fp8_rollout=True,
    use_int4_rollout=False,
    use_bridge=True,
    use_r3=False,
    num_gpus_per_node=8,
    cp_size=1,
    pp_size=2,
    tp_size=4,
    ep_size=4,
    rollout_num_gpus_per_engine=8,
    sglang_ep_size=8,
    max_tokens_per_gpu=2048,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE, need_fp8=CASE.use_fp8_rollout, need_int4=CASE.use_int4_rollout, all_bridge=CASE.use_bridge)
    execute(CASE, wandb_file=__file__)
