import os

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.megatron.test_qwen3_30B_A3B._common import CaseConfig, execute, prepare

# FIXME: fix here
register_cuda_ci(
    est_time=1800,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "replay"],
    disabled="Failed due to mismatch between fp8 rollout and bf16 training.",
)

CASE = CaseConfig(
    use_deepep=True,
    use_fp8_rollout=True,
    use_int4_rollout=False,
    use_bridge=False,
    use_r3=True,
    num_gpus_per_node=4,
    cp_size=2,
    pp_size=1,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE, need_fp8=CASE.use_fp8_rollout, need_int4=CASE.use_int4_rollout, all_bridge=CASE.use_bridge)
    execute(CASE, wandb_file=__file__)
