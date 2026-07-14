import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.e2e.megatron.test_glm47_flash._common import CaseConfig, execute, prepare

# FIXME: sglang deepep code path bug.
register_cuda_ci(
    est_time=900,
    suite="stage-c-8-gpu-h100",
    labels=["megatron"],
    disabled="Disabled due to sglang deepep code path bug.",
)
register_rocm_ci(
    est_time=900,
    suite="stage-c-8-gpu-mi300x",
    labels=["megatron"],
    disabled="Disabled due to sglang deepep code path bug.",
)

CASE = CaseConfig(
    use_deepep=True,
    num_gpus_per_node=8,
    cp_size=2,
    pp_size=2,
    tp_size=2,
    ep_size=4,
    sglang_ep_size=8,
    # GLM-4.7-Flash has 20 attention heads; non-EP SGLang TP must divide it.
    rollout_num_gpus_per_engine=4,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
