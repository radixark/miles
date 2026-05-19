import os

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.megatron.test_glm47_flash._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=900, suite="stage-c-4-gpu-h200", labels=["megatron"])

CASE = CaseConfig(use_deepep=False)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare()
    execute(CASE, wandb_file=__file__)
