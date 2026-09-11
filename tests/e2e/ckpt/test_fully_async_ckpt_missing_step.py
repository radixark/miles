from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ckpt.test_fully_async_ckpt import run

register_cuda_ci(est_time=1200, suite="stage-c-8-gpu-h100", labels=["ckpt", "fully-async"])


if __name__ == "__main__":
    run(missing_training_step=True)
