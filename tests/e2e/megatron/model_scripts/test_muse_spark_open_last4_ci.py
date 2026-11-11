import os
import sys

from tests.ci.ci_register import register_cuda_ci

import miles.utils.external_utils.command_utils.legacy as U

# The entry point imports scripts/muse_spark from its own directory, as `python scripts/run_muse_spark_64xh200.py` does.
sys.path.insert(0, str(U.repo_base_dir / "scripts"))
from muse_spark import common  # noqa: E402
from run_muse_spark_64xh200 import ScriptArgs  # noqa: E402

# Smoke test of Muse Spark Open on the last-four-layer partial on one node: download and torch_dist conversion,
# then three rollout/train/sync cycles with the --ci-test log-prob and KL checks and the post-update weight check.
# est_time is an upper bound until a /rerun-test run measures it.
register_cuda_ci(
    est_time=2400,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "model-scripts"],
    hardware=["hopper"],
    num_gpus=4,
)


def _args() -> ScriptArgs:
    # --partial selects the partial checkpoint, the one-node layout (TP2/PP2/EP2 on 4 GPUs), a random reward (the
    # partial is not a language model, so a math reward would be all zeros) and --ci-test.
    # The partial's router scores are nearly flat, so the top-k is decided by expert-bias gaps of about 4e-4 and
    # differs between engine and trainer; training replays the engine's routing, and this keeps the --ci-test
    # replay check from failing on that.
    return ScriptArgs(
        partial=True,
        num_gpus_per_node=4,
        num_rollout=3,
        rollout_batch_size=4,
        n_samples_per_prompt=4,
        extra_env_vars="MILES_TEST_R3_THRESHOLD=1",
    )


if __name__ == "__main__":
    args = _args()
    common.prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    common.execute(args, __file__)
