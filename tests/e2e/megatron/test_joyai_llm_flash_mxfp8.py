import os

from scripts.run_joy_ai_llm_flash import ScriptArgs, execute, prepare
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=3600, suite="stage-c-8-gpu-b200", labels=["megatron"], disabled="Blackwell CI not supported")

ARGS = ScriptArgs(
    rollout_mxfp8=True,
    train_mxfp8=True,
    ci_test=True,
    save_checkpoints=False,
    num_rollout=2,
    global_batch_size=32,
    data_pad_size_multiplier=4096,
    log_probs_chunk_size=1024,
)


if __name__ == "__main__":
    os.environ.setdefault("RAY_TMPDIR", "/tmp/ray")
    prepare(ARGS)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(ARGS, wandb_file=__file__)
