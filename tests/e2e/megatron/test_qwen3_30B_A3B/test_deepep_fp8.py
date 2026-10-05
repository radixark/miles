import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_30B_A3B._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=1500, suite="stage-c-4-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    use_deepep=True,
    use_fp8_rollout=True,
    use_int4_rollout=False,
    use_bridge=False,
    use_r3=False,
    # tp2/etp2 x dense dp2/ep2 on 4 GPUs: the Megatron DeepEP group is etp2 x ep2 = 4 ranks.
    # SGLang runs DP attention (dp4, attention TP1) with DeepEP EP4 = engine TP; 512 running
    # requests / dp4 = 128 tokens per rank, the DeepEP low-latency cap.
    num_gpus_per_node=4,
    cp_size=1,
    pp_size=1,
    tp_size=2,
    ep_size=2,
    etp_size=2,
    rollout_num_gpus_per_engine=4,
    sglang_ep_size=4,
    sglang_dp_size=4,
    sglang_enable_dp_attention=True,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE, need_fp8=CASE.use_fp8_rollout, need_int4=CASE.use_int4_rollout, all_bridge=CASE.use_bridge)
    execute(CASE, wandb_file=__file__)
