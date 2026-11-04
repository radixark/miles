"""E2E: Qwen3-30B-A3B on the torchtitan backend (the TP 2 / PP 2 / CP 2 / EP 2 layout of
``test_qwen3_30B_A3B_tp_pp_ep_cp.py``) on 8 Blackwell GPUs, with the routed-expert GEMMs on the Cake MoE grouped GEMM
of the miles-ext-kernels submodule: ``--moe-grouped-gemm-backend cake`` (``det``). Needs SM100a / SM103a GPUs and the
checked-out submodule.
"""

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.torchtitan._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=1800, suite="stage-c-8-gpu-b200", labels=["torchtitan"], hardware=["blackwell"], num_gpus=8)

CASE = CaseConfig(
    model_repo="Qwen/Qwen3-30B-A3B",
    titan_model_name="qwen3",
    titan_model_flavor="30B-A3B",
    num_gpus=8,
    tp_size=2,
    pp_size=2,
    cp_size=2,
    ep_size=2,
    seq_len=8192,
    max_response_len=4096,
    mem_fraction_static=0.55,
    num_rollout=4,
    extra_args=(
        "--sglang-moe-runner-backend triton "
        "--use-miles-ext-kernels --moe-grouped-gemm-backend cake --cake-moe-grouped-gemm-variant det "
    ),
)


if __name__ == "__main__":
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
