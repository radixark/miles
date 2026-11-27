"""The two workloads of the output store replay profile (README.md in this folder). No CI job runs them.

    python -m tests.manual.output_store_replay.launch {indexer,r3} {prepare,execute}

indexer: GLM-5 4-layer, indexer replay, trainer TP2 PP2 EP2 CP1 colocated with the engines on 4 GPUs.
r3: GLM-4.7-Flash, routing replay, trainer TP1 CP4 PP1 EP4 on 4 GPUs and one FP8 engine on each of 4 more.

Env: PROFILE_NUM_ROLLOUT (default 3), MOONCAKE_PROTOCOL (default tcp; rdma also needs MOONCAKE_DEVICE).
"""

import json
import os
import shlex
import sys

from scripts.run_glm5_744b_a40b import (
    ScriptArgs,
    _execute_train,
    _prepare_download,
    _prepare_megatron_ckpt,
    _validate_glm_checkpoint,
)
from tests.e2e.megatron.test_glm47_flash import _common as glm47_flash

from miles.utils.object_store_config import compute_mooncake_init_kwargs_vanilla

NUM_ROLLOUT = int(os.environ.get("PROFILE_NUM_ROLLOUT", "3"))


def _store_init_kwargs(global_segment_size: str) -> str:
    init_kwargs = compute_mooncake_init_kwargs_vanilla() | {"global_segment_size": global_segment_size}
    init_kwargs["protocol"] = os.environ.get("MOONCAKE_PROTOCOL", init_kwargs["protocol"])
    if init_kwargs["protocol"] == "rdma":
        init_kwargs["device_name"] = os.environ["MOONCAKE_DEVICE"]
    return f"--mooncake-store-init-kwargs {shlex.quote(json.dumps(init_kwargs))} "


def _indexer() -> ScriptArgs:
    # 64 prompts x 4 samples of up to 4096 tokens: about 35 GB of top-k rows per trainer rank and step.
    return ScriptArgs.from_env(
        hardware="H200",
        model_name="GLM-5_4layer",
        num_nodes=1,
        num_gpus_per_node=4,
        num_rollout=NUM_ROLLOUT,
        rollout_batch_size=64,
        n_samples_per_prompt=4,
        global_batch_size=256,
        enable_optimizer_offload=True,
        extra_args=(
            "--ci-test "
            "--ci-disable-logprobs-checker "
            "--use-rollout-indexer-replay "
            "--use-output-store --object-store-backend mooncake "
            f"{_store_init_kwargs('64gb')}"
            "--skip-actor-forward-only "
            "--rollout-max-response-len 4096 "
            "--sglang-max-total-tokens 1900000 "
            "--tensor-model-parallel-size 2 "
            "--pipeline-model-parallel-size 2 "
            "--expert-model-parallel-size 2 "
        ),
    )


def _r3() -> glm47_flash.CaseConfig:
    # 32 prompts x 16 samples of up to 16384 tokens: about 3.7 GB of routed experts per DP shard and step.
    return glm47_flash.CaseConfig(
        num_gpus_per_node=4,
        cp_size=4,
        pp_size=1,
        tp_size=1,
        ep_size=4,
        rollout_num_gpus_per_engine=1,
        use_fp8_rollout=True,
        use_output_store=True,
        max_tokens_per_gpu=16384,
        rollout_max_response_len=16384,
        colocate=False,
        rollout_num_gpus=4,
        update_weight_transfer_mode="broadcast",
        num_rollout=NUM_ROLLOUT,
        extra_args=(
            "--rollout-batch-size 32 --n-samples-per-prompt 16 --global-batch-size 512 "
            # 84 full-length requests fit the KV pool that 0.78 of an H200 leaves.
            "--sglang-mem-fraction-static 0.78 --sglang-max-running-requests 84 "
            f"{_store_init_kwargs('32gb')}"
        ),
    )


def main(workload: str, step: str) -> None:
    assert workload in ("indexer", "r3") and step in ("prepare", "execute"), (workload, step)
    if step == "execute":
        for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(proxy_var, None)
    if workload == "r3":
        case = _r3()
        glm47_flash.prepare(case) if step == "prepare" else glm47_flash.execute(case, wandb_file=__file__)
        return
    args = _indexer()
    if step == "execute":
        _execute_train(args)
        return
    args.create_backend().exec_command_cpu(f"mkdir -p {args.output_dir}")
    _prepare_download(args)
    _validate_glm_checkpoint(args)
    _prepare_megatron_ckpt(args)


if __name__ == "__main__":
    main(*sys.argv[1:])
