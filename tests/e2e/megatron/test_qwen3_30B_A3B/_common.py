import os
from dataclasses import dataclass

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-30B-A3B"
MODEL_TYPE = "qwen3-30B-A3B"

TIGHT_HOST_MEMORY = bool(int(os.environ.get("MILES_TEST_TIGHT_HOST_MEMORY", "1")))


@dataclass
class CaseConfig:
    num_gpus_per_node: int
    cp_size: int
    pp_size: int
    tp_size: int = None
    ep_size: int = None
    sglang_ep_size: int = None
    sglang_dp_size: int = None
    sglang_enable_dp_attention: bool = False
    use_deepep: bool = False
    use_fp8_rollout: bool = False
    use_int4_rollout: bool = False
    use_bridge: bool = False
    use_r3: bool = False
    max_tokens_per_gpu: int = 8192
    colocate: bool = True
    rollout_num_gpus: int = None
    rollout_num_gpus_per_engine: int = None
    update_weight_transfer_mode: str = None

    def __post_init__(self):
        if self.num_gpus_per_node % (self.cp_size * self.pp_size) != 0:
            raise ValueError(
                "num_gpus_per_node must be divisible by cp_size * pp_size: "
                f"{self.num_gpus_per_node=} {self.cp_size=} {self.pp_size=}"
            )
        if self.tp_size is None:
            self.tp_size = self.num_gpus_per_node // self.cp_size // self.pp_size
        if self.ep_size is None:
            self.ep_size = self.num_gpus_per_node // self.pp_size
        # Align with arguments.py: colocated rollout reuses the actor GPU pool, so
        # rollout_num_gpus resolves to num_gpus_per_node; disaggregated rollout keeps
        # its own standalone rollout_num_gpus.
        if self.colocate:
            self.rollout_num_gpus = self.num_gpus_per_node
        elif self.rollout_num_gpus is None:
            raise ValueError("rollout_num_gpus must be set when colocate is False")
        # EP spans the whole rollout pool so MoE experts shard across it, while
        # attention stays DP within each engine (rollout_num_gpus_per_engine).
        if self.rollout_num_gpus_per_engine is None:
            self.rollout_num_gpus_per_engine = 1 if self.use_int4_rollout else self.rollout_num_gpus
        if self.sglang_ep_size is None and self.use_deepep:
            self.sglang_ep_size = self.rollout_num_gpus
        if self.rollout_num_gpus % self.rollout_num_gpus_per_engine != 0:
            raise ValueError(
                "rollout_num_gpus must be divisible by rollout_num_gpus_per_engine: "
                f"{self.rollout_num_gpus=} {self.rollout_num_gpus_per_engine=}"
            )
        if self.update_weight_transfer_mode is not None:
            assert self.update_weight_transfer_mode == "broadcast"


def prepare(case: CaseConfig, *, need_fp8: bool, need_int4: bool, all_bridge: bool) -> None:
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command("hf download Qwen/Qwen3-30B-A3B --local-dir /root/models/Qwen3-30B-A3B")
    if need_fp8:
        U.exec_command("hf download Qwen/Qwen3-30B-A3B-FP8 --local-dir /root/models/Qwen3-30B-A3B-FP8")
    if need_int4:
        U.exec_command(
            f"python tools/convert_hf_to_int4_direct.py "
            f"--model-dir /root/models/{MODEL_NAME} "
            f"--save-dir /root/models/{MODEL_NAME}-INT4"
        )
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")

    # Bridge mode reads the HF checkpoint directly; non-bridge variants need
    # the torch_dist conversion. With one case per file, "all_bridge" reduces
    # to the single case being a bridge case.
    if not all_bridge:
        U.convert_checkpoint(
            model_name=MODEL_NAME,
            megatron_model_type=MODEL_TYPE,
            num_gpus_per_node=case.num_gpus_per_node,
        )


def build_train_args(case: CaseConfig, *, wandb_file: str) -> str:
    """Build the train_args string for `case`.

    Split out from `execute()` so the CPU-only golden test can inspect the
    string without monkeypatching `execute_train`.
    """
    if case.use_int4_rollout and case.use_fp8_rollout:
        raise ValueError("use_int4_rollout and use_fp8_rollout are mutually exclusive")

    enable_eval = bool(int(os.environ.get("MILES_TEST_ENABLE_EVAL", "0")))

    ref_load = f"/root/models/{MODEL_NAME}" if case.use_bridge else f"/root/{MODEL_NAME}_torch_dist"
    if case.use_int4_rollout:
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}-INT4/ " f"--ref-load {ref_load} "
    elif case.use_fp8_rollout:
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}-FP8 " f"--ref-load {ref_load} "
    else:
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} " f"--ref-load {ref_load} "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
        "--balance-data "
    )

    eval_args = (
        f"{'--eval-interval 20 ' if enable_eval else ''}"
        "--eval-prompt-data aime24 /root/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 16384 "
        "--eval-top-k 1 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {case.tp_size} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {case.pp_size} "
        f"--context-parallel-size {case.cp_size} "
        f"--expert-model-parallel-size {case.ep_size} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {case.max_tokens_per_gpu} "
    )

    if TIGHT_HOST_MEMORY:
        perf_args += "--exp-avg-dtype fp16 "
        perf_args += "--exp-avg-sq-dtype fp16 "
        perf_args += "--main-params-dtype fp16 "

    # r3 path uses --use-rollout-routing-replay; non-r3 uses --use-routing-replay.
    routing_flag = "--use-rollout-routing-replay" if case.use_r3 else "--use-routing-replay"
    grpo_args = (
        "--advantage-estimator gspo "
        f"{'' if case.use_bridge else '--use-kl-loss '}"
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
        "--use-tis "
        f"{routing_flag} "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {case.rollout_num_gpus_per_engine} "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-max-running-requests 512 "
        "--sglang-enable-metrics "
    )

    if case.use_deepep:
        sglang_args += "--sglang-moe-a2a-backend deepep --sglang-deepep-mode auto "
    if case.sglang_ep_size is not None:
        sglang_args += f"--sglang-expert-parallel-size {case.sglang_ep_size} "
    if case.sglang_dp_size is not None:
        sglang_args += f"--sglang-data-parallel-size {case.sglang_dp_size} "
    if case.sglang_enable_dp_attention:
        sglang_args += "--sglang-enable-dp-attention "

    ci_args = "--ci-test "
    if case.use_fp8_rollout or case.use_int4_rollout:
        ci_args += "--check-weight-update-allow-quant-error "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {case.num_gpus_per_node} "
    )
    if case.colocate:
        misc_args += "--colocate "
    else:
        misc_args += f"--rollout-num-gpus {case.rollout_num_gpus} "

    if case.update_weight_transfer_mode is not None:
        misc_args += f"--update-weight-transfer-mode {case.update_weight_transfer_mode} "

    if case.use_bridge:
        misc_args += "--megatron-to-hf-mode bridge "

    if case.use_deepep:
        misc_args += "--moe-token-dispatcher-type flex --moe-enable-deepep "
    else:
        misc_args += "--moe-token-dispatcher-type alltoall "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(wandb_file)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )
    return train_args


def execute(case: CaseConfig, *, wandb_file: str) -> None:
    train_args = build_train_args(case, wandb_file=wandb_file)

    extra_env_vars = {"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"}
    if case.use_int4_rollout:
        extra_env_vars |= {
            "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
            "OPEN_TRAINING_INT4_GROUP_SIZE": "128",
        }

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=case.num_gpus_per_node + (0 if case.colocate else case.rollout_num_gpus),
        megatron_model_type=MODEL_TYPE,
        extra_env_vars=extra_env_vars,
    )
