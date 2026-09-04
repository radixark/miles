"""Shared case harness for the torchtitan backend's end-to-end tests.

One file per case, all of them differing only in a topology and a model, so the
command line is built once here. The fields are named after torchtitan's own
parallelism degrees rather than Megatron's, because that is what the flags are:
the backend copies them straight into torchtitan's ``Parallelism`` config.

Context parallelism is deliberately not a field of the topology check below in
the way the others are: it is internal to the trainer, which shards the sequence
and gathers the logits back before the loss sees them, so the shared training
loop is told cp=1 either way.
"""

import os
from dataclasses import dataclass, field

import miles.utils.external_utils.command_utils as U


@dataclass
class CaseConfig:
    """One end-to-end configuration.

    Topology values are passed explicitly rather than inferred, so a case file
    reads as the configuration it is testing.
    """

    model_repo: str
    titan_model_name: str
    titan_model_flavor: str
    num_gpus: int
    seq_len: int
    max_response_len: int
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    dp_replicate: int = 1
    num_layers: int | None = None
    use_r3: bool = False
    with_ref: bool = False
    colocate: bool = True
    rollout_num_gpus: int | None = None
    rollout_num_gpus_per_engine: int | None = None
    fully_async: bool = False
    mem_fraction_static: float = 0.7
    use_mooncake: bool = False
    num_rollout: int = 2
    global_batch_size: int = 32
    extra_args: str = ""
    extra_env_vars: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.fully_async and self.colocate:
            raise ValueError(
                "fully_async cannot colocate: generation continues while training runs, "
                "so the two need separate GPUs"
            )
        if not self.colocate and self.rollout_num_gpus is None:
            raise ValueError("rollout_num_gpus must be set when colocate is False")
        if self.num_gpus % (self.tp_size * self.pp_size * self.cp_size) != 0:
            raise ValueError(
                "num_gpus must be divisible by tp * pp * cp: "
                f"{self.num_gpus=} {self.tp_size=} {self.pp_size=} {self.cp_size=}"
            )
        if self.seq_len <= self.max_response_len:
            raise ValueError(
                "seq_len must leave room for a prompt ahead of max_response_len: torchtitan "
                f"sizes its rotary tables from it ({self.seq_len=} {self.max_response_len=})"
            )
        rollout_pool = self.num_gpus if self.colocate else self.rollout_num_gpus
        if rollout_pool % self.engine_size != 0:
            raise ValueError(
                f"rollout pool {rollout_pool} is not divisible by the engine size {self.engine_size}"
            )

    @property
    def engine_size(self) -> int:
        if self.rollout_num_gpus_per_engine is not None:
            return self.rollout_num_gpus_per_engine
        return self.num_gpus if self.colocate else self.rollout_num_gpus

    @property
    def model_dir(self) -> str:
        return f"/root/models/{self.model_repo.split('/')[-1]}"


def prepare(case: CaseConfig) -> None:
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download {case.model_repo} --local-dir {case.model_dir}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")


def build_train_args(case: CaseConfig, *, wandb_file: str) -> str:
    """The whole command line for `case`.

    Separate from `execute` so a CPU-only test can read the string back without
    standing up a job.
    """
    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {3000 if U.get_env_enable_infinite_run() else case.num_rollout} "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        f"--rollout-max-response-len {case.max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {case.global_batch_size} "
        "--balance-data "
    )
    if case.fully_async:
        # retract, the default, can deadlock flush_cache under load in
        # fully_async, which is why the shipped recipes pin in_place.
        rollout_args += "--fully-async --pause-generation-mode in_place "

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    if case.use_r3:
        grpo_args += "--use-rollout-routing-replay "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {case.engine_size} "
        "--sglang-decode-log-interval 1000 "
        f"--sglang-mem-fraction-static {case.mem_fraction_static} "
    )

    titan_args = (
        "--train-backend torchtitan "
        f"--titan-model-name {case.titan_model_name} "
        f"--titan-model-flavor {case.titan_model_flavor} "
        f"--titan-seq-len {case.seq_len} "
        f"--titan-tensor-parallel-degree {case.tp_size} "
        f"--titan-pipeline-parallel-degree {case.pp_size} "
        f"--titan-context-parallel-degree {case.cp_size} "
        f"--titan-expert-parallel-degree {case.ep_size} "
        f"--titan-data-parallel-replicate-degree {case.dp_replicate} "
        "--micro-batch-size 1 "
        "--gradient-checkpointing "
        f"--update-weight-buffer-size {512 * 1024 * 1024} "
    )
    if case.num_layers is not None:
        titan_args += f"--titan-num-layers {case.num_layers} "

    misc_args = (
        f"--hf-checkpoint {case.model_dir} "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {case.num_gpus} "
    )
    misc_args += "--colocate " if case.colocate else f"--rollout-num-gpus {case.rollout_num_gpus} "
    if case.with_ref:
        misc_args += f"--ref-load {case.model_dir} --use-kl-loss "
    if case.use_mooncake:
        # The p2p weight transfer writes into a mooncake object store the
        # engines read from; the master is started before the job is submitted.
        misc_args += U.get_mooncake_object_store_args()

    ci_args = "--ci-test --ci-disable-kl-checker "

    return (
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{sglang_args} "
        f"{U.get_default_wandb_args(wandb_file)} "
        f"{titan_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{case.extra_args} "
    )


def execute(case: CaseConfig, *, wandb_file: str) -> None:
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    U.execute_train(
        train_args=build_train_args(case, wandb_file=wandb_file),
        num_gpus_per_node=case.num_gpus + (0 if case.colocate else case.rollout_num_gpus),
        megatron_model_type=None,
        train_script="train_async.py" if case.fully_async else "train.py",
        before_ray_job_submit=U.start_mooncake_master if case.use_mooncake else None,
        extra_env_vars=dict(case.extra_env_vars),
    )
