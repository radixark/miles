from miles.backends.sglang_utils.sglang_scaling_config import SglangScalingConfig
from miles.utils.args.schema import A, Arg, BaseConfig


class ScalingConfig(BaseConfig):
    sglang_scaling: SglangScalingConfig
    actor_num_nodes: A[int, Arg(help="Number of nodes for training actor")] = 1
    actor_num_gpus_per_node: A[int, Arg(help="Number of gpus per node for training actor")] = 8
    critic_num_nodes: A[int | None, Arg(help="Number of nodes for training actor")] = None
    critic_num_gpus_per_node: A[int | None, Arg(help="Number of gpus per node for training actor")] = None
    rollout_num_gpus: A[
        int | None,
        Arg(
            help=(
                "Number of GPUs for inference. Note that when using --colocate, "
                "i.e. the training and the inference engines are on the same gpus, this param will be ignored and will be set as "
                "actor_num_gpus_per_node * actor_num_nodes."
            )
        ),
    ] = None
    eval_num_gpus: A[
        int,
        Arg(
            help=(
                "Number of GPUs for a dedicated eval engine pool. When > 0, eval runs on "
                "its own engines behind its own router, using an independent GPU-delta "
                "receiver pool or loading HF checkpoint snapshots for other transfer modes. "
                "0 keeps shared-engine eval. The dedicated engines' "
                "settings inherit every --sglang-* value; override individually with "
                "--eval-sglang-* (e.g. --eval-sglang-mem-fraction-static 0.9)."
            )
        ),
    ] = 0
    eval_num_gpus_per_engine: A[
        int,
        Arg(help="GPUs per eval engine (TP size), independent of --rollout-num-gpus-per-engine."),
    ] = 1
